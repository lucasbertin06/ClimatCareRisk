"""Leakage-safe commune-level training for the real Var hazard closure."""

from __future__ import annotations

import json
import shutil
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import requests
from scipy.interpolate import RegularGridInterpolator

from wildfire.metrics import (
    average_precision,
    binary_nll,
    brier_score,
    expected_calibration_error,
)
from wildfire.real_scenario import (
    _expert_hazard_parameters,
    _projected_grid,
    _sample_coordinates,
    _target_lonlat,
    load_real_config,
    sha256_file,
)


def _training_sample_coordinates(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Use four archive nodes; reanalysis is coarser than the live forecast grid."""
    lats, lons = _sample_coordinates(config)
    return lats[[0, -1]], lons[[0, -1]]


def _download_training_weather(
    config: dict[str, Any], destination: Path, start_day: date, end_day: date
) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    lats, lons = _training_sample_coordinates(config)
    locations = [(lat, lon) for lat in lats for lon in lons]
    common = {
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "daily": (
            "temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_max,"
            "precipitation_sum,et0_fao_evapotranspiration"
        ),
        "timezone": "Europe/Paris",
    }
    payload: list[dict[str, Any]] = []
    session = requests.Session()
    for start in range(0, len(locations), 5):
        batch = locations[start : start + 5]
        params = {
            **common,
            "latitude": ",".join(f"{lat:.5f}" for lat, _ in batch),
            "longitude": ",".join(f"{lon:.5f}" for _, lon in batch),
        }
        response = None
        for attempt in range(4):
            response = session.get(
                config["sources"]["open_meteo_archive"],
                params=params,
                timeout=300,
                headers={"User-Agent": "IGNIS-Tesseract-Hackathon/1.0"},
            )
            if response.ok:
                break
            if attempt < 3:
                time.sleep(2**attempt)
        if response is None:
            raise RuntimeError("Open-Meteo returned no response")
        response.raise_for_status()
        batch_payload = response.json()
        payload.extend(batch_payload if isinstance(batch_payload, list) else [batch_payload])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return destination


def _interpolated_daily_weather(
    path: Path,
    config: dict[str, Any],
    commune_index: np.ndarray,
    commune_ids: np.ndarray,
) -> tuple[list[date], dict[str, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    lats, lons = _training_sample_coordinates(config)
    rows, columns = len(lats), len(lons)
    if len(payload) != rows * columns:
        raise ValueError("training weather response has an unexpected location count")
    days = [date.fromisoformat(value) for value in payload[0]["daily"]["time"]]
    variables = {
        "temperature": "temperature_2m_max",
        "humidity": "relative_humidity_2m_min",
        "wind": "wind_speed_10m_max",
        "precipitation": "precipitation_sum",
        "evapotranspiration": "et0_fao_evapotranspiration",
    }
    samples = {
        name: np.zeros((len(days), rows, columns), dtype=np.float32)
        for name in variables
    }
    for location_index, location in enumerate(payload):
        row, column = divmod(location_index, columns)
        for name, source_name in variables.items():
            samples[name][:, row, column] = np.asarray(
                location["daily"][source_name], dtype=np.float32
            )
    grid, _ = _projected_grid(config)
    lon_grid, lat_grid = _target_lonlat(grid)
    target_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    outputs = {
        name: np.zeros((len(days), len(commune_ids)), dtype=np.float32)
        for name in variables
    }
    masks = [commune_index == index for index in commune_ids]
    for day_index in range(len(days)):
        for name, values in samples.items():
            interpolator = RegularGridInterpolator(
                (lats, lons), values[day_index], bounds_error=False, fill_value=None
            )
            field = interpolator(target_points).reshape(grid.height, grid.width)
            outputs[name][day_index] = np.asarray(
                [float(field[mask].mean()) for mask in masks], dtype=np.float32
            )
    return days, outputs


def _event_targets(
    events_path: Path,
    days: list[date],
    commune_codes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    code_to_index = {code: index for index, code in enumerate(commune_codes)}
    day_to_index = {value: index for index, value in enumerate(days)}
    count = np.zeros((len(days), len(commune_codes)), dtype=np.float32)
    area = np.zeros_like(count)
    for event in json.loads(events_path.read_text(encoding="utf-8")):
        code = str(event.get("commune_code") or "")
        event_day = date.fromisoformat(str(event["event_date"])[:10])
        if code not in code_to_index or event_day not in day_to_index:
            continue
        row = day_to_index[event_day]
        column = code_to_index[code]
        count[row, column] += 1.0
        area[row, column] += float(event.get("burned_area") or 0.0)
    return count, area


def _commune_static_features(
    arrays: dict[str, np.ndarray], commune_ids: np.ndarray
) -> np.ndarray:
    commune_index = arrays["commune_index"]
    latest = arrays["features"][-1]
    return np.asarray(
        [latest[commune_index == index].mean(axis=0) for index in commune_ids],
        dtype=np.float32,
    )


def build_training_windows(
    config_path: Path,
    scenario_path: Path,
    *,
    events_path: Path | None = None,
    start_day: date = date(2025, 1, 1),
    end_day: date = date(2025, 12, 31),
) -> dict[str, Any]:
    """Build all-commune temporal windows without event-selected spatial masks."""
    config = load_real_config(config_path)
    with np.load(scenario_path, allow_pickle=False) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    commune_ids = np.unique(arrays["commune_index"])
    commune_ids = commune_ids[commune_ids > 0]
    codes = [str(arrays["commune_code"][arrays["commune_index"] == index][0]) for index in commune_ids]
    weather_path = config["raw_dir"] / (
        f"open-meteo-training-{start_day.isoformat()}-{end_day.isoformat()}-v4.json"
    )
    _download_training_weather(config, weather_path, start_day, end_day)
    days, weather = _interpolated_daily_weather(
        weather_path, config, arrays["commune_index"], commune_ids
    )
    event_count, event_area = _event_targets(
        events_path or config["bdiff_events"], days, codes
    )
    event_counts_by_year = {
        str(year): int(
            event_count[
                np.asarray([current.year == year for current in days], dtype=bool)
            ].sum()
        )
        for year in sorted({current.year for current in days})
    }
    static = _commune_static_features(arrays, commune_ids)
    features = np.broadcast_to(
        static[None, ...], (len(days), *static.shape)
    ).copy()
    features[..., 5] = np.clip((weather["temperature"] - 18.0) / 24.0, 0.0, 1.0)
    features[..., 6] = np.clip((70.0 - weather["humidity"]) / 60.0, 0.0, 1.0)
    features[..., 7] = np.clip(weather["wind"] / 65.0, 0.0, 1.0)
    features[..., 8] = np.exp(-weather["precipitation"] / 6.0)
    # The daily archive does not expose soil moisture. A leaky water-balance
    # state is a transparent, low-cost proxy for the production soil-dryness
    # channel and avoids downloading 26,000 hourly values per location/year.
    water_deficit = np.zeros_like(weather["precipitation"], dtype=np.float32)
    for day_index in range(1, len(days)):
        water_deficit[day_index] = np.clip(
            0.92 * water_deficit[day_index - 1]
            + weather["evapotranspiration"][day_index]
            - weather["precipitation"][day_index],
            0.0,
            20.0,
        )
    features[..., 9] = water_deficit / 20.0
    # The scenario's long-run density currently includes all 2025 BDIFF events.
    # Feeding it into a 2025 holdout would encode future target locations. Keep
    # the descriptive layer in the scenario, but force the learned closure to
    # ignore it until a strictly pre-2025 archive is available.
    features[..., 10] = 0.0
    features[..., 11] = static[None, :, 11]
    horizons = (1, 2, 3)
    ignition = np.zeros((len(days), len(codes), len(horizons)), dtype=np.float32)
    growth = np.zeros_like(ignition)
    burned_area = np.zeros_like(ignition)
    for day_index in range(len(days)):
        for horizon_index, horizon in enumerate(horizons):
            future = slice(day_index + 1, min(len(days), day_index + horizon + 1))
            future_count = event_count[future].sum(axis=0)
            future_area = event_area[future].sum(axis=0)
            ignition[day_index, :, horizon_index] = future_count > 0
            growth[day_index, :, horizon_index] = future_area >= 1.0
            burned_area[day_index, :, horizon_index] = future_area
    history = int(config["history_days"])
    valid_indices = np.arange(history - 1, len(days) - max(horizons), dtype=np.int32)
    windows = np.stack(
        [features[index - history + 1 : index + 1] for index in valid_indices]
    )
    target_ignition = ignition[valid_indices]
    target_growth = growth[valid_indices]
    target_area = burned_area[valid_indices]
    target_days = np.asarray([days[index].isoformat() for index in valid_indices])
    return {
        "windows": windows.astype(np.float32),
        "ignition": target_ignition,
        "growth": target_growth,
        "burned_area": target_area,
        "days": target_days,
        "commune_codes": np.asarray(codes),
        "event_counts_by_year": event_counts_by_year,
        "weather_path": weather_path,
    }


def _point_prediction(
    windows: jax.Array, weights: jax.Array, bias: jax.Array
) -> dict[str, jax.Array]:
    recent = jnp.mean(windows, axis=1)
    trend = windows[:, -1] - windows[:, 0]
    latest = windows[:, -1]
    signal = 0.25 * recent + 0.15 * trend + 0.60 * latest
    logits = jnp.einsum("bnc,co->bno", signal, weights) + bias
    horizon_logits = logits.reshape((*logits.shape[:-1], 3, 4))
    ignition_rate = jax.nn.sigmoid(horizon_logits[..., 0])
    growth_rate = jax.nn.sigmoid(horizon_logits[..., 1])
    ignition = 1.0 - jnp.cumprod(1.0 - ignition_rate, axis=-1)
    growth = 1.0 - jnp.cumprod(1.0 - growth_rate, axis=-1)
    area_mean = jnp.cumsum(jax.nn.softplus(horizon_logits[..., 2]), axis=-1)
    area_scale = jax.nn.softplus(horizon_logits[..., 3]) + 1e-4
    return {
        "ignition_probability": jnp.clip(ignition, 1e-5, 1.0 - 1e-5),
        "growth_probability": jnp.clip(growth, 1e-5, 1.0 - 1e-5),
        "burned_area_mean": area_mean,
        "burned_area_scale": area_scale,
    }


def _weighted_bce(
    probability: jax.Array, target: jax.Array, positive_weight: jax.Array
) -> jax.Array:
    return -jnp.mean(
        positive_weight * target * jnp.log(probability)
        + (1.0 - target) * jnp.log1p(-probability)
    )


def _training_loss(
    parameters: tuple[jax.Array, jax.Array],
    windows: jax.Array,
    ignition: jax.Array,
    growth: jax.Array,
    area: jax.Array,
    positive_weight: jax.Array,
) -> jax.Array:
    weights, bias = parameters
    prediction = _point_prediction(windows, weights, bias)
    ignition_loss = _weighted_bce(
        prediction["ignition_probability"], ignition, positive_weight
    )
    growth_loss = _weighted_bce(
        prediction["growth_probability"], growth, jnp.sqrt(positive_weight)
    )
    positive = area > 0
    area_error = jnp.where(
        positive,
        jnp.abs(jnp.log1p(prediction["burned_area_mean"]) - jnp.log1p(area)),
        0.0,
    )
    area_loss = jnp.sum(area_error) / (jnp.sum(positive) + 1.0)
    regularization = 2e-4 * jnp.mean(weights**2)
    return ignition_loss + 0.35 * growth_loss + 0.08 * area_loss + regularization


def _split_indices(
    days: np.ndarray, validation_start: date, test_start: date
) -> dict[str, np.ndarray]:
    if validation_start >= test_start:
        raise ValueError("validation_start must precede test_start")
    parsed = np.asarray([date.fromisoformat(str(value)) for value in days])
    return {
        "train": np.flatnonzero(parsed < validation_start),
        "validation": np.flatnonzero(
            (parsed >= validation_start) & (parsed < test_start)
        ),
        "test": np.flatnonzero(parsed >= test_start),
    }


def _adaptive_calibration_summary(
    probability: np.ndarray, target: np.ndarray, bins: int = 10
) -> tuple[float, list[dict[str, float | int]]]:
    """Return equal-count ECE and a reliability table for rare events."""
    scores = np.asarray(probability, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=np.float64).reshape(-1)
    groups = np.array_split(np.argsort(scores, kind="stable"), bins)
    table = []
    error = 0.0
    for group in groups:
        if not group.size:
            continue
        predicted = float(scores[group].mean())
        observed = float(labels[group].mean())
        weight = float(group.size / scores.size)
        error += weight * abs(predicted - observed)
        table.append(
            {
                "count": int(group.size),
                "mean_prediction": predicted,
                "observed_frequency": observed,
            }
        )
    return error, table


def _metrics(
    windows: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    parameters: tuple[jax.Array, jax.Array],
    training_prevalence: np.ndarray,
) -> dict[str, Any]:
    """Evaluate against a climatology fitted on training data only.

    ``oracle_climatology_brier`` is retained as a descriptive lower bound. It
    uses the evaluated split's prevalence and must not be interpreted as a
    deployable baseline.
    """
    prediction = _point_prediction(jnp.asarray(windows[indices]), *parameters)[
        "ignition_probability"
    ]
    labels = jnp.asarray(target[indices])
    values = []
    for horizon in range(3):
        probability = prediction[..., horizon].reshape(-1)
        actual = labels[..., horizon].reshape(-1)
        prevalence = jnp.mean(actual)
        climatology = jnp.full_like(probability, training_prevalence[horizon])
        oracle_climatology = jnp.full_like(probability, prevalence)
        brier = float(brier_score(probability, actual))
        climatology_brier = float(brier_score(climatology, actual))
        ap = float(average_precision(probability, actual))
        adaptive_ece, calibration_curve = _adaptive_calibration_summary(
            np.asarray(probability), np.asarray(actual), bins=10
        )
        values.append(
            {
                "horizon_hours": 24 * (horizon + 1),
                "sample_count": int(actual.size),
                "positive_count": int(jnp.sum(actual)),
                "prevalence": float(prevalence),
                "mean_prediction": float(jnp.mean(probability)),
                "brier": brier,
                "climatology_brier": climatology_brier,
                "brier_skill_vs_training_climatology": float(
                    1.0 - brier / max(climatology_brier, 1e-12)
                ),
                "oracle_climatology_brier": float(
                    brier_score(oracle_climatology, actual)
                ),
                "nll": float(binary_nll(probability, actual)),
                "ece_10_bins": float(
                    expected_calibration_error(probability, actual, bins=10)
                ),
                "adaptive_ece_10_bins": adaptive_ece,
                "calibration_curve_deciles": calibration_curve,
                "average_precision": ap,
                "climatology_average_precision": float(prevalence),
                "average_precision_lift": float(ap / max(float(prevalence), 1e-12)),
            }
        )
    monotonic_violations = int(
        jnp.sum(jnp.diff(prediction, axis=-1) < -1e-7)
    )
    return {
        "by_horizon": values,
        "monotonic_horizon_violations": monotonic_violations,
    }


def _mean_metric(metrics: dict[str, Any], name: str) -> float:
    return float(np.mean([row[name] for row in metrics["by_horizon"]]))


def _calibrate_bias(
    parameters: tuple[jax.Array, jax.Array],
    windows: np.ndarray,
    ignition: np.ndarray,
    growth: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any]]:
    """Fit six interval-logit offsets on validation data only.

    Calibrating interval rather than cumulative probabilities keeps the 24/48/72
    hour outputs monotone by construction. We calibrate growth as well because
    its probability later weights the burned-area closure.
    """
    weights, original_bias = parameters
    validation_windows = jnp.asarray(windows)
    validation_ignition = jnp.asarray(ignition)
    validation_growth = jnp.asarray(growth)
    offsets = jnp.zeros((3, 2), dtype=original_bias.dtype)
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(offsets)

    def calibrated_bias(values: jax.Array) -> jax.Array:
        result = original_bias
        result = result.at[0::4].add(values[:, 0])
        return result.at[1::4].add(values[:, 1])

    @jax.jit
    def calibration_step(
        values: jax.Array, optimizer_state: Any
    ) -> tuple[jax.Array, Any, jax.Array]:
        def loss_function(candidate: jax.Array) -> jax.Array:
            prediction = _point_prediction(
                validation_windows, weights, calibrated_bias(candidate)
            )
            ignition_loss = binary_nll(
                prediction["ignition_probability"], validation_ignition
            )
            growth_loss = binary_nll(
                prediction["growth_probability"], validation_growth
            )
            return ignition_loss + 0.30 * growth_loss + 2e-5 * jnp.mean(candidate**2)

        loss, gradient = jax.value_and_grad(loss_function)(values)
        updates, next_state = optimizer.update(gradient, optimizer_state, values)
        next_values = jnp.clip(optax.apply_updates(values, updates), -10.0, 4.0)
        return next_values, next_state, loss

    losses = []
    best_offsets = offsets
    best_loss = float("inf")
    for _ in range(steps):
        offsets, state, loss = calibration_step(offsets, state)
        value = float(loss)
        losses.append(value)
        if value < best_loss:
            best_loss = value
            best_offsets = offsets
    result = (weights, calibrated_bias(best_offsets))
    return result, {
        "method": "validation interval-logit offsets",
        "steps": steps,
        "learning_rate": learning_rate,
        "initial_loss": losses[0] if losses else None,
        "final_loss": best_loss,
        "ignition_offsets": np.asarray(best_offsets[:, 0]).tolist(),
        "growth_offsets": np.asarray(best_offsets[:, 1]).tolist(),
    }


def _load_parameters(path: Path) -> tuple[jax.Array, jax.Array] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as values:
        return jnp.asarray(values["weights"]), jnp.asarray(values["bias"])


def train_real_hazard(
    config_path: Path,
    scenario_path: Path,
    *,
    steps: int = 360,
    batch_size: int = 20,
    learning_rate: float = 0.015,
    max_positive_weight: float = 4.0,
    calibration_steps: int = 300,
    calibration_learning_rate: float = 0.04,
    seed: int = 20260805,
    candidate_path: Path | None = None,
    promote: bool = True,
    events_path: Path | None = None,
    data_start: date = date(2025, 1, 1),
    data_end: date = date(2025, 12, 31),
    validation_start: date = date(2025, 9, 1),
    test_start: date = date(2025, 11, 1),
) -> tuple[Path, dict[str, Any]]:
    """Train, calibrate, and conditionally promote the real hazard closure.

    Selection uses the September-October validation period only. The November-
    December test period is reported once and never participates in promotion.
    """
    config = load_real_config(config_path)
    data = build_training_windows(
        config_path,
        scenario_path,
        events_path=events_path,
        start_day=data_start,
        end_day=data_end,
    )
    splits = _split_indices(data["days"], validation_start, test_start)
    if any(not len(indices) for indices in splits.values()):
        raise ValueError("chronological train/validation/test splits must all be non-empty")
    canonical_output: Path = config["hazard_checkpoint"]
    incumbent_parameters = _load_parameters(canonical_output)
    weights, bias = _expert_hazard_parameters(data["windows"].shape[-1])
    weights[10, :] = 0.0
    parameters = (jnp.asarray(weights), jnp.asarray(bias))
    training_target = jnp.asarray(data["ignition"][splits["train"]])
    positive = jnp.sum(training_target, axis=(0, 1))
    total = training_target.shape[0] * training_target.shape[1]
    positive_weight = jnp.clip(
        (total - positive) / (positive + 1.0), 1.0, max_positive_weight
    )
    training_prevalence = np.asarray(jnp.mean(training_target, axis=(0, 1)))
    optimizer = optax.adamw(learning_rate, weight_decay=2e-4)
    state = optimizer.init(parameters)
    train_indices = splits["train"]
    losses = []
    validation_history = []
    rng = np.random.default_rng(seed)

    all_windows = jnp.asarray(data["windows"])
    all_ignition = jnp.asarray(data["ignition"])
    all_growth = jnp.asarray(data["growth"])
    all_area = jnp.asarray(data["burned_area"])

    @jax.jit
    def step(parameters: tuple[jax.Array, jax.Array], state: Any, indices: jax.Array):
        loss, gradient = jax.value_and_grad(_training_loss)(
            parameters,
            all_windows[indices],
            all_ignition[indices],
            all_growth[indices],
            all_area[indices],
            positive_weight,
        )
        updates, next_state = optimizer.update(gradient, state, parameters)
        return optax.apply_updates(parameters, updates), next_state, loss

    @jax.jit
    def validation_loss(current: tuple[jax.Array, jax.Array]) -> jax.Array:
        prediction = _point_prediction(
            all_windows[jnp.asarray(splits["validation"])], *current
        )
        return binary_nll(
            prediction["ignition_probability"],
            all_ignition[jnp.asarray(splits["validation"])],
        )

    best_parameters = parameters
    best_validation_loss = float(validation_loss(parameters))
    validation_history.append({"step": 0, "nll": best_validation_loss})
    for iteration in range(steps):
        indices = rng.choice(
            train_indices, size=min(batch_size, len(train_indices)), replace=False
        )
        parameters, state, loss = step(parameters, state, jnp.asarray(indices))
        losses.append(float(loss))
        if (iteration + 1) % 10 == 0 or iteration + 1 == steps:
            score = float(validation_loss(parameters))
            validation_history.append({"step": iteration + 1, "nll": score})
            if score < best_validation_loss:
                best_validation_loss = score
                best_parameters = parameters

    # Channel 10 is deliberately excluded for leakage safety; make that
    # invariant explicit in the deployed checkpoint as well.
    best_parameters = (
        best_parameters[0].at[10, :].set(0.0),
        best_parameters[1],
    )
    parameters, calibration = _calibrate_bias(
        best_parameters,
        data["windows"][splits["validation"]],
        data["ignition"][splits["validation"]],
        data["growth"][splits["validation"]],
        steps=calibration_steps,
        learning_rate=calibration_learning_rate,
    )
    output = candidate_path or canonical_output.with_name(
        f"{canonical_output.stem}-candidate.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        weights=np.asarray(parameters[0]),
        bias=np.asarray(parameters[1]),
        losses=np.asarray(losses, dtype=np.float32),
    )
    validation_metrics = _metrics(
        data["windows"],
        data["ignition"],
        splits["validation"],
        parameters,
        training_prevalence,
    )
    test_metrics = _metrics(
        data["windows"],
        data["ignition"],
        splits["test"],
        parameters,
        training_prevalence,
    )
    incumbent = None
    if incumbent_parameters is not None:
        incumbent = {
            "validation": _metrics(
                data["windows"],
                data["ignition"],
                splits["validation"],
                incumbent_parameters,
                training_prevalence,
            ),
            "test": _metrics(
                data["windows"],
                data["ignition"],
                splits["test"],
                incumbent_parameters,
                training_prevalence,
            ),
        }
    candidate_brier = _mean_metric(validation_metrics, "brier")
    candidate_ece = _mean_metric(validation_metrics, "adaptive_ece_10_bins")
    incumbent_brier = (
        _mean_metric(incumbent["validation"], "brier") if incumbent else float("inf")
    )
    incumbent_ece = (
        _mean_metric(incumbent["validation"], "adaptive_ece_10_bins")
        if incumbent
        else float("inf")
    )
    eligible = (
        validation_metrics["monotonic_horizon_violations"] == 0
        and np.isfinite(candidate_brier)
        and candidate_brier < incumbent_brier - 1e-7
        and candidate_ece <= incumbent_ece + 1e-7
    )
    promoted = bool(promote and eligible)
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "validation-calibrated monotone cumulative linear hazard closure",
        "dataset_scope": "all Var communes intersecting the real pilot; no event-selected mask",
        "selection_policy": (
            f"checkpoint promotion uses validation from {validation_start.isoformat()} "
            f"to the day before {test_start.isoformat()} only; later test metrics are "
            "never used for selection"
        ),
        "history_days": int(config["history_days"]),
        "event_catalogue": str(events_path or config["bdiff_events"]),
        "data_period": {
            "start": data_start.isoformat(),
            "end": data_end.isoformat(),
            "validation_start": validation_start.isoformat(),
            "test_start": test_start.isoformat(),
        },
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_uncalibrated_validation_nll": best_validation_loss,
        "validation_history": validation_history,
        "positive_weight": np.asarray(positive_weight).tolist(),
        "max_positive_weight": max_positive_weight,
        "training_prevalence": training_prevalence.tolist(),
        "commune_count": int(data["windows"].shape[1]),
        "mapped_catalogue_events_by_year": data["event_counts_by_year"],
        "target_positive_counts": {
            name: np.asarray(data["ignition"][indices].sum(axis=(0, 1))).astype(int).tolist()
            for name, indices in splits.items()
        },
        "calibration": calibration,
        "splits": {
            name: {
                "count": len(indices),
                "start": str(data["days"][indices[0]]),
                "end": str(data["days"][indices[-1]]),
            }
            for name, indices in splits.items()
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "incumbent_re_evaluated_on_same_features": incumbent,
        "promotion": {
            "requested": promote,
            "eligible": eligible,
            "promoted": promoted,
            "candidate_validation_mean_brier": candidate_brier,
            "incumbent_validation_mean_brier": (
                incumbent_brier if np.isfinite(incumbent_brier) else None
            ),
            "candidate_validation_mean_ece": candidate_ece,
            "incumbent_validation_mean_ece": (
                incumbent_ece if np.isfinite(incumbent_ece) else None
            ),
            "candidate_path": str(output),
            "canonical_path": str(canonical_output),
        },
        "weather_sha256": sha256_file(data["weather_path"]),
        "scenario_sha256": sha256_file(scenario_path),
        "limitations": [
            "Even three recent BDIFF campaigns are insufficient for operational calibration.",
            "BDIFF supplies commune-level events, not pixel ignition locations.",
            "Meteorological reanalysis captures seasonal state; no calendar-only feature is used.",
            "Training soil dryness is a daily precipitation-minus-ET0 water-balance proxy.",
            "Historical weather is bilinearly interpolated from four reanalysis nodes.",
            "The all-2025 historical-density channel is zeroed to prevent spatial target leakage.",
            "The oracle climatology is descriptive only; the deployable baseline uses training prevalence.",
            "The model is a transparent research closure, not an emergency forecast.",
        ],
    }
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    if promoted:
        shutil.copy2(output, canonical_output)
        canonical_metrics = canonical_output.with_suffix(".json")
        shutil.copy2(metrics_path, canonical_metrics)
        return canonical_output, metrics
    return output, metrics
