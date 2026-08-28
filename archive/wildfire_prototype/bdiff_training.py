"""Leakage-safe temporal windows for the real BDIFF hazard experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from wildfire.labels import temporal_split
from wildfire.metrics import (
    average_precision,
    binary_nll,
    brier_score,
    expected_calibration_error,
    top_k_recall,
)


@dataclass(frozen=True)
class BDIFFWindowData:
    """Daily BDIFF labels and chronological train/validation/test indices."""

    days: tuple[date, ...]
    event_count: np.ndarray
    burned_area: np.ndarray
    ignition: np.ndarray
    growth: np.ndarray
    history_days: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    event_scale: float
    area_scale: float
    train_climatology: np.ndarray
    active_mask: np.ndarray

    @property
    def feature_channels(self) -> int:
        """Number of leakage-safe historical and climatology channels."""
        return 9

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Spatial grid shape."""
        return tuple(int(value) for value in self.event_count.shape[1:3])


def _parse_days(values: np.ndarray) -> tuple[date, ...]:
    return tuple(date.fromisoformat(str(value)) for value in values.tolist())


def load_bdiff_windows(
    path: Path,
    history_days: int = 14,
    validation_start: date = date(2025, 10, 1),
    test_start: date = date(2025, 12, 1),
) -> BDIFFWindowData:
    """Load daily labels and define chronological windows without future leakage."""
    if history_days < 1:
        raise ValueError("history_days must be positive")
    with np.load(path) as data:
        days = _parse_days(data["days"])
        event_count = np.asarray(data["event_count"], dtype=np.float32)
        burned_area = np.asarray(data["burned_area"], dtype=np.float32)
        ignition = np.asarray(data["ignition"], dtype=np.float32)
        growth = np.asarray(data["growth"], dtype=np.float32)
        active_mask = np.asarray(
            data["active_mask"] if "active_mask" in data else np.ones(event_count.shape[1:3]),
            dtype=np.float32,
        )
    if event_count.ndim != 4 or burned_area.shape != event_count.shape:
        raise ValueError("BDIFF arrays must have shape (days, height, width, horizons)")
    if ignition.shape != event_count.shape or growth.shape != event_count.shape:
        raise ValueError("BDIFF targets must match event_count shape")
    if len(days) != event_count.shape[0] or len(days) <= history_days:
        raise ValueError("not enough daily observations for requested history")
    split = temporal_split(list(days), validation_start, test_start)
    valid_indices = tuple(index for index in range(history_days, len(days)))
    train = tuple(index for index in split["train"] if index in valid_indices)
    validation = tuple(index for index in split["validation"] if index in valid_indices)
    test = tuple(index for index in split["test"] if index in valid_indices)
    if not train or not validation or not test:
        raise ValueError("chronological split must contain train, validation, and test windows")
    # Scales are estimated from training context only; held-out periods never
    # influence feature normalization.
    context_days = np.asarray(train)[:, None] - np.arange(history_days)[None, :] - 1
    context_days = context_days.reshape(-1)
    event_scale = max(float(np.quantile(event_count[context_days, ..., 0], 0.99)), 1.0)
    area_scale = max(float(np.quantile(burned_area[context_days, ..., 0], 0.99)), 1.0)
    train_climatology = np.mean(ignition[np.asarray(train)], axis=0).astype(np.float32)
    return BDIFFWindowData(
        days=days,
        event_count=event_count,
        burned_area=burned_area,
        ignition=ignition,
        growth=growth,
        history_days=history_days,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        event_scale=event_scale,
        area_scale=area_scale,
        train_climatology=train_climatology,
        active_mask=active_mask,
    )


def window_batch(
    data: BDIFFWindowData,
    indices: tuple[int, ...] | list[int] | np.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Build a batch whose features end strictly before each target day."""
    indices_array = np.asarray(indices, dtype=np.int32)
    if indices_array.ndim != 1 or len(indices_array) == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    if np.any(indices_array < data.history_days) or np.any(
        indices_array >= len(data.days)
    ):
        raise ValueError("window index does not have enough historical context")
    feature_windows = []
    first_validation_index = min(data.validation_indices)
    for index in indices_array.tolist():
        history = np.arange(index - data.history_days, index)
        if index < first_validation_index:
            prior_train = [
                value for value in data.train_indices if value < index
            ]
            climatology = (
                np.mean(data.ignition[np.asarray(prior_train)], axis=0)
                if prior_train
                else np.zeros_like(data.train_climatology)
            )
        else:
            # Validation and test features use the frozen train-only prior so
            # held-out labels never alter the feature distribution.
            climatology = data.train_climatology
        event_series = data.event_count[..., 0]
        area_series = data.burned_area[..., 0]
        channels = []
        for position, day_index in enumerate(history):
            start_3 = max(0, position - 2)
            start_7 = max(0, position - 6)
            start_14 = max(0, position - 13)
            context_indices = history[: position + 1]
            event_signal = np.log1p(event_series[day_index]) / np.log1p(
                data.event_scale
            )
            event_3 = np.mean(
                np.log1p(event_series[history[start_3 : position + 1]])
                / np.log1p(data.event_scale),
                axis=0,
            )
            event_7 = np.mean(
                np.log1p(event_series[history[start_7 : position + 1]])
                / np.log1p(data.event_scale),
                axis=0,
            )
            event_14 = np.mean(
                np.log1p(event_series[history[start_14 : position + 1]])
                / np.log1p(data.event_scale),
                axis=0,
            )
            area_7 = np.mean(
                np.log1p(area_series[history[start_7 : position + 1]])
                / np.log1p(data.area_scale),
                axis=0,
            )
            target_day_of_year = data.days[index].timetuple().tm_yday
            del context_indices
            channels.append(
                np.stack(
                    [
                        np.clip(event_signal, 0.0, 1.0),
                        np.clip(event_3, 0.0, 1.0),
                        np.clip(event_7, 0.0, 1.0),
                        np.clip(event_14, 0.0, 1.0),
                        np.clip(area_7, 0.0, 1.0),
                        climatology[..., 0],
                        np.full(data.grid_shape, np.sin(2.0 * np.pi * target_day_of_year / 365.25), dtype=np.float32),
                        np.full(data.grid_shape, np.cos(2.0 * np.pi * target_day_of_year / 365.25), dtype=np.float32),
                        data.active_mask,
                    ],
                    axis=-1,
                )
            )
        feature_windows.append(np.stack(channels, axis=0))
    targets = {
        "ignition": jnp.asarray(data.ignition[indices_array]),
        "growth": jnp.asarray(data.growth[indices_array]),
        "burned_area": jnp.asarray(data.burned_area[indices_array]),
        "active_mask": jnp.asarray(data.active_mask)[None, ..., None],
    }
    return jnp.asarray(np.stack(feature_windows, axis=0)), targets


def evaluate_prediction(
    prediction: dict[str, jnp.ndarray],
    targets: dict[str, jnp.ndarray],
    horizon: int | None = None,
) -> dict[str, float | int]:
    """Return calibrated multi-horizon metrics and positive-event counts."""
    ignition_probability = prediction["ignition_probability"]
    target = targets["ignition"]
    active_mask = targets.get("active_mask", jnp.ones((*target.shape[:-1], 1)))
    active_mask = jnp.broadcast_to(active_mask, target.shape)
    if horizon is not None:
        if not 0 <= horizon < ignition_probability.shape[-1]:
            raise ValueError("horizon index out of range")
        ignition_probability = ignition_probability[..., horizon]
        target = target[..., horizon]
        active_mask = active_mask[..., horizon]
    active = active_mask.reshape(-1) > 0.5
    ignition_probability = ignition_probability.reshape(-1)[active]
    target = target.reshape(-1)[active]
    threshold = 0.5
    predicted_positive = ignition_probability >= threshold
    actual_positive = target >= threshold
    true_positive = jnp.sum(predicted_positive & actual_positive)
    actual_count = jnp.sum(actual_positive)
    return {
        "brier": float(brier_score(ignition_probability, target)),
        "nll": float(binary_nll(ignition_probability, target)),
        "ece": float(expected_calibration_error(ignition_probability, target)),
        "average_precision": float(average_precision(ignition_probability, target)),
        "top_1pct_recall": float(top_k_recall(ignition_probability, target, 0.01)),
        "horizon": -1 if horizon is None else horizon,
        "actual_positive_count": int(actual_count),
        "predicted_positive_count": int(jnp.sum(predicted_positive)),
        "recall_at_0_5": float(true_positive / (actual_count + 1e-6)),
        "precision_at_0_5": float(true_positive / (jnp.sum(predicted_positive) + 1e-6)),
    }
