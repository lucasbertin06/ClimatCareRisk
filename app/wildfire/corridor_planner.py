"""Differentiable planning over coherent candidate firebreak corridors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.ndimage import binary_dilation, maximum_filter

from wildfire.planner import hazard_prediction
from wildfire.scenario import WildfireScenario
from wildfire_shared.spread import spread_forward


@dataclass(frozen=True)
class CorridorConfig:
    """Dimensionless objective weights and real-world feasibility limits."""

    steps: int = 18
    iterations: int = 90
    learning_rate: float = 0.11
    budget_k_eur: float = 2_400.0
    construction_household_cap: float = 120.0
    area_weight: float = 1.0
    exposed_population_weight: float = 1.35
    exposed_households_weight: float = 0.70
    heat_weight: float = 0.45
    cvar_weight: float = 0.45
    construction_cost_weight: float = 0.035
    construction_households_weight: float = 0.055
    ecology_weight: float = 0.055
    binarization_weight: float = 0.025
    budget_penalty: float = 8.0
    household_penalty: float = 8.0
    area_scale_hectares: float = 5_000.0
    population_scale: float = 20_000.0
    household_scale: float = 10_000.0
    heat_scale: float = 2.0
    ecology_scale_hectares: float = 1_500.0
    corridor_effectiveness: float = 0.90


@dataclass(frozen=True)
class CorridorSet:
    """Candidate masks and their auditable construction attributes."""

    masks: jax.Array
    construction_cost_k_eur: jax.Array
    construction_households: jax.Array
    ecological_footprint_hectares: jax.Array
    length_km: jax.Array
    metadata: tuple[dict[str, Any], ...]


def _line_mask(
    height: int,
    width: int,
    row: int,
    column: int,
    angle_degrees: float,
    length_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(angle_degrees)
    half = (length_cells - 1) / 2.0
    positions = np.linspace(-half, half, length_cells * 3)
    rows = np.rint(row + positions * math.sin(angle)).astype(int)
    columns = np.rint(column + positions * math.cos(angle)).astype(int)
    valid = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    core = np.zeros((height, width), dtype=bool)
    core[rows[valid], columns[valid]] = True
    planning_cells = binary_dilation(core, iterations=1)
    return core, planning_cells


def generate_corridors(
    scenario: WildfireScenario,
    hazard: jax.Array,
    *,
    resolution_m: float,
    max_candidates: int = 72,
) -> CorridorSet:
    """Generate road-accessible, coherent corridor candidates near risk peaks."""
    fuel = np.asarray(scenario.fuel)
    forest = np.asarray(
        scenario.forest_fraction if scenario.forest_fraction is not None else scenario.fuel
    )
    road_access = np.asarray(
        scenario.road_access
        if scenario.road_access is not None
        else np.ones_like(fuel)
    )
    dwellings = np.asarray(
        scenario.dwelling_density
        if scenario.dwelling_density is not None
        else scenario.population / 2.15
    )
    slope = np.asarray(scenario.slope)
    ecology = np.asarray(scenario.ecological_cost)
    risk = np.asarray(hazard)
    score = risk * forest * (0.25 + 0.75 * road_access) * (fuel > 0.18)
    local_maximum = score >= maximum_filter(score, size=5, mode="nearest")
    seed_order = np.argsort(np.where(local_maximum, score, -np.inf).ravel())[::-1]
    seeds: list[tuple[int, int]] = []
    for flat_index in seed_order:
        if not np.isfinite(score.ravel()[flat_index]) or score.ravel()[flat_index] <= 0:
            break
        row, column = np.unravel_index(flat_index, score.shape)
        if all((row - old_row) ** 2 + (column - old_column) ** 2 >= 16 for old_row, old_column in seeds):
            seeds.append((int(row), int(column)))
        if len(seeds) >= 22:
            break
    candidates: list[np.ndarray] = []
    cores: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    signatures: set[bytes] = set()
    for seed_index, (row, column) in enumerate(seeds):
        for angle in (0.0, 45.0, 90.0, 135.0):
            length_cells = 7 if seed_index % 2 == 0 else 9
            core, planning = _line_mask(
                fuel.shape[0], fuel.shape[1], row, column, angle, length_cells
            )
            feasible = planning & (fuel > 0.16) & (slope < 0.88)
            if int(feasible.sum()) < 5 or float(forest[feasible].mean()) < 0.15:
                continue
            signature = np.packbits(feasible).tobytes()
            if signature in signatures:
                continue
            signatures.add(signature)
            candidates.append(feasible.astype(np.float32))
            cores.append((core & feasible).astype(np.float32))
            points = np.argwhere(core & feasible)
            start = points[0].tolist() if len(points) else [row, column]
            end = points[-1].tolist() if len(points) else [row, column]
            metadata.append(
                {
                    "corridor_id": f"C{len(candidates):03d}",
                    "seed_row": row,
                    "seed_column": column,
                    "angle_degrees": angle,
                    "start_cell": start,
                    "end_cell": end,
                }
            )
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
    if not candidates:
        raise ValueError("no feasible firebreak corridors were generated")
    masks = np.stack(candidates)
    core_masks = np.stack(cores)
    length_km = core_masks.sum(axis=(1, 2)) * resolution_m / 1000.0
    mean_slope = np.sum(masks * slope, axis=(1, 2)) / np.maximum(masks.sum(axis=(1, 2)), 1.0)
    mean_access = np.sum(masks * road_access, axis=(1, 2)) / np.maximum(masks.sum(axis=(1, 2)), 1.0)
    construction_cost = length_km * 48.0 * (
        0.85 + 1.35 * mean_slope + 0.45 * (1.0 - mean_access)
    )
    footprint_fraction = np.clip(35.0 / resolution_m, 0.008, 0.10)
    construction_households = np.sum(masks * dwellings, axis=(1, 2)) * footprint_fraction
    ecological_footprint = (
        np.sum(masks * forest * ecology, axis=(1, 2))
        * float(scenario.cell_area_hectares)
        * footprint_fraction
    )
    for index, item in enumerate(metadata):
        item.update(
            {
                "length_km": float(length_km[index]),
                "construction_cost_k_eur": float(construction_cost[index]),
                "construction_households": float(construction_households[index]),
                "ecological_footprint_hectares": float(ecological_footprint[index]),
                "mean_road_access": float(mean_access[index]),
                "mean_slope": float(mean_slope[index]),
            }
        )
    return CorridorSet(
        masks=jnp.asarray(masks),
        construction_cost_k_eur=jnp.asarray(construction_cost, dtype=jnp.float32),
        construction_households=jnp.asarray(
            construction_households, dtype=jnp.float32
        ),
        ecological_footprint_hectares=jnp.asarray(
            ecological_footprint, dtype=jnp.float32
        ),
        length_km=jnp.asarray(length_km, dtype=jnp.float32),
        metadata=tuple(metadata),
    )


def wind_ensemble(wind: jax.Array) -> jax.Array:
    """Return nine deterministic speed/direction perturbations for robust planning."""
    base = np.asarray(wind, dtype=np.float32)
    speed = max(float(np.linalg.norm(base)), 0.05)
    angle = math.atan2(float(base[1]), float(base[0]))
    values = []
    for multiplier, offset_degrees in (
        (0.72, -24.0),
        (0.82, 12.0),
        (0.92, -10.0),
        (1.0, 0.0),
        (1.05, 18.0),
        (1.12, -18.0),
        (1.20, 8.0),
        (1.28, 26.0),
        (1.38, -30.0),
    ):
        current = angle + math.radians(offset_degrees)
        values.append(
            [multiplier * speed * math.cos(current), multiplier * speed * math.sin(current)]
        )
    return jnp.asarray(values, dtype=jnp.float32)


def _spread_call(
    scenario: WildfireScenario,
    hazard: jax.Array,
    intervention: jax.Array,
    wind: jax.Array,
    *,
    backend: str,
    steps: int,
) -> dict[str, jax.Array]:
    inputs = {
        "hazard": hazard,
        "fuel": scenario.fuel,
        "wind": wind,
        "slope": scenario.slope,
        "intervention": intervention,
        "population": scenario.population,
        "ecological_cost": scenario.ecological_cost,
        "intervention_cost": scenario.intervention_cost,
        "steps": steps,
    }
    if backend == "pure":
        return spread_forward(**inputs)
    if backend != "tesseract":
        raise ValueError("backend must be 'pure' or 'tesseract'")
    from wildfire.bridge import torch_spread

    outputs = torch_spread(
        inputs["hazard"],
        inputs["fuel"],
        inputs["wind"],
        inputs["slope"],
        inputs["intervention"],
        inputs["population"],
        inputs["ecological_cost"],
        inputs["intervention_cost"],
        steps,
    )
    return {
        "burn_probability": outputs[0],
        "trajectory": outputs[1],
        "burned_area": outputs[2],
        "exposed_population": outputs[3],
        "ecological_penalty": outputs[4],
        "intervention_cost": outputs[5],
    }


def activation_mask(
    activation: jax.Array,
    corridors: CorridorSet,
    effectiveness: float,
) -> jax.Array:
    """Compose corridor activations into a differentiable union mask."""
    individual = jnp.clip(
        effectiveness * activation[:, None, None] * corridors.masks,
        0.0,
        0.98,
    )
    return 1.0 - jnp.prod(1.0 - individual, axis=0)


def _hazard_with_intervention(
    scenario: WildfireScenario,
    intervention: jax.Array,
    *,
    backend: str,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Re-evaluate ignition after differentiable local fuel treatment."""
    channel_effect = jnp.zeros((scenario.features.shape[-1],), dtype=jnp.float32)
    channel_effect = channel_effect.at[0].set(0.82)
    if scenario.features.shape[-1] > 1:
        channel_effect = channel_effect.at[1].set(0.62)
    if scenario.features.shape[-1] > 2:
        channel_effect = channel_effect.at[2].set(0.68)
    adjusted_features = scenario.features * (
        1.0
        - intervention[None, ..., None]
        * channel_effect[None, None, None, :]
    )
    inputs = {
        "features": adjusted_features,
        "weights": scenario.hazard_weights,
        "bias": scenario.hazard_bias,
        "horizons": int(scenario.horizon_hours.shape[0]),
    }
    if backend == "pure":
        from wildfire_shared.hazard import hazard_apply

        prediction = hazard_apply(inputs)
    elif backend == "tesseract":
        from wildfire.runtime import hazard_apply

        prediction = hazard_apply(inputs)
    else:
        raise ValueError("backend must be 'pure' or 'tesseract'")
    field = jnp.clip(
        prediction["ignition_probability"][..., -1]
        * (0.55 + 0.45 * prediction["growth_probability"][..., -1]),
        0.0,
        1.0,
    )
    return field, prediction


def evaluate_activation(
    activation: jax.Array,
    scenario: WildfireScenario,
    corridors: CorridorSet,
    hazard_field: jax.Array,
    winds: jax.Array,
    config: CorridorConfig,
    *,
    backend: str,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Evaluate a relaxed or binary corridor portfolio."""
    del hazard_field  # retained in the public signature for benchmark attribution
    intervention = activation_mask(
        activation, corridors, config.corridor_effectiveness
    )
    current_hazard, hazard_prediction_result = _hazard_with_intervention(
        scenario, intervention, backend=backend
    )
    results = jax.vmap(
        lambda current_wind: _spread_call(
            scenario,
            current_hazard,
            intervention,
            current_wind,
            backend=backend,
            steps=config.steps,
        )
    )(winds)
    forest = (
        scenario.forest_fraction
        if scenario.forest_fraction is not None
        else scenario.fuel
    )
    dwellings = (
        scenario.dwelling_density
        if scenario.dwelling_density is not None
        else scenario.population / 2.15
    )
    burned_hectares = jnp.sum(
        results["burn_probability"]
        * forest[None, ...]
        * float(scenario.cell_area_hectares),
        axis=(-2, -1),
    )
    exposed_population = jnp.sum(
        results["burn_probability"] * scenario.population[None, ...], axis=(-2, -1)
    )
    exposed_households = jnp.sum(
        results["burn_probability"] * dwellings[None, ...], axis=(-2, -1)
    )
    compound_heat = jnp.sum(
        results["burn_probability"] * scenario.heat_health_burden[None, ...],
        axis=(-2, -1),
    )
    scenario_loss = (
        config.area_weight * burned_hectares / config.area_scale_hectares
        + config.exposed_population_weight
        * exposed_population
        / config.population_scale
        + config.exposed_households_weight
        * exposed_households
        / config.household_scale
        + config.heat_weight * compound_heat / config.heat_scale
    )
    sorted_loss = jnp.sort(scenario_loss)
    tail_count = max(1, math.ceil(0.2 * int(winds.shape[0])))
    cvar = jnp.mean(sorted_loss[-tail_count:])
    construction_cost = jnp.sum(
        activation * corridors.construction_cost_k_eur
    )
    construction_households = jnp.sum(
        activation * corridors.construction_households
    )
    ecological_footprint = jnp.sum(
        activation * corridors.ecological_footprint_hectares
    )
    budget_violation = jax.nn.relu(
        construction_cost - config.budget_k_eur
    ) / config.budget_k_eur
    household_violation = jax.nn.relu(
        construction_households - config.construction_household_cap
    ) / max(config.construction_household_cap, 1.0)
    objective = (
        jnp.mean(scenario_loss)
        + config.cvar_weight * cvar
        + config.construction_cost_weight
        * construction_cost
        / config.budget_k_eur
        + config.construction_households_weight
        * construction_households
        / max(config.construction_household_cap, 1.0)
        + config.ecology_weight
        * ecological_footprint
        / config.ecology_scale_hectares
        + config.binarization_weight * jnp.mean(activation * (1.0 - activation))
        + config.budget_penalty * budget_violation**2
        + config.household_penalty * household_violation**2
    )
    return objective, {
        "activation": activation,
        "intervention": intervention,
        "scenario_loss": scenario_loss,
        "burned_hectares": burned_hectares,
        "exposed_population": exposed_population,
        "exposed_households": exposed_households,
        "compound_heat_burden": compound_heat,
        "construction_cost_k_eur": construction_cost,
        "construction_households": construction_households,
        "ecological_footprint_hectares": ecological_footprint,
        "cvar": cvar,
        "budget_violation": budget_violation,
        "household_violation": household_violation,
        "burn_probability": results["burn_probability"],
        "trajectory": results["trajectory"],
        "hazard_field": current_hazard,
        "hazard_by_horizon": hazard_prediction_result["ignition_probability"],
    }


def _feasible_selection(
    order: np.ndarray,
    corridors: CorridorSet,
    config: CorridorConfig,
) -> np.ndarray:
    selected = np.zeros(len(corridors.metadata), dtype=np.float32)
    cost = 0.0
    households = 0.0
    costs = np.asarray(corridors.construction_cost_k_eur)
    relocations = np.asarray(corridors.construction_households)
    for index in order:
        next_cost = cost + float(costs[index])
        next_households = households + float(relocations[index])
        if next_cost <= config.budget_k_eur and next_households <= config.construction_household_cap:
            selected[index] = 1.0
            cost = next_cost
            households = next_households
    return selected


def optimize_corridors(
    scenario: WildfireScenario,
    corridors: CorridorSet,
    *,
    config: CorridorConfig | None = None,
    backend: str = "pure",
) -> dict[str, Any]:
    """Optimize relaxed corridor activations and project them to a feasible plan."""
    config = config or CorridorConfig()
    hazard = hazard_prediction(scenario, backend=backend)
    hazard_field = jnp.clip(
        hazard["ignition_probability"][..., -1]
        * (0.55 + 0.45 * hazard["growth_probability"][..., -1]),
        0.0,
        1.0,
    )
    winds = wind_ensemble(scenario.wind)
    latent = jnp.full((len(corridors.metadata),), -2.6, dtype=jnp.float32)
    optimizer = optax.adam(config.learning_rate)
    state = optimizer.init(latent)

    def loss_function(values: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        return evaluate_activation(
            jax.nn.sigmoid(values),
            scenario,
            corridors,
            hazard_field,
            winds,
            config,
            backend=backend,
        )

    value_and_grad = jax.value_and_grad(loss_function, has_aux=True)
    if backend == "pure":
        value_and_grad = jax.jit(value_and_grad)
    losses = []
    for _ in range(config.iterations):
        (loss, _), gradient = value_and_grad(latent)
        updates, state = optimizer.update(gradient, state, latent)
        latent = optax.apply_updates(latent, updates)
        losses.append(float(loss))
    relaxed_activation = np.asarray(jax.nn.sigmoid(latent))
    _, zero_gradient = jax.value_and_grad(lambda values: loss_function(values)[0])(
        jnp.full_like(latent, -8.0)
    )
    sensitivity = -np.asarray(zero_gradient)
    ranking_score = 0.72 * relaxed_activation + 0.28 * (
        (sensitivity - sensitivity.min()) / (np.ptp(sensitivity) + 1e-8)
    )
    ranked = np.argsort(ranking_score)[::-1]
    projection_evaluations = 0

    @jax.jit
    def projected_losses_batch(activations: jax.Array) -> jax.Array:
        """Evaluate projected portfolios in parallel with the reference kernels."""
        return jax.vmap(
            lambda activation: evaluate_activation(
                activation,
                scenario,
                corridors,
                hazard_field,
                winds,
                config,
                backend="pure",
            )[0]
        )(activations)

    def evaluate_projected_trials(trials: np.ndarray) -> np.ndarray:
        """Bound peak memory while keeping the discrete repair vectorized."""
        nonlocal projection_evaluations
        if len(trials) == 0:
            return np.empty((0,), dtype=np.float64)
        projection_evaluations += len(trials)
        batches = []
        for start in range(0, len(trials), 96):
            values = projected_losses_batch(
                jnp.asarray(trials[start : start + 96], dtype=jnp.float32)
            )
            batches.append(np.asarray(values, dtype=np.float64))
        return np.concatenate(batches)

    running = np.zeros(len(corridors.metadata), dtype=np.float32)
    selected = running.copy()
    best_projected_loss = float(evaluate_projected_trials(running[None, :])[0])
    costs = np.asarray(corridors.construction_cost_k_eur)
    relocations = np.asarray(corridors.construction_households)
    running_cost = 0.0
    running_households = 0.0
    for index in ranked:
        if (
            running_cost + float(costs[index]) > config.budget_k_eur
            or running_households + float(relocations[index])
            > config.construction_household_cap
        ):
            continue
        running[index] = 1.0
        running_cost += float(costs[index])
        running_households += float(relocations[index])
        projected_loss = float(evaluate_projected_trials(running[None, :])[0])
        if projected_loss < best_projected_loss:
            best_projected_loss = projected_loss
            selected = running.copy()
    # Repair the relaxed projection with a small, auditable one-swap search.
    # Gradients reduce the candidate pool; the discrete pass only restores hard
    # budget/household feasibility at the non-convex boundary.
    individual_losses = np.full(len(ranked), np.inf, dtype=np.float64)
    individual_trials = []
    individual_indices = []
    for index in ranked:
        trial = np.zeros(len(ranked), dtype=np.float32)
        trial[index] = 1.0
        if (
            float(costs[index]) > config.budget_k_eur
            or float(relocations[index]) > config.construction_household_cap
        ):
            continue
        individual_trials.append(trial)
        individual_indices.append(int(index))
    individual_values = evaluate_projected_trials(np.asarray(individual_trials))
    for index, trial_loss in zip(individual_indices, individual_values, strict=True):
        individual_losses[index] = float(trial_loss)
    marginal_ranked = np.argsort(individual_losses)
    repair_pool = np.unique(
        np.concatenate(
            [
                ranked[: min(32, len(ranked))],
                marginal_ranked[: min(24, len(marginal_ranked))],
                np.flatnonzero(selected),
            ]
        )
    )
    for _ in range(3):
        best_trial = selected
        best_trial_loss = best_projected_loss
        removals = np.concatenate([np.asarray([-1]), np.flatnonzero(selected)])
        for remove_index in removals:
            base = selected.copy()
            if remove_index >= 0:
                base[remove_index] = 0.0
            base_cost = float(np.sum(base * costs))
            base_households = float(np.sum(base * relocations))
            for add_index in repair_pool:
                if base[add_index] > 0.5:
                    continue
                if (
                    base_cost + float(costs[add_index]) > config.budget_k_eur
                    or base_households + float(relocations[add_index])
                    > config.construction_household_cap
                ):
                    continue
                trial = base.copy()
                trial[add_index] = 1.0
                trial_loss = float(evaluate_projected_trials(trial[None, :])[0])
                if trial_loss + 1e-7 < best_trial_loss:
                    best_trial_loss = trial_loss
                    best_trial = trial
        if best_trial_loss + 1e-7 >= best_projected_loss:
            break
        selected = best_trial
        best_projected_loss = best_trial_loss
    # A single two-for-two exchange captures corridor interactions that neither
    # independent marginal scores nor a one-swap neighbourhood can observe. The
    # candidates remain gradient-screened and all trials retain both hard limits.
    selected_indices = np.flatnonzero(selected > 0.5)
    if len(selected_indices) >= 2:
        pair_pool = np.unique(
            np.concatenate(
                [
                    ranked[: min(20, len(ranked))],
                    marginal_ranked[: min(24, len(marginal_ranked))],
                    selected_indices,
                ]
            )
        )
        pair_trials: list[np.ndarray] = []
        for first_position in range(len(selected_indices) - 1):
            for second_position in range(first_position + 1, len(selected_indices)):
                base = selected.copy()
                base[selected_indices[first_position]] = 0.0
                base[selected_indices[second_position]] = 0.0
                for first_add_position in range(len(pair_pool) - 1):
                    for second_add_position in range(first_add_position + 1, len(pair_pool)):
                        first_add = int(pair_pool[first_add_position])
                        second_add = int(pair_pool[second_add_position])
                        if base[first_add] > 0.5 or base[second_add] > 0.5:
                            continue
                        trial = base.copy()
                        trial[first_add] = 1.0
                        trial[second_add] = 1.0
                        if (
                            float(np.sum(trial * costs)) > config.budget_k_eur
                            or float(np.sum(trial * relocations))
                            > config.construction_household_cap
                        ):
                            continue
                        pair_trials.append(trial)
        if pair_trials:
            unique_trials = np.unique(np.stack(pair_trials), axis=0)
            trial_losses = evaluate_projected_trials(unique_trials)
            best_index = int(np.argmin(trial_losses))
            if float(trial_losses[best_index]) + 1e-7 < best_projected_loss:
                selected = unique_trials[best_index]
                best_projected_loss = float(trial_losses[best_index])
    final_loss, final = evaluate_activation(
        jnp.asarray(selected),
        scenario,
        corridors,
        hazard_field,
        winds,
        config,
        backend=backend,
    )
    baseline_loss, baseline = evaluate_activation(
        jnp.zeros_like(jnp.asarray(selected)),
        scenario,
        corridors,
        hazard_field,
        winds,
        config,
        backend=backend,
    )
    return {
        "hazard": hazard,
        "hazard_field": hazard_field,
        "winds": winds,
        "latent": latent,
        "relaxed_activation": relaxed_activation,
        "sensitivity": sensitivity,
        "selected": selected,
        "losses": np.asarray(losses, dtype=np.float32),
        "projection_evaluations": projection_evaluations,
        "loss": final_loss,
        "diagnostics": final,
        "baseline_loss": baseline_loss,
        "baseline": baseline,
    }


def benchmark_corridor_strategies(
    scenario: WildfireScenario,
    corridors: CorridorSet,
    optimized: dict[str, Any],
    *,
    config: CorridorConfig,
    include_greedy: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compare equal-budget corridor portfolios using the pure reference solver."""
    hazard_field = optimized["hazard_field"]
    winds = optimized["winds"]
    corridor_risk = np.sum(
        np.asarray(corridors.masks) * np.asarray(hazard_field)[None, ...], axis=(1, 2)
    )
    corridor_population = np.sum(
        np.asarray(corridors.masks) * np.asarray(scenario.population)[None, ...],
        axis=(1, 2),
    )
    costs = np.asarray(corridors.construction_cost_k_eur)
    strategies = {
        "no_intervention": np.zeros(len(corridors.metadata), dtype=np.float32),
        "top_risk_per_cost": _feasible_selection(
            np.argsort(corridor_risk / np.maximum(costs, 1e-6))[::-1], corridors, config
        ),
        "risk_population_per_cost": _feasible_selection(
            np.argsort(
                corridor_risk * (1.0 + corridor_population / 10_000.0)
                / np.maximum(costs, 1e-6)
            )[::-1],
            corridors,
            config,
        ),
        "gradient_portfolio": np.asarray(optimized["selected"], dtype=np.float32),
    }
    rng = np.random.default_rng(20260805)
    strategies["random_feasible"] = _feasible_selection(
        rng.permutation(len(corridors.metadata)), corridors, config
    )
    if include_greedy:
        selected = np.zeros(len(corridors.metadata), dtype=np.float32)
        remaining = set(range(len(corridors.metadata)))
        current_loss, _ = evaluate_activation(
            jnp.asarray(selected),
            scenario,
            corridors,
            hazard_field,
            winds,
            config,
            backend="pure",
        )
        current_loss = float(current_loss)
        running_cost = 0.0
        running_households = 0.0
        costs = np.asarray(corridors.construction_cost_k_eur)
        relocations = np.asarray(corridors.construction_households)
        while remaining:
            best_index = None
            best_loss = current_loss
            for index in sorted(remaining):
                if (
                    running_cost + float(costs[index]) > config.budget_k_eur
                    or running_households + float(relocations[index])
                    > config.construction_household_cap
                ):
                    continue
                trial = selected.copy()
                trial[index] = 1.0
                loss, _ = evaluate_activation(
                    jnp.asarray(trial),
                    scenario,
                    corridors,
                    hazard_field,
                    winds,
                    config,
                    backend="pure",
                )
                current_loss = float(loss)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_index = index
            if best_index is None:
                break
            selected[best_index] = 1.0
            running_cost += float(costs[best_index])
            running_households += float(relocations[best_index])
            current_loss = best_loss
            remaining.remove(best_index)
        strategies["greedy_marginal"] = selected
    results: dict[str, dict[str, Any]] = {}
    for name, activation in strategies.items():
        loss, diagnostics = evaluate_activation(
            jnp.asarray(activation),
            scenario,
            corridors,
            hazard_field,
            winds,
            config,
            backend="pure",
        )
        results[name] = {
            "objective": float(loss),
            "selected_corridors": np.flatnonzero(activation > 0.5).tolist(),
            "burned_hectares_mean": float(jnp.mean(diagnostics["burned_hectares"])),
            "burned_hectares_p90": float(jnp.quantile(diagnostics["burned_hectares"], 0.9)),
            "exposed_population_mean": float(
                jnp.mean(diagnostics["exposed_population"])
            ),
            "exposed_households_mean": float(
                jnp.mean(diagnostics["exposed_households"])
            ),
            "construction_cost_k_eur": float(
                diagnostics["construction_cost_k_eur"]
            ),
            "construction_households": float(
                diagnostics["construction_households"]
            ),
            "ecological_footprint_hectares": float(
                diagnostics["ecological_footprint_hectares"]
            ),
        }
    return results
