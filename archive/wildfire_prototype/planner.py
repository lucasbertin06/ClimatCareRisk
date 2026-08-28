"""Differentiable intervention planning for wildfire-risk scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax

from wildfire.scenario import WildfireScenario
from wildfire_shared.hazard import hazard_apply
from wildfire_shared.spread import spread_forward


@dataclass(frozen=True)
class PlannerConfig:
    """Weights and constraints for candidate intervention planning."""

    budget_fraction: float = 0.14
    area_weight: float = 1.0
    population_weight: float = 1.8
    ecology_weight: float = 0.35
    cost_weight: float = 0.2
    smoothness_weight: float = 0.08
    cvar_weight: float = 0.25
    health_weight: float = 2.4
    steps: int = 24


_DEFAULT_PLANNER_CONFIG = PlannerConfig()


def hazard_prediction(
    scenario: WildfireScenario, backend: str = "pure"
) -> dict[str, jax.Array]:
    """Predict hazard from the explicit scenario model parameters."""
    if backend not in {"pure", "tesseract"}:
        raise ValueError("backend must be 'pure' or 'tesseract'")
    if backend == "tesseract":
        from wildfire.runtime import hazard_apply as runtime_hazard_apply

        apply = runtime_hazard_apply
    else:
        apply = hazard_apply
    return apply(
        {
            "features": scenario.features,
            "weights": scenario.hazard_weights,
            "bias": scenario.hazard_bias,
            "horizons": scenario.horizon_hours.shape[0],
        }
    )


def _total_variation(mask: jax.Array) -> jax.Array:
    return jnp.mean(jnp.abs(mask[:, 1:] - mask[:, :-1])) + jnp.mean(
        jnp.abs(mask[1:, :] - mask[:-1, :])
    )


def _spread_inputs(
    scenario: WildfireScenario,
    hazard: jax.Array,
    mask: jax.Array,
    wind: jax.Array,
) -> dict[str, jax.Array]:
    return {
        "hazard": hazard,
        "fuel": scenario.fuel,
        "wind": wind,
        "slope": scenario.slope,
        "intervention": mask,
        "population": scenario.population,
        "ecological_cost": scenario.ecological_cost,
        "intervention_cost": scenario.intervention_cost,
    }


def planning_loss(
    latent_mask: jax.Array,
    scenario: WildfireScenario,
    config: PlannerConfig = _DEFAULT_PLANNER_CONFIG,
    hazard: dict[str, jax.Array] | None = None,
    wind_scenarios: jax.Array | None = None,
    backend: str = "pure",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Evaluate expected social, ecological, and budget-adjusted intervention loss."""
    if hazard is None:
        hazard = hazard_prediction(scenario, backend=backend)
    if wind_scenarios is None:
        wind_scenarios = jnp.stack(
            [scenario.wind, scenario.wind * jnp.array([1.15, 0.9]), scenario.wind * 0.8]
        )
    mask = jax.nn.sigmoid(latent_mask)
    total_budget = config.budget_fraction * jnp.sum(scenario.intervention_cost)
    if backend not in {"pure", "tesseract"}:
        raise ValueError("backend must be 'pure' or 'tesseract'")
    spread_function = spread_forward
    if backend == "tesseract":
        from wildfire.bridge import torch_spread

        def spread_function(**inputs: Any) -> dict[str, jax.Array]:
            outputs = torch_spread(
                inputs["hazard"],
                inputs["fuel"],
                inputs["wind"],
                inputs["slope"],
                inputs["intervention"],
                inputs["population"],
                inputs["ecological_cost"],
                inputs["intervention_cost"],
                int(inputs.get("steps", config.steps)),
            )
            return {
                "burn_probability": outputs[0],
                "trajectory": outputs[1],
                "burned_area": outputs[2],
                "exposed_population": outputs[3],
                "ecological_penalty": outputs[4],
                "intervention_cost": outputs[5],
            }

    results = jax.vmap(
        lambda wind: spread_function(
            hazard=jnp.mean(hazard["growth_probability"], axis=-1),
            fuel=scenario.fuel,
            wind=wind,
            slope=scenario.slope,
            intervention=mask,
            population=scenario.population,
            ecological_cost=scenario.ecological_cost,
            intervention_cost=scenario.intervention_cost,
            steps=config.steps,
        )
    )(wind_scenarios)
    baseline_results = jax.vmap(
        lambda wind: spread_function(
            hazard=jnp.mean(hazard["growth_probability"], axis=-1),
            fuel=scenario.fuel,
            wind=wind,
            slope=scenario.slope,
            intervention=jnp.zeros_like(mask),
            population=scenario.population,
            ecological_cost=scenario.ecological_cost,
            intervention_cost=scenario.intervention_cost,
            steps=config.steps,
        )
    )(wind_scenarios)
    health_burden = scenario.heat_health_burden
    if backend == "tesseract":
        from wildfire.runtime import health_apply as runtime_health_apply

        health = runtime_health_apply(
            {
                "temperature_c": scenario.temperature_history,
                "relative_humidity": scenario.relative_humidity_history,
                "vulnerable_population": scenario.vulnerable_population,
                "baseline_rate": scenario.health_baseline_rate,
            }
        )
        health_burden = health["expected_excess_burden"]
    compound_health_burden = jnp.sum(
        results["burn_probability"] * health_burden,
        axis=(-2, -1),
    )
    scenario_loss = (
        config.area_weight * results["burned_area"]
        + config.population_weight * results["exposed_population"]
        + config.health_weight * compound_health_burden
        + config.ecology_weight * results["ecological_penalty"]
    )
    sorted_loss = jnp.sort(scenario_loss)
    tail_count = max(1, scenario_loss.shape[0] // 3)
    cvar = jnp.mean(sorted_loss[-tail_count:])
    cost = results["intervention_cost"][0]
    budget_violation = jax.nn.relu(cost - total_budget) ** 2 / (total_budget + 1e-6)
    loss = (
        jnp.mean(scenario_loss)
        + config.cvar_weight * cvar
        + config.cost_weight * cost
        + config.smoothness_weight * _total_variation(mask)
        + 10.0 * budget_violation
    )
    diagnostics = {
        "mask": mask,
        "scenario_loss": scenario_loss,
        "burned_area": results["burned_area"],
        "exposed_population": results["exposed_population"],
        "compound_health_burden": compound_health_burden,
        "heat_health_burden": jnp.sum(health_burden),
        "ecological_penalty": results["ecological_penalty"],
        "intervention_cost": cost,
        "budget": total_budget,
        "cvar": cvar,
        "budget_violation": jnp.maximum(cost - total_budget, 0.0),
        "baseline_burn_probability": baseline_results["burn_probability"],
        "planned_burn_probability": results["burn_probability"],
        "baseline_trajectory": baseline_results["trajectory"],
        "planned_trajectory": results["trajectory"],
        "burn_probability_reduction": jnp.mean(
            baseline_results["burn_probability"] - results["burn_probability"], axis=0
        ),
        "hazard_by_horizon": hazard["growth_probability"],
    }
    return loss, diagnostics


def _project_mask(
    mask: jax.Array, scenario: WildfireScenario, config: PlannerConfig
) -> jax.Array:
    """Apply hard feasibility projection after relaxed gradient optimization."""
    budget = config.budget_fraction * jnp.sum(scenario.intervention_cost)
    allowed = (scenario.fuel > 0.35) & (scenario.slope < 0.82)
    scores = mask * allowed.astype(mask.dtype) / (scenario.intervention_cost + 1e-4)
    flat_order = jnp.argsort(scores.reshape(-1))[::-1]
    allowed_flat = allowed.reshape(-1)[flat_order]
    sorted_cost = scenario.intervention_cost.reshape(-1)[flat_order] * allowed_flat
    cumulative = jnp.cumsum(sorted_cost)
    selected = ((cumulative <= budget) & allowed_flat).astype(mask.dtype)
    projected = jnp.zeros_like(mask).reshape(-1).at[flat_order].set(selected)
    return projected.reshape(mask.shape)


def optimize_intervention(
    scenario: WildfireScenario,
    iterations: int = 80,
    learning_rate: float = 0.15,
    config: PlannerConfig = _DEFAULT_PLANNER_CONFIG,
    wind_scenarios: jax.Array | None = None,
    backend: str = "pure",
) -> dict[str, jax.Array]:
    """Optimize a relaxed intervention mask and return a feasible candidate mask."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    hazard = hazard_prediction(scenario, backend=backend)
    latent = jnp.full(scenario.fuel.shape, -4.0, dtype=jnp.float32)
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(latent)

    def step(
        values: tuple[jax.Array, optax.OptState], _: jax.Array
    ) -> tuple[tuple[jax.Array, optax.OptState], jax.Array]:
        current, opt_state = values
        (loss, _), gradient = jax.value_and_grad(planning_loss, has_aux=True)(
            current, scenario, config, hazard, wind_scenarios, backend
        )
        updates, next_state = optimizer.update(gradient, opt_state, current)
        return (optax.apply_updates(current, updates), next_state), loss

    (latent, _), losses = jax.lax.scan(step, (latent, state), jnp.arange(iterations))
    relaxed_loss, relaxed = planning_loss(
        latent,
        scenario,
        config,
        hazard,
        wind_scenarios=wind_scenarios,
        backend=backend,
    )
    feasible_mask = _project_mask(relaxed["mask"], scenario, config)
    final_loss, final = planning_loss(
        jnp.where(feasible_mask > 0.0, 8.0, -8.0),
        scenario,
        config,
        hazard,
        wind_scenarios=wind_scenarios,
        backend=backend,
    )
    return {
        "latent": latent,
        "losses": losses,
        "relaxed_loss": relaxed_loss,
        "relaxed_mask": relaxed["mask"],
        "feasible_mask": feasible_mask,
        "loss": final_loss,
        "hazard": jnp.mean(hazard["growth_probability"], axis=-1),
        "burned_area": final["burned_area"],
        "exposed_population": final["exposed_population"],
        "compound_health_burden": final["compound_health_burden"],
        "heat_health_burden": final["heat_health_burden"],
        "scenario_loss": final["scenario_loss"],
        "intervention_cost": final["intervention_cost"],
        "budget": final["budget"],
        "baseline_burn_probability": final["baseline_burn_probability"],
        "planned_burn_probability": final["planned_burn_probability"],
        "baseline_trajectory": final["baseline_trajectory"],
        "planned_trajectory": final["planned_trajectory"],
        "burn_probability_reduction": final["burn_probability_reduction"],
        "hazard_by_horizon": final["hazard_by_horizon"],
    }
