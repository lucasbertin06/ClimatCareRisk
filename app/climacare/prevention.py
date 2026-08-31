from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from climacare.pipeline import TesseractPipeline


def prevention_field(
    level: jax.Array | float,
    pipeline: TesseractPipeline,
) -> jax.Array:
    # Uniform C0 intervention: one scalar investment level controls the
    # prevention field over every cell carrying baseline fuel.
    clipped = jnp.clip(jnp.asarray(level, dtype=jnp.float64), 0.0, 1.0)
    fuel_mask = (jnp.asarray(pipeline.config.fuel_base) > 0.0).astype(jnp.float64)
    return clipped * fuel_mask


def prevention_outputs(
    level: jax.Array | float,
    pipeline: TesseractPipeline,
    theta: jax.Array,
) -> dict[str, Any]:
    field = prevention_field(level, pipeline)
    return pipeline.direct(theta, frame_count=1, fuel_prevention=field)


def prevention_objective(
    level: jax.Array,
    pipeline: TesseractPipeline,
    theta: jax.Array,
    burn_weight: float = 1.0,
    smoke_weight: float = 1.0,
    investment_weight: float = 0.05,
) -> jax.Array:
    # This scalar contains both Tesseract calls, so dJ/du_fuel crosses the
    # PyTorch FireSpread VJP and the C++ SmokeTransport adjoint.
    out = prevention_outputs(level, pipeline, theta)
    smoke_exposure = jnp.mean(out["sensor_concentration"])
    investment_cost = investment_weight * jnp.square(level)
    return (
        burn_weight * out["burned_area"]
        + smoke_weight * smoke_exposure
        + investment_cost
    )


def optimize_fuel_prevention(
    pipeline: TesseractPipeline,
    theta: jax.Array,
    initial_level: float = 0.2,
    steps: int = 20,
    learning_rate: float = 0.2,
    burn_weight: float = 1.0,
    smoke_weight: float = 1.0,
    investment_weight: float = 0.05,
) -> dict[str, Any]:
    objective = lambda level: prevention_objective(
        level,
        pipeline,
        theta,
        burn_weight,
        smoke_weight,
        investment_weight,
    )
    value_and_grad = jax.value_and_grad(objective)

    level = jnp.asarray(initial_level, dtype=jnp.float64)
    objective_history = [float(objective(level))]
    level_history = [float(level)]

    initial = prevention_outputs(level, pipeline, theta)
    burned_area_initial = float(initial["burned_area"])
    smoke_exposure_initial = float(jnp.mean(initial["sensor_concentration"]))

    for _ in range(steps):
        value, gradient = value_and_grad(level)
        step = learning_rate
        candidate = jnp.clip(level - step * gradient, 0.0, 1.0)
        candidate_value = objective(candidate)

        while float(candidate_value) >= float(value) and step > 1e-5:
            step *= 0.5
            candidate = jnp.clip(level - step * gradient, 0.0, 1.0)
            candidate_value = objective(candidate)

        if float(candidate_value) >= float(value):
            objective_history.extend([float(value)] * (steps - len(level_history) + 1))
            level_history.extend([float(level)] * (steps - len(level_history) + 1))
            break

        level = candidate
        objective_history.append(float(candidate_value))
        level_history.append(float(level))

    if len(objective_history) < steps + 1:
        objective_history.extend([objective_history[-1]] * (steps + 1 - len(objective_history)))
        level_history.extend([level_history[-1]] * (steps + 1 - len(level_history)))

    final = prevention_outputs(level, pipeline, theta)

    return {
        "level": float(level),
        "level_history": level_history,
        "objective_history": objective_history,
        "burned_area_initial": burned_area_initial,
        "burned_area_final": float(final["burned_area"]),
        "smoke_exposure_initial": smoke_exposure_initial,
        "smoke_exposure_final": float(jnp.mean(final["sensor_concentration"])),
    }
