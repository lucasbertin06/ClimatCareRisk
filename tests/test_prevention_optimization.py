from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from climacare.prevention import optimize_fuel_prevention, prevention_objective
from conftest import requires_containers


@requires_containers
def test_prevention_objective_has_a_valid_gradient(pipeline) -> None:
    theta = jnp.asarray(pipeline.config.truth)
    level = jnp.asarray(0.2, dtype=jnp.float64)

    value, gradient = jax.value_and_grad(prevention_objective)(
        level,
        pipeline,
        theta,
    )

    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))
    assert abs(float(gradient)) > 1e-12


@requires_containers
def test_optimize_fuel_prevention_decreases_the_coupled_objective(pipeline) -> None:
    theta = jnp.asarray(pipeline.config.truth)

    result = optimize_fuel_prevention(
        pipeline,
        theta,
        initial_level = 0.2,
        steps = 12,
        learning_rate = 0.2,
    )

    assert 0.0 <= result["level"] <= 1.0
    assert len(result["objective_history"]) == 13
    assert result["objective_history"][-1] < result["objective_history"][0]
    assert result["burned_area_final"] <= result["burned_area_initial"]
    assert result["smoke_exposure_final"] <= result["smoke_exposure_initial"]
