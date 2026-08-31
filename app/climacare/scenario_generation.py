from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from climacare.health_model import health_impact
from climacare.objective import PARAMETER_NAMES, decode_free_parameters

__all__ = ["generate_scenario"]


def generate_scenario(
    pipeline,
    config,
    zones,
    n_scenarios,
    *,
    dt,
    cell_area,
    filter_level=0.0,
    seed=0,
    frame_count=None,
    theta_samples=None,
):
    """Generate fixed fire and health scenarios for portfolio optimization."""
    count = int(n_scenarios)
    if count < 1:
        raise ValueError("n_scenarios must be >= 1")
    if frame_count is None:
        frame_count = config.frame_count
    if theta_samples is not None:
        draws = np.asarray(theta_samples)
        if draws.ndim != 2 or draws.shape[1] != len(PARAMETER_NAMES):
            raise ValueError("theta_samples must have shape (S, 4)")
        if draws.shape[0] < count:
            raise ValueError("theta_samples contains fewer rows than n_scenarios")
        draws = draws[:count]
    else:
        draws = np.random.default_rng(seed).standard_normal((count, len(PARAMETER_NAMES)))

    fires = []
    concentrations = []
    for draw in draws:
        theta = (
            jnp.asarray(draw, dtype=jnp.float64)
            if theta_samples is not None
            else decode_free_parameters(jnp.asarray(draw), config)
        )
        output = pipeline.direct(theta, frame_count=frame_count)
        fires.append(output["burned_area"])
        concentrations.append(output["concentration_frames"])

    fire_scenarios = jnp.stack(fires)
    health_scenarios = health_impact(
        jnp.stack(concentrations),
        zones,
        dt=dt,
        cell_area=cell_area,
        filter_level=filter_level,
    )
    return fire_scenarios, health_scenarios
