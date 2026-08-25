from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from climacare.config import PARAMETER_NAMES, TinyConfig
from climacare.health import HealthZone, incremental_health_impact, mean_exposure
from climacare.pipeline import TesseractPipeline

__all__ = ["posterior_theta_samples", "simulate_scenarios"]


def posterior_theta_samples(
    laplace : dict,
    rng_key : jax.Array,
    num_samples : int = 200,) -> np.ndarray :
    # Draw theta samples from the Laplace approximation returned by uq.laplace_approx()
    # (a dict with keys "theta_map" and "covariance"). This is the single sampling API :
    # uq.laplace_approx -> scenarios.posterior_theta_samples -> generate_scenario(theta_samples=...)
    mean = jnp.asarray(laplace["theta_map"])
    cov = jnp.asarray(laplace["covariance"])
    draws = jax.random.multivariate_normal(rng_key, mean = mean, cov = cov, shape = (num_samples,),)
    return np.asarray(draws)


def simulate_scenarios(
    pipeline : TesseractPipeline,
    thetas : np.ndarray,
    zones : list[HealthZone],
    cell_area : float,
    filter_level: float = 0.0,) -> dict :
    # Thin wrapper around src.generate_scenario (the single scenario generator),
    # kept so older call sites keep working. Returns a dict instead of a class.
    from src.generate_scenario import generate_scenario

    config = pipeline.config
    scenarios_fire, scenarios_H_r = generate_scenario(
        pipeline, config, zones, int(thetas.shape[0]),
        dt = config.dt, cell_area = cell_area,
        filter_level = filter_level, theta_samples = thetas,
    )
    return {"scenarios_fire": scenarios_fire, "scenarios_H_r" : scenarios_H_r}
