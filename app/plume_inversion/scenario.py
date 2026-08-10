"""Canonical synthetic urban-canyon scenario."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from plume_inversion_shared.physics import simulate_plume
from plume_inversion_shared.sensor import sensor_apply_jax


@dataclass(frozen=True)
class Scenario:
    """All fixed inputs and observations for one inversion experiment."""

    times: jnp.ndarray
    sensor_positions: jnp.ndarray
    obstacle_mask: jnp.ndarray
    wind: jnp.ndarray
    diffusivity: jnp.ndarray
    decay_rate: jnp.ndarray
    spatial_sigma: jnp.ndarray
    temporal_sigma: jnp.ndarray
    dt: jnp.ndarray
    observations: jnp.ndarray
    noise_scale: jnp.ndarray
    true_parameters: jnp.ndarray


def make_obstacle_mask(size: int = 32) -> jnp.ndarray:
    """Create two grid-aligned blocked building strips."""
    mask = jnp.zeros((size, size), dtype=jnp.float32)
    mask = mask.at[size // 3 : 2 * size // 3, size // 5 : size // 5 + 2].set(1.0)
    mask = mask.at[size // 4 : 3 * size // 4, 3 * size // 5 : 3 * size // 5 + 2].set(
        1.0
    )
    return mask


def make_scenario(
    seed: int = 0,
    grid_size: int = 32,
    steps: int = 300,
) -> Scenario:
    """Build a deterministic noisy tracer-release scenario."""
    times = jnp.arange(steps, dtype=jnp.float32) * 0.005
    sensor_positions = jnp.array(
        [[0.45, 0.42], [0.55, 0.42], [0.65, 0.42], [0.55, 0.55], [0.65, 0.55]],
        dtype=jnp.float32,
    )
    obstacle_mask = make_obstacle_mask(grid_size)
    wind = jnp.array([0.08, 0.015], dtype=jnp.float32)
    diffusivity = jnp.array(0.00035, dtype=jnp.float32)
    decay_rate = jnp.array(0.015, dtype=jnp.float32)
    spatial_sigma = jnp.array(0.045, dtype=jnp.float32)
    temporal_sigma = jnp.array(0.035, dtype=jnp.float32)
    dt = jnp.array(0.001, dtype=jnp.float32)
    noise_scale = jnp.array(0.02, dtype=jnp.float32)
    true_parameters = jnp.array([0.32, 0.42, 0.25, 2.0], dtype=jnp.float32)
    plume = simulate_plume(
        source_position=true_parameters[:2],
        source_time=true_parameters[2],
        source_rate=true_parameters[3],
        wind=wind,
        diffusivity=diffusivity,
        decay_rate=decay_rate,
        times=times,
        sensor_positions=sensor_positions,
        obstacle_mask=obstacle_mask,
        spatial_sigma=spatial_sigma,
        temporal_sigma=temporal_sigma,
        dt=dt,
    )
    sensor_parameters = {
        "concentration": plume["sensor_concentration"],
        "gains": jnp.ones(sensor_positions.shape[0], dtype=jnp.float32),
        "biases": jnp.zeros(sensor_positions.shape[0], dtype=jnp.float32),
        "time_constants": jnp.ones(sensor_positions.shape[0], dtype=jnp.float32)
        * 0.025,
        "observations": jnp.zeros(
            (steps, sensor_positions.shape[0]), dtype=jnp.float32
        ),
        "dt": times[1] - times[0],
        "noise_scale": noise_scale,
    }
    clean_observations = sensor_apply_jax(sensor_parameters)["predicted"]
    noise = (
        0.35
        * noise_scale
        * jax.random.normal(jax.random.PRNGKey(seed), clean_observations.shape)
    )
    return Scenario(
        times=times,
        sensor_positions=sensor_positions,
        obstacle_mask=obstacle_mask,
        wind=wind,
        diffusivity=diffusivity,
        decay_rate=decay_rate,
        spatial_sigma=spatial_sigma,
        temporal_sigma=temporal_sigma,
        dt=dt,
        observations=clean_observations + noise,
        noise_scale=noise_scale,
        true_parameters=true_parameters,
    )
