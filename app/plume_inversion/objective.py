"""Forward model and inverse objective for plume source recovery."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from plume_inversion.bridge import sensor_nll
from plume_inversion.scenario import Scenario
from plume_inversion.tesseract_runtime import plume_sensor_concentration
from plume_inversion_shared.sensor import sensor_apply_jax


def unpack_parameters(
    raw: jax.Array, sensor_count: int, noise_scale: jax.Array
) -> dict[str, jax.Array]:
    """Map unconstrained optimizer coordinates to physical parameters."""
    return {
        "source_position": jax.nn.sigmoid(raw[:2]),
        "source_time": 0.35 * jax.nn.sigmoid(raw[2]),
        "source_rate": jnp.exp(raw[3]),
        "gains": jnp.exp(raw[4 : 4 + sensor_count]),
        "biases": jnp.zeros(sensor_count, dtype=raw.dtype),
        "time_constants": jnp.ones(sensor_count, dtype=raw.dtype) * 0.025,
        "noise_scale": noise_scale.astype(raw.dtype),
    }


def _plume_inputs(raw: jax.Array, scenario: Scenario) -> dict[str, jax.Array]:
    parameters = unpack_parameters(
        raw, scenario.sensor_positions.shape[0], scenario.noise_scale
    )
    return {
        "source_position": parameters["source_position"],
        "source_time": parameters["source_time"],
        "source_rate": parameters["source_rate"],
        "wind": scenario.wind,
        "diffusivity": scenario.diffusivity,
        "decay_rate": scenario.decay_rate,
        "times": scenario.times,
        "sensor_positions": scenario.sensor_positions,
        "obstacle_mask": scenario.obstacle_mask,
        "spatial_sigma": scenario.spatial_sigma,
        "temporal_sigma": scenario.temporal_sigma,
        "dt": scenario.dt,
    }


def forward_concentration(raw: jax.Array, scenario: Scenario) -> jax.Array:
    """Simulate concentration at each sensor for raw optimizer parameters."""
    return plume_sensor_concentration(_plume_inputs(raw, scenario))


def loss_jax(raw: jax.Array, scenario: Scenario) -> jax.Array:
    """Evaluate the all-JAX reference negative log posterior."""
    parameters = unpack_parameters(
        raw, scenario.sensor_positions.shape[0], scenario.noise_scale
    )
    result = sensor_apply_jax(
        {
            "concentration": forward_concentration(raw, scenario),
            "gains": parameters["gains"],
            "biases": parameters["biases"],
            "time_constants": parameters["time_constants"],
            "observations": scenario.observations,
            "dt": scenario.times[1] - scenario.times[0],
            "noise_scale": parameters["noise_scale"],
        }
    )
    return result["nll"] + 0.02 * jnp.mean(raw[4:] ** 2)


def loss_hybrid(raw: jax.Array, scenario: Scenario) -> jax.Array:
    """Evaluate the composed plume and sensor negative log posterior."""
    parameters = unpack_parameters(
        raw, scenario.sensor_positions.shape[0], scenario.noise_scale
    )
    nll = sensor_nll(
        forward_concentration(raw, scenario),
        parameters["gains"],
        parameters["biases"],
        parameters["time_constants"],
        scenario.observations,
        scenario.times[1] - scenario.times[0],
        parameters["noise_scale"],
    )
    return nll + 0.02 * jnp.mean(raw[4:] ** 2)
