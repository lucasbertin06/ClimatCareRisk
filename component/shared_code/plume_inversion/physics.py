"""Differentiable two-dimensional tracer transport."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

DEFAULT_GRID_SIZE = 32


def _grid(size: int = DEFAULT_GRID_SIZE) -> tuple[jax.Array, jax.Array]:
    axis = jnp.linspace(0.0, 1.0, size, dtype=jnp.float32)
    return jnp.meshgrid(axis, axis, indexing="xy")


def _sample_sensor_field(field: jax.Array, sensor_positions: jax.Array) -> jax.Array:
    size = field.shape[-1]
    coordinates = jnp.clip(sensor_positions, 0.0, 1.0) * (size - 1)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, size - 1)
    y1 = jnp.minimum(y0 + 1, size - 1)
    wx = x - x0
    wy = y - y0
    top_left = field[:, y0, x0]
    top_right = field[:, y0, x1]
    bottom_left = field[:, y1, x0]
    bottom_right = field[:, y1, x1]
    return (
        (1.0 - wx)[None, :] * (1.0 - wy)[None, :] * top_left
        + wx[None, :] * (1.0 - wy)[None, :] * top_right
        + (1.0 - wx)[None, :] * wy[None, :] * bottom_left
        + wx[None, :] * wy[None, :] * bottom_right
    )


def simulate_plume(
    source_position: jax.Array,
    source_time: jax.Array,
    source_rate: jax.Array,
    wind: jax.Array,
    diffusivity: jax.Array,
    decay_rate: jax.Array,
    times: jax.Array,
    sensor_positions: jax.Array,
    obstacle_mask: jax.Array,
    spatial_sigma: jax.Array,
    temporal_sigma: jax.Array,
    dt: jax.Array,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> dict[str, jax.Array]:
    """Integrate a smooth advection-diffusion-release model."""
    grid_x, grid_y = _grid(grid_size)
    dx = 1.0 / (grid_size - 1)
    mask = 1.0 - jnp.clip(obstacle_mask, 0.0, 1.0)
    source_shape = jnp.exp(
        -0.5
        * (
            ((grid_x - source_position[0]) / spatial_sigma) ** 2
            + ((grid_y - source_position[1]) / spatial_sigma) ** 2
        )
    )
    source_shape = source_shape / (source_shape.sum() * dx * dx + 1e-6)

    def step(concentration: jax.Array, time: jax.Array) -> tuple[jax.Array, jax.Array]:
        left = jnp.concatenate(
            [jnp.zeros_like(concentration[:, :1]), concentration[:, :-1]], axis=1
        )
        right = jnp.concatenate(
            [concentration[:, 1:], jnp.zeros_like(concentration[:, :1])], axis=1
        )
        down = jnp.concatenate(
            [jnp.zeros_like(concentration[:1, :]), concentration[:-1, :]], axis=0
        )
        up = jnp.concatenate(
            [concentration[1:, :], jnp.zeros_like(concentration[:1, :])], axis=0
        )
        gradient_x = (
            jnp.where(wind[0] >= 0.0, concentration - left, right - concentration) / dx
        )
        gradient_y = (
            jnp.where(wind[1] >= 0.0, concentration - down, up - concentration) / dx
        )
        laplacian = (left + right + down + up - 4.0 * concentration) / (dx * dx)
        temporal_profile = jnp.exp(-0.5 * ((time - source_time) / temporal_sigma) ** 2)
        source = source_rate * temporal_profile * source_shape * mask
        updated = concentration + dt * (
            -wind[0] * gradient_x
            - wind[1] * gradient_y
            + diffusivity * laplacian
            - decay_rate * concentration
            + source
        )
        updated = updated * mask
        return updated, updated

    initial = jnp.zeros((grid_size, grid_size), dtype=jnp.float32)
    _, fields = jax.lax.scan(step, initial, times)
    sensor_concentration = _sample_sensor_field(fields, sensor_positions)
    return {"field": fields, "sensor_concentration": sensor_concentration}


def plume_apply(inputs: dict[str, Any]) -> dict[str, jax.Array]:
    """Adapt a serialized Tesseract input mapping to the transport solver."""
    return simulate_plume(
        source_position=inputs["source_position"],
        source_time=inputs["source_time"],
        source_rate=inputs["source_rate"],
        wind=inputs["wind"],
        diffusivity=inputs["diffusivity"],
        decay_rate=inputs["decay_rate"],
        times=inputs["times"],
        sensor_positions=inputs["sensor_positions"],
        obstacle_mask=inputs["obstacle_mask"],
        spatial_sigma=inputs["spatial_sigma"],
        temporal_sigma=inputs["temporal_sigma"],
        dt=inputs["dt"],
    )
