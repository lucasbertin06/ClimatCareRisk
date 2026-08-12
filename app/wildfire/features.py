"""Feature construction on a common projected wildfire grid."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from wildfire.schema import GridSpec


@dataclass(frozen=True)
class FeatureBundle:
    """Static and dynamic features with their names and provenance."""

    values: jnp.ndarray
    names: tuple[str, ...]
    cutoff_time: str
    grid: GridSpec


def normalize_feature(
    value: jnp.ndarray, low: float = 0.0, high: float = 1.0
) -> jnp.ndarray:
    """Clip a feature into a stable range for the hazard model."""
    return jnp.clip((value - low) / (high - low + 1e-8), 0.0, 1.0)


def build_features(
    weather_history: jnp.ndarray,
    fuel: jnp.ndarray,
    slope: jnp.ndarray,
    distance_to_access: jnp.ndarray,
    population: jnp.ndarray,
    historical_frequency: jnp.ndarray,
    grid: GridSpec,
    cutoff_time: str,
) -> FeatureBundle:
    """Combine weather and static layers without future observations.

    ``weather_history`` has shape ``(history, height, width, 4)`` and stores
    temperature, humidity, precipitation, and wind speed. All static layers
    must already be rasterized to the common grid.
    """
    if weather_history.ndim != 4 or weather_history.shape[-1] != 4:
        raise ValueError("weather_history must have shape (time, height, width, 4)")
    spatial_shape = weather_history.shape[1:3]
    static_layers = (fuel, slope, distance_to_access, population, historical_frequency)
    if any(layer.shape != spatial_shape for layer in static_layers):
        raise ValueError("all static layers must match the weather grid")
    temperature = normalize_feature(weather_history[..., 0], -10.0, 45.0)
    humidity = normalize_feature(weather_history[..., 1], 0.0, 100.0)
    precipitation = normalize_feature(weather_history[..., 2], 0.0, 100.0)
    wind_speed = normalize_feature(weather_history[..., 3], 0.0, 40.0)
    dry_spell = 1.0 - jnp.clip(jnp.mean(precipitation, axis=0), 0.0, 1.0)
    static = jnp.stack(
        [
            normalize_feature(fuel),
            normalize_feature(slope),
            1.0 - normalize_feature(distance_to_access),
            normalize_feature(population),
            normalize_feature(historical_frequency),
            dry_spell,
        ],
        axis=-1,
    )
    dynamic = jnp.stack(
        [temperature, 1.0 - humidity, precipitation, wind_speed], axis=-1
    )
    values = jnp.concatenate(
        [
            dynamic,
            jnp.broadcast_to(static, (weather_history.shape[0], *spatial_shape, 6)),
        ],
        axis=-1,
    )
    names = (
        "temperature",
        "humidity_deficit",
        "precipitation",
        "wind_speed",
        "fuel",
        "slope",
        "access_proximity",
        "population",
        "historical_frequency",
        "dry_spell",
    )
    return FeatureBundle(values=values, names=names, cutoff_time=cutoff_time, grid=grid)


def grid_coordinates(
    latitude: jnp.ndarray,
    longitude: jnp.ndarray,
    grid: GridSpec,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map normalized pilot coordinates into integer grid indices.

    This helper intentionally expects coordinates already transformed to the
    configured projected CRS; it does not silently treat degrees as metres.
    """
    x = jnp.floor((longitude - grid.x_min_m) / grid.resolution_m).astype(jnp.int32)
    y = jnp.floor((latitude - grid.y_min_m) / grid.resolution_m).astype(jnp.int32)
    return (
        jnp.clip(y, 0, grid.height - 1),
        jnp.clip(x, 0, grid.width - 1),
    )
