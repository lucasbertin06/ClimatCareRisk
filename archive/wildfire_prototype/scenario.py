"""Synthetic wildfire scenario used for deterministic development and tests."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class WildfireScenario:
    """Gridded weather, landscape, labels, and intervention inputs."""

    features: jax.Array
    targets: dict[str, jax.Array]
    fuel: jax.Array
    slope: jax.Array
    wind: jax.Array
    population: jax.Array
    vulnerable_population: jax.Array
    temperature_history: jax.Array
    relative_humidity_history: jax.Array
    heat_stress: jax.Array
    heat_health_burden: jax.Array
    health_baseline_rate: jax.Array
    ecological_cost: jax.Array
    intervention_cost: jax.Array
    hazard_weights: jax.Array
    hazard_bias: jax.Array
    horizon_hours: jax.Array
    region: str
    crs: str
    dwelling_density: jax.Array | None = None
    forest_fraction: jax.Array | None = None
    road_access: jax.Array | None = None
    cell_area_hectares: float = 1.0


def make_scenario(
    seed: int = 0,
    history: int = 14,
    size: int = 20,
    channels: int = 8,
    horizons: int = 3,
) -> WildfireScenario:
    """Create a reproducible Mediterranean-style synthetic scenario."""
    if history < 2 or size < 5 or channels < 4 or horizons < 1:
        raise ValueError("scenario dimensions are too small")
    key = jax.random.PRNGKey(seed)
    feature_key, noise_key = jax.random.split(key)
    yy, xx = jnp.meshgrid(
        jnp.linspace(0.0, 1.0, size, dtype=jnp.float32),
        jnp.linspace(0.0, 1.0, size, dtype=jnp.float32),
        indexing="ij",
    )
    forest = jnp.clip(
        0.25
        + 0.7 * jnp.exp(-((xx - 0.62) ** 2 + (yy - 0.48) ** 2) / 0.18)
        + 0.1 * jnp.sin(8.0 * xx) * jnp.sin(5.0 * yy),
        0.0,
        1.0,
    )
    ridge = 0.35 * jnp.sin(2.5 * xx) + 0.2 * jnp.cos(3.5 * yy)
    slope = jnp.clip(jnp.abs(ridge + 0.25 * (yy - 0.5)), 0.0, 1.0)
    roads = jnp.exp(-((yy - (0.25 + 0.3 * xx)) ** 2) / 0.002)
    population = 0.15 + 2.5 * jnp.exp(-((xx - 0.78) ** 2 + (yy - 0.32) ** 2) / 0.03)
    ecological_cost = jnp.clip(
        0.25 + 0.7 * jnp.exp(-((xx - 0.38) ** 2 + (yy - 0.7) ** 2) / 0.04), 0.0, 1.0
    )
    intervention_cost = 0.5 + 1.5 * slope + 0.2 * (1.0 - roads)
    vulnerable_population = population * (
        0.2 + 0.8 * jnp.exp(-((xx - 0.7) ** 2 + (yy - 0.28) ** 2) / 0.08)
    )
    temperature_history = (
        jnp.broadcast_to(
            29.0 + 5.5 * forest + 0.4 * jnp.sin(4.0 * xx), (history, size, size)
        )
        + jnp.linspace(-1.0, 2.0, history, dtype=jnp.float32)[:, None, None]
    )
    relative_humidity_history = jnp.broadcast_to(
        45.0 - 12.0 * forest + 3.0 * yy, (history, size, size)
    )
    baseline_rate = jnp.array(0.08, dtype=jnp.float32)
    from wildfire_shared.health import heat_health_forward

    health = heat_health_forward(
        temperature_history,
        relative_humidity_history,
        vulnerable_population,
        baseline_rate,
    )
    static = jnp.stack([forest, slope, roads, population, ecological_cost], axis=-1)
    weather_base = jnp.stack(
        [
            0.65 + 0.1 * jnp.sin(2.0 * jnp.pi * xx),
            0.35 + 0.1 * yy,
            0.18 + 0.15 * (1.0 - forest),
        ],
        axis=-1,
    )
    weather = weather_base[None, ...] + 0.025 * jax.random.normal(
        feature_key, (history, size, size, 3)
    )
    temporal = jnp.linspace(-0.08, 0.08, history, dtype=jnp.float32)[
        :, None, None, None
    ]
    dynamic = (
        weather
        + temporal
        * jnp.stack([forest, -forest, jnp.ones_like(forest)], axis=-1)[None, ...]
    )
    features = jnp.concatenate(
        [dynamic, jnp.broadcast_to(static, (history, size, size, static.shape[-1]))],
        axis=-1,
    )
    true_weights = jnp.array(
        [
            [1.6, 1.2, 0.9, 0.4],
            [0.5, 0.8, 0.4, 0.7],
            [0.2, 0.4, 0.1, 0.2],
            [0.7, 1.1, 0.2, 0.5],
            [1.0, 0.3, 0.9, 0.2],
            [0.4, 0.2, 0.4, 0.2],
            [0.3, 0.2, 0.5, 0.1],
            [0.2, 0.4, 0.2, 0.5],
        ],
        dtype=jnp.float32,
    )
    true_bias = jnp.array([-2.1, -1.6, 0.0, -0.25], dtype=jnp.float32)
    signal = jnp.mean(features, axis=0)
    logits = jnp.einsum("hwc,co->hwo", signal, true_weights) + true_bias
    ignition = jax.nn.sigmoid(logits[..., 0])[..., None] * jnp.linspace(
        1.0, 0.7, horizons
    )
    growth = jax.nn.sigmoid(logits[..., 1])[..., None] * jnp.linspace(
        1.0, 0.72, horizons
    )
    clean_area = jax.nn.softplus(logits[..., 0] + logits[..., 1])[..., None]
    clean_area = clean_area * jnp.linspace(1.0, 0.75, horizons)
    noise = 0.04 * jax.random.uniform(noise_key, (size, size, horizons))
    burned_area = jnp.where(ignition > 0.5, clean_area + noise, 0.0)
    targets = {
        "ignition": (ignition > 0.55).astype(jnp.float32),
        "growth": (growth > 0.4).astype(jnp.float32),
        "burned_area": burned_area,
    }
    return WildfireScenario(
        features=features,
        targets=targets,
        fuel=forest,
        slope=slope,
        wind=jnp.array([0.8, 0.25], dtype=jnp.float32),
        population=population,
        vulnerable_population=vulnerable_population,
        temperature_history=temperature_history,
        relative_humidity_history=relative_humidity_history,
        heat_stress=health["heat_stress"],
        heat_health_burden=health["expected_excess_burden"],
        health_baseline_rate=baseline_rate,
        ecological_cost=ecological_cost,
        intervention_cost=intervention_cost,
        hazard_weights=true_weights * 0.75,
        hazard_bias=true_bias * 0.7,
        horizon_hours=jnp.array([24.0, 48.0, 72.0], dtype=jnp.float32)[:horizons],
        region="Var / Bouches-du-Rhône / Hérault pilot",
        crs="EPSG:2154",
        dwelling_density=population / 2.15,
        forest_fraction=forest,
        road_access=roads,
        cell_area_hectares=100.0,
    )
