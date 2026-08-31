r"""Smoke exposure and incremental health impact, specification section 10.

The coefficients are synthetic. Nothing here is a clinical prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = ["HealthZone", "incremental_health_impact", "mean_exposure", "zone_population"]


@dataclass(frozen=True)
class HealthZone:
    """Synthetic exposure zone with a finite, non-negative density field."""

    name: str
    density: jax.Array
    baseline: float
    slope: float
    filter_efficiency: float

    def __post_init__(self) -> None:
        if getattr(self.density, "ndim", None) != 2:
            raise ValueError(f"zone {self.name}: density must be a two-dimensional field")
        if not bool(jnp.all(jnp.isfinite(self.density))):
            raise ValueError(f"zone {self.name}: density must be finite")
        if bool(jnp.any(self.density < 0.0)):
            raise ValueError(f"zone {self.name}: density must be non-negative")
        if not math.isfinite(self.baseline) or not math.isfinite(self.slope):
            raise ValueError(f"zone {self.name}: coefficients must be finite")
        if not math.isfinite(self.filter_efficiency):
            raise ValueError(f"zone {self.name}: filter efficiency must be finite")
        if self.slope < 0.0:
            raise ValueError(f"zone {self.name}: slope b_r must be non-negative")
        if not 0.0 <= self.filter_efficiency <= 1.0:
            raise ValueError(f"zone {self.name}: filter efficiency must lie in [0, 1]")


def zone_population(zone: HealthZone, cell_area: float) -> jax.Array:
    r"""Return :math:`N_r = \Delta x\Delta y\sum_{ij} p_{r,ij}`."""
    if not math.isfinite(cell_area) or cell_area <= 0.0:
        raise ValueError(f"cell_area must be finite and positive, got {cell_area}")
    total = cell_area * jnp.sum(zone.density)
    if float(total) <= 0.0:
        raise ValueError(f"zone {zone.name}: population must be strictly positive")
    return total


def mean_exposure(
    concentration: jax.Array,
    zone: HealthZone,
    *,
    dt: float,
    cell_area: float,
    filter_level: jax.Array | float = 0.0,
) -> jax.Array:
    r"""Return the mean dose per person :math:`e_r` of section 10."""
    if getattr(concentration, "ndim", None) != 3:
        raise ValueError("concentration must have shape (n_levels, ny, nx)")
    if tuple(concentration.shape[1:]) != tuple(zone.density.shape):
        raise ValueError(
            "concentration and density grids disagree: "
            f"{tuple(concentration.shape[1:])} vs {tuple(zone.density.shape)}"
        )
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    population = zone_population(zone, cell_area)
    weighted = jnp.tensordot(concentration, zone.density, axes=((1, 2), (0, 1)))
    attenuation = 1.0 - zone.filter_efficiency * jnp.clip(filter_level, 0.0, 1.0)
    integral = dt * cell_area * jnp.sum(weighted) * attenuation
    return integral / population


def incremental_health_impact(
    exposure: jax.Array, zone: HealthZone, cell_area: float
) -> jax.Array:
    r"""Return :math:`\Delta H_r` of section 10, exactly zero at zero exposure."""
    population = zone_population(zone, cell_area)
    activated = jax.nn.softplus(zone.baseline + zone.slope * exposure)
    baseline = jax.nn.softplus(jnp.asarray(zone.baseline, dtype=exposure.dtype))
    return population * (activated - baseline)
