r"""Smoke exposure and incremental health impact, specification section 10.

Native JAX, downstream of the pipeline and deliberately outside the MAP
likelihood. The two invariants that matter are:

* the dose is a **mean per person**, so it divides by the zone population
  :math:`N_r = \Delta x\Delta y\sum p_{r,ij}`;
* the incremental impact vanishes exactly at zero exposure, because it is
  written as a softplus difference.

The coefficients are synthetic. Nothing here is a clinical prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = ["HealthZone", "incremental_health_impact", "mean_exposure", "zone_population"]


@dataclass(frozen=True)
class HealthZone:
    """Synthetic exposure zone."""

    name: str
    density: jax.Array  # (ny, nx), non-negative synthetic population density
    baseline: float  # a_r
    slope: float  # b_r >= 0
    filter_efficiency: float  # eta_filter,r in [0, 1]

    def __post_init__(self) -> None:
        if self.slope < 0.0:
            raise ValueError(f"zone {self.name}: slope b_r must be non-negative")
        if not 0.0 <= self.filter_efficiency <= 1.0:
            raise ValueError(f"zone {self.name}: filter efficiency must lie in [0, 1]")


def zone_population(zone: HealthZone, cell_area: float) -> jax.Array:
    r"""Return :math:`N_r = \Delta x\Delta y\sum_{ij} p_{r,ij}`."""
    total = cell_area * jnp.sum(zone.density)
    return total


def mean_exposure(
    concentration: jax.Array,
    zone: HealthZone,
    *,
    dt: float,
    cell_area: float,
    filter_level: jax.Array | float = 0.0,
) -> jax.Array:
    r"""Return the mean dose per person :math:`e_r` of section 10.

    Args:
        concentration: history of shape ``(n_levels, ny, nx)``.
        zone: exposure zone carrying the synthetic density and coefficients.
        dt: integration step between two levels.
        cell_area: :math:`\Delta x\Delta y`.
        filter_level: continuous filtration intensity :math:`u_{filter,r}`.

    Returns:
        The scalar mean dose, zero when the concentration vanishes.

    Raises:
        ValueError: if the zone population is not strictly positive.
    """
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
