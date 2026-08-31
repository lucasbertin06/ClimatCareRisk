from __future__ import annotations

import jax
import jax.numpy as jnp

from climacare.health import HealthZone, incremental_health_impact, mean_exposure

__all__ = ["HealthZone", "health_impact", "incremental_health_impact", "mean_exposure"]


def health_impact(
    concentration_scenarios,
    zones,
    *,
    dt,
    cell_area,
    filter_level=0.0,
):
    """Convert scenario concentration histories into zone health impacts."""

    def per_scenario(concentration):
        return jnp.stack(
            [
                incremental_health_impact(
                    mean_exposure(
                        concentration,
                        zone,
                        dt=dt,
                        cell_area=cell_area,
                        filter_level=filter_level,
                    ),
                    zone,
                    cell_area,
                )
                for zone in zones
            ]
        )

    return jax.vmap(per_scenario)(concentration_scenarios)
