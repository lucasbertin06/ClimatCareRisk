"""Compatibility wrapper for the installable health model."""

from climacare.health_model import (
    HealthZone,
    health_impact,
    incremental_health_impact,
    mean_exposure,
)

__all__ = ["HealthZone", "health_impact", "incremental_health_impact", "mean_exposure"]
