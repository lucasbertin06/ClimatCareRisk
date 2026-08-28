"""Runtime adapters for the composed wildfire Tesseracts."""

from __future__ import annotations

import atexit

import jax

from wildfire.bridge import torch_spread
from wildfire_shared.hazard import hazard_apply as pure_hazard_apply
from wildfire_shared.spread import spread_forward

_hazard_client = None
_health_client = None


def tesseract_enabled() -> bool:
    """Return whether real Tesseract images should be used."""
    import os

    return os.getenv("WILDFIRE_USE_TESSERACT", "0") == "1"


def _get_hazard_client():
    global _hazard_client
    if _hazard_client is None:
        from tesseract_core import Tesseract

        _hazard_client = Tesseract.from_image("wildfire_hazard_jax")
        _hazard_client.serve()
    return _hazard_client


def _get_health_client():
    global _health_client
    if _health_client is None:
        from tesseract_core import Tesseract

        _health_client = Tesseract.from_image("heat_health_jax")
        _health_client.serve()
    return _health_client


def _teardown() -> None:
    if _hazard_client is not None:
        _hazard_client.teardown()
    if _health_client is not None:
        _health_client.teardown()


atexit.register(_teardown)


def health_apply(inputs: dict) -> dict:
    """Run the heat-health component through Tesseract or its pure reference."""
    from wildfire_shared.health import health_apply as pure_health_apply

    if not tesseract_enabled():
        return pure_health_apply(inputs)
    from tesseract_jax import apply_tesseract

    return apply_tesseract(_get_health_client(), inputs)


def hazard_apply(inputs: dict) -> dict:
    """Run the hazard component through Tesseract or its pure reference."""
    if not tesseract_enabled():
        return pure_hazard_apply(inputs)
    from tesseract_jax import apply_tesseract

    return apply_tesseract(_get_hazard_client(), inputs)


def spread_apply(inputs: dict) -> dict:
    """Run the pure spread kernel for non-differentiated inspection."""
    return spread_forward(**inputs)


def composed_spread_objective(
    hazard: jax.Array,
    fuel: jax.Array,
    wind: jax.Array,
    slope: jax.Array,
    intervention: jax.Array,
    population: jax.Array,
    ecological_cost: jax.Array,
    intervention_cost: jax.Array,
    steps: int = 24,
) -> jax.Array:
    """Evaluate the Torch Tesseract objective through the explicit VJP bridge."""
    outputs = torch_spread(
        hazard,
        fuel,
        wind,
        slope,
        intervention,
        population,
        ecological_cost,
        intervention_cost,
        steps,
    )
    return outputs[2] + 1.8 * outputs[3] + 0.35 * outputs[4] + 0.2 * outputs[5]
