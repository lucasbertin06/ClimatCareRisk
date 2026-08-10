"""Runtime adapters for executing the composed Tesseract images."""

from __future__ import annotations

import atexit

import jax

from plume_inversion_shared.physics import plume_apply as pure_plume_apply

_plume_client = None


def tesseract_enabled() -> bool:
    """Return whether application calls should use built Tesseract images."""
    import os

    return os.getenv("PLUME_INVERSION_USE_TESSERACT", "0") == "1"


def _get_plume_client():
    global _plume_client
    if _plume_client is None:
        from tesseract_core import Tesseract

        _plume_client = Tesseract.from_image("plume_solver_jax")
        _plume_client.serve()
    return _plume_client


def _teardown() -> None:
    if _plume_client is not None:
        _plume_client.teardown()


atexit.register(_teardown)


def plume_apply(inputs: dict) -> dict:
    """Run the plume solver through a Tesseract or its pure reference."""
    if not tesseract_enabled():
        return pure_plume_apply(inputs)
    from tesseract_jax import apply_tesseract

    return apply_tesseract(_get_plume_client(), inputs)


def plume_sensor_concentration(inputs: dict) -> jax.Array:
    """Run the plume component and return its sensor traces."""
    return plume_apply(inputs)["sensor_concentration"]
