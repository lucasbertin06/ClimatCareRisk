"""Composed FireSpread to SmokeTransport pipeline, differentiated end to end.

A single JAX function calls two Tesseract containers in sequence:

1. ``fire_spread_torch`` — PyTorch, CPU, differentiated by PyTorch autodiff;
2. ``smoke_transport_cpp`` — C++20/OpenMP, differentiated by a hand-written
   discrete adjoint.

``jax.value_and_grad`` of the loss therefore triggers the C++ adjoint first, and
feeds its smoke-source cotangent straight into the PyTorch VJP. No intermediate
file is involved and no solver runs in the host process.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from climacare.config import TinyConfig

__all__ = [
    "FIRE_IMAGE",
    "SMOKE_IMAGE",
    "TesseractPipeline",
    "enable_float64",
    "open_pipeline",
    "pipeline_versions",
]

FIRE_IMAGE = os.environ.get("CLIMACARE_FIRE_IMAGE", "fire_spread_torch")
SMOKE_IMAGE = os.environ.get("CLIMACARE_SMOKE_IMAGE", "smoke_transport_cpp")


def enable_float64() -> None:
    """Turn on JAX double precision, the reference for every C0 gradient."""
    jax.config.update("jax_enable_x64", True)


def pipeline_versions() -> dict[str, str]:
    """Return the versions actually loaded in the host process."""
    import numpy
    import tesseract_core
    import tesseract_jax

    return {
        "python": os.environ.get("PYTHON_VERSION", ""),
        "jax": jax.__version__,
        "numpy": numpy.__version__,
        "tesseract_core": tesseract_core.__version__,
        "tesseract_jax": tesseract_jax.__version__,
        "fire_image": FIRE_IMAGE,
        "smoke_image": SMOKE_IMAGE,
    }


@dataclass
class TesseractPipeline:
    """Two served Tesseracts plus the configuration that binds them."""

    config: TinyConfig
    fire_client: Any
    smoke_client: Any

    # -- input construction ------------------------------------------------ #
    def _direction(self, delta_phi: jax.Array) -> jax.Array:
        angle = self.config.wind.phi_base + delta_phi
        return jnp.stack([jnp.cos(angle), jnp.sin(angle)])

    def fire_inputs(
        self,
        theta: jax.Array,
        frame_count: int = 1,
        fuel_prevention: jax.Array | None = None,
    ) -> dict[str, Any]:
        """Return the FireSpread payload for a physical parameter vector."""
        config = self.config
        fire = config.fire
        direction = self._direction(theta[3])
        prevention = (
            jnp.asarray(config.fuel_prevention)
            if fuel_prevention is None
            else jnp.asarray(fuel_prevention)
        )
        return {
            "ignition": jnp.stack([theta[0], theta[1], theta[2]]),
            "wind": config.wind.fire_speed * direction,
            "moisture": jnp.asarray(config.moisture),
            "fuel_base": jnp.asarray(config.fuel_base),
            "fuel_prevention": prevention,
            "dt": config.dt,
            "n_steps": config.n_steps,
            "diffusivity": fire.diffusivity,
            "heat_loss": fire.heat_loss,
            "heat_release": fire.heat_release,
            "reaction_rate": fire.reaction_rate,
            "moisture_sensitivity": fire.moisture_sensitivity,
            "ignition_threshold": fire.ignition_threshold,
            "ignition_width": fire.ignition_width,
            "source_sigma": fire.source_sigma,
            "smoke_yield": fire.smoke_yield,
            "wind_speed_bound": config.wind.fire_speed,
            "frame_count": frame_count,
        }

    def smoke_inputs(
        self, theta: jax.Array, smoke_source: jax.Array, frame_count: int = 1
    ) -> dict[str, Any]:
        """Return the SmokeTransport payload for a source tensor."""
        config = self.config
        direction = self._direction(theta[3])
        return {
            "smoke_source": smoke_source,
            "wind": config.wind.smoke_speed * direction,
            "sensor_positions": jnp.asarray(config.sensors.positions),
            "sensor_bias": jnp.asarray(config.sensors.bias),
            "diffusivity": config.smoke.diffusivity,
            "decay": config.smoke.decay,
            "dt": config.dt,
            "frame_count": frame_count,
        }

    # -- composed evaluations ---------------------------------------------- #
    def sensor_predictions(self, theta: jax.Array) -> jax.Array:
        """Return the sensor predictions of shape ``(N_t, S)``.

        This is the differentiated composition: both Tesseract calls live inside
        one JAX function, so ``jax.grad`` crosses both container boundaries.
        """
        from tesseract_jax import apply_tesseract

        fire = apply_tesseract(self.fire_client, self.fire_inputs(theta, 1), vmap_method="sequential")
        smoke = apply_tesseract(
            self.smoke_client, self.smoke_inputs(theta, fire["smoke_source"], 1),
            vmap_method="sequential",
        )
        return smoke["sensor_concentration"]

    def direct(
        self,
        theta: jax.Array,
        frame_count: int | None = None,
        fuel_prevention: jax.Array | None = None,
    ) -> dict[str, Any]:
        """Run both Tesseracts once and return every diagnostic field."""
        from tesseract_jax import apply_tesseract

        frames = self.config.frame_count if frame_count is None else frame_count
        fire = apply_tesseract(
            self.fire_client,
            self.fire_inputs(theta, frames, fuel_prevention),
            vmap_method="sequential",
        )
        smoke = apply_tesseract(
            self.smoke_client, self.smoke_inputs(theta, fire["smoke_source"], frames),
            vmap_method="sequential",
        )
        return {
            "smoke_source": fire["smoke_source"],
            "intensity_frames": fire["intensity_frames"],
            "fuel_final": fire["fuel_final"],
            "burned_area": fire["burned_area"],
            "sensor_concentration": smoke["sensor_concentration"],
            "concentration_frames": smoke["concentration_frames"],
            "smoke_cfl_number": smoke["cfl_number"],
        }


@contextlib.contextmanager
def open_pipeline(config: TinyConfig) -> Iterator[TesseractPipeline]:
    """Serve both Tesseract images for the whole command and tear them down.

    Containers stay alive across every loss evaluation, so no container start-up
    is charged to a gradient step (ADR-001 section 6).
    """
    from tesseract_core import Tesseract

    enable_float64()
    fire = Tesseract.from_image(FIRE_IMAGE)
    smoke = Tesseract.from_image(SMOKE_IMAGE)
    fire.serve()
    try:
        smoke.serve()
        try:
            yield TesseractPipeline(config=config, fire_client=fire, smoke_client=smoke)
        finally:
            smoke.teardown()
    finally:
        fire.teardown()
