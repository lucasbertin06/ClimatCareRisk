"""Explicit JAX-to-PyTorch differentiation boundary."""

from __future__ import annotations

import atexit
from typing import Any

import jax
import numpy as np
import torch

from plume_inversion_shared.sensor import sensor_apply_torch

_sensor_client = None


def _as_torch(value: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    tensor = torch.as_tensor(np.array(value, copy=True), dtype=torch.float32)
    return tensor.requires_grad_(requires_grad)


def _tesseract_enabled() -> bool:
    import os

    return os.getenv("PLUME_INVERSION_USE_TESSERACT", "0") == "1"


def _get_sensor_client():
    global _sensor_client
    if _sensor_client is None:
        from tesseract_core import Tesseract

        _sensor_client = Tesseract.from_image("sensor_model_torch")
        _sensor_client.serve()
    return _sensor_client


def _teardown() -> None:
    if _sensor_client is not None:
        _sensor_client.teardown()


atexit.register(_teardown)


def _sensor_result(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not _tesseract_enabled():
        return sensor_apply_torch(inputs)
    from tesseract_torch import apply_tesseract

    return apply_tesseract(_get_sensor_client(), inputs)


def _forward_callback(
    concentration: np.ndarray,
    gains: np.ndarray,
    biases: np.ndarray,
    time_constants: np.ndarray,
    observations: np.ndarray,
    dt: np.ndarray,
    noise_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = {
        "concentration": _as_torch(concentration),
        "gains": _as_torch(gains),
        "biases": _as_torch(biases),
        "time_constants": _as_torch(time_constants),
        "observations": _as_torch(observations),
        "dt": _as_torch(dt),
        "noise_scale": _as_torch(noise_scale),
    }
    result = _sensor_result(inputs)
    return result["predicted"].detach().numpy(), result["nll"].detach().reshape(
        ()
    ).numpy()


def _vjp_callback(
    concentration: np.ndarray,
    gains: np.ndarray,
    biases: np.ndarray,
    time_constants: np.ndarray,
    observations: np.ndarray,
    dt: np.ndarray,
    noise_scale: np.ndarray,
    predicted_cotangent: np.ndarray,
    nll_cotangent: np.ndarray,
) -> tuple[np.ndarray, ...]:
    differentiable = [
        _as_torch(concentration, True),
        _as_torch(gains, True),
        _as_torch(biases, True),
        _as_torch(time_constants, True),
        _as_torch(noise_scale, True),
    ]
    inputs = {
        "concentration": differentiable[0],
        "gains": differentiable[1],
        "biases": differentiable[2],
        "time_constants": differentiable[3],
        "observations": _as_torch(observations),
        "dt": _as_torch(dt),
        "noise_scale": differentiable[4],
    }
    result = _sensor_result(inputs)
    cotangent_loss = (
        result["predicted"] * _as_torch(predicted_cotangent)
    ).sum() + result["nll"] * _as_torch(nll_cotangent)
    gradients = torch.autograd.grad(cotangent_loss, differentiable, allow_unused=True)
    values = [
        concentration,
        gains,
        biases,
        time_constants,
        observations,
        dt,
        noise_scale,
    ]
    gradient_values = [
        np.zeros_like(np.asarray(value), dtype=np.float32) for value in values
    ]
    for index, gradient in zip((0, 1, 2, 3, 6), gradients, strict=True):
        if gradient is not None:
            gradient_values[index] = gradient.detach().numpy()
    return tuple(gradient_values)


def _result_shapes(concentration: jax.Array) -> tuple[Any, Any]:
    return (
        jax.ShapeDtypeStruct(concentration.shape, concentration.dtype),
        jax.ShapeDtypeStruct((), concentration.dtype),
    )


def _torch_sensor_impl(
    concentration: jax.Array,
    gains: jax.Array,
    biases: jax.Array,
    time_constants: jax.Array,
    observations: jax.Array,
    dt: jax.Array,
    noise_scale: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return jax.pure_callback(
        _forward_callback,
        _result_shapes(concentration),
        concentration,
        gains,
        biases,
        time_constants,
        observations,
        dt,
        noise_scale,
        vmap_method="sequential",
    )


@jax.custom_vjp
def torch_sensor(
    concentration: jax.Array,
    gains: jax.Array,
    biases: jax.Array,
    time_constants: jax.Array,
    observations: jax.Array,
    dt: jax.Array,
    noise_scale: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the PyTorch sensor component from a JAX program."""
    return _torch_sensor_impl(
        concentration,
        gains,
        biases,
        time_constants,
        observations,
        dt,
        noise_scale,
    )


def _torch_sensor_forward(
    concentration: jax.Array,
    gains: jax.Array,
    biases: jax.Array,
    time_constants: jax.Array,
    observations: jax.Array,
    dt: jax.Array,
    noise_scale: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
    result = _torch_sensor_impl(
        concentration,
        gains,
        biases,
        time_constants,
        observations,
        dt,
        noise_scale,
    )
    return result, (
        concentration,
        gains,
        biases,
        time_constants,
        observations,
        dt,
        noise_scale,
    )


def _torch_sensor_backward(
    residuals: tuple[jax.Array, ...], cotangents: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, ...]:
    concentration, gains, biases, time_constants, observations, dt, noise_scale = (
        residuals
    )
    shapes = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype) for value in residuals
    )
    return jax.pure_callback(
        _vjp_callback,
        shapes,
        concentration,
        gains,
        biases,
        time_constants,
        observations,
        dt,
        noise_scale,
        cotangents[0],
        cotangents[1],
        vmap_method="sequential",
    )


torch_sensor.defvjp(_torch_sensor_forward, _torch_sensor_backward)


def sensor_nll(
    concentration: jax.Array,
    gains: jax.Array,
    biases: jax.Array,
    time_constants: jax.Array,
    observations: jax.Array,
    dt: jax.Array,
    noise_scale: jax.Array,
) -> jax.Array:
    """Return the PyTorch sensor likelihood while preserving its VJP."""
    return torch_sensor(
        concentration,
        gains,
        biases,
        time_constants,
        observations,
        dt,
        noise_scale,
    )[1]
