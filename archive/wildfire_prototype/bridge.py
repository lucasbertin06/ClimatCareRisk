"""Explicit JAX-to-PyTorch differentiation boundary for fire spread."""

from __future__ import annotations

import atexit
from functools import partial
from typing import Any

import jax
import numpy as np
import torch

from wildfire_shared.spread_torch import spread_forward_torch

_spread_client = None


def _as_torch(value: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    tensor = torch.as_tensor(np.array(value, copy=True), dtype=torch.float32)
    return tensor.requires_grad_(requires_grad)


def _tesseract_enabled() -> bool:
    import os

    return os.getenv("WILDFIRE_USE_TESSERACT", "0") == "1"


def _get_spread_client() -> Any:
    global _spread_client
    if _spread_client is None:
        from tesseract_core import Tesseract

        _spread_client = Tesseract.from_image("ignis_spread_torch")
        _spread_client.serve()
    return _spread_client


def _teardown() -> None:
    if _spread_client is not None:
        _spread_client.teardown()


atexit.register(_teardown)


def _spread_result(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not _tesseract_enabled():
        return spread_forward_torch(inputs)
    from tesseract_torch import apply_tesseract

    return apply_tesseract(_get_spread_client(), inputs)


def _forward_callback(
    hazard: np.ndarray,
    fuel: np.ndarray,
    wind: np.ndarray,
    slope: np.ndarray,
    intervention: np.ndarray,
    population: np.ndarray,
    ecological_cost: np.ndarray,
    intervention_cost: np.ndarray,
    steps: np.ndarray,
) -> tuple[np.ndarray, ...]:
    inputs = {
        "hazard": _as_torch(hazard),
        "fuel": _as_torch(fuel),
        "wind": _as_torch(wind),
        "slope": _as_torch(slope),
        "intervention": _as_torch(intervention),
        "population": _as_torch(population),
        "ecological_cost": _as_torch(ecological_cost),
        "intervention_cost": _as_torch(intervention_cost),
        "steps": int(np.asarray(steps)),
    }
    result = _spread_result(inputs)
    return (
        result["burn_probability"].detach().numpy(),
        result["trajectory"].detach().numpy(),
        result["burned_area"].detach().reshape(()).numpy(),
        result["exposed_population"].detach().reshape(()).numpy(),
        result["ecological_penalty"].detach().reshape(()).numpy(),
        result["intervention_cost"].detach().reshape(()).numpy(),
    )


def _vjp_callback(
    hazard: np.ndarray,
    fuel: np.ndarray,
    wind: np.ndarray,
    slope: np.ndarray,
    intervention: np.ndarray,
    population: np.ndarray,
    ecological_cost: np.ndarray,
    intervention_cost: np.ndarray,
    steps: np.ndarray,
    *cotangents: np.ndarray,
) -> tuple[np.ndarray, ...]:
    differentiable = [
        _as_torch(hazard, True),
        _as_torch(wind, True),
        _as_torch(intervention, True),
        _as_torch(intervention_cost, True),
    ]
    inputs = {
        "hazard": differentiable[0],
        "fuel": _as_torch(fuel),
        "wind": differentiable[1],
        "slope": _as_torch(slope),
        "intervention": differentiable[2],
        "population": _as_torch(population),
        "ecological_cost": _as_torch(ecological_cost),
        "intervention_cost": differentiable[3],
        "steps": int(np.asarray(steps)),
    }
    result = _spread_result(inputs)
    cotangent_loss = (result["burn_probability"] * _as_torch(cotangents[0])).sum()
    cotangent_loss = (
        cotangent_loss + (result["trajectory"] * _as_torch(cotangents[1])).sum()
    )
    cotangent_loss = cotangent_loss + result["burned_area"] * _as_torch(cotangents[2])
    cotangent_loss = cotangent_loss + result["exposed_population"] * _as_torch(
        cotangents[3]
    )
    cotangent_loss = cotangent_loss + result["ecological_penalty"] * _as_torch(
        cotangents[4]
    )
    cotangent_loss = cotangent_loss + result["intervention_cost"] * _as_torch(
        cotangents[5]
    )
    gradients = torch.autograd.grad(cotangent_loss, differentiable, allow_unused=True)
    values = [
        hazard,
        fuel,
        wind,
        slope,
        intervention,
        population,
        ecological_cost,
        intervention_cost,
    ]
    gradient_values = [
        np.zeros_like(np.asarray(value), dtype=np.float32) for value in values
    ]
    for index, gradient in zip((0, 2, 4, 7), gradients, strict=True):
        if gradient is not None:
            gradient_values[index] = gradient.detach().numpy()
    return tuple(gradient_values)


def _result_shapes(hazard: jax.Array, steps: int) -> tuple[Any, ...]:
    return (
        jax.ShapeDtypeStruct(hazard.shape, hazard.dtype),
        jax.ShapeDtypeStruct((steps, *hazard.shape), hazard.dtype),
        jax.ShapeDtypeStruct((), hazard.dtype),
        jax.ShapeDtypeStruct((), hazard.dtype),
        jax.ShapeDtypeStruct((), hazard.dtype),
        jax.ShapeDtypeStruct((), hazard.dtype),
    )


def _spread_impl(
    hazard: jax.Array,
    fuel: jax.Array,
    wind: jax.Array,
    slope: jax.Array,
    intervention: jax.Array,
    population: jax.Array,
    ecological_cost: jax.Array,
    intervention_cost: jax.Array,
    steps: int,
) -> tuple[jax.Array, ...]:
    return jax.pure_callback(
        _forward_callback,
        _result_shapes(hazard, steps),
        hazard,
        fuel,
        wind,
        slope,
        intervention,
        population,
        ecological_cost,
        intervention_cost,
        np.int32(steps),
        vmap_method="sequential",
    )


@partial(jax.custom_vjp, nondiff_argnums=(8,))
def torch_spread(
    hazard: jax.Array,
    fuel: jax.Array,
    wind: jax.Array,
    slope: jax.Array,
    intervention: jax.Array,
    population: jax.Array,
    ecological_cost: jax.Array,
    intervention_cost: jax.Array,
    steps: int = 24,
) -> tuple[jax.Array, ...]:
    """Evaluate the Torch spread component from a JAX program."""
    return _spread_impl(
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


def _torch_spread_forward(
    hazard: jax.Array,
    fuel: jax.Array,
    wind: jax.Array,
    slope: jax.Array,
    intervention: jax.Array,
    population: jax.Array,
    ecological_cost: jax.Array,
    intervention_cost: jax.Array,
    steps: int,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    inputs = (
        hazard,
        fuel,
        wind,
        slope,
        intervention,
        population,
        ecological_cost,
        intervention_cost,
    )
    return _spread_impl(*inputs, steps), inputs


def _torch_spread_backward(
    steps: int,
    residuals: tuple[jax.Array, ...],
    cotangents: tuple[jax.Array, ...],
) -> tuple[jax.Array, ...]:
    shapes = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype) for value in residuals
    )
    return jax.pure_callback(
        _vjp_callback,
        shapes,
        *residuals,
        np.int32(steps),
        *cotangents,
        vmap_method="sequential",
    )


torch_spread.defvjp(_torch_spread_forward, _torch_spread_backward)


def spread_objective(
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
    """Return a scalar social objective through the Torch boundary."""
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
