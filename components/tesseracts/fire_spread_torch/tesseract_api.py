"""Tesseract A — FireSpread.

PyTorch, CPU, float64. Differentiated with PyTorch autodiff through
``torch.func.vjp``. This container owns the coupled thermal-intensity and fuel
model of ``docs/mathematical_specification.md`` section 4 and emits the full
space-time smoke source tensor consumed by Tesseract B.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from climacare_shared.fire_model import fire_forward
from climacare_shared.grid import Grid, fire_cfl_number
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths
from torch.utils._pytree import tree_map

_DTYPE = torch.float64


class InputSchema(BaseModel):
    """Inputs of the FireSpread step of the C0 pipeline."""

    ignition: Differentiable[Array[(3,), Float64]] = Field(
        description="Continuous ignition parameters (x0, y0, log A0)"
    )
    wind: Differentiable[Array[(2,), Float64]] = Field(
        description="Shared thermal advection velocity v = s_T d(delta_phi)"
    )
    moisture: Array[(None, None), Float64] = Field(
        description="Normalised moisture map M, fixed in C0"
    )
    fuel_base: Array[(None, None), Float64] = Field(
        description="Baseline fuel fraction F_base in [0, 1]"
    )
    fuel_prevention: Differentiable[Array[(None, None), Float64]] = Field(
        description="Continuous fuel prevention level u_fuel in [0, 1]"
    )
    dt: Float64 = Field(description="Explicit Euler time step")
    n_steps: int = Field(description="Number of explicit steps N_t", ge=1)
    diffusivity: Float64 = Field(description="Thermal diffusivity D_T > 0")
    heat_loss: Float64 = Field(description="Linear heat loss h >= 0")
    heat_release: Float64 = Field(description="Heat release Q > 0")
    reaction_rate: Float64 = Field(description="Reaction rate k_r > 0")
    moisture_sensitivity: Float64 = Field(description="Moisture sensitivity alpha_M")
    ignition_threshold: Float64 = Field(description="Ignition threshold T_ign")
    ignition_width: Float64 = Field(description="Ignition smoothing width eps_T > 0")
    source_sigma: Float64 = Field(description="Initial thermal spot width sigma_0 > 0")
    smoke_yield: Float64 = Field(description="Smoke yield eta_smoke > 0")
    wind_speed_bound: Float64 = Field(
        description="Angle-independent bound on |v| used by the CFL check"
    )
    frame_count: int = Field(
        default=1,
        ge=1,
        description="Number of evenly spaced intensity frames to return",
    )


_DIFFERENTIABLE_INPUTS = ("ignition", "wind", "fuel_prevention")
_DIFFERENTIABLE_OUTPUTS = ("smoke_source", "intensity_frames", "fuel_final", "burned_area")


class OutputSchema(BaseModel):
    """Outputs of the FireSpread step."""

    smoke_source: Differentiable[Array[(None, None, None), Float64]] = Field(
        description="Smoke source S[n, j, i] = eta_smoke R^n, shape (N_t, ny, nx)"
    )
    intensity_frames: Differentiable[Array[(None, None, None), Float64]] = Field(
        description="Thermal intensity snapshots, shape (frame_count, ny, nx)"
    )
    fuel_final: Differentiable[Array[(None, None), Float64]] = Field(
        description="Remaining fuel fraction F^{N_t}"
    )
    burned_area: Differentiable[Float64] = Field(
        description="Burned fraction A_burned = dx dy sum(F^0 - F^{N_t})"
    )


def _to_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(_DTYPE)
    if isinstance(value, (np.ndarray, np.generic, float, int)):
        return torch.as_tensor(np.array(value, copy=True), dtype=_DTYPE)
    return value


def evaluate(inputs: dict) -> dict:
    """Run the solver on a dumped input mapping."""
    return fire_forward(tree_map(_to_tensor, inputs))


def apply(inputs: InputSchema) -> OutputSchema:
    return evaluate(inputs.model_dump())


def abstract_eval(abstract_inputs: Any) -> dict:
    """Return output shapes and dtypes without running the solver.

    Tesseract-JAX requires this endpoint before any JAX transformation; the
    PyTorch template does not provide it, so it is derived analytically from the
    fixed-map shape and the requested step and frame counts.
    """
    payload = abstract_inputs.model_dump()
    moisture = payload["moisture"]
    shape = tuple(moisture["shape"] if isinstance(moisture, dict) else moisture.shape)
    if len(shape) != 2:
        raise ValueError(f"moisture must be a 2-D map, got shape {shape}")
    ny, nx = int(shape[0]), int(shape[1])
    n_steps = int(payload["n_steps"])
    frame_count = int(payload["frame_count"])
    dtype = "float64"
    return {
        "smoke_source": {"shape": (n_steps, ny, nx), "dtype": dtype},
        "intensity_frames": {"shape": (frame_count, ny, nx), "dtype": dtype},
        "fuel_final": {"shape": (ny, nx), "dtype": dtype},
        "burned_area": {"shape": (), "dtype": dtype},
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict:
    """Return PyTorch cotangents for the requested differentiable inputs."""
    input_paths = tuple(sorted(vjp_inputs))
    output_paths = tuple(sorted(vjp_outputs))
    tensor_inputs = tree_map(_to_tensor, inputs.model_dump())
    positional = tuple(flatten_with_paths(tensor_inputs, input_paths).values())
    filtered = filter_func(
        evaluate,
        tensor_inputs,
        output_paths,
        input_paths=input_paths,
    )
    _, pullback = torch.func.vjp(filtered, *positional)
    cotangents = tree_map(
        _to_tensor, {path: cotangent_vector[path] for path in output_paths}
    )
    gradients = pullback(cotangents)
    return dict(zip(input_paths, gradients, strict=True))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict:
    """Return forward-mode products, used only by the component tests."""
    input_paths = tuple(sorted(jvp_inputs))
    output_paths = tuple(sorted(jvp_outputs))
    tensor_inputs = tree_map(_to_tensor, inputs.model_dump())
    positional = tuple(flatten_with_paths(tensor_inputs, input_paths).values())
    tangents = tuple(_to_tensor(tangent_vector[path]) for path in input_paths)
    filtered = filter_func(
        evaluate,
        tensor_inputs,
        output_paths,
        input_paths=input_paths,
    )
    return torch.func.jvp(filtered, positional, tangents)[1]


def cfl_report(inputs: InputSchema) -> dict[str, float]:
    """Return the stability budgets of section 4.6 for diagnostics."""
    payload = inputs.model_dump()
    moisture = np.asarray(payload["moisture"])
    grid = Grid(nx=int(moisture.shape[1]), ny=int(moisture.shape[0]))
    return {
        "nu_T": fire_cfl_number(
            dt=float(payload["dt"]),
            grid=grid,
            wind_speed=float(payload["wind_speed_bound"]),
            diffusivity=float(payload["diffusivity"]),
            heat_loss=float(payload["heat_loss"]),
        ),
        "dt_kr": float(payload["dt"]) * float(payload["reaction_rate"]),
    }
