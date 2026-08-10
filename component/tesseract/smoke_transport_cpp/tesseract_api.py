"""Tesseract B — SmokeTransport.

Python is only the mandatory Tesseract glue: the explicit scheme of
``docs/mathematical_specification.md`` section 5 and the discrete adjoint of
section 6 both live in ``cpp/smoke_kernel.cpp``, compiled to a C++20/OpenMP
extension module at image build time. Nothing here differentiates the solver;
the VJP endpoint forwards cotangents to the hand-written adjoint.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64

# The compiled extension is copied next to this file inside the image.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from climacare_shared.kernel import load_smoke_kernel

_DIFFERENTIABLE_INPUTS = ("smoke_source", "wind")
_DIFFERENTIABLE_OUTPUTS = ("sensor_concentration",)


class InputSchema(BaseModel):
    """Inputs of the SmokeTransport step of the C0 pipeline."""

    smoke_source: Differentiable[Array[(None, None, None), Float64]] = Field(
        description="Space-time smoke source S[n, j, i] from FireSpread"
    )
    wind: Differentiable[Array[(2,), Float64]] = Field(
        description="Shared smoke advection velocity w = s_c d(delta_phi)"
    )
    sensor_positions: Array[(None, 2), Float64] = Field(
        description="Fixed sensor coordinates in the unit square"
    )
    sensor_bias: Array[(None,), Float64] = Field(
        description="Known additive sensor bias b_j, zero in Tiny"
    )
    diffusivity: Float64 = Field(description="Smoke diffusivity D_c > 0")
    decay: Float64 = Field(description="First-order decay lambda_c >= 0")
    dt: Float64 = Field(description="Explicit Euler time step, shared with the fire")
    frame_count: int = Field(
        default=1,
        ge=1,
        description="Number of evenly spaced concentration frames to return",
    )


class OutputSchema(BaseModel):
    """Outputs of the SmokeTransport step."""

    sensor_concentration: Differentiable[Array[(None, None), Float64]] = Field(
        description="Predictions H_j c^n + b_j for levels 1..N_t, shape (N_t, S)"
    )
    concentration_frames: Array[(None, None, None), Float64] = Field(
        description="Concentration snapshots, shape (frame_count, ny, nx)"
    )
    cfl_number: Float64 = Field(
        description="Realised stability budget nu_c of section 5.4"
    )


def _frame_levels(n_steps: int, frame_count: int) -> np.ndarray:
    """Return ``frame_count`` evenly spaced history levels in ``[0, n_steps]``."""
    if frame_count == 1:
        return np.array([n_steps], dtype=np.int32)
    step = n_steps / (frame_count - 1)
    return np.array(
        [min(n_steps, round(index * step)) for index in range(frame_count)],
        dtype=np.int32,
    )


def _prepare(payload: dict[str, Any]) -> dict[str, Any]:
    source = np.ascontiguousarray(payload["smoke_source"], dtype=np.float64)
    if source.ndim != 3:
        raise ValueError(
            f"smoke_source must have shape (N_t, ny, nx), got {source.shape}"
        )
    wind = np.asarray(payload["wind"], dtype=np.float64).reshape(2)
    sensors = np.ascontiguousarray(payload["sensor_positions"], dtype=np.float64)
    bias = np.ascontiguousarray(payload["sensor_bias"], dtype=np.float64).reshape(-1)
    if sensors.ndim != 2 or sensors.shape[1] != 2:
        raise ValueError(
            f"sensor_positions must have shape (S, 2), got {sensors.shape}"
        )
    if bias.shape[0] != sensors.shape[0]:
        raise ValueError(
            "sensor_bias and sensor_positions disagree: "
            f"{bias.shape[0]} vs {sensors.shape[0]}"
        )
    return {
        "source": source,
        "wind": wind,
        "sensors": sensors,
        "bias": bias,
        "diffusivity": float(payload["diffusivity"]),
        "decay": float(payload["decay"]),
        "dt": float(payload["dt"]),
        "frame_count": int(payload["frame_count"]),
    }


def apply(inputs: InputSchema) -> OutputSchema:
    kernel = load_smoke_kernel()
    prepared = _prepare(inputs.model_dump())
    levels = _frame_levels(prepared["source"].shape[0], prepared["frame_count"])
    observations, frames = kernel.forward(
        prepared["source"],
        prepared["sensors"],
        prepared["bias"],
        float(prepared["wind"][0]),
        float(prepared["wind"][1]),
        prepared["diffusivity"],
        prepared["decay"],
        prepared["dt"],
        levels,
    )
    budget = kernel.cfl_number(
        prepared["source"],
        float(prepared["wind"][0]),
        float(prepared["wind"][1]),
        prepared["diffusivity"],
        prepared["decay"],
        prepared["dt"],
    )
    return {
        "sensor_concentration": observations,
        "concentration_frames": frames,
        "cfl_number": np.float64(budget),
    }


def abstract_eval(abstract_inputs: Any) -> dict:
    """Return output shapes and dtypes without touching the C++ kernel."""
    payload = abstract_inputs.model_dump()

    def shape_of(value: Any) -> tuple[int, ...]:
        if isinstance(value, dict):
            return tuple(int(entry) for entry in value["shape"])
        return tuple(int(entry) for entry in value.shape)

    source_shape = shape_of(payload["smoke_source"])
    if len(source_shape) != 3:
        raise ValueError(
            f"smoke_source must have shape (N_t, ny, nx), got {source_shape}"
        )
    n_steps, ny, nx = source_shape
    n_sensors = shape_of(payload["sensor_positions"])[0]
    frame_count = int(payload["frame_count"])
    dtype = "float64"
    return {
        "sensor_concentration": {"shape": (n_steps, n_sensors), "dtype": dtype},
        "concentration_frames": {"shape": (frame_count, ny, nx), "dtype": dtype},
        "cfl_number": {"shape": (), "dtype": dtype},
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict:
    """Return the discrete-adjoint cotangents of the requested inputs.

    Only ``sensor_concentration`` carries a cotangent: the frames and the CFL
    diagnostic are not part of the differentiated path.
    """
    unknown_inputs = set(vjp_inputs) - set(_DIFFERENTIABLE_INPUTS)
    if unknown_inputs:
        raise ValueError(
            f"SmokeTransport cannot differentiate with respect to {sorted(unknown_inputs)}"
        )
    unknown_outputs = set(vjp_outputs) - set(_DIFFERENTIABLE_OUTPUTS)
    if unknown_outputs:
        raise ValueError(
            f"SmokeTransport has no adjoint for outputs {sorted(unknown_outputs)}"
        )

    kernel = load_smoke_kernel()
    prepared = _prepare(inputs.model_dump())
    cotangent = np.ascontiguousarray(
        cotangent_vector["sensor_concentration"], dtype=np.float64
    )
    source_bar, wind_bar, _, _ = kernel.vector_jacobian_product(
        prepared["source"],
        prepared["sensors"],
        float(prepared["wind"][0]),
        float(prepared["wind"][1]),
        prepared["diffusivity"],
        prepared["decay"],
        prepared["dt"],
        cotangent,
    )
    available = {"smoke_source": source_bar, "wind": wind_bar}
    return {path: available[path] for path in sorted(vjp_inputs)}
