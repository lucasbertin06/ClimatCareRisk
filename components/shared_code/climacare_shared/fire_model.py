r"""FireSpread: coupled thermal intensity and fuel model in PyTorch.

Implements specification section 4 exactly:

.. math::

    \partial_t T &= \nabla\cdot(D_T\nabla T) - v\cdot\nabla T - hT + QR, \\
    \partial_t F &= -R, \\
    R &= k_r F e^{-\alpha_M M}\,\sigma\!\left(\frac{T-T_{ign}}{\varepsilon_T}\right), \\
    S &= \eta_{smoke} R .

Discretisation: cell-centred uniform grid, explicit Euler, five-point centred
Laplacian with zero normal flux, first-order upwind advection with homogeneous
inflow and discrete convective outflow. No clamp is applied anywhere on the
nominal path; positivity comes from the CFL budget checked before the loop.

Everything is written functionally on ``torch.float64`` tensors so that
``torch.func.vjp`` can differentiate it without an autograd tape.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from climacare_shared.grid import Grid, check_fire_stability

__all__ = ["FIRE_FIELDS", "fire_forward", "fire_step"]

FIRE_FIELDS = ("smoke_source", "intensity_frames", "fuel_final", "burned_area")

_DTYPE = torch.float64


# --------------------------------------------------------------------------- #
# Finite-difference stencils
# --------------------------------------------------------------------------- #
def _shift_west(field: torch.Tensor) -> torch.Tensor:
    """Return ``q[j, i-1]`` with a homogeneous value outside the domain."""
    return F.pad(field, (1, 0, 0, 0))[..., :, :-1]


def _shift_east(field: torch.Tensor) -> torch.Tensor:
    """Return ``q[j, i+1]`` with a homogeneous value outside the domain."""
    return F.pad(field, (0, 1, 0, 0))[..., :, 1:]


def _shift_south(field: torch.Tensor) -> torch.Tensor:
    """Return ``q[j-1, i]`` with a homogeneous value outside the domain."""
    return F.pad(field, (0, 0, 1, 0))[..., :-1, :]


def _shift_north(field: torch.Tensor) -> torch.Tensor:
    """Return ``q[j+1, i]`` with a homogeneous value outside the domain."""
    return F.pad(field, (0, 0, 0, 1))[..., 1:, :]


def _laplacian(field: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Five-point Laplacian with an explicit zero normal flux boundary."""
    padded = F.pad(field[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    d2x = (padded[1:-1, 2:] - 2.0 * field + padded[1:-1, :-2]) / (dx * dx)
    d2y = (padded[2:, 1:-1] - 2.0 * field + padded[:-2, 1:-1]) / (dy * dy)
    return d2x + d2y


def _upwind_advection(
    field: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    r"""Return :math:`v\cdot\nabla_h^{up} q` with homogeneous advective inflow.

    On the downwind boundary the one-sided upwind difference only reads interior
    cells, which is exactly the discrete convective outflow of section 4.4.
    """
    vx_plus = torch.clamp(vx, min=0.0)
    vx_minus = torch.clamp(vx, max=0.0)
    vy_plus = torch.clamp(vy, min=0.0)
    vy_minus = torch.clamp(vy, max=0.0)

    backward_x = (field - _shift_west(field)) / dx
    forward_x = (_shift_east(field) - field) / dx
    backward_y = (field - _shift_south(field)) / dy
    forward_y = (_shift_north(field) - field) / dy

    return (
        vx_plus * backward_x
        + vx_minus * forward_x
        + vy_plus * backward_y
        + vy_minus * forward_y
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _reaction(
    intensity: torch.Tensor,
    fuel: torch.Tensor,
    moisture: torch.Tensor,
    *,
    reaction_rate: float,
    moisture_sensitivity: float,
    ignition_threshold: float,
    ignition_width: float,
) -> torch.Tensor:
    r"""Return :math:`R(T, F, M)` of section 4.1."""
    activation = torch.sigmoid((intensity - ignition_threshold) / ignition_width)
    return reaction_rate * fuel * torch.exp(-moisture_sensitivity * moisture) * activation


def fire_step(
    intensity: torch.Tensor,
    fuel: torch.Tensor,
    moisture: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    *,
    dt: float,
    dx: float,
    dy: float,
    diffusivity: float,
    heat_loss: float,
    heat_release: float,
    reaction_rate: float,
    moisture_sensitivity: float,
    ignition_threshold: float,
    ignition_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one explicit Euler step, returning ``(T_next, F_next, R)``."""
    reaction = _reaction(
        intensity,
        fuel,
        moisture,
        reaction_rate=reaction_rate,
        moisture_sensitivity=moisture_sensitivity,
        ignition_threshold=ignition_threshold,
        ignition_width=ignition_width,
    )
    laplacian = _laplacian(intensity, dx, dy)
    advection = _upwind_advection(intensity, vx, vy, dx, dy)
    rhs = (
        diffusivity * laplacian
        - advection
        - heat_loss * intensity
        + heat_release * reaction
    )
    return intensity + dt * rhs, fuel - dt * reaction, reaction


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(_DTYPE)
    return torch.as_tensor(value, dtype=_DTYPE)


def _frame_indices(n_steps: int, frame_count: int) -> list[int]:
    """Return ``frame_count`` evenly spaced indices in ``[0, n_steps]``."""
    if frame_count < 1:
        raise ValueError(f"frame_count must be >= 1, got {frame_count}")
    if frame_count == 1:
        return [n_steps]
    step = n_steps / (frame_count - 1)
    return [min(n_steps, round(index * step)) for index in range(frame_count)]


def fire_forward(inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Run the FireSpread solver on a dumped ``InputSchema`` mapping.

    Args:
        inputs: mapping with the differentiable entries ``ignition`` (shape
            ``(3,)``, holding ``x0``, ``y0`` and ``log A0``) and ``wind`` (shape
            ``(2,)``), plus the fixed fields and scalars of section 2.3.

    Returns:
        Mapping with ``smoke_source`` of shape ``(n_steps, ny, nx)``,
        ``intensity_frames`` of shape ``(frame_count, ny, nx)``, ``fuel_final``
        and the scalar ``burned_area``.

    Raises:
        ValueError: if the grid, the CFL budget or the shapes are invalid.
    """
    moisture = _as_tensor(inputs["moisture"])
    fuel_base = _as_tensor(inputs["fuel_base"])
    prevention = _as_tensor(inputs["fuel_prevention"])
    if moisture.shape != fuel_base.shape or moisture.shape != prevention.shape:
        raise ValueError(
            "moisture, fuel_base and fuel_prevention must share a shape, got "
            f"{tuple(moisture.shape)}, {tuple(fuel_base.shape)}, "
            f"{tuple(prevention.shape)}"
        )

    ny, nx = int(moisture.shape[0]), int(moisture.shape[1])
    grid = Grid(nx=nx, ny=ny)
    dt = float(inputs["dt"])
    n_steps = int(inputs["n_steps"])
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    diffusivity = float(inputs["diffusivity"])
    heat_loss = float(inputs["heat_loss"])
    heat_release = float(inputs["heat_release"])
    reaction_rate = float(inputs["reaction_rate"])
    moisture_sensitivity = float(inputs["moisture_sensitivity"])
    ignition_threshold = float(inputs["ignition_threshold"])
    ignition_width = float(inputs["ignition_width"])
    source_sigma = float(inputs["source_sigma"])
    smoke_yield = float(inputs["smoke_yield"])
    wind_speed_bound = float(inputs["wind_speed_bound"])
    if source_sigma <= 0.0:
        raise ValueError(f"source_sigma must be positive, got {source_sigma}")
    if smoke_yield <= 0.0:
        raise ValueError(f"smoke_yield must be positive, got {smoke_yield}")

    check_fire_stability(
        dt=dt,
        grid=grid,
        wind_speed=wind_speed_bound,
        diffusivity=diffusivity,
        heat_loss=heat_loss,
        reaction_rate=reaction_rate,
        ignition_width=ignition_width,
    )

    ignition = _as_tensor(inputs["ignition"]).reshape(3)
    wind = _as_tensor(inputs["wind"]).reshape(2)
    x0, y0, log_amplitude = ignition[0], ignition[1], ignition[2]
    vx, vy = wind[0], wind[1]

    centres_x = torch.as_tensor(grid.centres_x(), dtype=_DTYPE)
    centres_y = torch.as_tensor(grid.centres_y(), dtype=_DTYPE)
    mesh_x = centres_x[None, :].expand(ny, nx)
    mesh_y = centres_y[:, None].expand(ny, nx)

    squared_distance = (mesh_x - x0) ** 2 + (mesh_y - y0) ** 2
    intensity = torch.exp(log_amplitude) * torch.exp(
        -squared_distance / (2.0 * source_sigma**2)
    )
    fuel_initial = fuel_base * (1.0 - prevention)
    fuel = fuel_initial

    wanted_frames = _frame_indices(n_steps, int(inputs["frame_count"]))
    frames: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    for index in range(n_steps):
        if index in wanted_frames:
            frames.extend([intensity] * wanted_frames.count(index))
        intensity, fuel, reaction = fire_step(
            intensity,
            fuel,
            moisture,
            vx,
            vy,
            dt=dt,
            dx=grid.dx,
            dy=grid.dy,
            diffusivity=diffusivity,
            heat_loss=heat_loss,
            heat_release=heat_release,
            reaction_rate=reaction_rate,
            moisture_sensitivity=moisture_sensitivity,
            ignition_threshold=ignition_threshold,
            ignition_width=ignition_width,
        )
        sources.append(smoke_yield * reaction)
    frames.extend([intensity] * wanted_frames.count(n_steps))

    burned_area = grid.cell_area * torch.sum(fuel_initial - fuel)
    return {
        "smoke_source": torch.stack(sources, dim=0),
        "intensity_frames": torch.stack(frames, dim=0),
        "fuel_final": fuel,
        "burned_area": burned_area.reshape(()),
    }
