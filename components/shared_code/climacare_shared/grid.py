r"""Grid geometry, CFL budgets and sensor stencils shared by both Tesseracts.

Everything here follows ``docs/mathematical_specification.md`` sections 4.5,
4.6, 5.3, 5.4 and 5.5. The domain is always the unit square with
cell-centred coordinates :math:`x_i = (i + 1/2)\Delta x`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Grid",
    "bilinear_weights",
    "check_fire_stability",
    "check_smoke_stability",
    "fire_cfl_number",
    "smoke_cfl_number",
    "wind_vector",
]


@dataclass(frozen=True)
class Grid:
    """Uniform cell-centred grid on the unit square."""

    nx: int
    ny: int

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError(
                f"grid must have at least 3 cells per axis, got nx={self.nx}, ny={self.ny}"
            )

    @property
    def dx(self) -> float:
        """Cell width along x."""
        return 1.0 / float(self.nx)

    @property
    def dy(self) -> float:
        """Cell height along y."""
        return 1.0 / float(self.ny)

    @property
    def cell_area(self) -> float:
        r"""Return :math:`\Delta x\,\Delta y`."""
        return self.dx * self.dy

    def centres_x(self) -> np.ndarray:
        """Return the cell-centre x coordinates."""
        return (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dx

    def centres_y(self) -> np.ndarray:
        """Return the cell-centre y coordinates."""
        return (np.arange(self.ny, dtype=np.float64) + 0.5) * self.dy

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, Y)`` broadcast over ``(ny, nx)``."""
        return np.meshgrid(self.centres_x(), self.centres_y(), indexing="xy")

    def interior_bounds(
        self, margin_cells: float = 1.0
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return bilinear-safe coordinate windows for the x and y axes."""
        return (
            (margin_cells * self.dx, 1.0 - margin_cells * self.dx),
            (margin_cells * self.dy, 1.0 - margin_cells * self.dy),
        )


def wind_vector(speed: float, phi_base: float, delta_phi: float) -> tuple[float, float]:
    """Return the shared wind direction of specification section 3."""
    angle = phi_base + delta_phi
    return (speed * math.cos(angle), speed * math.sin(angle))


def _transport_budget(
    *,
    dt: float,
    grid: Grid,
    speed_bound: float,
    diffusivity: float,
    linear_sink: float,
) -> float:
    """Return the sufficient explicit-stability budget of sections 4.6 and 5.4.

    ``speed_bound`` is an angle-independent upper bound of the wind norm, so the
    returned budget never depends on a traced parameter. It is therefore safe to
    evaluate before the first step and to compare against one.
    """
    advective = speed_bound * (1.0 / grid.dx + 1.0 / grid.dy)
    diffusive = 2.0 * diffusivity * (1.0 / grid.dx**2 + 1.0 / grid.dy**2)
    return dt * (advective + diffusive + linear_sink)


def fire_cfl_number(
    *,
    dt: float,
    grid: Grid,
    wind_speed: float,
    diffusivity: float,
    heat_loss: float,
) -> float:
    r"""Return :math:`\nu_T` of specification section 4.6."""
    return _transport_budget(
        dt=dt,
        grid=grid,
        speed_bound=wind_speed,
        diffusivity=diffusivity,
        linear_sink=heat_loss,
    )


def smoke_cfl_number(
    *,
    dt: float,
    grid: Grid,
    wind_speed: float,
    diffusivity: float,
    decay: float,
) -> float:
    r"""Return :math:`\nu_c` of specification section 5.4."""
    return _transport_budget(
        dt=dt,
        grid=grid,
        speed_bound=wind_speed,
        diffusivity=diffusivity,
        linear_sink=decay,
    )


def check_fire_stability(
    *,
    dt: float,
    grid: Grid,
    wind_speed: float,
    diffusivity: float,
    heat_loss: float,
    reaction_rate: float,
    ignition_width: float,
) -> float:
    r"""Validate the FireSpread positivity constraints, return :math:`\nu_T`.

    Raises:
        ValueError: if any sufficient condition of section 4.6 is violated.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if diffusivity <= 0.0:
        raise ValueError(f"fire diffusivity D_T must be positive, got {diffusivity}")
    if heat_loss < 0.0:
        raise ValueError(f"heat loss h must be non-negative, got {heat_loss}")
    if reaction_rate <= 0.0:
        raise ValueError(f"reaction rate k_r must be positive, got {reaction_rate}")
    if ignition_width <= 0.0:
        raise ValueError(f"ignition width eps_T must be positive, got {ignition_width}")

    nu_t = fire_cfl_number(
        dt=dt,
        grid=grid,
        wind_speed=wind_speed,
        diffusivity=diffusivity,
        heat_loss=heat_loss,
    )
    if nu_t > 1.0:
        raise ValueError(
            "FireSpread CFL violated: nu_T = "
            f"{nu_t:.6f} > 1 for dt={dt}, nx={grid.nx}, ny={grid.ny}, "
            f"D_T={diffusivity}, |v|<={wind_speed}, h={heat_loss}"
        )
    if dt * reaction_rate > 1.0:
        raise ValueError(
            "FireSpread fuel positivity violated: dt * k_r = "
            f"{dt * reaction_rate:.6f} > 1"
        )
    return nu_t


def check_smoke_stability(
    *,
    dt: float,
    grid: Grid,
    wind_speed: float,
    diffusivity: float,
    decay: float,
) -> float:
    r"""Validate the SmokeTransport constraints, return :math:`\nu_c`.

    Raises:
        ValueError: if any sufficient condition of section 5.4 is violated.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if diffusivity <= 0.0:
        raise ValueError(f"smoke diffusivity D_c must be positive, got {diffusivity}")
    if decay < 0.0:
        raise ValueError(f"smoke decay lambda_c must be non-negative, got {decay}")

    nu_c = smoke_cfl_number(
        dt=dt,
        grid=grid,
        wind_speed=wind_speed,
        diffusivity=diffusivity,
        decay=decay,
    )
    if nu_c > 1.0:
        raise ValueError(
            "SmokeTransport CFL violated: nu_c = "
            f"{nu_c:.6f} > 1 for dt={dt}, nx={grid.nx}, ny={grid.ny}, "
            f"D_c={diffusivity}, |w|<={wind_speed}, lambda_c={decay}"
        )
    return nu_c


def bilinear_weights(
    positions: np.ndarray, grid: Grid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(i0, j0, weights)`` for the bilinear observation operator ``H``.

    ``weights`` has shape ``(n_sensors, 2, 2)`` indexed as ``[s, dj, di]`` so
    that ``H_s c = sum_{dj,di} weights[s, dj, di] * c[j0 + dj, i0 + di]``.

    Raises:
        ValueError: if a sensor is too close to a boundary for the four-cell
            stencil to stay inside the domain.
    """
    coords = np.asarray(positions, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"sensor positions must have shape (S, 2), got {coords.shape}")
    if not np.all(np.isfinite(coords)):
        raise ValueError("sensor positions must all be finite")

    gx = coords[:, 0] / grid.dx - 0.5
    gy = coords[:, 1] / grid.dy - 0.5
    i0 = np.floor(gx).astype(np.int64)
    j0 = np.floor(gy).astype(np.int64)
    fx = gx - i0
    fy = gy - j0

    x_bounds, y_bounds = grid.interior_bounds()
    if np.any(i0 < 0) or np.any(i0 + 1 > grid.nx - 1):
        raise ValueError(
            "sensor x coordinate outside the bilinear-safe window "
            f"{x_bounds} : {coords[:, 0].tolist()}"
        )
    if np.any(j0 < 0) or np.any(j0 + 1 > grid.ny - 1):
        raise ValueError(
            "sensor y coordinate outside the bilinear-safe window "
            f"{y_bounds} : {coords[:, 1].tolist()}"
        )

    weights = np.empty((coords.shape[0], 2, 2), dtype=np.float64)
    weights[:, 0, 0] = (1.0 - fy) * (1.0 - fx)
    weights[:, 0, 1] = (1.0 - fy) * fx
    weights[:, 1, 0] = fy * (1.0 - fx)
    weights[:, 1, 1] = fy * fx
    return i0, j0, weights
