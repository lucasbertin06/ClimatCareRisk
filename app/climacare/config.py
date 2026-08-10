"""Typed loader for the ClimaCare-Risk Tiny configuration.

The configuration is the single source of truth for the three Tiny commands.
Loading it always verifies the stability budgets of specification sections 4.6
and 5.4, so an invalid file fails before any solver runs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from climacare_shared.grid import (
    Grid,
    check_fire_stability,
    check_smoke_stability,
    wind_vector,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "FireParams",
    "SmokeParams",
    "TinyConfig",
    "WindParams",
    "load_tiny_config",
]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "tiny.yaml"

PARAMETER_NAMES = ("x0", "y0", "log_amplitude", "delta_phi")


@dataclass(frozen=True)
class WindParams:
    """Shared wind of specification section 3."""

    phi_base: float
    fire_speed: float
    smoke_speed: float
    delta_phi_max: float


@dataclass(frozen=True)
class FireParams:
    """Fixed FireSpread coefficients of specification section 2.3."""

    diffusivity: float
    heat_loss: float
    heat_release: float
    reaction_rate: float
    moisture_sensitivity: float
    ignition_threshold: float
    ignition_width: float
    source_sigma: float
    smoke_yield: float
    moisture: float
    fuel_base: float
    fuel_prevention: float


@dataclass(frozen=True)
class SmokeParams:
    """Fixed SmokeTransport coefficients."""

    diffusivity: float
    decay: float


@dataclass(frozen=True)
class PriorParams:
    """Priors of specification section 8."""

    margin: float
    log_amplitude_mean: float
    log_amplitude_std: float
    delta_phi_std: float


@dataclass(frozen=True)
class SensorParams:
    """Fixed sensor network of specification section 5.5."""

    positions: np.ndarray
    bias: np.ndarray
    noise_std: np.ndarray
    mask_fraction: float

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


@dataclass(frozen=True)
class GradientCheckParams:
    """Finite-difference settings of specification section 9."""

    epsilon: float
    step_factors: tuple[float, ...]
    absolute_floor: float


@dataclass(frozen=True)
class TinyConfig:
    """Fully validated Tiny configuration."""

    case: str
    seed: int
    grid: Grid
    dt: float
    n_steps: int
    wind: WindParams
    fire: FireParams
    smoke: SmokeParams
    truth: np.ndarray
    initial_guess: np.ndarray
    priors: PriorParams
    sensors: SensorParams
    frame_count: int
    iterations: int
    learning_rate: float
    optimizer: str
    gradient_check: GradientCheckParams
    source_path: Path
    nu_fire: float = field(default=0.0)
    nu_smoke: float = field(default=0.0)

    # -- derived quantities ------------------------------------------------ #
    def fire_wind(self, delta_phi: float) -> tuple[float, float]:
        """Return the fire-wind vector for an angular perturbation."""
        return wind_vector(self.wind.fire_speed, self.wind.phi_base, delta_phi)

    def smoke_wind(self, delta_phi: float) -> tuple[float, float]:
        """Return the smoke-wind vector for an angular perturbation."""
        return wind_vector(self.wind.smoke_speed, self.wind.phi_base, delta_phi)

    def uniform_map(self, value: float) -> np.ndarray:
        """Return a grid filled with one scalar value."""
        return np.full((self.grid.ny, self.grid.nx), float(value), dtype=np.float64)

    @property
    def moisture(self) -> np.ndarray:
        """Return the uniform moisture field."""
        return self.uniform_map(self.fire.moisture)

    @property
    def fuel_base(self) -> np.ndarray:
        """Return the uniform baseline-fuel field."""
        return self.uniform_map(self.fire.fuel_base)

    @property
    def fuel_prevention(self) -> np.ndarray:
        """Return the uniform prevention-fuel field."""
        return self.uniform_map(self.fire.fuel_prevention)

    @property
    def position_bounds(self) -> tuple[float, float]:
        """Return the admissible ignition-position interval."""
        return (self.priors.margin, 1.0 - self.priors.margin)

    @property
    def final_time(self) -> float:
        """Return the simulated duration."""
        return self.dt * self.n_steps

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable provenance block."""
        return {
            "case": self.case,
            "seed": self.seed,
            "config_path": str(self.source_path),
            "grid": {"nx": self.grid.nx, "ny": self.grid.ny},
            "time": {
                "dt": self.dt,
                "n_steps": self.n_steps,
                "final_time": self.final_time,
            },
            "output": {"frame_count": self.frame_count},
            "stability": {
                "nu_fire": self.nu_fire,
                "nu_smoke": self.nu_smoke,
                "dt_times_reaction_rate": self.dt * self.fire.reaction_rate,
            },
            "wind": asdict(self.wind),
            "fire": asdict(self.fire),
            "smoke": asdict(self.smoke),
            "priors": asdict(self.priors),
            "sensors": {
                "positions": self.sensors.positions.tolist(),
                "bias": self.sensors.bias.tolist(),
                "noise_std": self.sensors.noise_std.tolist(),
                "mask_fraction": self.sensors.mask_fraction,
            },
            "parameters": list(PARAMETER_NAMES),
            "optimization": {
                "optimizer": self.optimizer,
                "iterations": self.iterations,
                "learning_rate": self.learning_rate,
            },
            "truth": dict(zip(PARAMETER_NAMES, self.truth.tolist(), strict=True)),
            "initial_guess": dict(
                zip(PARAMETER_NAMES, self.initial_guess.tolist(), strict=True)
            ),
        }


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing key {key!r} in {context} of the Tiny configuration")
    return mapping[key]


def _vector(mapping: dict[str, Any], context: str) -> np.ndarray:
    return np.array(
        [float(_require(mapping, name, context)) for name in PARAMETER_NAMES],
        dtype=np.float64,
    )


def load_tiny_config(path: str | Path | None = None) -> TinyConfig:
    """Load, validate and return the Tiny configuration.

    Args:
        path: configuration file, defaulting to ``configs/tiny.yaml``.

    Returns:
        The validated configuration, including the realised CFL budgets.

    Raises:
        ValueError: if a stability budget, a bound or a sensor placement is
            invalid, or if the truth or the starting point leave their bounds.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    grid_raw = _require(raw, "grid", "root")
    grid = Grid(nx=int(grid_raw["nx"]), ny=int(grid_raw["ny"]))
    time_raw = _require(raw, "time", "root")
    dt = float(time_raw["dt"])
    n_steps = int(time_raw["n_steps"])

    wind_raw = _require(raw, "wind", "root")
    wind = WindParams(
        phi_base=math.radians(float(wind_raw["phi_base_deg"])),
        fire_speed=float(wind_raw["fire_speed"]),
        smoke_speed=float(wind_raw["smoke_speed"]),
        delta_phi_max=float(wind_raw["delta_phi_max"]),
    )
    fire = FireParams(**{key: float(value) for key, value in raw["fire"].items()})
    smoke = SmokeParams(**{key: float(value) for key, value in raw["smoke"].items()})
    priors = PriorParams(**{key: float(value) for key, value in raw["priors"].items()})

    sensors_raw = _require(raw, "sensors", "root")
    sensors = SensorParams(
        positions=np.asarray(sensors_raw["positions"], dtype=np.float64),
        bias=np.asarray(sensors_raw["bias"], dtype=np.float64),
        noise_std=np.asarray(sensors_raw["noise_std"], dtype=np.float64),
        mask_fraction=float(sensors_raw["mask_fraction"]),
    )
    check_raw = _require(raw, "gradient_check", "root")
    gradient_check = GradientCheckParams(
        epsilon=float(check_raw["epsilon"]),
        step_factors=tuple(float(value) for value in check_raw["step_factors"]),
        absolute_floor=float(check_raw["absolute_floor"]),
    )

    nu_fire = check_fire_stability(
        dt=dt,
        grid=grid,
        wind_speed=wind.fire_speed,
        diffusivity=fire.diffusivity,
        heat_loss=fire.heat_loss,
        reaction_rate=fire.reaction_rate,
        ignition_width=fire.ignition_width,
    )
    nu_smoke = check_smoke_stability(
        dt=dt,
        grid=grid,
        wind_speed=wind.smoke_speed,
        diffusivity=smoke.diffusivity,
        decay=smoke.decay,
    )

    config = TinyConfig(
        case=str(raw.get("case", "tiny")),
        seed=int(_require(raw, "seed", "root")),
        grid=grid,
        dt=dt,
        n_steps=n_steps,
        wind=wind,
        fire=fire,
        smoke=smoke,
        truth=_vector(_require(raw, "truth", "root"), "truth"),
        initial_guess=_vector(_require(raw, "initial_guess", "root"), "initial_guess"),
        priors=priors,
        sensors=sensors,
        frame_count=int(_require(raw, "output", "root")["frame_count"]),
        iterations=int(raw["optimization"]["iterations"]),
        learning_rate=float(raw["optimization"]["learning_rate"]),
        optimizer=str(raw["optimization"].get("optimizer", "lbfgs")),
        gradient_check=gradient_check,
        source_path=config_path,
        nu_fire=nu_fire,
        nu_smoke=nu_smoke,
    )
    _validate_bounds(config)
    return config


def _validate_bounds(config: TinyConfig) -> None:
    """Check that both parameter vectors sit strictly inside their bounds."""
    low, high = config.position_bounds
    for label, vector in (
        ("truth", config.truth),
        ("initial_guess", config.initial_guess),
    ):
        x0, y0, _, delta_phi = vector.tolist()
        if not (low < x0 < high) or not (low < y0 < high):
            raise ValueError(
                f"{label} ignition position ({x0}, {y0}) is not strictly inside "
                f"({low}, {high})"
            )
        if abs(delta_phi) >= config.wind.delta_phi_max:
            raise ValueError(
                f"{label} delta_phi {delta_phi} is not strictly inside "
                f"+/-{config.wind.delta_phi_max}"
            )
    if config.sensors.count < 3:
        raise ValueError("the C0 identifiability measures require at least 3 sensors")
    if np.any(config.sensors.noise_std <= 0.0):
        raise ValueError("sensor noise standard deviations must be positive")
    if not 0.0 <= config.sensors.mask_fraction < 1.0:
        raise ValueError("mask_fraction must lie in [0, 1)")
    # Sensors must not be collinear, otherwise the localisation degenerates.
    positions = config.sensors.positions
    first = positions[1] - positions[0]
    second = positions[2] - positions[0]
    cross = float(first[0] * second[1] - first[1] * second[0])
    if abs(cross) < 1e-6:
        raise ValueError(
            f"the first three sensors are collinear (cross product {cross:.3e})"
        )
