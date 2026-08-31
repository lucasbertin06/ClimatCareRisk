"""FireSpread invariants required by specification section 16."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from climacare_shared.fire_model import _reaction, fire_forward
from climacare_shared.grid import Grid, check_fire_stability

BASE = {
    "dt": 0.02,
    "n_steps": 12,
    "diffusivity": 2.0e-4,
    "heat_loss": 0.20,
    "heat_release": 0.80,
    "reaction_rate": 6.0,
    "moisture_sensitivity": 1.2,
    "ignition_threshold": 0.50,
    "ignition_width": 0.05,
    "source_sigma": 0.055,
    "smoke_yield": 1.0,
    "wind_speed_bound": 0.35,
    "frame_count": 3,
}


def make_inputs(**overrides: object) -> dict:
    """Return a small deterministic FireSpread payload."""
    nx = ny = 24
    ones = torch.ones((ny, nx), dtype=torch.float64)
    payload = dict(BASE)
    payload.update(
        {
            "ignition": torch.tensor([0.35, 0.35, math.log(1.2)], dtype=torch.float64),
            "wind": torch.tensor([0.26, 0.23], dtype=torch.float64),
            "moisture": 0.15 * ones,
            "fuel_base": ones.clone(),
            "fuel_prevention": torch.zeros_like(ones),
        }
    )
    payload.update(overrides)
    return payload


def test_zero_fuel_gives_zero_reaction() -> None:
    intensity = torch.full((4, 4), 2.0, dtype=torch.float64)
    fuel = torch.zeros((4, 4), dtype=torch.float64)
    moisture = torch.full((4, 4), 0.1, dtype=torch.float64)
    reaction = _reaction(
        intensity,
        fuel,
        moisture,
        reaction_rate=6.0,
        moisture_sensitivity=1.2,
        ignition_threshold=0.5,
        ignition_width=0.05,
    )
    assert torch.all(reaction == 0.0)


def test_higher_moisture_reduces_reaction() -> None:
    intensity = torch.full((4, 4), 1.0, dtype=torch.float64)
    fuel = torch.full((4, 4), 0.8, dtype=torch.float64)
    kwargs = {
        "reaction_rate": 6.0,
        "moisture_sensitivity": 1.2,
        "ignition_threshold": 0.5,
        "ignition_width": 0.05,
    }
    dry = _reaction(intensity, fuel, torch.full((4, 4), 0.1, dtype=torch.float64), **kwargs)
    wet = _reaction(intensity, fuel, torch.full((4, 4), 0.6, dtype=torch.float64), **kwargs)
    assert torch.all(wet < dry)


def test_fuel_is_monotone_and_bounded() -> None:
    result = fire_forward(make_inputs(n_steps=30, frame_count=2))
    fuel_final = result["fuel_final"].numpy()
    source = result["smoke_source"].numpy()
    assert np.all(fuel_final >= 0.0)
    assert np.all(fuel_final <= 1.0)
    # F^{n+1} = F^n - dt R with R >= 0, so the fuel never increases.
    assert np.all(source >= 0.0)
    assert float(result["burned_area"]) > 0.0

def test_invalid_normalized_fields_are_rejected() -> None:
    invalid_fuel = torch.full((24, 24), 1.1, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"fuel_base must lie in \[0, 1\]"):
        fire_forward(make_inputs(fuel_base=invalid_fuel))
    invalid_moisture = torch.full((24, 24), -0.1, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"moisture must lie in \[0, 1\]"):
        fire_forward(make_inputs(moisture=invalid_moisture))


def test_without_reaction_a_zero_field_stays_zero() -> None:
    """An exactly zero intensity is a fixed point when the reaction vanishes."""
    from climacare_shared.fire_model import fire_step

    zeros = torch.zeros((8, 8), dtype=torch.float64)
    intensity = zeros.clone()
    fuel = zeros.clone()  # F = 0 forces R = 0 exactly
    for _ in range(15):
        intensity, fuel, reaction = fire_step(
            intensity,
            fuel,
            torch.full((8, 8), 0.15, dtype=torch.float64),
            torch.tensor(0.26, dtype=torch.float64),
            torch.tensor(0.23, dtype=torch.float64),
            dt=0.02,
            dx=1 / 8,
            dy=1 / 8,
            diffusivity=2.0e-4,
            heat_loss=0.20,
            heat_release=0.80,
            reaction_rate=6.0,
            moisture_sensitivity=1.2,
            ignition_threshold=0.50,
            ignition_width=0.05,
        )
        assert torch.all(reaction == 0.0)
        assert torch.all(intensity == 0.0)
        assert torch.all(fuel == 0.0)


def test_without_reaction_a_positive_field_decays() -> None:
    inputs = make_inputs(
        reaction_rate=1.0e-12,
        wind=torch.zeros(2, dtype=torch.float64),
        n_steps=20,
        frame_count=2,
    )
    frames = fire_forward(inputs)["intensity_frames"].numpy()
    # Zero-flux diffusion conserves the integral; the linear sink strictly
    # reduces it.
    assert frames[-1].sum() < frames[0].sum()


def test_diffusion_is_symmetric_without_wind() -> None:
    inputs = make_inputs(
        ignition=torch.tensor([0.5, 0.5, math.log(0.2)], dtype=torch.float64),
        wind=torch.zeros(2, dtype=torch.float64),
        reaction_rate=1.0e-12,
        n_steps=15,
        frame_count=1,
        source_sigma=0.12,
    )
    field = fire_forward(inputs)["intensity_frames"].numpy()[-1]
    assert np.allclose(field, field[::-1, :], atol=1e-12)
    assert np.allclose(field, field[:, ::-1], atol=1e-12)
    assert np.allclose(field, field.T, atol=1e-12)


def test_wind_displaces_the_thermal_centroid_downwind() -> None:
    grid = Grid(nx=24, ny=24)
    mesh_x, mesh_y = grid.meshgrid()

    def centroid(wind: torch.Tensor) -> tuple[float, float]:
        field = fire_forward(
            make_inputs(
                wind=wind,
                reaction_rate=1.0e-12,
                n_steps=25,
                frame_count=1,
                ignition=torch.tensor([0.4, 0.4, math.log(1.0)], dtype=torch.float64),
            )
        )["intensity_frames"].numpy()[-1]
        mass = field.sum()
        return float((field * mesh_x).sum() / mass), float((field * mesh_y).sum() / mass)

    still = centroid(torch.zeros(2, dtype=torch.float64))
    blown = centroid(torch.tensor([0.30, 0.15], dtype=torch.float64))
    assert blown[0] > still[0]
    assert blown[1] > still[1]
    # The x wind is twice the y wind, so the x displacement must dominate.
    assert (blown[0] - still[0]) > (blown[1] - still[1])


@pytest.mark.parametrize("axis", [0, 1])
def test_sensitivity_to_the_ignition_position_is_continuous(axis: int) -> None:
    grid = Grid(nx=24, ny=24)
    base = 0.35
    offsets = np.linspace(-0.5, 0.5, 9) * grid.dx
    totals = []
    for offset in offsets:
        ignition = [0.35, 0.35, math.log(1.2)]
        ignition[axis] = base + float(offset)
        result = fire_forward(
            make_inputs(
                ignition=torch.tensor(ignition, dtype=torch.float64),
                n_steps=8,
                frame_count=1,
            )
        )
        totals.append(float(result["smoke_source"].numpy().sum()))
    differences = np.abs(np.diff(totals))
    # A sub-cell shift must never produce a jump: no thresholding, no snapping.
    assert differences.max() < 5.0 * differences.mean() + 1e-12


def test_gradient_flows_to_every_differentiable_input() -> None:
    inputs = make_inputs(n_steps=10, frame_count=1)
    ignition = inputs["ignition"].clone()
    wind = inputs["wind"].clone()

    def scalar(ignition_value: torch.Tensor, wind_value: torch.Tensor) -> torch.Tensor:
        payload = dict(inputs)
        payload["ignition"] = ignition_value
        payload["wind"] = wind_value
        return fire_forward(payload)["smoke_source"].sum()

    value, pullback = torch.func.vjp(scalar, ignition, wind)
    grad_ignition, grad_wind = pullback(torch.ones_like(value))
    assert torch.all(torch.isfinite(grad_ignition))
    assert torch.all(torch.isfinite(grad_wind))
    assert torch.any(grad_ignition != 0.0)
    assert torch.any(grad_wind != 0.0)


def test_invalid_cfl_is_rejected_before_the_first_step() -> None:
    grid = Grid(nx=32, ny=32)
    with pytest.raises(ValueError, match="CFL violated"):
        check_fire_stability(
            dt=0.5,
            grid=grid,
            wind_speed=0.35,
            diffusivity=2.0e-4,
            heat_loss=0.2,
            reaction_rate=6.0,
            ignition_width=0.05,
        )
    with pytest.raises(ValueError, match="fuel positivity"):
        check_fire_stability(
            dt=0.02,
            grid=grid,
            wind_speed=0.35,
            diffusivity=2.0e-4,
            heat_loss=0.2,
            reaction_rate=500.0,
            ignition_width=0.05,
        )
    with pytest.raises(ValueError, match="CFL violated"):
        fire_forward(make_inputs(dt=0.5))


def test_tiny_run_is_stable_and_unsaturated(tiny_config: object) -> None:
    config = tiny_config
    truth = np.asarray(config.truth)
    result = fire_forward(
        {
            "ignition": torch.tensor(truth[:3], dtype=torch.float64),
            "wind": torch.tensor(config.fire_wind(truth[3]), dtype=torch.float64),
            "moisture": torch.as_tensor(config.moisture),
            "fuel_base": torch.as_tensor(config.fuel_base),
            "fuel_prevention": torch.as_tensor(config.fuel_prevention),
            "dt": config.dt,
            "n_steps": config.n_steps,
            "diffusivity": config.fire.diffusivity,
            "heat_loss": config.fire.heat_loss,
            "heat_release": config.fire.heat_release,
            "reaction_rate": config.fire.reaction_rate,
            "moisture_sensitivity": config.fire.moisture_sensitivity,
            "ignition_threshold": config.fire.ignition_threshold,
            "ignition_width": config.fire.ignition_width,
            "source_sigma": config.fire.source_sigma,
            "smoke_yield": config.fire.smoke_yield,
            "wind_speed_bound": config.wind.fire_speed,
            "frame_count": config.frame_count,
        }
    )
    frames = result["intensity_frames"].numpy()
    fuel = result["fuel_final"].numpy()
    assert np.all(np.isfinite(frames))
    assert frames.max() < 10.0, "the explicit scheme must not blow up on Tiny"
    assert fuel.min() > 0.0, "no cell may reach the fuel floor exactly"
    assert fuel.max() < 1.0 - 1e-9, "the reaction must consume fuel somewhere"
