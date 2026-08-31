"""Health and finance invariants required by specification section 16."""

from __future__ import annotations

from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import pytest
from climacare.finance import (
    FinanceParams,
    budget_allocation,
    conditional_value_at_risk,
    insurance_payout,
    liquidity_requirement,
    net_loss,
    optimal_cvar,
    physical_loss,
    smooth_plus,
)
from climacare.health import (
    HealthZone,
    incremental_health_impact,
    mean_exposure,
    zone_population,
)

GRID = 12
CELL_AREA = 1.0 / (GRID * GRID)


def make_zone(**overrides: object) -> HealthZone:
    """Return a small synthetic exposure zone."""
    settings = {
        "name": "test",
        "density": jnp.full((GRID, GRID), 4.0),
        "baseline": -2.0,
        "slope": 3.0,
        "filter_efficiency": 0.5,
    }
    settings.update(overrides)
    return HealthZone(**settings)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def test_zero_exposure_gives_exactly_zero_impact() -> None:
    zone = make_zone()
    concentration = jnp.zeros((5, GRID, GRID))
    exposure = mean_exposure(concentration, zone, dt=0.1, cell_area=CELL_AREA)
    assert float(exposure) == 0.0
    impact = incremental_health_impact(exposure, zone, CELL_AREA)
    assert float(impact) == 0.0


def test_exposure_is_a_mean_per_person() -> None:
    """Doubling the density at constant concentration leaves the dose unchanged."""
    concentration = jnp.full((4, GRID, GRID), 0.3)
    thin = make_zone(density=jnp.full((GRID, GRID), 2.0))
    thick = make_zone(density=jnp.full((GRID, GRID), 20.0))
    dose_thin = float(mean_exposure(concentration, thin, dt=0.1, cell_area=CELL_AREA))
    dose_thick = float(mean_exposure(concentration, thick, dt=0.1, cell_area=CELL_AREA))
    assert abs(dose_thin - dose_thick) < 1e-12
    assert float(zone_population(thick, CELL_AREA)) > float(
        zone_population(thin, CELL_AREA)
    )


def test_more_filtration_cannot_increase_exposure() -> None:
    zone = make_zone(filter_efficiency=0.8)
    concentration = jnp.full((4, GRID, GRID), 0.3)
    doses = [
        float(
            mean_exposure(
                concentration, zone, dt=0.1, cell_area=CELL_AREA, filter_level=level
            )
        )
        for level in (0.0, 0.25, 0.5, 1.0)
    ]
    assert all(later <= earlier for earlier, later in pairwise(doses))
    assert doses[-1] < doses[0]


def test_impact_is_monotone_in_exposure() -> None:
    zone = make_zone()
    values = [
        float(incremental_health_impact(jnp.asarray(dose), zone, CELL_AREA))
        for dose in (0.0, 0.1, 0.5, 1.0)
    ]
    assert all(later >= earlier for earlier, later in pairwise(values))
    assert values[-1] > values[0]


def test_negative_slope_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_zone(slope=-1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_zone(filter_efficiency=1.5)

def test_zero_population_is_rejected() -> None:
    zone = make_zone(density=jnp.zeros((GRID, GRID)))
    concentration = jnp.ones((2, GRID, GRID))
    with pytest.raises(ValueError, match="population must be strictly positive"):
        mean_exposure(concentration, zone, dt=0.1, cell_area=CELL_AREA)


def test_negative_density_is_rejected() -> None:
    with pytest.raises(ValueError, match="density must be non-negative"):
        make_zone(density=jnp.full((GRID, GRID), -1.0))


# --------------------------------------------------------------------------- #
# Finance
# --------------------------------------------------------------------------- #
def test_insurance_does_not_appear_in_the_physical_loss() -> None:
    params = FinanceParams()
    impacts = jnp.asarray([1.0, 2.0])
    burned = jnp.asarray(0.3)
    baseline = float(physical_loss(impacts, burned, 0.0, params))
    for level in (0.0, 0.5, 1.0):
        payout = insurance_payout(jnp.asarray(1.0), level, params)
        assert float(physical_loss(impacts, burned, 0.0, params)) == baseline
        # The payout only shifts the net position, never the physical one.
        shifted = float(
            net_loss(jnp.asarray(baseline), jnp.asarray(1.0), level, 0.0, 0.0, params)
        )
        assert abs(shifted - (baseline - float(payout) + 0.12 * level)) < 1e-12


def test_prevention_can_reduce_the_physical_loss() -> None:
    params = FinanceParams()
    burned_without = jnp.asarray(0.40)
    burned_with = jnp.asarray(0.25)
    impacts = jnp.asarray([1.0])
    assert float(physical_loss(impacts, burned_with, 0.0, params)) < float(
        physical_loss(impacts, burned_without, 0.0, params)
    )


def test_payout_is_capped_and_smooth() -> None:
    params = FinanceParams()
    # Sample finely enough for the trigger width: a step function would still
    # jump by the full coverage between two neighbouring samples.
    indices = jnp.linspace(-2.0, 4.0, 4001)
    payouts = np.asarray(
        [float(insurance_payout(index, 1.0, params)) for index in indices]
    )
    step = float(indices[1] - indices[0])
    assert payouts.max() <= params.coverage
    assert payouts.min() >= 0.0
    assert np.all(np.diff(payouts) >= -1e-12), "the trigger must be monotone"
    # A smooth trigger has a bounded derivative: sigmoid'(0) / eps_trigger.
    bound = 1.05 * step * params.coverage / (4.0 * params.trigger_width)
    assert np.abs(np.diff(payouts)).max() < bound, "the trigger must not be a step"


def test_liquidity_requirement_is_non_negative() -> None:
    params = FinanceParams()
    for loss in (0.0, 0.1, 1.0, 5.0):
        value = float(
            liquidity_requirement(jnp.asarray(loss), jnp.asarray(1.0), 1.0, params)
        )
        assert value >= 0.0


def test_cvar_is_at_least_the_mean_and_permutation_invariant() -> None:
    params = FinanceParams(alpha=0.9, smoothing=0.01)
    generator = np.random.default_rng(3)
    losses = jnp.asarray(np.sort(generator.exponential(size=64)))
    cvar = float(optimal_cvar(losses, params))
    assert cvar >= float(jnp.mean(losses))
    shuffled = jnp.asarray(generator.permutation(np.asarray(losses)))
    assert abs(cvar - float(optimal_cvar(shuffled, params))) < 1e-9


def test_cvar_uses_no_sort_and_stays_finite_on_extreme_values() -> None:
    params = FinanceParams(smoothing=0.05)
    losses = jnp.asarray([0.0, 1.0, 1e4])
    value = float(conditional_value_at_risk(losses, 1.0, params))
    assert np.isfinite(value)
    assert np.isfinite(float(smooth_plus(jnp.asarray(1e5), params.smoothing)))
    assert float(smooth_plus(jnp.asarray(-1e5), params.smoothing)) >= 0.0


def test_budget_is_respected_exactly_by_the_softmax() -> None:
    generator = np.random.default_rng(11)
    free = jnp.asarray(generator.normal(size=4) * 3.0)
    scales = jnp.asarray([1.0, 2.0, 0.5, 4.0])
    budget, intensity = budget_allocation(free, 7.5, scales)
    assert abs(float(jnp.sum(budget)) - 7.5) < 1e-12
    assert np.all(np.asarray(budget) >= 0.0)
    assert np.all(np.asarray(intensity) >= 0.0)
    assert np.all(np.asarray(intensity) < 1.0)


def test_finance_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="alpha"):
        FinanceParams(alpha=1.0)
    with pytest.raises(ValueError, match="smoothing"):
        FinanceParams(smoothing=0.0)
    with pytest.raises(ValueError, match="trigger width"):
        FinanceParams(trigger_width=-1.0)
