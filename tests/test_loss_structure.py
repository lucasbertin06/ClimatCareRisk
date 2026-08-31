import jax.numpy as jnp
import pytest

from loss_structure import (
    COST,
    evaluate_policy,
    generate_efficient_frontier,
    optimize_portfolio,
    robust_objective,
    run_stress_tests,
    total_loss,
)


def _fake_scenarios(n=20):
    # Generates n dummy scenarios for testing 
    H_r = jnp.ones((n, 3)) * 2.0  # 3 regions with a constant health impact of 2.0
    fire = jnp.linspace(0.1, 0.9, n)  # Fire intensity scaling from 0.1 to 0.9 across n scenarios
    return H_r, fire


def test_total_loss_is_nonnegative():
    # Net loss must never be negative 
    H_r, fire = _fake_scenarios()
    u = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])  # "Neutral" portfolio allocation (20% per lever)
    losses = jnp.array([total_loss(u, h, f) for h, f in zip(H_r, fire)])
    assert jnp.all(losses >= 0.0)


def test_more_insurance_reduces_loss():
    # Higher insurance coverage (u_insurance) must decrease net loss, never increase it
    H_r, fire = _fake_scenarios()
    u_low = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])  # No active mitigation levers
    u_high = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0])  # Insurance fully maxed out
    loss_low = total_loss(u_low, H_r[0], fire[10])  # Evaluation at ~0.5 fire intensity
    loss_high = total_loss(u_high, H_r[0], fire[10])
    assert loss_high <= loss_low  # Purchasing insurance should reduce net financial exposure


def test_optimize_portfolio_respects_bounds_and_budget():
    # Verifies that gradient descent produces a valid optimal portfolio vector u*
    H_r, fire = _fake_scenarios()
    u_opt = optimize_portfolio(H_r, fire, steps=50)  # Reduced steps (50) for fast test runtime
    assert jnp.all(u_opt >= 0.0) and jnp.all(u_opt <= 1.0)  # Decision bounds [0, 1] must hold
    total_capex = jnp.sum(u_opt * COST["unit_costs"])
    assert total_capex <= COST["unit_costs"].sum() * 1.01  # 1% numerical tolerance margin for soft budget penalty

def test_run_stress_tests_increases_risk():
    # Verifies that stress conditions degrade risk metrics (higher loss/CVaR) relative to nominal
    H_r, fire = _fake_scenarios()
    basis_noises = jnp.zeros_like(fire)
    u_opt = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])

    results = run_stress_tests(u_opt, H_r, fire, basis_noises)

    # Check that all stress scenarios are evaluated
    assert "nominal" in results
    assert "wind_strong" in results

    # Strong wind must increase expected loss (EL) compared to nominal
    assert results["wind_strong"]["EL"] >= results["nominal"]["EL"]


def test_objective_includes_investment_cost():
    health = jnp.zeros((4, 3))
    fire = jnp.zeros(4)
    assert robust_objective(jnp.ones(5), health, fire, None, 0.0) > robust_objective(
        jnp.zeros(5), health, fire, None, 0.0
    )


def test_optimizer_history_is_monotone_and_budget_is_exact():
    health, fire = _fake_scenarios()
    budget = 200_000.0
    portfolio, history = optimize_portfolio(
        health, fire, budget_max=budget, steps=5, return_history=True
    )
    assert jnp.all(history[1:] <= history[:-1] + 1e-6)
    assert jnp.sum(portfolio * COST["unit_costs"]) <= budget + 1e-6


def test_negative_budget_is_rejected():
    health, fire = _fake_scenarios()
    with pytest.raises(ValueError, match="non-negative"):
        optimize_portfolio(health, fire, budget_max=-1.0)


def test_frontier_total_loss_is_non_increasing():
    health, fire = _fake_scenarios()
    portfolios, _, _, _, _ = generate_efficient_frontier(health, fire, 3)
    expected = jnp.asarray([evaluate_policy(u, health, fire)[0] for u in portfolios])
    assert jnp.all(expected[1:] <= expected[:-1] + 1e-6)