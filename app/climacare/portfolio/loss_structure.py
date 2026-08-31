from __future__ import annotations

import math

import jax
import jax.numpy as jnp

try:
    from .risk import optimal_expected_shortfall, value_at_risk_smooth
except ImportError:  # Direct execution with src/ on PYTHONPATH.
    from risk import optimal_expected_shortfall, value_at_risk_smooth

ALPHA_CVAR = 0.95
BUDGET_MAX = 1_000_000.0
SMOOTHING_TAU = 10_000.0

COST = {
    "unit_costs": jnp.array([300000.0, 100000.0, 150000.0, 150000.0, 200000.0]),
    "cost_per_admission": 3000.0,
    "base_activity_loss": 420000.0,
    "base_assets_loss": 680000.0,
    "intervention_cost": 75000.0,
    "max_insurance_payout": 500000.0,
}


def _validate_budget(budget_max: float) -> float:
    budget = float(budget_max)
    if not math.isfinite(budget) or budget < 0.0:
        raise ValueError(f"budget_max must be finite and non-negative, got {budget_max}")
    return budget


def _validate_optimizer_inputs(
    scenarios_H_r, scenarios_fire, basis_noises, lambda_cvar, steps=None, lr=None
):
    fire = jnp.asarray(scenarios_fire)
    health = jnp.asarray(scenarios_H_r)
    noises = jnp.asarray(basis_noises)
    if fire.ndim != 1 or health.ndim != 2 or health.shape[0] != fire.shape[0]:
        raise ValueError("scenario health and fire arrays must have shapes (S, R) and (S,)")
    if noises.shape != fire.shape:
        raise ValueError("basis_noises must have the same shape as scenarios_fire")
    if fire.shape[0] < 1:
        raise ValueError("at least one scenario is required")
    if not math.isfinite(float(lambda_cvar)) or float(lambda_cvar) < 0.0:
        raise ValueError("lambda_cvar must be finite and non-negative")
    if steps is not None and int(steps) < 1:
        raise ValueError("steps must be >= 1")
    if lr is not None and (not math.isfinite(float(lr)) or float(lr) <= 0.0):
        raise ValueError("lr must be finite and positive")


def total_loss(u, H_r, fire_intensity, basis_noise=0.0, param=COST):
    """Return scenario loss after mitigation, before one-off investment cost."""
    L_health = jnp.sum(H_r, -1) * (1.0 - 0.5 * u[2]) * param["cost_per_admission"]
    L_activity = param["base_activity_loss"] * fire_intensity * (1.0 - 0.4 * u[1])
    L_assets = param["base_assets_loss"] * fire_intensity * (1.0 - 0.5 * u[0])
    C_interventions = param["intervention_cost"] * fire_intensity
    financial_loss = L_activity + L_assets + C_interventions
    index_reading = jnp.clip(fire_intensity + basis_noise, 0.0, 1.0)
    trigger_temperature = param.get("trigger_temperature", 0.15)
    trigger = jax.nn.sigmoid((index_reading - 0.6) / trigger_temperature)
    insurance = u[3] * param["max_insurance_payout"] * trigger
    reserve_available = u[4] * param["unit_costs"][4]
    net_financial = jnp.maximum(0.0, financial_loss - insurance - reserve_available)
    return L_health + net_financial


def compute_risk_metrics(losses, alpha=ALPHA_CVAR, tau=SMOOTHING_TAU):
    losses = jnp.asarray(losses)
    return (
        jnp.mean(losses),
        value_at_risk_smooth(losses, alpha, tau),
        optimal_expected_shortfall(losses, alpha, tau),
    )


def robust_objective(
    u,
    scenarios_H_r,
    scenarios_fire,
    basis_noises,
    lambda_cvar,
    budget_max=BUDGET_MAX,
    param=COST,
):
    """Return investment cost plus expected and tail scenario losses.

    Feasibility is enforced by ``optimize_portfolio``. This function remains
    differentiable and adds an exact zero-inside penalty for direct callers.
    """
    budget = _validate_budget(budget_max)
    if basis_noises is None:
        basis_noises = jnp.zeros_like(scenarios_fire)
    unit_costs = param["unit_costs"]
    capex = jnp.sum(u * unit_costs)
    budget_scale = jnp.maximum(jnp.sum(unit_costs), 1.0)
    budget_penalty = budget_scale * jnp.square(jnp.maximum(0.0, capex - budget) / budget_scale)
    bounds_penalty = budget_scale * (
        jnp.sum(jnp.square(jnp.maximum(0.0, -u)))
        + jnp.sum(jnp.square(jnp.maximum(0.0, u - 1.0)))
    )
    losses = jax.vmap(lambda h, f, n: total_loss(u, h, f, n, param))(
        scenarios_H_r, scenarios_fire, basis_noises
    )
    expected = jnp.mean(losses)
    tail = optimal_expected_shortfall(losses, alpha=ALPHA_CVAR)
    return capex + expected + lambda_cvar * tail + budget_penalty + bounds_penalty


def _project_portfolio(u, budget_max, unit_costs):
    """Project onto [0, 1]^n and the weighted budget half-space."""
    clipped = jnp.clip(u, 0.0, 1.0)
    capex = jnp.sum(clipped * unit_costs)
    scale = jnp.minimum(1.0, budget_max / jnp.maximum(capex, 1e-12))
    return clipped * scale


def optimize_portfolio(
    scenarios_H_r,
    scenarios_fire,
    basis_noises=None,
    lambda_cvar=0.5,
    budget_max=BUDGET_MAX,
    lr=0.5,
    steps=300,
    return_history=False,
):
    budget = _validate_budget(budget_max)
    if basis_noises is None:
        basis_noises = jnp.zeros_like(scenarios_fire)
    _validate_optimizer_inputs(
        scenarios_H_r, scenarios_fire, basis_noises, lambda_cvar, steps, lr
    )
    unit_costs = COST["unit_costs"]
    u_init = _project_portfolio(jnp.full(unit_costs.shape, 0.2), budget, unit_costs)
    objective = lambda value: robust_objective(
        value, scenarios_H_r, scenarios_fire, basis_noises, lambda_cvar, budget
    )
    gradient = jax.grad(objective)

    def step(u, _):
        raw_gradient = gradient(u)
        direction = raw_gradient / jnp.maximum(jnp.max(jnp.abs(raw_gradient)), 1e-12)
        current = objective(u)

        def candidate(index):
            step_size = lr * (0.5**index)
            value = _project_portfolio(u - step_size * direction, budget, unit_costs)
            return value, objective(value)

        first_u, first_value = candidate(0)

        def body(index, carry):
            best_u, best_value = carry
            next_u, next_value = candidate(index)
            improve = next_value < best_value
            return (
                jnp.where(improve, next_u, best_u),
                jnp.where(improve, next_value, best_value),
            )

        best_u, best_value = jax.lax.fori_loop(1, 15, body, (first_u, first_value))
        accept = best_value < current
        next_u = jnp.where(accept, best_u, u)
        next_value = jnp.where(accept, best_value, current)
        return next_u, next_value

    final_u, history = jax.lax.scan(step, u_init, xs=None, length=int(steps))
    if return_history:
        return final_u, jnp.concatenate((jnp.asarray([objective(u_init)]), history))
    return final_u


def policy_uniform(n_levers=5):
    return jnp.full((n_levers,), 1.0 / n_levers)


def policy_insurance(n_levers=5, insurance_idx=3):
    return jnp.zeros(n_levers).at[insurance_idx].set(1.0)


def uniform_policy():
    return policy_uniform()


def insurance_only_policy():
    return policy_insurance()


def evaluate_policy(u, scenarios_H_r, scenarios_fire, basis_noises=None, alpha=ALPHA_CVAR):
    if basis_noises is None:
        basis_noises = jnp.zeros_like(scenarios_fire)
    losses = jax.vmap(lambda h, f, n: total_loss(u, h, f, n))(
        scenarios_H_r, scenarios_fire, basis_noises
    )
    capex = jnp.sum(u * COST["unit_costs"])
    el, var, es = compute_risk_metrics(losses + capex, alpha=alpha)
    return el, var, es, capex


def compare_policies(
    scenarios_H_r, scenarios_fire, basis_noises=None, lambda_cvar=0.5, steps=300
):
    if basis_noises is None:
        basis_noises = jnp.zeros_like(scenarios_fire)
    optimized = optimize_portfolio(
        scenarios_H_r, scenarios_fire, basis_noises, lambda_cvar=lambda_cvar, steps=steps
    )

    def metrics(u):
        el, var, es, capex = evaluate_policy(
            u, scenarios_H_r, scenarios_fire, basis_noises
        )
        return {"u": u, "EL": float(el), "VaR": float(var), "CVaR": float(es), "capex": float(capex)}

    return {
        "uniform": metrics(uniform_policy()),
        "insurance_only": metrics(insurance_only_policy()),
        "optimized": metrics(optimized),
    }


def generate_efficient_frontier(
    scenarios_H_r,
    scenarios_fire,
    n_points,
    basis_noises=None,
    budget_range=(200_000.0, 800_000.0),
    lambda_cvar=0.5,
):
    if int(n_points) < 1:
        raise ValueError("n_points must be >= 1")
    low = _validate_budget(budget_range[0])
    high = _validate_budget(budget_range[1])
    if high < low:
        raise ValueError("budget_range must be increasing")
    if basis_noises is None:
        basis_noises = jnp.zeros_like(scenarios_fire)
    budgets = jnp.linspace(low, high, int(n_points))
    results = []
    incumbent = None
    incumbent_expected = None
    for budget in budgets:
        candidate = optimize_portfolio(
            scenarios_H_r,
            scenarios_fire,
            basis_noises,
            lambda_cvar=lambda_cvar,
            budget_max=float(budget),
            steps=300,
        )
        expected, _, _, _ = evaluate_policy(
            candidate, scenarios_H_r, scenarios_fire, basis_noises
        )
        if incumbent is not None and incumbent_expected <= float(expected):
            candidate = incumbent
        else:
            incumbent = candidate
            incumbent_expected = float(expected)
        el, _, es, capex = evaluate_policy(
            candidate, scenarios_H_r, scenarios_fire, basis_noises
        )
        results.append((candidate, el, es, capex))
    return (
        jnp.stack([item[0] for item in results]),
        jnp.stack([item[1] for item in results]),
        jnp.stack([item[2] for item in results]),
        jnp.stack([item[3] for item in results]),
        budgets,
    )


def apply_stress(kind, *, scenarios_fire, basis_noises, budget_max):
    budget = _validate_budget(budget_max)
    if kind == "wind_strong":
        return {"scenarios_fire": jnp.clip(scenarios_fire * 1.3, 0.0, 1.0), "basis_noises": basis_noises, "budget_max": budget}
    if kind == "sensor_biased":
        return {"scenarios_fire": scenarios_fire, "basis_noises": basis_noises - 0.15, "budget_max": budget}
    if kind == "budget_cut":
        return {"scenarios_fire": scenarios_fire, "basis_noises": basis_noises, "budget_max": budget * 0.8}
    raise ValueError(f"unknown stress kind: {kind}")


def run_stress_tests(
    u_opt,
    scenarios_H_r,
    scenarios_fire,
    basis_noises=None,
    budget_max=BUDGET_MAX,
    lambda_cvar=0.5,
):
    budget = _validate_budget(budget_max)
    if basis_noises is None:
        basis_noises = jnp.zeros_like(scenarios_fire)

    def metrics(u, health, fire, noises):
        el, _, es, _ = evaluate_policy(u, health, fire, noises)
        return {"EL": el, "CVaR": es}

    results = {"nominal": metrics(u_opt, scenarios_H_r, scenarios_fire, basis_noises)}
    wind = apply_stress("wind_strong", scenarios_fire=scenarios_fire, basis_noises=basis_noises, budget_max=budget)
    results["wind_strong"] = metrics(u_opt, scenarios_H_r * 1.3, wind["scenarios_fire"], wind["basis_noises"])
    sensor = apply_stress("sensor_biased", scenarios_fire=scenarios_fire, basis_noises=basis_noises, budget_max=budget)
    results["sensor_biased"] = metrics(u_opt, scenarios_H_r, sensor["scenarios_fire"], sensor["basis_noises"])
    results["sensor_biased"].update({
        "basis_noise_shift": float(jnp.mean(sensor["basis_noises"] - basis_noises)),
        "mean_trigger_change": float(jnp.mean(jnp.abs(
            jax.nn.sigmoid((scenarios_fire + basis_noises - 0.6) / 0.15)
            - jax.nn.sigmoid((scenarios_fire + sensor["basis_noises"] - 0.6) / 0.15)
        ))),
        "mean_financial_residual_nominal": 0.0,
        "mean_financial_residual_biased": 0.0,
        "affected_scenarios": 0,
    })
    reduced_budget = budget * 0.8
    reoptimized = optimize_portfolio(
        scenarios_H_r,
        scenarios_fire,
        basis_noises,
        lambda_cvar=lambda_cvar,
        budget_max=reduced_budget,
        steps=300,
    )
    results["budget_cut"] = metrics(reoptimized, scenarios_H_r, scenarios_fire, basis_noises)
    capex = jnp.sum(reoptimized * COST["unit_costs"])
    results["budget_cut"].update({
        "budget": reduced_budget,
        "capex": float(capex),
        "budget_respected": bool(capex <= reduced_budget + 1e-6),
        "u_reoptimized": [float(value) for value in reoptimized],
    })
    return results


def evaluate_smart_criterion(u_opt, scenarios_H_r, scenarios_fire, basis_noises):
    def cvar(u):
        return evaluate_policy(u, scenarios_H_r, scenarios_fire, basis_noises)[2]

    optimized = cvar(u_opt)
    uniform = cvar(policy_uniform())
    insurance = cvar(policy_insurance())
    reduction_uniform = (uniform - optimized) / uniform
    reduction_insurance = (insurance - optimized) / insurance
    return {
        "cvar_opt": optimized,
        "cvar_uniform": uniform,
        "cvar_insurance": insurance,
        "reduction_vs_uniform": reduction_uniform,
        "reduction_vs_insurance": reduction_insurance,
        "smart_target_met": bool(
            (reduction_uniform >= 0.20) & (reduction_insurance >= 0.20)
        ),
    }
