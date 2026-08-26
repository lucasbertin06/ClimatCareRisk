from __future__ import annotations

import jax
import jax.numpy as jnp

from risk import expected_shortfall, optimal_expected_shortfall, value_at_risk_smooth

# L_tot = L_santé + L_activité + L_actifs + C_interventions - I_assurance
# u = (u_fuel, u_sensor, u_filter, u_capacity, u_insurance, u_reserve)

ALPHA_CVAR = 0.95
BUDGET_MAX = 1_000_000.0
SMOOTHING_TAU = 10_000.0

COST= {
    'unit_costs': jnp.array([ # u
        300000.0,  # u_fuel, based on ONF and DGFIP (3000.0 per Ha)
        100000.0,  # u_sensor 
        150000.0,  # u_filter, 1 filter = 50-60k   
        150000.0,  # u_insurance, based on EIOPA studies
        200000.0,   # u_reserve, based on PCS recommandation  
    ]),
    'cost_per_admission' : 3000.0, # based on GHM studies
    'base_activity_loss' : 420000.0, # based on INSEE/CCI studies
    'base_assets_loss' : 680000.0, # based on the French Federation of the assurance
    'intervention_cost' : 75000.0, # based on SDIS reimbursement grid
    'max_insurance_payout' : 500000.0, # based on contracts of differents assurance (CCR, AXA CLIMATE...)
}

def total_loss(u, H_r, fire_intensity, basis_noise = 0.0, param = COST) :

    L_health = jnp.sum(H_r, -1) * (1.0 - 0.5 * u[2]) * param['cost_per_admission'] # added -1 to jnp.sum so we only add the last dimension of the array
    L_activity = param['base_activity_loss'] * fire_intensity * (1.0 - 0.4 * u[1])
    L_assets = param['base_assets_loss'] * fire_intensity * (1.0 - 0.5 * u[0])
    C_interventions = param['intervention_cost'] * fire_intensity
    
    financial_loss = L_activity + L_assets + C_interventions

    index_reading = jnp.clip(fire_intensity + basis_noise, 0.0, 1.0) # Basis risk handling: Account for noise between ground truth and satellite/weather index
    temp = param.get('trigger_temperature', 0.15) # temp = 0.15 instead of 0.05 so the gradients can propagate on the interval [0.4; 0.8]
    trigger = jax.nn.sigmoid((index_reading - 0.6) / temp) # with a smooth sigmoid to ensure continuous differentiability for JAX (jax.grad)

    I_assurance = u[3] * param['max_insurance_payout'] * trigger
    reserve_available = u[4] * param['unit_costs'][4]

    net_financial = jnp.maximum(0.0, financial_loss - I_assurance - reserve_available)

    L_tot = L_health + net_financial  # health impact always fully present, never netted by insurance/reserve

    return L_tot

def compute_risk_metrics(losses, alpha = ALPHA_CVAR, tau = SMOOTHING_TAU) : # This function returns EL, VaR, Expected Shortfall/CVaR 
    el = jnp.mean(losses) # the avg annual loss
    var = value_at_risk_smooth(losses, alpha, tau)
    es = optimal_expected_shortfall(losses, alpha, tau) # we know used the optimized version of the last functions
    
    return el, var, es

def robust_objective(u, scenarios_H_r, scenarios_fire, basis_noises, lambda_cvar, budget_max = BUDGET_MAX, param = COST) : # Function that implements the Robust Objective Function, adds EL and CVaR et applies penalities if the investment is over budget_max  
    # J(u) = Expected Loss + lambda * CVaR + Penalities (Budget and Bounds [0,1])
    # basis_noise : L'erreur ou l'écart de mesure entre l'indice satellite/météo et la réalité du terrain : 0 = mesure parfaite
    if basis_noises is None :    
        basis_noises = jnp.zeros_like(scenarios_fire) 

    # penalties
    total_capex = jnp.sum(u * param['unit_costs'])
    budget_penalty = 1e6 * jnp.square(jnp.maximum(0.0, total_capex - budget_max))
    
    # Évaluation vectorisée sur l'ensemble des scénarios
    losses = jax.vmap(
        lambda h, f, n: total_loss(u, h, f, n, param)
    )(scenarios_H_r, scenarios_fire, basis_noises)
    
    el = jnp.mean(losses)
    es = optimal_expected_shortfall(losses, alpha = ALPHA_CVAR)
    
    return el + lambda_cvar * es + budget_penalty 

def optimize_portfolio(scenarios_H_r, scenarios_fire , basis_noises = None, lambda_cvar = 0.5, budget_max = BUDGET_MAX, lr = 0.5, steps = 300, return_history = False) : # finds the optimal protfolio u* by gradient descent with JAX .
    # lr : vitesse a laquelle la descente de gradient modifie u a chaque etape
    # steps : Nombre total d'itérations d'ajustement du portefeuille. À chaque itération, JAX calcule le gradient de J(u) et met à jour u
    # return_history : if True, also returns the trajectory of J(u) at each step (for convergence plots)
    if basis_noises is None:
        basis_noises = jnp.zeros_like(scenarios_fire)

    u_init = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
    grad_fn = jax.grad(robust_objective, argnums = 0)
    obj_fn = lambda u: robust_objective(u, scenarios_H_r, scenarios_fire, basis_noises, lambda_cvar, budget_max)

    def optimized_range(u, _) :
        raw_grads = grad_fn(u, scenarios_H_r, scenarios_fire, basis_noises, lambda_cvar, budget_max)
        # Normalize the step : the objective mixes currency-scale terms (~1e6),
        # so raw gradients are huge and a fixed lr diverges (NaN). Scaling by
        # the inf-norm keeps the descent direction while bounding the step
        # to at most lr per lever.
        step_scale = jnp.maximum(jnp.max(jnp.abs(raw_grads)), 1e-12)
        grads = raw_grads / step_scale
        # Backtracking : reject any step that increases J (the CVaR term is only
        # locally smooth, so a full lr step can overshoot near the kink and
        # diverge after ~150 iterations). Halving until J decreases keeps the
        # descent monotone without changing the fixed point.
        j_current = obj_fn(u)
        new_u = jnp.clip(u - lr * grads, 0.0, 1.0)
        j_new = obj_fn(new_u)

        def backtrack(carry) :
            u_try, j_try, half = carry
            u_next = jnp.clip(u - 0.5 * half * lr * grads, 0.0, 1.0)
            j_next = obj_fn(u_next)
            accept = j_next < j_try
            return (jnp.where(accept, u_next, u_try),
                    jnp.where(accept, j_next, j_try),
                    half * 0.5)

        def cond_fn(carry) :
            _, j_try, half = carry
            return jnp.logical_and(j_try >= j_current, half > 1e-4)

        final_u, final_j, _ = jax.lax.while_loop(cond_fn, backtrack,
                                                 (new_u, j_new, jnp.array(1.0)))
        return final_u, obj_fn(final_u) # keep the objective at each step for the convergence curve

    final_u, history = jax.lax.scan(optimized_range, u_init, xs = None, length = steps)

    if return_history :
        # prepend the initial objective so the curve has steps + 1 points
        j_init = obj_fn(u_init)
        return final_u, jnp.concatenate([jnp.array([j_init]), history])

    return final_u

# ---------------------------------------------------------------------------
# Baseline policies required by the specification (section 6.1) :
# the optimized portfolio must be compared against a uniform allocation and an
# insurance-only strategy on the same scenario set.
# ---------------------------------------------------------------------------

def uniform_policy() -> jax.Array :
    """Equal 20% allocation across the five decision levers."""
    return policy_uniform()

def insurance_only_policy() -> jax.Array :
    """All capital on the parametric insurance lever, nothing on prevention."""
    return policy_insurance()

def evaluate_policy(u, scenarios_H_r, scenarios_fire, basis_noises = None, alpha = ALPHA_CVAR) :
    """Return (EL, VaR, CVaR, capex) for a given portfolio on the scenario set."""
    if basis_noises is None :
        basis_noises = jnp.zeros_like(scenarios_fire)
    losses = jax.vmap(lambda h, f, n: total_loss(u, h, f, n))(scenarios_H_r, scenarios_fire, basis_noises)
    el, var, es = compute_risk_metrics(losses, alpha = alpha)
    capex = jnp.sum(u * COST["unit_costs"])
    return el, var, es, capex

def compare_policies(scenarios_H_r, scenarios_fire, basis_noises = None, lambda_cvar = 0.5, steps = 300) :
    """Compare uniform / insurance-only / optimized portfolios.

    Returns a dict mapping policy name to its metrics dict. Used by experiment
    E5 of the experimental plan (which portfolio dominates naive strategies ?).
    """
    if basis_noises is None :
        basis_noises = jnp.zeros_like(scenarios_fire)

    u_opt = optimize_portfolio(scenarios_H_r, scenarios_fire, basis_noises,
                               lambda_cvar = lambda_cvar, steps = steps)

    def _metrics(name, u) :
        el, var, es, capex = evaluate_policy(u, scenarios_H_r, scenarios_fire, basis_noises)
        return {"u": u, "EL": float(el), "VaR": float(var), "CVaR": float(es), "capex": float(capex)}

    return {
        "uniform": _metrics("uniform", uniform_policy()),
        "insurance_only": _metrics("insurance_only", insurance_only_policy()),
        "optimized": _metrics("optimized", u_opt),
    }

def generate_efficient_frontier(scenarios_H_r, scenarios_fire, n_points, basis_noises = None) : # Computes the Pareto frontier (EL vs. CVaR) by sweeping through n_points values of lambda_cvar
    # n_points : The number of optimal portfolios to compute along the frontier.
    if basis_noises is None :
        basis_noises = jnp.zeros_like(scenarios_fire)

    lambdas = jnp.linspace(0.0, 2.0, n_points)

    # vmap over lambda_cvar only : one compile handles all n_points portfolios.
    # scenarios_H_r / scenarios_fire / basis_noises stay fixed (in_axes=None).
    # NOTE : optimize_portfolio is not jitted, so this runs the scan eagerly
    # per lambda ; wrap with jax.jit if profiling shows it dominates runtime.
    def _solve(l) :
        return optimize_portfolio(scenarios_H_r, scenarios_fire, basis_noises,
                                  lambda_cvar = l, steps = 300)

    def _single(l) :
        u = _solve(l)
        el, var, es, capex = evaluate_policy(u, scenarios_H_r, scenarios_fire, basis_noises)[:2]
        return u, el, es

    results = [(_single(float(l))) for l in lambdas]  # plain loop keeps NaNs from one lambda out of the others
    portfolios = jnp.stack([r[0] for r in results])
    frontier_el = jnp.stack([r[1] for r in results])
    frontier_es = jnp.stack([r[2] for r in results])

    return portfolios, frontier_el, frontier_es

def apply_stress(kind, *, scenarios_fire, basis_noises, budget_max) : # it's a Stress-Testing function, its objective is to verify the robustness of the optimal investment portfolio u* when subjected to unforeseen degradations
    if kind == "wind_strong" : # intense fire
        return dict(scenarios_fire = jnp.clip(scenarios_fire * 1.3, 0.0, 1.0), basis_noises = basis_noises, budget_max = budget_max) # jnp.clip only here to cap the intesity to 1.0 (100%)
    
    if kind == "sensor_biased" : # sensors systematically underestimate
        return dict(scenarios_fire = scenarios_fire, basis_noises = basis_noises - 0.15, budget_max = budget_max)
    
    if kind == "budget_cut" : # -20% budget
        return dict(scenarios_fire = scenarios_fire, basis_noises = basis_noises, budget_max = budget_max * 0.8)
    
    raise ValueError(f"unknown stress kind : {kind}")

def run_stress_tests(u_opt, scenarios_H_r, scenarios_fire, basis_noises, budget_max = BUDGET_MAX) : # Evaluate a fixed optimal portfolio u_opt under stress scenarios without re-optimizing 
    def metrics(fire, noises) :
        losses = jax.vmap(lambda h, f, n: total_loss(u_opt, h, f, n))(scenarios_H_r, fire, noises)
        el, _, es = compute_risk_metrics(losses)
        capex = jnp.sum(u_opt * COST["unit_costs"])
        feasible = capex <= budget  # u_opt reste il finançable sous le budget réduit
        return {"EL" : el, "CVaR" : es, "feasible" : bool(feasible), "capex" : float(capex)}

    results = {"nominal": metrics(scenarios_fire, basis_noises, budget_max)}

    for kind in ["wind_strong", "sensor_biased", "budget_cut"] : # Stress testing across predefined perturbation kinds
        s = apply_stress(kind, scenarios_fire = scenarios_fire, basis_noises = basis_noises, budget_max = budget_max)
        results[kind] = metrics(s["scenarios_fire"], s["basis_noises"], s["budget_max"])
    return results

def policy_uniform(n_levers = 5) : # it's an equivalent repartition
    return jnp.full((n_levers,), 1.0 / n_levers) # it ensures the sum of lever costs stays within budget_max constraint

def policy_insurance(n_levers = 5, insurance_idx = 3) : # we based the budget on insurance 
    u = jnp.zeros(n_levers)
    return u.at[insurance_idx].set(1.0)

def evaluate_smart_criterion(u_opt, scenarios_H_r, scenarios_fire, basis_noises) : # implementaiton point 10.3 du pdf, we compare the baseline with the optimized portfolio
    # Evaluates portfolio reduction against SMART criteria (se baser sur le pdf page 6)

    # this is where the comparison starts

    def compute_cvar(u) :
        losses = jax.vmap(lambda h, f, n: total_loss(u, h, f, n))(scenarios_H_r, scenarios_fire, basis_noises)
        _, _, cvar = compute_risk_metrics(losses)
        return cvar

    cvar_opt = compute_cvar(u_opt)
    cvar_uniform = compute_cvar(policy_uniform())
    cvar_insurance = compute_cvar(policy_insurance())

    # Compute percentage reduction relative to baselines
    reduction_vs_uniform = (cvar_uniform - cvar_opt) / cvar_uniform
    reduction_vs_insurance = (cvar_insurance - cvar_opt) / cvar_insurance

    return {
        "cvar_opt" : cvar_opt,
        "cvar_uniform" : cvar_uniform,
        "cvar_insurance" : cvar_insurance,
        "reduction_vs_uniform" : reduction_vs_uniform,
        "reduction_vs_insurance" : reduction_vs_insurance,
        "smart_target_met" : (reduction_vs_uniform >= 0.20) and (reduction_vs_insurance >= 0.20), # TRUE only if the optimization reduce risk by 20% compared to the 2 baselines
    }
