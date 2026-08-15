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
    
    brut_loss = L_health + L_activity + L_assets + C_interventions

    index_reading = jnp.clip(fire_intensity + basis_noise, 0.0, 1.0) # Basis risk handling: Account for noise between ground truth and satellite/weather index
    temp = param.get('trigger_temperature', 0.15) # temp = 0.15 instead of 0.05 so the gradients can propagate on the interval [0.4; 0.8]
    trigger = jax.nn.sigmoid((index_reading - 0.6) / temp) # with a smooth sigmoid to ensure continuous differentiability for JAX (jax.grad)
    
    I_assurance = u[3] * param['max_insurance_payout'] * trigger

    reserve_available = u[4] * param['unit_costs'][4]
    L_tot = jnp.maximum(0.0, brut_loss - I_assurance - reserve_available)
    
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

@jax.jit
def optimize_portfolio(scenarios_H_r, scenarios_fire , basis_noises = None, lambda_cvar = 0.5, budget_max = BUDGET_MAX, lr = 0.01, steps = 300) : # finds the optimal protfolio u* by gradient descent with JAX .
    # lr : vitesse a laquelle la descente de gradient modifie u a chaque etape
    # steps : Nombre total d'itérations d'ajustement du portefeuille. À chaque itération, JAX calcule le gradient de J(u) et met à jour u   
    if basis_noises is None:
        basis_noises = jnp.zeros_like(scenarios_fire)

    u_init = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
    grad_fn = jax.grad(robust_objective, argnums = 0)
    
    def optimized_range(u, _) :
        grads = grad_fn(u, scenarios_H_r, scenarios_fire, basis_noises, lambda_cvar, budget_max)
        new_u = u - lr * grads
        new_u = jnp.clip(new_u, 0.0, 1.0)
        return new_u, None # we don't want the 300 proofs, only the optimized u*
        
    final_u, v = jax.lax.scan(optimized_range, u_init, xs = None, length = steps)    

    return final_u

def generate_efficient_frontier(scenarios_H_r, scenarios_fire, n_points, basis_noises = None) : # Computes the Pareto frontier (EL vs. CVaR) by sweeping through n_points values of lambda_cvar
    # n_points : The number of optimal portfolios to compute along the frontier.
    if basis_noises is None:
        basis_noises = jnp.zeros_like(scenarios_fire)
        
    lambdas = jnp.linspace(0.0, 2.0, n_points)
    frontier_portfolios, frontier_el, frontier_es = [], [], []
    
    for lmbda in lambdas:
        u_opt = optimize_portfolio(scenarios_H_r, scenarios_fire, basis_noises, lambda_cvar = lmbda)
        
        losses = jax.vmap(lambda h, f, n: total_loss(u_opt, h, f, n))(scenarios_H_r, scenarios_fire, basis_noises)
        el, _, es = compute_risk_metrics(losses)
        
        frontier_portfolios.append(u_opt)
        frontier_el.append(el)
        frontier_es.append(es)
        
    return jnp.array(frontier_portfolios), jnp.array(frontier_el), jnp.array(frontier_es)

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
        return {"EL": el, "CVaR": es}

    results = {"nominal": metrics(scenarios_fire, basis_noises)}

    for kind in ["wind_strong", "sensor_biased", "budget_cut"] : # Stress testing across predefined perturbation kinds
        s = apply_stress(kind, scenarios_fire = scenarios_fire, basis_noises = basis_noises, budget_max = budget_max)
        results[kind] = metrics(s["scenarios_fire"], s["basis_noises"])
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