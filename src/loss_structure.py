import jax
import jax.numpy as jnp

# L_tot = L_santé + L_activité + L_actifs + C_interventions - I_assurance
# u = (u_fuel, u_sensor, u_filter, u_capacity, u_insurance, u_reserve)

COST= {
    'unit_costs': jnp.array([ # u
        300000.0,  # u_fuel, based on ONF and DGFIP (3000.0 per Ha)
        100000.0,  # u_sensor 
        150000.0,  # u_filter, 1 filter = 50-60k   
        150000.0,  # u_insurance, based on EIOPA studies
        200000.0   # u_reserve, based on PCS recommandation  
    ])
    'cost_per_admission' : 3000.0 # based on GHM studies
    'base_activity_loss' : 420000.0 # based on INSEE/CCI studies
    'base_assets_loss' : 680000.0 # based on the French Federation of the assurance
    'intervention_lost' : 75000.0 # based on SDIS reimbursement grid
    'max_insurance_payout' : 500000.0 # based on contracts of differents assurance (CCR, AXA CLIMATE...)
}

def total_loss(u, H_r, param = COST) :
    
    L_health = H_r * (1.0 - 0.5 * u[2]) * param['cost_per_admission']
    L_activity = param['base_activity_loss'] * fire_intensity * (1.0 - 0.4 * u[1])
    L_assets = param['base_assets_loss'] * fire_intensity * (1.0 - 0.5 * u[0])
    C_interventions = param['intervention_cost'] * fire_intensity
    brut_loss = L_health + L_activity + L_assets + C_interventions
    trigger = jnp.where(fire_intensity >= 0.6, 1.0, 0.0)
    I_assurance = u[3] * param['max_insurance_payout'] * trigger
    reserve_available = u[4] * param['unit_costs'][4]
    L_tot = jnp.maximum(0.0, perte_brute - I_assurance - reserve_available)
    
    return L_tot