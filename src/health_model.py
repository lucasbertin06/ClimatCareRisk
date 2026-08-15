from climacare.health import HealthZone, mean_exposure, incremental_health_impact

all = ["HealthZone", "mean_exposure", "incremental_health_impact"]

def health_impact(concentration_scenarios, zones, *, dt, cell_area, filter_level = 0.0) : # Compute hospital admission impacts across all scenarios and health zones
    # concentration_scenarios : it's an array of shape (n_scenarios, n_levels, ny, nx)
    # zones : List of HealthZone objects
    # dt : Simulation time step
    # cell_area : Area of a single grid cell
    # filter_level : Mitigation lever u_filter in [0, 1].

    # To sum up : it converts smoke concentration fields into quantified health impacts

    C_eff = concentration_scenarios / (1.0 - filter_level)

    zone_masks = jnp.stack([z.mask for z in zones]) # stack is great to pass from 2D to 3D : (n_zones, ny, nx)

    pop_grid = zones[0].pop_grid # (ny, nx)
    
    exposure = jnp.einsum('styx, ryx, yx -> sr', C_eff, zone_masks, pop_grid)

    # sanitary impact per zone and scenario
    baseline_rates = jnp.array([z.baseline_rate for z in zones])
    hospital_impact = exposure * baseline_rates

    return hospital_impact
