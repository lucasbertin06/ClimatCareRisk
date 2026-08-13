from climacare.health import HealthZone, mean_exposure, incremental_health_impact

all = ["HealthZone", "mean_exposure", "incremental_health_impact"]

def health_impact(concentration_scenarios, zones, *, dt, cell_area, filter_level = 0.0) : # Compute hospital admission impacts across all scenarios and health zones
    # concentration_scenarios : it's an array of shape (n_scenarios, n_levels, ny, nx)
    # zones : List of HealthZone objects
    # dt : Simulation time step
    # cell_area : Area of a single grid cell
    # filter_level : Mitigation lever u_filter in [0, 1].

    # all this returns a jax.Array of shape (n_scenarios, n_zones) expected by robust_objective
    # To sum up : it converts smoke concentration fields into quantified health impacts

    def per_scenario(concentration) : # Auxiliary function computing health impacts across all zones for a single scenario tensor
        impacts = [incremental_health_impact(
            mean_exposure(concentration, zone, dt = dt, cell_area = cell_area, filter_level = filter_level), 
            zone, cell_area, 
        )
        for zone in zones ]

        return jnp.stack(impacts) # Pack the list of per-zone scalar impacts into a 1D JAX array of shape (n_zones,)

    return jax.vmap(per_scenario)(concentration_scenarios) # Automatically vectorize 'per_scenario' over the leading axis (n_scenarios) using JAX compilation.

    # jax.vmap transforms single-instance array logic into batch operations executed in parallel

