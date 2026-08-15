import jax
import jax.numpy as jnp

from climacare.health import HealthZone, mean_exposure, incremental_health_impact

all = ["HealthZone", "mean_exposure", "incremental_health_impact", "health_impact"]

def health_impact(concentration_scenarios, zones, *, dt, cell_area, filter_level = 0.0) : # Compute hospital admission impacts across all scenarios and health zones
    # concentration_scenarios : it's an array of shape (n_scenarios, n_levels, ny, nx)
    # zones : List of HealthZone objects
    # dt : Simulation time step
    # cell_area : Area of a single grid cell
    # filter_level : Mitigation lever u_filter in [0, 1].

    # To sum up : it converts smoke concentration fields into quantified health impacts

    def per_scenario(concentration):  # une seule concentration (n_levels, ny, nx)
        impacts = []
        for zone in zones:
            e_r = mean_exposure(concentration, zone, dt = dt, cell_area = cell_area, filter_level = filter_level)
            dH_r = incremental_health_impact(e_r, zone, cell_area)
            impacts.append(dH_r)
        return jnp.stack(impacts) # (n_zones,)

    return jax.vmap(per_scenario)(concentration_scenarios) # (n_scenarios, n_zones)

