import numpy as np
import jax.numpy as jnp

from climacare.objective import PARAMETER_NAMES, decode_free_parameters
from health_model import health_impact

def generate_scenario(pipeline, config, zones, n_scenarios, *, dt, cell_area, filter_level = 0.0, seed = 0, frame_count = None) :
    # Draw n_scenarios theta from the prior, run pipeline.direct() for each, and return (scenarios_fire, scenarios_H_r) ready for optimize_portfolio()/generate_efficient_frontier()
    rng = np.random.default_rng(seed)
    z_draws = rng.standard_normal((n_scenarios, len(PARAMETER_NAMES))) # output is a matrix (n_scenarios, n_params)

    fires, concentrations = [], [] # we will stock the output in those

    for z in z_draws :
        theta = decode_free_parameters(jnp.asarray(z), config) # converts the vector z into a physical parameter
        out = pipeline.direct(theta, frame_count=frame_count) # to make work the simulation
        fires.append(out["burned_area"])
        concentrations.append(out["concentration_frames"])