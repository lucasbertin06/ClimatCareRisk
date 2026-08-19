import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from climacare.config import PARAMETER_NAMES
from climacare.inverse import make_physical_loss
from climacare.objective import Observations
from climacare.pipeline import TesseractPipeline

def probabilistic_model( # y = G(theta) + noise, Allows fixing certain parameters
    pipeline: TesseractPipeline,
    observations: Observations, # Structure containing field data collected from sensors
    fixed_theta: jnp.ndarray = None, 
    free_indices: tuple = (0, 1, 2, 3), 
) :

    # we have : y = G(θ, u) + ε, ε ∼ N (0, Σ) 

    config = pipeline.config # Extract global configuration from pipeline
    priors = config.priors # Extract prior distribution hyperparameters (means, stds)
    low, high = config.position_bounds # Extract grid spatial bounds [min, max]

    components = [] # to reconstruct the theta vector 

    for i, name in enumerate(PARAMETER_NAMES) :  
        if i not in free_indices:  #
            components.append(fixed_theta[i]) # Keep constant value for fixed parameter
        elif name in ("x0", "y0") : 
            components.append(numpyro.sample(name, dist.Uniform(low, high))) # Uninformative uniform prior
        elif name == "log_amplitude" : # Case for smoke emission intensity (log scale)
            components.append(numpyro.sample(name, dist.Normal(priors.log_amplitude_mean, priors.log_amplitude_std),
                )
            )
        elif name == "delta_phi" :  # Case for wind direction correction/deviation
            components.append(numpyro.sample(name, dist.Normal(0.0, priors.delta_phi_std))) # Sample according to Normal prior centered at zero
           
    theta = jnp.stack(components) # Stack 4 scalars to form the JAX theta vector
    predictions = pipeline.sensor_predictions(theta) # Simulate the direct physical model

    sigma = jnp.asarray(observations.noise_std) # Convert sensor noise standard deviation to JAX array
    numpyro.sample(  # Define likelihood p(y|theta) comparing simulation and observations
        "obs", 
        dist.Normal(predictions, sigma),  
        obs = jnp.asarray(observations.values), # Fix actual field observations measured by sensors
    )