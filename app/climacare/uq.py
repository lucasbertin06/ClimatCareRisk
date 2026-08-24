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
    pipeline : TesseractPipeline,
    observations : Observations, # Structure containing field data collected from sensors
    fixed_theta : jnp.ndarray = None, 
    free_indices : tuple = (0, 1, 2, 3), 
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

def laplace_approx(
    pipeline : TesseractPipeline, 
    observations : Observations, 
    theta_map : jnp.ndarray, # Optimal point previously found by MAP algorithm (theta_MAP)
    jitter : float = 1e-8, # Regularization term to prevent division by zero
) -> dict : 
    
    loss_fn = make_physical_loss(pipeline, observations) # Create deterministic loss function L(theta)

    # Finite-difference Hessian instead of jax.hessian : the smoke Tesseract
    # only implements the VJP endpoint (no JVP), so second-order autodiff
    # through it is not supported. Central differences on the exact gradient
    # give the same matrix up to O(h^2), at the cost of 2n extra gradient
    # evaluations (8 for four parameters).
    theta = jnp.asarray(theta_map, dtype = jnp.float64) # Ensure theta_MAP is in float64 JAX format
    grad_fn = jax.grad(loss_fn)
    n = theta.shape[0]
    h = 1e-4
    H = jnp.zeros((n, n), dtype = jnp.float64)
    for i in range(n) :
        e_i = jnp.zeros(n).at[i].set(h)
        g_plus = grad_fn(theta + e_i)
        g_minus = grad_fn(theta - e_i)
        H = H.at[:, i].set((g_plus - g_minus) / (2.0 * h)) # column i of the Hessian
    H = 0.5 * (H + H.T) # symmetrize to remove finite-difference asymmetry

    H_stable = H + jitter * jnp.eye(H.shape[0]) # Add small identity matrix (H + jitter * I)
    covariance = jnp.linalg.inv(H_stable) # Invert Hessian to get covariance (Sigma = H^-1)
    std_dev = jnp.sqrt(jnp.diag(covariance)) # Extract square root of diagonal to get standard deviations

    return { 
        "theta_map": theta_map, # Estimated MAP values
        "covariance": covariance, # Estimated 4x4 covariance matrix
        "std_dev": std_dev, # Standard deviation (uncertainty) for each parameter
    }

def run_nuts(
    pipeline: TesseractPipeline,
    observations: Observations, 
    theta_map: jnp.ndarray,  
    rng_key: jax.Array,  # JAX random key for MCMC sampling
    free_indices: tuple = (0, 1),  # Default: sample only x0 (index 0) and y0 (index 1)
    num_warmup: int = 200, # Number of warmup/burn-in steps to adapt simulation step size
    num_samples: int = 400, # Number of retained samples for posterior distribution
) -> dict :  # Returns a dictionary containing MCMC chain draws
    def model() : 
        probabilistic_model( # Call probabilistic model while restricting parameters
            pipeline, # Pass physical pipeline
            observations, # Pass observations
            fixed_theta = theta_map, # Fix non-sampled parameters to their MAP values
            free_indices = free_indices, # Specify which parameters remain random variables
        )

    kernel = NUTS(model) # Initialize No-U-Turn Sampler MCMC algorithm
    mcmc = MCMC( # Configure MCMC execution engine
        kernel, num_warmup = num_warmup, num_samples = num_samples, progress_bar = False # Execution parameters
    )
    mcmc.run(rng_key) # Run MCMC simulation chain with JAX random key

    return mcmc.get_samples() # Return dictionary containing samples drawn from posterior