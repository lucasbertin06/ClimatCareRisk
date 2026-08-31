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

def probabilistic_model(
    pipeline: TesseractPipeline,
    observations: Observations,
    fixed_theta: jnp.ndarray | None = None,
    free_indices: tuple = (0, 1, 2, 3),
) -> None:
    """Define the masked posterior model for the physical parameters."""
    config = pipeline.config
    priors = config.priors
    low, high = config.position_bounds
    components = []

    for i, name in enumerate(PARAMETER_NAMES):
        if i not in free_indices:
            components.append(fixed_theta[i])
        elif name in ("x0", "y0"):
            components.append(numpyro.sample(name, dist.Uniform(low, high)))
        elif name == "log_amplitude":
            components.append(
                numpyro.sample(
                    name,
                    dist.Normal(priors.log_amplitude_mean, priors.log_amplitude_std),
                )
            )
        elif name == "delta_phi":
            components.append(
                numpyro.sample(
                    name,
                    dist.TruncatedNormal(
                        0.0,
                        priors.delta_phi_std,
                        low=-config.wind.delta_phi_max,
                        high=config.wind.delta_phi_max,
                    ),
                )
            )

    theta = jnp.stack(components)
    predictions = pipeline.sensor_predictions(theta)
    sigma = jnp.asarray(observations.noise_std)
    mask = jnp.asarray(observations.mask, dtype=bool)
    with numpyro.handlers.mask(mask=mask):
        numpyro.sample(
            "obs",
            dist.Normal(predictions, sigma),
            obs=jnp.asarray(observations.values),
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