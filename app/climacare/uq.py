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
    observations: Observations,
    fixed_theta: jnp.ndarray = None,
    free_indices: tuple = (0, 1, 2, 3),
) :

    # we have : y = G(θ, u) + ε, ε ∼ N (0, Σ) 

    if bounds is None : 
        bounds = {
            "x0": (0.0, 100.0),       # Ignition x-position (km)
            "y0": (0.0, 100.0),       # Ignition y-position (km)
            "wind_bias": (-2.0, 2.0), # Wind speed/direction bias
            "intensity": (0.5, 5.0)   # Fire intensity multiplier
        }

    x0 = numpyro.sample("x0", dist.Uniform(bounds["x0"][0], bounds["x0"][1])) # this is for the before we take the captors datas
    y0 = numpyro.sample("y0", dist.Uniform(bounds["y0"][0], bounds["y0"][1]))
    wind_bias = numpyro.sample("wind_bias", dist.Uniform(bounds["wind_bias"][0], bounds["wind_bias"][1]))
    intensity = numpyro.sample("intensity", dist.Uniform(bounds["intensity"][0], bounds["intensity"][1]))

    theta = { # Parameter bundle of θ
        "x0": x0,
        "y0": y0,
        "wind_bias": wind_bias,
        "intensity": intensity
    }