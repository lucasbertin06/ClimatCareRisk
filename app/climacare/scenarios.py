from __future__ import annotations
 
import jax
import jax.numpy as jnp
import numpy as np
 
from climacare.config import PARAMETER_NAMES, TinyConfig
from climacare.health import HealthZone, incremental_health_impact, mean_exposure
from climacare.pipeline import TesseractPipeline
from climacare.uq import LaplaceResult, NutsResult, SviResult
 
__all__ = ["ScenarioBatch", "posterior_theta_samples", "simulate_scenarios"]
 
def posterior_theta_samples(
    result : LaplaceResult | SviResult | NutsResult,
    rng_key : jax.Array,
    num_samples: int = 200,) -> np.ndarray :
   
    if isinstance(result, LaplaceResult) :
        mean = result.theta_map
        draws = jax.random.multivariate_normal(rng_key, mean = jnp.asarray(mean), cov = jnp.asarray(result.covariance), shape = (num_samples,),)
        return np.asarray(draws)
 
    if isinstance(result, SviResult) :
        samples = result.posterior_samples(rng_key, num_samples = num_samples)
        return np.stack([samples[name] for name in PARAMETER_NAMES], axis = -1)
 
    if isinstance(result, NutsResult) :
        n_available = len(next(iter(result.samples.values())))
        n = min(num_samples, n_available)
        columns = []
        for name in PARAMETER_NAMES:
            if name in result.samples :
                columns.append(result.samples[name][:n])
            else:
                raise ValueError(
                    f"NutsResult is missing '{name}': pass theta_map separately "
                    "for the parameters that were held fixed, or use "
                    "Laplace/SVI for a full four-parameter scenario batch."
                )
        return np.stack(columns, axis = -1)
 
    raise TypeError(f"unsupported posterior result type: {type(result)!r}")
 
class ScenarioBatch :
    # Container matching the arrays src.loss_structure expects
 
    def __init__(self, scenarios_H_r : jax.Array, scenarios_fire : jax.Array) -> None:
        self.scenarios_H_r = scenarios_H_r  # shape (n_scenarios, n_zones)
        self.scenarios_fire = scenarios_fire  # shape (n_scenarios,)
 
 def simulate_scenarios(
    pipeline: TesseractPipeline,
    thetas: np.ndarray,
    zones: list[HealthZone],
    cell_area: float,
    filter_level: float = 0.0,) -> ScenarioBatch:
    
    config = pipeline.config
    n_scenarios = thetas.shape[0]
 
    fire_intensities = []
    zone_impacts = []
    for i in range(n_scenarios):
        theta = jnp.asarray(thetas[i], dtype=jnp.float64)
        out = pipeline.direct(theta)
 
        fire_intensities.append(out["burned_area"])
 
        h_per_zone = []
        
        for zone in zones :
            exposure = mean_exposure(
                out["concentration_frames"],
                zone,
                dt = config.dt,
                cell_area = cell_area,
                filter_level = filter_level,
            )
            h_per_zone.append(incremental_health_impact(exposure, zone, cell_area))
        zone_impacts.append(jnp.stack(h_per_zone))
 
    scenarios_fire = jnp.stack(fire_intensities)
    scenarios_H_r = jnp.stack(zone_impacts)
    return ScenarioBatch(scenarios_H_r = scenarios_H_r, scenarios_fire = scenarios_fire)