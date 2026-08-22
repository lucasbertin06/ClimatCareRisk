from __future__ import annotations
 
import numpy as np
 
from climacare.config import TinyConfig
from climacare.health import HealthZone
 
__all__ = ["default_zones"]
 
 
def _gaussian_density(grid, center: tuple[float, float], sigma: float, peak: float) -> np.ndarray : # Return a (ny, nx) synthetic population-density blob
    xs = grid.centres_x()[None, :]
    ys = grid.centres_y()[:, None]
    cx, cy = center
    return peak * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))
 
 
def default_zones(config : TinyConfig) -> list[HealthZone] : # Returns: [urban, rural, vulnerable], in that order.
    
    grid = config.grid
 
    urban = HealthZone(
        name = "urban",
        density = _gaussian_density(grid, center=(0.3, 0.3), sigma=0.12, peak=800.0),
        baseline = -2.0,
        slope = 4.0,
        filter_efficiency = 0.3,
    )
    
    rural = HealthZone(
        name = "rural",
        density = _gaussian_density(grid, center=(0.75, 0.25), sigma=0.18, peak=80.0),
        baseline = -2.5,
        slope = 3.0,
        filter_efficiency = 0.1,
    )
    
    vulnerable = HealthZone(
        name = "vulnerable", # e.g. hospital / care home, section 3.1 "population vulnérable"
        density = _gaussian_density(grid, center=(0.6, 0.7), sigma=0.06, peak=250.0),
        baseline = -1.0, # higher baseline sensitivity
        slope = 6.0, # steeper response to exposure
        filter_efficiency = 0.0, # no filtration until u_filter is invested
    )
    return [urban, rural, vulnerable]
 