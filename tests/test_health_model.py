import jax.numpy as jnp

from health_model import HealthZone, health_impact
from climacare.health import mean_exposure, incremental_health_impact

def _zones():
    return [
        HealthZone(name="z1", density=jnp.ones((10, 10)), baseline=0.1, slope=0.5, filter_efficiency=0.3),
        HealthZone(name="z2", density=jnp.ones((10, 10)) * 2, baseline=0.2, slope=0.4, filter_efficiency=0.1),
    ]

def test_zero_concentration_gives_zero_impact():
    concentration = jnp.zeros((5, 3, 10, 10))
    H_r = health_impact(concentration, _zones(), dt=1.0, cell_area=1.0)
    assert jnp.allclose(H_r, 0.0, atol=1e-6)

def test_output_shape():
    concentration = jnp.ones((5, 3, 10, 10)) * 0.01
    H_r = health_impact(concentration, _zones(), dt=1.0, cell_area=1.0)
    assert H_r.shape == (5, 2)  # (n_scenarios, n_zones)

def test_more_filtration_reduces_impact():
    concentration = jnp.ones((1, 3, 10, 10)) * 0.05
    H_low = health_impact(concentration, _zones(), dt=1.0, cell_area=1.0, filter_level=0.0)
    H_high = health_impact(concentration, _zones(), dt=1.0, cell_area=1.0, filter_level=1.0)
    assert jnp.all(H_high <= H_low)

def test_matches_reference_implementation():
    zones = _zones()
    concentration = jnp.ones((1, 3, 10, 10)) * 0.03

    H_r_vectorized = health_impact(concentration, zones, dt=1.0, cell_area=1.0)

    H_r_reference = jnp.stack([
        incremental_health_impact(
            mean_exposure(concentration[0], zone, dt=1.0, cell_area=1.0), zone, 1.0
        )
        for zone in zones
    ])

    assert jnp.allclose(H_r_vectorized[0], H_r_reference, atol=1e-4)    