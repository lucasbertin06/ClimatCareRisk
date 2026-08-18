import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
from unittest.mock import MagicMock

import jax.numpy as jnp

from climacare.config import load_tiny_config
from health_model import HealthZone
from generate_scenario import generate_scenario

config = load_tiny_config()  # aucun docker requis, juste configs/tiny.yaml

zones = [
    HealthZone(name="z1", density=jnp.ones((10, 10)), baseline=0.1, slope=0.5, filter_efficiency=0.3),
    HealthZone(name="z2", density=jnp.ones((10, 10)) * 2, baseline=0.2, slope=0.4, filter_efficiency=0.1),
]

# pipeline factice : simule pipeline.direct() sans lancer de conteneur Docker
fake_pipeline = MagicMock()
fake_pipeline.direct.return_value = {
    "burned_area": jnp.array(0.3),
    "concentration_frames": jnp.ones((3, 10, 10)) * 0.01,  # (n_levels, ny, nx)
}

for n in [1, 5, 10, 20, 50]:
    t0 = time.perf_counter()
    generate_scenario(fake_pipeline, config, zones, n_scenarios=n, dt=1.0, cell_area=1.0)
    elapsed = time.perf_counter() - t0
    print(f"n_scenarios={n:3d} -> {elapsed:.4f}s ({elapsed/n*1000:.2f}ms/scenario)")