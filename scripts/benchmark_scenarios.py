from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "components" / "shared_code"))

import jax.numpy as jnp

from climacare.config import load_tiny_config
from climacare.health import HealthZone
from climacare.scenario_generation import generate_scenario

config = load_tiny_config()
zones = [
    HealthZone("z1", jnp.ones((10, 10)), 0.1, 0.5, 0.3),
    HealthZone("z2", jnp.ones((10, 10)) * 2, 0.2, 0.4, 0.1),
]
fake_pipeline = MagicMock()
fake_pipeline.direct.return_value = {
    "burned_area": jnp.array(0.3),
    "concentration_frames": jnp.ones((3, 10, 10)) * 0.01,
}

for count in [1, 5, 10, 20, 50]:
    started = time.perf_counter()
    generate_scenario(
        fake_pipeline,
        config,
        zones,
        n_scenarios=count,
        dt=1.0,
        cell_area=1.0,
        frame_count=3,
    )
    elapsed = time.perf_counter() - started
    print(
        f"n_scenarios={count:3d} -> {elapsed:.4f}s "
        f"({elapsed / count * 1000:.2f}ms/scenario)"
    )
