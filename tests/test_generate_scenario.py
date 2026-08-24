import time
import pytest
import jax.numpy as jnp

# 1. Correction de l'import (si le fichier generate_scenarios.py est à la racine)
from generate_scenario import generate_scenario


def test_generate_scenario_performance(pipeline, tiny_config):
    # 1. Définition des paramètres de test
    n_scenarios = 3
    max_seconds_allowed = 10.0
    
    # Création de zones fictives si aucune fixture zones n'existe
    zones = jnp.zeros((10, 10))

    # 2. Mesure du temps d'exécution
    start_time = time.perf_counter()

    scenarios_fire, scenarios_H_r = generate_scenario(
        pipeline,
        tiny_config,
        zones,
        n_scenarios=n_scenarios,
        dt=1.0,
        cell_area=100.0,
    )

    # Force la fin des calculs asynchrones JAX
    scenarios_fire.block_until_ready()
    scenarios_H_r.block_until_ready()

    duration = time.perf_counter() - start_time
    print(f"\n[BENCHMARK] Temps pour {n_scenarios} scénarios : {duration:.3f}s")

    # 3. Assertions
    assert scenarios_fire.shape[0] == n_scenarios
    assert duration < max_seconds_allowed, f"Exécution trop lente : {duration:.2f}s (max: {max_seconds_allowed}s)"