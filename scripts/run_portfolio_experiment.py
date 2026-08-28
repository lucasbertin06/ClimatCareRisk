# Experiment E5 : pipeline -> posterior -> scenarios -> portfolio optimization
#
# Two modes :
#   python scripts/run_portfolio_experiment.py --fake   # synthetic scenarios, no Docker needed
#   python scripts/run_portfolio_experiment.py          # full chain, needs the two Tesseract images (make build-c0)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))  # repo root : needed for "from src.generate_scenario import ..."

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np


def parse_args() :
    p = argparse.ArgumentParser(description = "E5/E6 : portfolio optimization experiment")
    p.add_argument("--n", type = int, default = 20) # number of scenarios
    p.add_argument("--steps", type = int, default = 300) # optimizer iterations
    p.add_argument("--lambda-cvar", type = float, default = 0.5)
    p.add_argument("--num-samples", type = int, default = 20) # posterior draws (real mode)
    p.add_argument("--spread", type = float, default = 1.0,
                   help = "covariance inflation factor for posterior draws; "
                          "values > 1 widen the sample spread and make the "
                          "EL/CVaR frontier non-degenerate (spec section 10.3)")
    p.add_argument("--seed", type = int, default = 0)
    p.add_argument("--fake", action = "store_true") # synthetic scenarios, no Docker needed
    return p.parse_args()


def fake_scenarios(n) : # same shapes as tests/test_loss_structure.py
    H_r = jnp.ones((n, 3)) * 2.0 # constant health impact of 2.0 per zone
    fire = jnp.linspace(0.1, 0.9, n) # fire intensity from 0.1 to 0.9
    return H_r, fire


def real_scenarios(args) :
    # full chain : MAP -> Laplace -> posterior samples -> generate_scenario(theta_samples=...)
    from climacare.config import load_tiny_config
    from climacare.pipeline import open_pipeline
    from climacare.inverse import run_map
    from climacare.objective import make_observations
    from climacare.uq import laplace_approx
    from climacare.zones import default_zones
    from src.generate_scenario import generate_scenario

    config = load_tiny_config()
    zones = default_zones(config)
    dt = config.dt
    cell_area = config.grid.cell_area if hasattr(config.grid, "cell_area") else 1.0

    with open_pipeline(config) as pipeline :
        # observations at the ground truth (synthetic case, spec section 5.5)
        clean = np.asarray(pipeline.sensor_predictions(jnp.asarray(config.truth)))
        observations = make_observations(config, clean)

        print("[chain] running MAP...")
        map_result = run_map(pipeline, observations)
        print("        theta_MAP =", np.round(map_result.estimate, 4))

        print("[chain] Laplace approximation...")
        laplace = laplace_approx(pipeline, observations, map_result.estimate)

        rng_key = jax.random.PRNGKey(args.seed)
        cov = jnp.asarray(laplace["covariance"])
        if args.spread != 1.0 :
            cov = cov * float(args.spread) ** 2
            print(f"[chain] posterior spread inflated x{args.spread}")
        mean = jnp.asarray(laplace["theta_map"])
        thetas = np.asarray(jax.random.multivariate_normal(rng_key, mean, cov,
            shape = (args.num_samples,),))
        # note : we draw from the Laplace dict directly ; scenarios.posterior_theta_samples
        # is the equivalent helper for reuse outside this script

        print(f"[chain] generating {len(thetas)} scenarios through the pipeline...")
        return generate_scenario(pipeline, config, zones, len(thetas),
            dt = dt, cell_area = cell_area, theta_samples = thetas)


def main() :
    args = parse_args()

    if args.fake :
        scenarios_H_r, scenarios_fire = fake_scenarios(args.n)
        print("[scenarios] using synthetic data (--fake)")
    else :
        scenarios_fire, scenarios_H_r = real_scenarios(args)  # generate_scenario returns (fire, H_r)
        print("[scenarios] generated from the Tesseract pipeline")

    basis_noises = jnp.zeros_like(scenarios_fire)

    from loss_structure import (
        compare_policies,
        evaluate_smart_criterion,
        generate_efficient_frontier,
        optimize_portfolio,
        run_stress_tests,
    )

    u_opt, history = optimize_portfolio(scenarios_H_r, scenarios_fire, basis_noises,
        lambda_cvar = args.lambda_cvar, steps = args.steps, return_history = True)
    print(f"\noptimize_portfolio : J {float(history[0]):.0f} -> {float(history[-1]):.0f}"
          f"  (-{100 * (1 - float(history[-1]) / max(float(history[0]), 1e-12)):.1f}%)")
    print("u* =", [round(float(x), 3) for x in u_opt])

    comparison = compare_policies(scenarios_H_r, scenarios_fire, basis_noises,
        lambda_cvar = args.lambda_cvar, steps = args.steps)
    for name, m in comparison.items() :
        print(f"{name:15s} EL={m['EL']:>10.0f}  VaR={m['VaR']:>10.0f}  CVaR={m['CVaR']:>10.0f}  capex={m['capex']:.0f}")

    smart = evaluate_smart_criterion(comparison["optimized"]["u"], scenarios_H_r, scenarios_fire, basis_noises)
    print("\nSMART target met :", bool(smart["smart_target_met"]),
          f"(vs uniform -{100*float(smart['reduction_vs_uniform']):.1f}%"
          f", vs insurance-only -{100*float(smart['reduction_vs_insurance']):.1f}%)")

    stress = run_stress_tests(comparison["optimized"]["u"], scenarios_H_r, scenarios_fire, basis_noises)
    for kind, v in stress.items() :
        print(f"stress {kind:14s} EL={float(v['EL']):>10.0f}  CVaR={float(v['CVaR']):>10.0f}")

    portfolios, els, ess, capex, budgets = generate_efficient_frontier(scenarios_H_r, scenarios_fire, 8, basis_noises)

    print("\nbudget frontier :")
    for i in range(len(budgets)) :
        u_list = [round(float(x), 3) for x in portfolios[i]]
        budget_ok = float(capex[i]) <= float(budgets[i]) * 1.01
        print(f"budget={float(budgets[i]):.0f}  capex={float(capex[i]):.0f}  EL={float(els[i]):.0f}  CVaR={float(ess[i]):.0f}  budget_ok={budget_ok}  u={u_list}")

    results = {
        "mode": "fake" if args.fake else "pipeline",
        "config": {"n": args.n, "steps": args.steps, "lambda_cvar": args.lambda_cvar, "seed": args.seed},
        "u_opt": [float(x) for x in u_opt],
        "J_init": float(history[0]),
        "J_final": float(history[-1]),
        "policies": {k: {"EL": m["EL"], "VaR": m["VaR"], "CVaR": m["CVaR"], "capex": m["capex"]}
                     for k, m in comparison.items()},
        "smart_target_met": bool(smart["smart_target_met"]),
        "stress": {
            k: {
                key: ([float(x) for x in value] if key == "u_reoptimized" else bool(value) if key == "budget_respected" else float(value))
                for key, value in v.items()
            }
            for k, v in stress.items()
        },
        "budget_frontier" : {
            "budgets": [float(x) for x in budgets],
            "capex": [float(x) for x in capex],
            "EL": [float(x) for x in els],
            "CVaR": [float(x) for x in ess],
            "portfolios": [[float(x) for x in u] for u in portfolios],
        },
    }
    out = ROOT / "results"
    out.mkdir(exist_ok = True)
    path = out / f"portfolio_e5_{'fake' if args.fake else 'pipeline'}.json"
    path.write_text(json.dumps(results, indent = 2))
    print("\nsaved :", path)


if __name__ == "__main__" :
    main()
