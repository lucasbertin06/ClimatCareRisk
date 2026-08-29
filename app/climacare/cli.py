"""Command-line entry point for the C0 slice.

Usage :
    python -m climacare.cli direct    --output-dir results/tiny_direct
    python -m climacare.cli map       --output-dir results/tiny_map
    python -m climacare.cli uq        --output-dir results/tiny_uq [--nuts]
    python -m climacare.cli portfolio --output-dir results/portfolio [--fake]

Runs one direct pipeline evaluation at the ground truth, checks numerical
stability, computes the downstream physical loss and writes a complete,
reproducible JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from climacare.config import load_tiny_config
from climacare.finance import FinanceParams, physical_loss
from climacare.pipeline import open_pipeline, pipeline_versions


def _git_commit() -> str :
    # short hash of the current repository state ("unknown" outside a git repo)
    try :
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output = True, text = True, check = True,
            cwd = Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception :
        return "unknown"


def run_direct(output_dir : Path) -> None :
    config = load_tiny_config()

    with open_pipeline(config) as pipeline :
        theta_truth = jnp.asarray(config.truth)
        out = pipeline.direct(theta_truth)

        # health impact of the truth scenario across the three zones
        from climacare.health import mean_exposure, incremental_health_impact
        from climacare.zones import default_zones

        zones = default_zones(config)
        dt = config.dt
        cell_area = 1.0
        impacts = []
        for zone in zones :
            exposure = mean_exposure(out["concentration_frames"], zone,
                                     dt = dt, cell_area = cell_area)
            impacts.append(float(incremental_health_impact(exposure, zone, cell_area)))

        loss = physical_loss(jnp.asarray(impacts), float(out["burned_area"]), 0.0, FinanceParams())

    payload = {
        "command" : "tiny-direct",
        "configuration" : {
            "case" : config.case,
            "seed" : config.seed,
            "grid" : {"nx" : config.grid.nx, "ny" : config.grid.ny},
            "dt" : dt,
            "n_steps" : config.n_steps,
            "nu_fire_realised" : float(config.fire.reaction_rate * 0.0 + getattr(config, "nu_fire", 0.0)),
            "nu_smoke_realised" : float(getattr(config, "nu_smoke", 0.0)),
        },
        "versions" : pipeline_versions(),
        "git_commit" : _git_commit(),
        "truth_theta" : [float(x) for x in np.asarray(theta_truth)],
        "diagnostics" : {
            "burned_area" : float(out["burned_area"]),
            "smoke_cfl_number" : float(out["smoke_cfl_number"]),
            "health_impacts_per_zone" : impacts,
        },
        "stability" : {
            # CFL numbers must stay below one for the explicit schemes to be stable
            "nu_smoke_realised" : float(out["smoke_cfl_number"]),
        },
        "downstream" : {
            "loss_physical" : float(loss),
        },
    }

    output_dir.mkdir(parents = True, exist_ok = True)
    artifact = output_dir / "tiny_direct.json"
    artifact.write_text(json.dumps(payload, indent = 2))
    print(f"tiny_direct.json written to {artifact}")


def _observations_at_truth(pipeline, config) :
    # synthetic dataset of spec section 5.5, generated once at the ground truth
    from climacare.objective import make_observations

    clean = np.asarray(pipeline.sensor_predictions(jnp.asarray(config.truth)))
    return make_observations(config, clean)


def run_map_command(output_dir : Path, iterations : int | None, optimizer : str | None) -> None :
    # E2 (gradient check) + E3 (MAP) : reconstruct theta from noisy sensors
    from climacare.inverse import gradient_check, run_map

    config = load_tiny_config()
    with open_pipeline(config) as pipeline :
        observations = _observations_at_truth(pipeline, config)

        print("[map] gradient check...")
        check = gradient_check(pipeline, observations)
        print(f"       median relative error = {check.median_relative_error:.2e}")

        print("[map] optimising...")
        result = run_map(pipeline, observations, iterations = iterations, optimizer = optimizer)
        print(f"       loss {result.loss_history[0]:.4f} -> {result.loss_history[-1]:.4f}"
              f"  position error {result.initial_position_error:.3f} -> {result.final_position_error:.3f}")

    payload = {
        "command" : "map",
        "versions" : pipeline_versions(),
        "git_commit" : _git_commit(),
        "gradient_check" : check.summary(),
        "map" : result.summary(),
    }
    output_dir.mkdir(parents = True, exist_ok = True)
    artifact = output_dir / "map.json"
    artifact.write_text(json.dumps(payload, indent = 2))
    print(f"map.json written to {artifact}")


def run_uq_command(output_dir : Path, nuts : bool, num_warmup : int, num_samples : int) -> None :
    # E3/E4 : MAP -> Laplace approximation, optionally refined with NUTS
    from climacare.inverse import run_map
    from climacare.uq import laplace_approx, run_nuts

    config = load_tiny_config()
    with open_pipeline(config) as pipeline :
        observations = _observations_at_truth(pipeline, config)

        print("[uq] running MAP...")
        map_result = run_map(pipeline, observations)

        print("[uq] Laplace approximation...")
        laplace = laplace_approx(pipeline, observations, map_result.estimate)

        payload = {
            "command" : "uq",
            "versions" : pipeline_versions(),
            "git_commit" : _git_commit(),
            "map_estimate" : [float(x) for x in map_result.estimate],
            "laplace" : {
                "theta_map" : [float(x) for x in np.asarray(laplace["theta_map"])],
                "std_dev" : [float(x) for x in np.asarray(laplace["std_dev"])],
                "covariance" : [[float(x) for x in row] for row in np.asarray(laplace["covariance"])],
            },
        }

        if nuts :
            print(f"[uq] NUTS ({num_warmup} warmup, {num_samples} samples)...")
            rng_key = jax.random.PRNGKey(0)
            samples = run_nuts(pipeline, observations, jnp.asarray(map_result.estimate), rng_key,
                num_warmup = num_warmup, num_samples = num_samples)
            payload["nuts"] = {
                name : [float(x) for x in np.asarray(values)]
                for name, values in samples.items()
            }

    output_dir.mkdir(parents = True, exist_ok = True)
    artifact = output_dir / "uq.json"
    artifact.write_text(json.dumps(payload, indent = 2))
    print(f"uq.json written to {artifact}")


def run_portfolio_command(output_dir : Path, fake : bool, n : int, steps : int,
                           lambda_cvar : float, num_samples : int, seed : int) -> None :
    # E5/E6 : posterior scenarios -> robust portfolio optimisation -> stress tests
    from loss_structure import (
        compare_policies,
        evaluate_smart_criterion,
        generate_efficient_frontier,
        optimize_portfolio,
        run_stress_tests,
    )

    if fake :
        scenarios_H_r = jnp.ones((n, 3)) * 2.0
        scenarios_fire = jnp.linspace(0.1, 0.9, n)
        print("[portfolio] using synthetic data (--fake)")
    else :
        from climacare.inverse import run_map
        from climacare.uq import laplace_approx
        from climacare.zones import default_zones
        from generate_scenario import generate_scenario

        config = load_tiny_config()
        zones = default_zones(config)
        dt = config.dt
        cell_area = config.grid.cell_area if hasattr(config.grid, "cell_area") else 1.0

        with open_pipeline(config) as pipeline :
            observations = _observations_at_truth(pipeline, config)
            print("[portfolio] running MAP...")
            map_result = run_map(pipeline, observations)
            print("[portfolio] Laplace approximation...")
            laplace = laplace_approx(pipeline, observations, map_result.estimate)

            rng_key = jax.random.PRNGKey(seed)
            thetas = np.asarray(jax.random.multivariate_normal(
                rng_key, jnp.asarray(laplace["theta_map"]), jnp.asarray(laplace["covariance"]),
                shape = (num_samples,),
            ))
            print(f"[portfolio] generating {len(thetas)} scenarios through the pipeline...")
            scenarios_fire, scenarios_H_r = generate_scenario(
                pipeline, config, zones, len(thetas), dt = dt, cell_area = cell_area,
                theta_samples = thetas,
            )
        print("[portfolio] generated from the Tesseract pipeline")

    basis_noises = jnp.zeros_like(scenarios_fire)

    u_opt, history = optimize_portfolio(scenarios_H_r, scenarios_fire, basis_noises,
        lambda_cvar = lambda_cvar, steps = steps, return_history = True)
    print(f"[portfolio] J {float(history[0]):.0f} -> {float(history[-1]):.0f}")

    comparison = compare_policies(scenarios_H_r, scenarios_fire, basis_noises,
        lambda_cvar = lambda_cvar, steps = steps)
    smart = evaluate_smart_criterion(comparison["optimized"]["u"], scenarios_H_r, scenarios_fire, basis_noises)
    stress = run_stress_tests(comparison["optimized"]["u"], scenarios_H_r, scenarios_fire, basis_noises)
    portfolios, els, ess, capex, budgets = generate_efficient_frontier(
        scenarios_H_r, scenarios_fire, 8, basis_noises)

    def _stress_to_json(block) :
        return {
            k : {
                key : ([float(x) for x in value] if key == "u_reoptimized"
                       else bool(value) if key == "budget_respected"
                       else float(value))
                for key, value in v.items()
            }
            for k, v in block.items()
        }

    payload = {
        "command" : "portfolio",
        "mode" : "fake" if fake else "pipeline",
        "config" : {"n" : n, "steps" : steps, "lambda_cvar" : lambda_cvar, "seed" : seed},
        "u_opt" : [float(x) for x in u_opt],
        "J_init" : float(history[0]),
        "J_final" : float(history[-1]),
        "policies" : {k : {"EL" : m["EL"], "VaR" : m["VaR"], "CVaR" : m["CVaR"], "capex" : m["capex"]}
                      for k, m in comparison.items()},
        "smart_target_met" : bool(smart["smart_target_met"]),
        "stress" : _stress_to_json(stress),
        "budget_frontier" : {
            "budgets" : [float(x) for x in budgets],
            "capex" : [float(x) for x in capex],
            "EL" : [float(x) for x in els],
            "CVaR" : [float(x) for x in ess],
            "portfolios" : [[float(x) for x in u] for u in portfolios],
        },
    }

    output_dir.mkdir(parents = True, exist_ok = True)
    artifact = output_dir / f"portfolio_{'fake' if fake else 'pipeline'}.json"
    artifact.write_text(json.dumps(payload, indent = 2))
    print(f"portfolio json written to {artifact}")


def main() -> None :
    parser = argparse.ArgumentParser(prog = "climacare.cli")
    sub = parser.add_subparsers(dest = "command", required = True)

    direct = sub.add_parser("direct", help = "run one direct evaluation at the truth (E1)")
    direct.add_argument("--output-dir", type = Path, required = True)

    map_parser = sub.add_parser("map", help = "gradient check + MAP inverse problem (E2/E3)")
    map_parser.add_argument("--output-dir", type = Path, required = True)
    map_parser.add_argument("--iterations", type = int, default = None)
    map_parser.add_argument("--optimizer", choices = ["lbfgs", "adam"], default = None)

    uq_parser = sub.add_parser("uq", help = "Laplace posterior, optionally refined with NUTS (E4)")
    uq_parser.add_argument("--output-dir", type = Path, required = True)
    uq_parser.add_argument("--nuts", action = "store_true")
    uq_parser.add_argument("--num-warmup", type = int, default = 200)
    uq_parser.add_argument("--num-samples", type = int, default = 400)

    portfolio_parser = sub.add_parser("portfolio", help = "robust portfolio + stress tests (E5/E6)")
    portfolio_parser.add_argument("--output-dir", type = Path, required = True)
    portfolio_parser.add_argument("--fake", action = "store_true",
        help = "synthetic scenarios, no Docker images needed")
    portfolio_parser.add_argument("--n", type = int, default = 20)
    portfolio_parser.add_argument("--steps", type = int, default = 300)
    portfolio_parser.add_argument("--lambda-cvar", type = float, default = 0.5)
    portfolio_parser.add_argument("--num-samples", type = int, default = 20)
    portfolio_parser.add_argument("--seed", type = int, default = 0)

    args = parser.parse_args()

    if args.command == "direct" :
        run_direct(args.output_dir)
    elif args.command == "map" :
        run_map_command(args.output_dir, args.iterations, args.optimizer)
    elif args.command == "uq" :
        run_uq_command(args.output_dir, args.nuts, args.num_warmup, args.num_samples)
    elif args.command == "portfolio" :
        run_portfolio_command(args.output_dir, args.fake, args.n, args.steps,
            args.lambda_cvar, args.num_samples, args.seed)


if __name__ == "__main__" :
    main()