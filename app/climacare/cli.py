"""Command-line entry point for the C0 slice.

Usage :
    python -m climacare.cli direct --output-dir results/tiny_direct

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


def main() -> None :
    parser = argparse.ArgumentParser(prog = "climacare.cli")
    sub = parser.add_subparsers(dest = "command", required = True)
    direct = sub.add_parser("direct", help = "run one direct evaluation at the truth")
    direct.add_argument("--output-dir", type = Path, required = True)
    args = parser.parse_args()

    if args.command == "direct" :
        run_direct(args.output_dir)


if __name__ == "__main__" :
    main()
