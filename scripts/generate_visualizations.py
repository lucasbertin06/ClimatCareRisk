"""Generate the README visual assets from the reproducible Tiny artifacts.

The animated fields are recomputed with the local PyTorch fire solver and the
compiled C++ smoke kernel. The outcome charts read the committed JSON artifacts.
No Docker service is required.
"""

from __future__ import annotations

import argparse
import math
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "components" / "shared_code"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from climacare.config import TinyConfig, load_tiny_config
from climacare_shared.fire_model import fire_forward, frame_indices
from climacare_shared.kernel import KernelNotBuiltError, load_smoke_kernel

DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets"
PORTFOLIO_RESULT = ROOT / "results" / "portfolio_e5_fake.json"
PREVENTION_RESULT = ROOT / "results" / "prevention" / "prevention.json"

INK = "#0B1820"
PANEL = "#102630"
GRID = "#2B4650"
TEXT = "#F3F7F5"
MUTED = "#9EB4B8"
FIRE = "#FF6846"
EMBER = "#FFC857"
SMOKE = "#6FD0C7"
SKY = "#7FA9FF"

FIRE_MAP = LinearSegmentedColormap.from_list(
    "climacare_fire", [INK, "#352231", "#9B3436", FIRE, EMBER, "#FFF2BE"]
)
SMOKE_MAP = LinearSegmentedColormap.from_list(
    "climacare_smoke", [INK, "#12333B", "#1E6E73", SMOKE, "#D5F5E9"]
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic README figures from the Tiny case."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"asset directory (default: {DEFAULT_OUTPUT_DIR.relative_to(ROOT)})",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fire_inputs(config: TinyConfig) -> dict[str, Any]:
    theta = config.truth
    fire = config.fire
    return {
        "ignition": theta[:3],
        "wind": np.asarray(config.fire_wind(float(theta[3]))),
        "moisture": config.moisture,
        "fuel_base": config.fuel_base,
        "fuel_prevention": config.fuel_prevention,
        "dt": config.dt,
        "n_steps": config.n_steps,
        "diffusivity": fire.diffusivity,
        "heat_loss": fire.heat_loss,
        "heat_release": fire.heat_release,
        "reaction_rate": fire.reaction_rate,
        "moisture_sensitivity": fire.moisture_sensitivity,
        "ignition_threshold": fire.ignition_threshold,
        "ignition_width": fire.ignition_width,
        "source_sigma": fire.source_sigma,
        "smoke_yield": fire.smoke_yield,
        "wind_speed_bound": config.wind.fire_speed,
        "frame_count": config.frame_count,
    }




def _simulate_fields(config: TinyConfig) -> tuple[np.ndarray, np.ndarray]:
    fire_output = fire_forward(_fire_inputs(config))
    source = np.ascontiguousarray(
        fire_output["smoke_source"].detach().cpu().numpy(), dtype=np.float64
    )
    fire_frames = fire_output["intensity_frames"].detach().cpu().numpy()

    try:
        kernel = load_smoke_kernel()
    except KernelNotBuiltError as error:
        raise SystemExit(
            "The local smoke kernel is required. Run `make smoke-kernel`, then "
            "rerun this command."
        ) from error

    wind_x, wind_y = config.smoke_wind(float(config.truth[3]))
    _, smoke_frames = kernel.forward(
        source,
        np.ascontiguousarray(config.sensors.positions),
        np.ascontiguousarray(config.sensors.bias),
        wind_x,
        wind_y,
        config.smoke.diffusivity,
        config.smoke.decay,
        config.dt,
        np.asarray(frame_indices(config.n_steps, config.frame_count), dtype=np.int32),
    )
    return np.asarray(fire_frames), np.asarray(smoke_frames)


def _health_payload(
    config: TinyConfig, smoke_frames: np.ndarray
) -> dict[str, Any]:
    """Compute health diagnostics from the current smoke simulation."""
    frame_dt = config.final_time / (len(smoke_frames) - 1)
    impacts = []
    zone_specs = (
        ((0.3, 0.3), 0.12, 800.0, -2.0, 4.0),
        ((0.75, 0.25), 0.18, 80.0, -2.5, 3.0),
        ((0.6, 0.7), 0.06, 250.0, -1.0, 6.0),
    )
    mesh_x, mesh_y = config.grid.meshgrid()
    for (center_x, center_y), sigma, peak, baseline, slope in zone_specs:
        density = peak * np.exp(
            -((mesh_x - center_x) ** 2 + (mesh_y - center_y) ** 2)
            / (2.0 * sigma**2)
        )
        population = config.grid.cell_area * density.sum()
        exposure = (
            frame_dt
            * config.grid.cell_area
            * np.sum(smoke_frames * density[None, :, :])
            / population
        )
        impact = population * (
            np.logaddexp(0.0, baseline + slope * exposure)
            - np.logaddexp(0.0, baseline)
        )
        impacts.append(float(impact))
    return {"diagnostics": {"health_impacts_per_zone": impacts}}

def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.tick_params(colors=MUTED, labelsize=8, length=0, pad=5)
    axis.grid(color=GRID, linewidth=0.55, alpha=0.45)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def _add_wind(axis: plt.Axes, config: TinyConfig) -> None:
    angle = config.wind.phi_base + float(config.truth[3])
    dx = 0.13 * math.cos(angle)
    dy = 0.13 * math.sin(angle)
    axis.annotate(
        "",
        xy=(0.12 + dx, 0.88 + dy),
        xytext=(0.12, 0.88),
        arrowprops={"arrowstyle": "-|>", "color": TEXT, "lw": 1.5},
    )
    axis.text(0.08, 0.93, "WIND", color=MUTED, fontsize=7, weight="bold")


def _add_sensor_network(axis: plt.Axes, config: TinyConfig) -> None:
    positions = config.sensors.positions
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=42,
        facecolor=INK,
        edgecolor=TEXT,
        linewidth=1.2,
        marker="D",
        zorder=5,
    )
    for index, (x_pos, y_pos) in enumerate(positions, start=1):
        axis.text(
            x_pos + 0.025,
            y_pos + 0.022,
            f"S{index}",
            color=TEXT,
            fontsize=7,
            weight="bold",
            zorder=6,
        )


def render_field_animation(
    config: TinyConfig,
    fire_frames: np.ndarray,
    smoke_frames: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 5.7), facecolor=INK)
    figure.subplots_adjust(left=0.055, right=0.965, bottom=0.12, top=0.82, wspace=0.13)
    figure.text(
        0.055,
        0.94,
        "ONE IGNITION, TWO COUPLED FIELDS",
        color=TEXT,
        fontsize=19,
        weight="bold",
    )
    figure.text(
        0.055,
        0.885,
        "Differentiable fire spread feeds an adjoint smoke-transport solver",
        color=MUTED,
        fontsize=10.5,
    )
    time_text = figure.text(
        0.945,
        0.925,
        "",
        color=EMBER,
        fontsize=11,
        weight="bold",
        ha="right",
    )

    fire_peak = max(float(np.max(fire_frames)), 1e-12)
    smoke_peak = max(float(np.max(smoke_frames)), 1e-12)
    fire_image = axes[0].imshow(
        fire_frames[0],
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=FIRE_MAP,
        vmin=0,
        vmax=fire_peak,
        interpolation="bilinear",
    )
    smoke_image = axes[1].imshow(
        smoke_frames[0],
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=SMOKE_MAP,
        vmin=0,
        vmax=smoke_peak,
        interpolation="bilinear",
    )

    for axis, title, label in zip(
        axes,
        ("FIRE INTENSITY", "SMOKE CONCENTRATION"),
        ("PYTORCH / AUTODIFF", "C++20 / DISCRETE ADJOINT"),
        strict=True,
    ):
        _style_axis(axis)
        axis.set_title(title, loc="left", color=TEXT, fontsize=11, weight="bold", pad=13)
        axis.text(
            1.0,
            1.035,
            label,
            transform=axis.transAxes,
            ha="right",
            color=MUTED,
            fontsize=7,
            weight="bold",
        )
        _add_wind(axis, config)

    axes[0].scatter(
        [config.truth[0]],
        [config.truth[1]],
        s=70,
        facecolor=EMBER,
        edgecolor=INK,
        linewidth=1.2,
        marker="*",
        zorder=6,
    )
    axes[0].text(
        config.truth[0] + 0.025,
        config.truth[1] - 0.055,
        "IGNITION",
        color=TEXT,
        fontsize=7,
        weight="bold",
        zorder=6,
    )
    _add_sensor_network(axes[1], config)

    progress_axis = figure.add_axes((0.055, 0.045, 0.89, 0.018), facecolor=GRID)
    progress_axis.set_xlim(0, len(fire_frames) - 1)
    progress_axis.set_ylim(0, 1)
    progress_axis.set_xticks([])
    progress_axis.set_yticks([])
    for spine in progress_axis.spines.values():
        spine.set_visible(False)
    progress = progress_axis.barh(0.5, 0, height=1.0, color=EMBER, align="center")[0]

    def update(frame_index: int) -> None:
        fire_image.set_data(fire_frames[frame_index])
        smoke_image.set_data(smoke_frames[frame_index])
        simulation_time = config.final_time * frame_index / (len(fire_frames) - 1)
        time_text.set_text(f"t = {simulation_time:0.2f} / {config.final_time:0.2f}")
        progress.set_width(frame_index)
    animation = FuncAnimation(
        figure,
        update,
        frames=len(fire_frames),
        blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=6), dpi=115)
    plt.close(figure)


def _setup_chart_figure(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=(10.4, 4.8), facecolor=INK)
    figure.subplots_adjust(left=0.09, right=0.96, bottom=0.19, top=0.72)
    figure.text(0.065, 0.91, title, color=TEXT, fontsize=19, weight="bold")
    figure.text(0.065, 0.84, subtitle, color=MUTED, fontsize=10.5)
    axis.set_facecolor(INK)
    axis.tick_params(colors=MUTED, labelsize=9, length=0)
    axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return figure, axis


def render_health_impacts(direct: dict[str, Any], output_path: Path) -> None:
    values = np.asarray(direct["diagnostics"]["health_impacts_per_zone"], dtype=float)
    labels = ["Urban", "Rural", "Vulnerable"]
    colors = [FIRE, SKY, SMOKE]
    figure, axis = _setup_chart_figure(
        "WHO BEARS THE EXPOSURE?",
        "Synthetic incremental health impact by population zone · Tiny direct run",
    )
    y_positions = np.arange(len(values))
    axis.hlines(y_positions, 0, values, color=colors, linewidth=5, alpha=0.3)
    axis.scatter(values, y_positions, s=125, color=colors, edgecolor=TEXT, linewidth=1.1)
    axis.set_yticks(y_positions, labels)
    axis.set_xlim(0, max(values) * 1.28)
    axis.set_xlabel("Incremental impact · dimensionless", color=MUTED, labelpad=14)
    axis.invert_yaxis()
    for y_pos, value, color in zip(y_positions, values, colors, strict=True):
        axis.text(
            value + max(values) * 0.035,
            y_pos,
            f"{value:,.1f}",
            va="center",
            color=color,
            fontsize=12,
            weight="bold",
        )

    total = float(values.sum())
    axis.text(
        0.98,
        1.20,
        f"TOTAL  {total:,.1f}",
        transform=axis.transAxes,
        ha="right",
        color=EMBER,
        fontsize=11,
        weight="bold",
    )
    figure.savefig(output_path, dpi=180, facecolor=INK, metadata={"Software": "ClimaCare-Risk"})
    plt.close(figure)


def render_portfolio_outcomes(portfolio: dict[str, Any], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), facecolor=INK)
    figure.subplots_adjust(left=0.075, right=0.965, bottom=0.18, top=0.71, wspace=0.3)
    figure.text(
        0.055,
        0.91,
        "FROM HAZARD TO CAPITAL ALLOCATION",
        color=TEXT,
        fontsize=19,
        weight="bold",
    )
    figure.text(
        0.055,
        0.84,
        "Expected total loss and tail risk under corrected synthetic scenarios",
        color=MUTED,
        fontsize=10.5,
    )

    policy_names = ["Uniform", "Insurance only", "Optimized"]
    policy_keys = ["uniform", "insurance_only", "optimized"]
    expected = np.array([portfolio["policies"][key]["EL"] for key in policy_keys]) / 1000
    tail = np.array([portfolio["policies"][key]["CVaR"] for key in policy_keys]) / 1000
    positions = np.arange(len(policy_names))
    width = 0.32
    axes[0].bar(positions - width / 2, expected, width, color=SMOKE, label="Expected loss")
    axes[0].bar(positions + width / 2, tail, width, color=FIRE, label="CVaR")
    axes[0].set_xticks(positions, policy_names)
    axes[0].set_ylabel("Loss · thousands", color=MUTED, labelpad=10)
    axes[0].set_title("POLICY COMPARISON", loc="left", color=TEXT, fontsize=10, weight="bold")
    axes[0].legend(loc="upper right", frameon=False, labelcolor=TEXT, fontsize=8)
    frontier = portfolio["budget_frontier"]
    budgets = np.asarray(frontier["budgets"]) / 1000
    frontier_el = np.asarray(frontier["EL"]) / 1000
    frontier_cvar = np.asarray(frontier["CVaR"]) / 1000
    axes[1].plot(budgets, frontier_cvar, color=FIRE, linewidth=2.4, marker="o", label="CVaR")
    axes[1].plot(
        budgets,
        frontier_el,
        color=SMOKE,
        linewidth=2.4,
        marker="o",
        label="Expected loss",
    )
    axes[1].fill_between(budgets, frontier_el, frontier_cvar, color=FIRE, alpha=0.08)
    axes[1].set_xlabel("Available budget · thousands", color=MUTED, labelpad=10)
    axes[1].set_ylabel("Loss · thousands", color=MUTED, labelpad=10)
    axes[1].set_title("BUDGET FRONTIER", loc="left", color=TEXT, fontsize=10, weight="bold")
    axes[1].legend(loc="upper right", frameon=False, labelcolor=TEXT, fontsize=8)

    for axis in axes:
        axis.set_facecolor(INK)
        axis.tick_params(colors=MUTED, labelsize=8.5, length=0, pad=6)
        axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
    reduction = 100 * (1 - expected[-1] / expected[0])
    badge_text = (
        f"OPTIMIZED EXPECTED LOSS  −{reduction:.1f}%"
        if reduction >= 0.05
        else "OPTIMIZED TOTAL LOSS ≈ UNIFORM"
    )
    badge = FancyBboxPatch(
        (0.675, 0.745),
        0.29,
        0.045,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        transform=figure.transFigure,
        facecolor=PANEL,
        edgecolor=GRID,
        linewidth=0.8,
    )
    figure.add_artist(badge)
    figure.text(
        0.692,
        0.758,
        badge_text,
        color=EMBER,
        fontsize=8.2,
        weight="bold",
    )
    figure.savefig(output_path, dpi=180, facecolor=INK, metadata={"Software": "ClimaCare-Risk"})
    plt.close(figure)


def render_prevention_optimization(prevention: dict[str, Any], output_path: Path) -> None:
    result = prevention["result"]
    iterations = np.arange(len(result["objective_history"]))
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), facecolor=INK)
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.78, wspace=0.3)
    figure.suptitle("END-TO-END FUEL PREVENTION OPTIMIZATION", color=TEXT, fontsize=15, weight="bold")

    for axis in axes:
        axis.set_facecolor(PANEL)
        axis.tick_params(colors=MUTED)
        axis.grid(color=GRID, alpha=0.45, linewidth=0.8)
        for spine in axis.spines.values():
            spine.set_color(GRID)

    axes[0].plot(iterations, result["objective_history"], color=EMBER, linewidth=2.5)
    axes[0].set_xlabel("Gradient iteration", color=MUTED)
    axes[0].set_ylabel("Coupled objective", color=MUTED)
    axes[0].set_title("GRADIENTS DO THE WORK", loc="left", color=TEXT, fontsize=10, weight="bold")

    labels = ["Burned area", "Smoke exposure"]
    initial = [result["burned_area_initial"], result["smoke_exposure_initial"]]
    final = [result["burned_area_final"], result["smoke_exposure_final"]]
    x = np.arange(len(labels))
    width = 0.34
    axes[1].bar(x - width / 2, initial, width, color=FIRE, label="Initial")
    axes[1].bar(x + width / 2, final, width, color=SMOKE, label="Optimized")
    axes[1].set_xticks(x, labels, color=TEXT)
    axes[1].set_ylabel("Normalized physical impact", color=MUTED)
    axes[1].set_title("PHYSICAL IMPACT", loc="left", color=TEXT, fontsize=10, weight="bold")
    axes[1].legend(frameon=False, labelcolor=TEXT)

    figure.text(
        0.5,
        0.04,
        f"u_fuel {result['level_history'][0]:.3f} → {result['level']:.3f} · "
        f"objective {result['objective_history'][0]:.3f} → {result['objective_history'][-1]:.3f}",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    figure.savefig(output_path, dpi=180, facecolor=INK, metadata={"Software": "ClimaCare-Risk"})
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_tiny_config()
    portfolio = _load_json(PORTFOLIO_RESULT)
    prevention = _load_json(PREVENTION_RESULT)
    fire_frames, smoke_frames = _simulate_fields(config)
    direct = _health_payload(config, smoke_frames)

    animation_path = output_dir / "tiny_fire_smoke.gif"
    health_path = output_dir / "health_impacts.png"
    portfolio_path = output_dir / "portfolio_outcomes.png"
    prevention_path = output_dir / "prevention_optimization.png"

    render_field_animation(config, fire_frames, smoke_frames, animation_path)
    render_health_impacts(direct, health_path)
    render_portfolio_outcomes(portfolio, portfolio_path)
    render_prevention_optimization(prevention, prevention_path)

    for path in (animation_path, health_path, portfolio_path, prevention_path):
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        print(f"wrote {display_path}")


if __name__ == "__main__":
    main()
