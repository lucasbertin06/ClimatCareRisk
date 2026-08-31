"""Direct composition of the two Tesseracts, specification section 16."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from climacare.config import TinyConfig, load_tiny_config
from climacare.objective import make_observations
from climacare.pipeline import TesseractPipeline
from conftest import requires_containers

REPOSITORY = Path(__file__).resolve().parents[1]


def test_configuration_verifies_both_stability_budgets(tiny_config: TinyConfig) -> None:
    assert 0.0 < tiny_config.nu_fire <= 1.0
    assert 0.0 < tiny_config.nu_smoke <= 1.0
    assert tiny_config.dt * tiny_config.fire.reaction_rate <= 1.0
    assert tiny_config.seed == 20260805


def test_configuration_rejects_an_unstable_time_step(tmp_path: Path) -> None:
    source = (REPOSITORY / "configs" / "tiny.yaml").read_text(encoding="utf-8")
    broken = tmp_path / "unstable.yaml"
    broken.write_text(source.replace("dt: 0.02", "dt: 0.5"), encoding="utf-8")
    with pytest.raises(ValueError, match="CFL violated"):
        load_tiny_config(broken)

def test_configuration_rejects_invalid_sensor_shapes(tmp_path: Path) -> None:
    source = (REPOSITORY / "configs" / "tiny.yaml").read_text(encoding="utf-8")
    broken = tmp_path / "bad-sensors.yaml"
    broken.write_text(source.replace("bias: [0.0, 0.0, 0.0]", "bias: [0.0]"), encoding="utf-8")
    with pytest.raises(ValueError, match="must both have shape"):
        load_tiny_config(broken)


def test_configuration_rejects_non_finite_scalars(tmp_path: Path) -> None:
    source = (REPOSITORY / "configs" / "tiny.yaml").read_text(encoding="utf-8")
    broken = tmp_path / "nan.yaml"
    broken.write_text(source.replace("heat_loss: 0.20", "heat_loss: .nan"), encoding="utf-8")
    with pytest.raises(ValueError, match="must all be finite"):
        load_tiny_config(broken)


def _abstract(payload: dict) -> dict:
    """Turn a concrete payload into the shape/dtype form abstract_eval expects."""
    abstract = {}
    for name, value in payload.items():
        if isinstance(value, (bool, int)) and not isinstance(value, float):
            abstract[name] = value
            continue
        array = np.asarray(value)
        abstract[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    return abstract


@requires_containers
def test_schemas_and_shapes_agree_with_abstract_eval(
    pipeline: TesseractPipeline,
) -> None:
    """Both endpoints must agree, otherwise JAX cannot trace the container."""
    config = pipeline.config
    theta = jnp.asarray(config.truth)
    for client, inputs in (
        (pipeline.fire_client, pipeline.fire_inputs(theta, config.frame_count)),
        (
            pipeline.smoke_client,
            pipeline.smoke_inputs(
                theta,
                jnp.zeros((config.n_steps, config.grid.ny, config.grid.nx)),
                config.frame_count,
            ),
        ),
    ):
        declared = client.abstract_eval(_abstract(inputs))
        produced = client.apply(inputs)
        assert set(declared) == set(produced)
        for name, meta in declared.items():
            assert tuple(meta["shape"]) == np.asarray(produced[name]).shape, name
            assert meta["dtype"] == "float64", name


@requires_containers
def test_direct_run_produces_consistent_fields(pipeline: TesseractPipeline) -> None:
    config = pipeline.config
    result = pipeline.direct(jnp.asarray(config.truth))

    source = np.asarray(result["smoke_source"])
    assert source.shape == (config.n_steps, config.grid.ny, config.grid.nx)
    assert np.all(source >= 0.0)

    frames = np.asarray(result["concentration_frames"])
    assert frames.shape == (config.frame_count, config.grid.ny, config.grid.nx)
    assert np.all(frames >= 0.0)
    assert np.allclose(frames[0], 0.0), "the smoke initial condition must vanish"

    predictions = np.asarray(result["sensor_concentration"])
    assert predictions.shape == (config.n_steps, config.sensors.count)
    assert np.all(np.isfinite(predictions))
    assert predictions.max() > 10.0 * config.sensors.noise_std.max(), (
        "the sensors must see a signal well above the noise floor"
    )
    assert float(result["smoke_cfl_number"]) <= 1.0
    assert float(result["burned_area"]) > 0.0


@requires_containers
def test_composition_is_deterministic_at_a_fixed_seed(
    pipeline: TesseractPipeline,
) -> None:
    theta = jnp.asarray(pipeline.config.truth)
    first = np.asarray(pipeline.sensor_predictions(theta))
    second = np.asarray(pipeline.sensor_predictions(theta))
    assert np.array_equal(first, second)

    observations_a = make_observations(pipeline.config, first)
    observations_b = make_observations(pipeline.config, first)
    assert np.array_equal(observations_a.values, observations_b.values)
    assert np.array_equal(observations_a.mask, observations_b.mask)


@requires_containers
def test_observation_mask_matches_the_configured_fraction(
    pipeline: TesseractPipeline,
) -> None:
    config = pipeline.config
    clean = np.asarray(pipeline.sensor_predictions(jnp.asarray(config.truth)))
    observations = make_observations(config, clean)
    total = observations.mask.size
    masked = total - observations.count
    assert 0 < masked < total
    assert abs(masked / total - config.sensors.mask_fraction) < 0.05


@requires_containers
def test_no_intermediate_file_is_written_by_the_differentiated_path(
    pipeline: TesseractPipeline, tmp_path: Path
) -> None:
    """The composed evaluation must not touch the working directory."""
    before = {path.name for path in tmp_path.iterdir()}
    pipeline.sensor_predictions(jnp.asarray(pipeline.config.truth))
    after = {path.name for path in tmp_path.iterdir()}
    assert before == after


@requires_containers
def test_tiny_direct_command_writes_a_complete_artifact(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "climacare.cli", "direct", "--output-dir", str(tmp_path)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=True,
        env=_environment(),
        timeout=1800,
    )
    assert "tiny_direct.json" in completed.stdout
    payload = json.loads((tmp_path / "tiny_direct.json").read_text(encoding="utf-8"))
    assert payload["command"] == "tiny-direct"
    assert payload["configuration"]["seed"] == 20260805
    assert payload["versions"]["tesseract_core"] == "1.11.0"
    assert payload["versions"]["tesseract_jax"] == "0.4.1"
    assert payload["git_commit"] != "unknown"
    assert payload["stability"]["nu_smoke_realised"] <= 1.0
    assert payload["downstream"]["loss_physical"] > 0.0


def _environment() -> dict[str, str]:
    import os

    return dict(
        os.environ,
        # os.pathsep : ";" on Windows, ":" on Linux
        PYTHONPATH=os.pathsep.join([str(REPOSITORY / "components" / "shared_code"), str(REPOSITORY / "app")]),
    )
