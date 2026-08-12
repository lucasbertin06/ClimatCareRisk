"""Shared fixtures for the slice C0 test-suite.

This test root runs in double precision. The frozen IGNIS suite under
``app/tests`` runs in single precision, so the two roots are executed in
separate pytest sessions (see the ``test`` target of the Makefile). Enabling
``jax_enable_x64`` is a global JAX switch and cannot be scoped per module.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from climacare.config import TinyConfig, load_tiny_config
from climacare.pipeline import (
    FIRE_IMAGE,
    SMOKE_IMAGE,
    TesseractPipeline,
    enable_float64,
    open_pipeline,
)

enable_float64()


def _docker_images_available() -> bool:
    """Return whether both C0 Tesseract images can be listed by Docker."""
    if shutil.which("docker") is None:
        return False
    import subprocess

    try:
        listed = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.split()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return FIRE_IMAGE in listed and SMOKE_IMAGE in listed


requires_containers = pytest.mark.skipif(
    not _docker_images_available(),
    reason=(
        f"Tesseract images {FIRE_IMAGE} and {SMOKE_IMAGE} must be built first "
        "(make build-c0)"
    ),
)


@pytest.fixture(scope="session")
def tiny_config() -> TinyConfig:
    """Return the validated Tiny configuration."""
    return load_tiny_config()


@pytest.fixture(scope="session")
def pipeline(tiny_config: TinyConfig) -> Iterator[TesseractPipeline]:
    """Serve both Tesseract containers once for the whole session."""
    with open_pipeline(tiny_config) as served:
        yield served
