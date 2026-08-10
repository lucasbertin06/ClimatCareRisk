"""SmokeTransport invariants and adjoint test required by specification section 16."""

from __future__ import annotations

import numpy as np
import pytest
from climacare_shared.grid import Grid, bilinear_weights, check_smoke_stability
from climacare_shared.kernel import load_smoke_kernel


@pytest.fixture(scope="module")
def smoke_kernel() -> object:
    """Return the compiled C++ kernel, skipping when it is not built."""
    try:
        return load_smoke_kernel()
    except ImportError as error:  # pragma: no cover - environment dependent
        pytest.skip(str(error))


NX = NY = 16
NT = 9
SENSORS = np.array([[0.32, 0.42], [0.61, 0.55], [0.44, 0.71]], dtype=np.float64)
BIAS = np.zeros(3, dtype=np.float64)
WIND = (0.37, 0.26)
DIFFUSIVITY = 1.5e-4
DECAY = 0.5
DT = 0.004
LEVELS = np.array([0, NT], dtype=np.int32)


def run(smoke_kernel: object, source: np.ndarray, **overrides: object) -> tuple:
    """Run the forward kernel with the module defaults."""
    settings = {
        "wind_x": WIND[0],
        "wind_y": WIND[1],
        "diffusivity": DIFFUSIVITY,
        "decay": DECAY,
        "dt": DT,
    }
    settings.update(overrides)
    return smoke_kernel.forward(
        np.ascontiguousarray(source),
        SENSORS,
        BIAS,
        settings["wind_x"],
        settings["wind_y"],
        settings["diffusivity"],
        settings["decay"],
        settings["dt"],
        LEVELS,
    )


def gaussian_source(centre: tuple[float, float] = (0.3, 0.3)) -> np.ndarray:
    """Return a compact positive source, constant in time."""
    grid = Grid(nx=NX, ny=NY)
    mesh_x, mesh_y = grid.meshgrid()
    blob = np.exp(-(((mesh_x - centre[0]) ** 2 + (mesh_y - centre[1]) ** 2) / (2 * 0.06**2)))
    return np.repeat(blob[None, :, :], NT, axis=0)


def test_zero_source_gives_zero_concentration(smoke_kernel: object) -> None:
    observations, frames = run(smoke_kernel, np.zeros((NT, NY, NX)))
    assert np.all(observations == 0.0)
    assert np.all(frames == 0.0)


def test_decay_reduces_the_mass_without_source(smoke_kernel: object) -> None:
    source = np.zeros((NT, NY, NX))
    source[0] = gaussian_source()[0]
    _, frames_decay = run(smoke_kernel, source, decay=2.0)
    _, frames_none = run(smoke_kernel, source, decay=0.0)
    assert frames_decay[-1].sum() < frames_none[-1].sum()


def test_transport_follows_the_wind(smoke_kernel: object) -> None:
    grid = Grid(nx=NX, ny=NY)
    mesh_x, mesh_y = grid.meshgrid()
    source = np.zeros((NT, NY, NX))
    source[0] = gaussian_source((0.4, 0.4))[0]

    def centroid(**overrides: object) -> tuple[float, float]:
        _, frames = run(smoke_kernel, source, dt=0.004, **overrides)
        field = frames[-1]
        mass = field.sum()
        return float((field * mesh_x).sum() / mass), float((field * mesh_y).sum() / mass)

    still = centroid(wind_x=0.0, wind_y=0.0)
    east = centroid(wind_x=1.5, wind_y=0.0)
    north = centroid(wind_x=0.0, wind_y=1.5)
    assert east[0] > still[0]
    assert abs(east[1] - still[1]) < 1e-9
    assert north[1] > still[1]
    assert abs(north[0] - still[0]) < 1e-9


def test_higher_diffusivity_spreads_the_plume(smoke_kernel: object) -> None:
    grid = Grid(nx=NX, ny=NY)
    mesh_x, mesh_y = grid.meshgrid()
    source = np.zeros((NT, NY, NX))
    source[0] = gaussian_source((0.5, 0.5))[0]

    def variance(diffusivity: float) -> float:
        _, frames = run(
            smoke_kernel, source, wind_x=0.0, wind_y=0.0, decay=0.0,
            diffusivity=diffusivity,
        )
        field = frames[-1]
        mass = field.sum()
        centre_x = (field * mesh_x).sum() / mass
        centre_y = (field * mesh_y).sum() / mass
        return float(
            (field * ((mesh_x - centre_x) ** 2 + (mesh_y - centre_y) ** 2)).sum() / mass
        )

    assert variance(4.0e-4) > variance(1.0e-5)


def test_bilinear_interpolation_is_exact_on_an_affine_field(smoke_kernel: object) -> None:
    grid = Grid(nx=NX, ny=NY)
    mesh_x, mesh_y = grid.meshgrid()
    affine = 0.3 + 1.7 * mesh_x - 0.9 * mesh_y

    # One step with a zero operator: c^1 = dt * S, so pick S to realise `affine`.
    source = np.zeros((1, NY, NX))
    source[0] = affine / DT
    observations, _ = smoke_kernel.forward(
        np.ascontiguousarray(source),
        SENSORS,
        BIAS,
        0.0,
        0.0,
        1.0e-12,
        0.0,
        DT,
        np.array([1], dtype=np.int32),
    )
    expected = 0.3 + 1.7 * SENSORS[:, 0] - 0.9 * SENSORS[:, 1]
    assert np.allclose(observations[0], expected, atol=1e-9)


def test_invalid_cfl_is_rejected(smoke_kernel: object) -> None:
    source = gaussian_source()
    with pytest.raises(Exception, match="CFL"):
        run(smoke_kernel, source, dt=1.0)
    with pytest.raises(Exception, match="D_c"):
        run(smoke_kernel, source, diffusivity=0.0)
    with pytest.raises(Exception, match="lambda_c"):
        run(smoke_kernel, source, decay=-1.0)
    with pytest.raises(ValueError, match="CFL violated"):
        check_smoke_stability(
            dt=1.0,
            grid=Grid(nx=NX, ny=NY),
            wind_speed=0.45,
            diffusivity=DIFFUSIVITY,
            decay=DECAY,
        )


def test_sensor_outside_the_bilinear_window_is_rejected(smoke_kernel: object) -> None:
    source = gaussian_source()
    with pytest.raises(Exception, match="bilinear-safe"):
        smoke_kernel.forward(
            np.ascontiguousarray(source),
            np.array([[0.001, 0.5]], dtype=np.float64),
            np.zeros(1),
            WIND[0],
            WIND[1],
            DIFFUSIVITY,
            DECAY,
            DT,
            LEVELS,
        )


def test_adjoint_dot_product_identity(smoke_kernel: object) -> None:
    r"""Check :math:`\langle Jv, q\rangle = \langle v, J^\top q\rangle`.

    The forward operator is affine in the source with a zero initial condition,
    so ``forward(v)`` is exactly ``J v`` and no finite difference is involved.
    """
    generator = np.random.default_rng(20260805)
    source = generator.normal(size=(NT, NY, NX))
    direction = generator.normal(size=source.shape)
    cotangent = generator.normal(size=(NT, SENSORS.shape[0]))

    jacobian_action, _ = run(smoke_kernel, direction)
    source_bar, _, _, _ = smoke_kernel.vector_jacobian_product(
        np.ascontiguousarray(source),
        SENSORS,
        WIND[0],
        WIND[1],
        DIFFUSIVITY,
        DECAY,
        DT,
        np.ascontiguousarray(cotangent),
    )
    left = float(np.sum(jacobian_action * cotangent))
    right = float(np.sum(direction * source_bar))
    denominator = max(1.0, abs(left), abs(right))
    assert abs(left - right) / denominator < 1e-6


def test_wind_cotangent_matches_central_differences(smoke_kernel: object) -> None:
    generator = np.random.default_rng(11)
    source = gaussian_source()
    cotangent = generator.normal(size=(NT, SENSORS.shape[0]))

    _, wind_bar, _, _ = smoke_kernel.vector_jacobian_product(
        np.ascontiguousarray(source),
        SENSORS,
        WIND[0],
        WIND[1],
        DIFFUSIVITY,
        DECAY,
        DT,
        np.ascontiguousarray(cotangent),
    )

    def objective(wind_x: float, wind_y: float) -> float:
        observations, _ = run(smoke_kernel, source, wind_x=wind_x, wind_y=wind_y)
        return float(np.sum(observations * cotangent))

    step = 1e-6
    finite_x = (objective(WIND[0] + step, WIND[1]) - objective(WIND[0] - step, WIND[1])) / (
        2 * step
    )
    finite_y = (objective(WIND[0], WIND[1] + step) - objective(WIND[0], WIND[1] - step)) / (
        2 * step
    )
    assert abs(wind_bar[0] - finite_x) <= 1e-6 * max(1.0, abs(finite_x))
    assert abs(wind_bar[1] - finite_y) <= 1e-6 * max(1.0, abs(finite_y))


def test_adjoint_is_independent_of_the_thread_count(smoke_kernel: object) -> None:
    """The coloured scatter must give identical results for any thread count."""
    import os
    import subprocess
    import sys

    script = (
        "import numpy as np, sys;"
        "sys.path[:0] = ['components/shared_code'];"
        "from climacare_shared.kernel import load_smoke_kernel;"
        "k = load_smoke_kernel();"
        "g = np.random.default_rng(7);"
        f"s = g.normal(size=({NT}, {NY}, {NX}));"
        f"q = g.normal(size=({NT}, 3));"
        f"sensors = np.array({SENSORS.tolist()});"
        f"out = k.vector_jacobian_product(np.ascontiguousarray(s), sensors, {WIND[0]},"
        f" {WIND[1]}, {DIFFUSIVITY}, {DECAY}, {DT}, np.ascontiguousarray(q));"
        "print(out[0].tobytes().hex()[:64], out[1].tolist())"
    )
    outputs = []
    for threads in ("1", "6"):
        environment = dict(os.environ, OMP_NUM_THREADS=threads)
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env=environment,
            ).stdout.strip()
        )
    assert outputs[0] == outputs[1]


def test_bilinear_weights_partition_unity() -> None:
    grid = Grid(nx=NX, ny=NY)
    _, _, weights = bilinear_weights(SENSORS, grid)
    assert np.allclose(weights.sum(axis=(1, 2)), 1.0)
    with pytest.raises(ValueError, match="bilinear-safe"):
        bilinear_weights(np.array([[0.0005, 0.5]]), grid)
