# ClimaCare-Risk

ClimaCare-Risk is a differentiable research pipeline that connects wildfire spread, smoke transport, exposure, and financial allocation in one computation graph.

![Tiny synthetic case: coupled fire intensity and smoke concentration over time](docs/assets/tiny_fire_smoke.gif)

The project was built for the Tesseract Hackathon 2026, in the *Differentiable inference and uncertainty quantification* track. Its core experiment tests reverse-mode differentiation across two heterogeneous numerical solvers without replacing either solver with a surrogate.

Tesseract-JAX composes the components. In reverse mode, the smoke cotangent passes through a hand-written C++ discrete adjoint before the gradient continues through the PyTorch VJP. The same computation chain supports source inversion and uncertainty approximation; synthetic health and finance models consume its outputs downstream.

> **Scope.** The Tiny case, observations, population fields, health coefficients, and financial parameters are synthetic and dimensionless. The outputs are benchmark results, not wildfire forecasts, clinical estimates, or financial advice.

## Results

The committed [direct-run artifact](results/tiny_direct/tiny_direct.json) predates the cell-area correction and is retained only as historical output. New direct runs use the grid cell area when computing health impacts. The committed synthetic portfolio artifact includes investment cost in expected and tail losses; policies are therefore compared on total cost rather than mitigation benefit alone.

### Coupled fields

The animation contains 13 evenly spaced snapshots from a 60-step run over $t \in [0, 1.2]$. Diamonds mark the three synthetic smoke sensors; the star marks the ignition point.

### Downstream diagnostics

![Historical health-impact figure from the pre-correction direct artifact](docs/assets/health_impacts.png)

![Historical portfolio figure; regenerate after producing a matching pipeline artifact](docs/assets/portfolio_outcomes.png)

## How the gradient crosses the pipeline

```text
physical parameters θ = (x₀, y₀, log A₀, Δφ)
            │
            ▼
┌──────────────────────────────┐
│ FireSpread                   │  PyTorch · autodiff
│ reaction–diffusion–advection │
└──────────────┬───────────────┘
               │ smoke source S[n, y, x]
               ▼
┌──────────────────────────────┐
│ SmokeTransport               │  C++20/OpenMP · discrete adjoint
│ advection–diffusion–decay    │
└──────────────┬───────────────┘
               │ sensor concentrations
               ▼
       MAP objective in JAX
               │
               ▼ reverse mode
       C++ adjoint → PyTorch VJP
```

| Layer | Responsibility | Differentiation |
|---|---|---|
| [`fire_spread_torch`](components/tesseracts/fire_spread_torch) | Evolves thermal intensity and fuel; emits the full smoke source history | PyTorch VJP |
| [`smoke_transport_cpp`](components/tesseracts/smoke_transport_cpp) | Transports smoke and samples the fixed sensor network | Hand-written C++ discrete adjoint |
| [`app/climacare`](app/climacare) | Composes both Tesseracts; runs MAP, Laplace UQ, and optional NUTS | JAX and Tesseract-JAX |
| [`src/loss_structure.py`](src/loss_structure.py) | Computes expected loss, VaR/CVaR, allocation, and stress tests | JAX |

Health and finance are downstream diagnostics. They do not enter the MAP likelihood and therefore cannot alter the inferred physical parameters.

## Setup

Requirements:

- Python 3.12 or newer
- GNU Make
- Docker, only for commands that serve the two Tesseract images
- A C++20 compiler, CMake, and OpenMP for the local smoke kernel

```bash
git clone https://github.com/lucasbertin06/ClimatCareRisk.git
cd ClimatCareRisk
python -m venv .venv
source .venv/bin/activate
pip install -e 'app[dev]' -e components/shared_code
```

### Local paths without Tesseract images

The synthetic portfolio path runs without Docker:

```bash
python -m climacare.cli portfolio --fake --output-dir results/portfolio_fake
```

The standalone plume inversion CLI exposes an explicit subcommand:

```bash
plume_inversion run --steps 120
```

The visual assets use the local PyTorch solver and compiled C++ smoke kernel:

```bash
make smoke-kernel
python scripts/generate_visualizations.py
```

Assets are written to `docs/assets/`. Use `--output-dir PATH` to render elsewhere.

### Full heterogeneous pipeline

Build and tag both Tesseract images:

```bash
make build-c0
```

Then run any experiment that crosses the served fire and smoke components:

```bash
python -m climacare.cli direct --output-dir results/tiny_direct
python -m climacare.cli map --output-dir results/tiny_map
python -m climacare.cli uq --output-dir results/tiny_uq
python -m climacare.cli uq --nuts --output-dir results/tiny_uq_nuts
python -m climacare.cli portfolio --output-dir results/portfolio
```

## Experiments

| Experiment | Command | Output | Images required |
|---|---|---|---|
| E1 · direct simulation | `python -m climacare.cli direct --output-dir results/tiny_direct` | `tiny_direct.json` | Yes |
| E2/E3 · gradient check and MAP | `python -m climacare.cli map --output-dir results/tiny_map` | `map.json` | Yes |
| E4 · Laplace posterior | `python -m climacare.cli uq --output-dir results/tiny_uq` | `uq.json` | Yes |
| E4 · NUTS refinement | `python -m climacare.cli uq --nuts --output-dir results/tiny_uq_nuts` | `uq.json` with samples | Yes |
| E5/E6 · pipeline portfolio | `python -m climacare.cli portfolio --output-dir results/portfolio` | `portfolio.json` | Yes |
| E5/E6 · synthetic portfolio | `python -m climacare.cli portfolio --fake --output-dir results/portfolio_fake` | `portfolio.json` | No |

`make tiny-direct`, `make tiny-gradient`, and `make tiny-map` provide shortcuts for the first three pipeline checks. `python scripts/run_portfolio_experiment.py --help` exposes the lower-level portfolio controls, including scenario count, posterior spread, CVaR weight, and optimizer steps.

The direct, MAP, and UQ commands record dependency versions and the source revision when Git metadata is available. The committed examples are:

- [`results/tiny_direct/tiny_direct.json`](results/tiny_direct/tiny_direct.json)
- [`results/portfolio_e5_pipeline.json`](results/portfolio_e5_pipeline.json)
- [`results/portfolio_e5_fake.json`](results/portfolio_e5_fake.json)

## Verification

```bash
python -m pytest tests -q
```

The test suite covers the local numerical kernels, direct composition, cross-component gradients, inversion, health calculations, smoke transport, fire spread, and portfolio loss structure. CI compiles and tests the local C++ smoke kernel explicitly; container-dependent tests run again after both Tesseract images are built.

For the mathematical model, discretizations, adjoint equations, stability conditions, and benchmark limits, see the [mathematical specification](docs/mathematical_specification.md).

## Repository map

```text
app/climacare/                         orchestration, inference, UQ, finance, CLI
components/tesseracts/fire_spread_torch/
                                      PyTorch FireSpread Tesseract
components/tesseracts/smoke_transport_cpp/
                                      C++ SmokeTransport Tesseract and adjoint
components/shared_code/               numerical code shared with the containers
configs/tiny.yaml                     frozen synthetic benchmark configuration
src/                                  scenario generation and portfolio loss model
scripts/                              experiments, benchmarks, asset generation
tests/                                numerical and end-to-end tests
results/                              committed reproducible JSON artifacts
docs/                                 mathematical specification and README assets
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
