# ClimaCare-Risk

Differentiable Digital Twin for Health and Financial Resilience Against Wildfires

A single differentiable computation chain linking fire spread → smoke transport → health exposure → financial risk and allocation, composed across four heterogeneous Tesseract components. Built for the **Tesseract Hackathon 2026** (track: *Differentiable inference & uncertainty quantification*).

---

## Key Features

- **Cross-Language Differentiable Composition**: `fire_spread_torch` (PyTorch, native autodiff) and `smoke_transport_cpp` (C++20/OpenMP, hand-written discrete adjoint) are composed inside a single `jax.value_and_grad` call via Tesseract-JAX. The gradient crosses the container boundary with no intermediate file and no host-side solver execution.
- **Verified Adjoint Correctness**: the C++ discrete adjoint passes a dot-product identity test (`⟨v, Ju⟩ = ⟨Jᵀv, u⟩`) and its cotangents match centred finite differences on the wind parameter.
- **Full Bayesian Inference Stack**: MAP estimation (L-BFGS/Adam), Laplace approximation, and NUTS posterior refinement (NumPyro) directly on top of the composed gradient — not on a surrogate.
- **Gradients That Do Real Work**: the composed pipeline drives a robust portfolio allocation under CVaR risk, reducing expected loss from **$88.2k (uniform policy)** to **$1.4k (optimized)** on the real physics pipeline — a **~63× reduction** — subject to a capex budget constraint.
- **Reproducible by Design**: every experiment (`direct`, `map`, `uq`, `portfolio`) runs through one CLI entry point and writes a self-describing JSON artifact (git commit, package versions, CFL numbers included).

For the full derivation (discretization, adjoint, likelihood, CVaR, budget constraints), see [`docs/mathematical_specification.md`](docs/mathematical_specification.md).

---

## Table of Contents

- [Key Features](#key-features)
- [About this Project](#about-this-project)
  - [Architecture](#architecture)
  - [Differentiable Composition](#differentiable-composition)
- [Numerical Experiments](#numerical-experiments)
  - [Performance Summary](#performance-summary)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation and Build](#installation-and-build)
  - [Reproducing the Results](#reproducing-the-results)
  - [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [Tech Stack](#tech-stack)
- [License](#license)
- [Status](#status)

---

## About this Project

### Architecture

Four heterogeneous components, composed via [Tesseract](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/):

| Component | Role | Language / differentiation |
|---|---|---|
| `fire_spread_torch` | Fire propagation (reaction–diffusion–advection) | PyTorch, native autodiff |
| `smoke_transport_cpp` | Atmospheric smoke transport | C++20/OpenMP, hand-written discrete adjoint |
| HealthImpact | Exposure and health burden | JAX, native autodiff |
| ResilienceFinance | Expected loss, CVaR, portfolio allocation | JAX/Python |

The orchestrator (`app/climacare`) serves the first two as Tesseracts and drives Bayesian inference and robust optimization on top.

### Differentiable Composition

θ (ignition, wind) -> [fire_spread_torch] -> smoke_source -> [smoke_transport_cpp] -> sensor_concentration -> loss
PyTorch autodiff C++ hand-written adjoint

A single JAX function (`TesseractPipeline.sensor_predictions`) calls both containers through `tesseract_jax.apply_tesseract`. `jax.value_and_grad` of the downstream loss therefore triggers the C++ adjoint first, and feeds its smoke-source cotangent straight into the PyTorch VJP — the composition genuinely crosses the language and differentiation-strategy boundary the hackathon asks for, it is not two solvers glued by a shared script.

---

## Numerical Experiments

The Tiny case (32×32 grid, 60 steps, single ignition point) validates the pipeline end to end:

| Experiment | Metric | Result |
|---|---|---|
| E1 — Direct simulation | Burned area | 0.125 (fraction of domain) |
| E1 — Direct simulation | Smoke transport CFL number | 0.429 (stable) |
| E2 — Gradient check | Composed VJP vs. centred finite differences | matches within tolerance, all components finite and sign-consistent |
| E3 — MAP inversion | Ignition position recovery | converges from the prior guess to the synthetic truth |
| E4 — UQ | Laplace + NUTS posterior over 4 physical parameters | — |

### Performance Summary

Robust portfolio allocation (E5/E6), real pipeline (not the synthetic fallback), 20 candidate mitigation actions, CVaR level λ = 0.5:

| Policy | Expected Loss | CVaR | Capex |
|---|---|---|---|
| Uniform | $88,182 | $146,466 | $180,000 |
| Insurance-only | $124,847 | $183,093 | $150,000 |
| **Optimized (ours)** | **$1,393** | **$41,107** | $561,297 |

Objective value $\mathcal{J}$ (loss + CVaR penalty) drops from **$161,415 → $21,947** over the optimization (initial → final). Stress tests (strong wind, biased sensors, 20% budget cut) confirm the optimized allocation stays within budget and degrades gracefully.

---

## Repository Structure

ClimatCareRisk/  
├── app/climacare/ # Orchestrator: pipeline, inference, UQ, finance, CLI  
│ ├── pipeline.py # Composed FireSpread → SmokeTransport, jax.value_and_grad  
│ ├── inverse.py # Gradient check + MAP (L-BFGS / Adam)  
│ ├── uq.py # Laplace approximation + NUTS (NumPyro)  
│ ├── finance.py # CVaR, portfolio optimization, stress tests  
│ ├── health.py # Exposure → health impact  
│ └── cli.py # python -m climacare.cli {direct,map,uq,portfolio}  
│  
├── components/  
│ ├── tesseracts/  
│ │ ├── fire_spread_torch/ # Tesseract A — PyTorch autodiff  
│ │ └── smoke_transport_cpp/ # Tesseract B — C++20/OpenMP, hand-written adjoint  
│ └── shared_code/ # Physics kernels shared between components and orchestrator  
│  
├── src/ # Scenario generation, loss structure, health model  
├── scripts/ # Portfolio experiments, benchmarks  
├── tests/ # pytest suite (unit + container-dependent, auto-skipped)  
├── docs/mathematical_specification.md  
├── configs/tiny.yaml # Tiny case configuration  
├── results/ # Reproducible experiment outputs (JSON)  
└── Makefile   

## Getting Started

### Prerequisites

- Python ≥ 3.12
- [Docker](https://docs.docker.com/get-docker/) (to build and serve the Tesseract images)
- GNU Make
- A C++20 compiler + [CMake](https://cmake.org/) ≥ 3.22 (to compile the smoke-transport kernel locally, e.g. for tests)

### Installation and Build

```bash
git clone https://github.com/lucasbertin06/ClimatCareRisk.git
cd ClimatCareRisk
pip install -e app -e components/shared_code

make build-c0     # build + tag the fire_spread_torch and smoke_transport_cpp Docker images
make smoke-kernel # compile the C++ kernel locally (cmake), used by the unit tests
```

> [!NOTE]
> `tesseract-core` also exposes a no-Docker debugging mode (`Tesseract.from_tesseract_api`) that imports a component's `tesseract_api.py` directly in-process. It's how the composed gradient can be sanity-checked without building any image, and it's what the test suite falls back to when Docker images aren't available (skipping the container-only tests instead of failing).

### Reproducing the Results

All experiments run through a single entry point, `climacare.cli`:

```bash
python -m climacare.cli direct    --output-dir results/tiny_direct         # E1: direct fire -> smoke simulation
python -m climacare.cli map       --output-dir results/tiny_map            # E2/E3: gradient check + inverse problem (MAP)
python -m climacare.cli uq        --output-dir results/tiny_uq [--nuts]    # E4: Laplace posterior, optional NUTS refinement
python -m climacare.cli portfolio --output-dir results/portfolio [--fake]  # E5/E6: robust portfolio + stress tests
```

Equivalent shortcuts via `make`:

```bash
make tiny-direct    # E1
make tiny-gradient  # E2 (dedicated gradient test)
make tiny-map       # E2 + E3
make test           # full test suite
```

`portfolio --fake` uses synthetic scenarios and needs no Docker images; without `--fake`, the command chains MAP → Laplace → posterior sampling → simulation through the real Tesseract pipeline and requires `make build-c0` beforehand. For finer-grained variants (number of scenarios, posterior spread, CVaR level...):

```bash
python scripts/run_portfolio_experiment.py --help
```

Results are written as reproducible JSON files under `results/` (git commit, package versions, and CFL numbers are recorded alongside every run).

### Troubleshooting

| Issue | Solution |
|---|---|
| `Tesseract images fire_spread_torch and smoke_transport_cpp must be built first` | Run `make build-c0` (needs Docker), or rely on the container-dependent tests being skipped |
| `compiled kernel 'smoke_kernel_cpp' not found` | Run `make smoke-kernel`, or set `CLIMACARE_SMOKE_KERNEL_DIR` to the build directory |
| `pybind11 is required to build smoke_transport_cpp` | `pip install pybind11` before `make smoke-kernel` |
| Import error on `tesseract_core.runtime.cli` (missing `lz4` etc.) | `pip install "tesseract-core[runtime]"` |
| Docker unavailable in a sandbox/CI environment | Use `Tesseract.from_tesseract_api(...)` to run components in-process for debugging (see above) |

---

## Future Work

- Extend the physical model beyond the Tiny case (larger grids, multiple ignition points, time-varying wind fields).
- Replace the fixed sensor network with a differentiable sensor-placement optimization.
- Explore full-posterior (NUTS-only) portfolio optimization instead of the current Laplace-approximation shortcut.

---

## Tech Stack

- **Composition**: [Tesseract-Core](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/) 1.11.0, [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) 0.4.1
- **Differentiable computing**: JAX 0.11.0, PyTorch (autodiff), C++20/OpenMP (hand-written adjoint)
- **Inference**: NumPyro (SVI, NUTS), SciPy (L-BFGS-B), Optax (Adam)
- **Build**: CMake ≥ 3.22, pybind11, Docker

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Status

Research prototype built for the Tesseract Hackathon 2026 (August 4–31, 2026). The health, insurance, and economic results shown are demonstration hypotheses: they do not constitute an operational forecast, nor clinical or financial advice.