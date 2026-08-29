# ClimaCare-Risk

Differentiable digital twin for health and financial resilience against wildfires.

A scientific pipeline linking, in a single differentiable computation chain: fire spread → smoke transport → health exposure → financial risk and allocation. Built for the **Tesseract Hackathon 2026** (main track: *Differentiable inference & uncertainty quantification*).

## Architecture

Four heterogeneous components, composed via [Tesseract](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/):

| Component | Role | Language / differentiation |
|---|---|---|
| `fire_spread_torch` | Fire propagation (reaction-diffusion-advection) | PyTorch, autodiff |
| `smoke_transport_cpp` | Atmospheric smoke transport | C++20/OpenMP, hand-written discrete adjoint |
| HealthImpact | Exposure and health burden | JAX, native autodiff |
| ResilienceFinance | Expected loss, CVaR, portfolio allocation | JAX/Python |

The orchestrator (`app/climacare`) calls the first two through Tesseract-JAX and drives Bayesian inference (MAP, Laplace, NUTS) and robust optimization.

## Requirements

- Python ≥ 3.10
- [Docker](https://docs.docker.com/get-docker/) (to build the Tesseract images)
- `pip install tesseract-core`
- GNU Make

## Installation and build

```bash
git clone https://github.com/lucasbertin06/ClimatCareRisk.git
cd ClimatCareRisk
pip install -e app -e components/shared_code

make build-c0     # build + tag the fire_spread_torch and smoke_transport_cpp images
make smoke-kernel # compile the C++ kernel (cmake) outside the image, for local tests
```

## Reproducing the results

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

`portfolio --fake` uses synthetic scenarios and needs no Docker images; without `--fake`, the command chains MAP → Laplace → posterior sampling → simulation through the Tesseract pipeline, and requires `make build-c0` beforehand.

For finer-grained variants (number of scenarios, posterior spread, CVaR level...), the original script is still available:
```bash
python scripts/run_portfolio_experiment.py --help
```

Results are written as reproducible JSON files under `results/`.

## Repository structure

app/climacare/ # orchestrator: pipeline, inference, UQ, finance, CLI  
components/tesseracts/ # Tesseract components (fire_spread_torch, smoke_transport_cpp)  
components/shared_code/ # Python code shared between components and the orchestrator  
src/ # scenario generation, loss structure, health model  
scripts/ # experiments (portfolio, benchmarks)  
tests/ # test suite (pytest)  
docs/ # mathematical specification  
results/ # experiment outputs  
configs/ # scenario configurations (e.g. tiny.yaml)

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Status

Research prototype built for the Tesseract Hackathon 2026 (August 4–31, 2026). The health, insurance, and economic results shown are demonstration hypotheses: they do not constitute an operational forecast, nor clinical or financial advice.