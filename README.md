# ClimaCare-Risk

Jumeau numérique différentiable pour la résilience sanitaire et financière face aux incendies de forêt.

Pipeline scientifique reliant, dans une seule chaîne de calcul différentiable : propagation du feu → transport des fumées → exposition sanitaire → risque et allocation financière. Projet réalisé dans le cadre du **Tesseract Hackathon 2026** (track principal : *Differentiable inference & uncertainty quantification*).

## Architecture

Quatre composants hétérogènes, composés via [Tesseract](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/) :

| Composant | Rôle | Langage / différentiation |
|---|---|---|
| `fire_spread_torch` | Propagation du feu (réaction-diffusion-advection) | PyTorch, autodiff |
| `smoke_transport_cpp` | Transport atmosphérique des fumées | C++20/OpenMP, adjoint discret écrit à la main |
| HealthImpact | Exposition et pression sanitaire | JAX, autodiff native |
| ResilienceFinance | Perte attendue, CVaR, allocation de portefeuille | JAX/Python |

L'orchestrateur (`app/climacare`) appelle les deux premiers via Tesseract-JAX et porte l'inférence bayésienne (MAP, Laplace, NUTS) et l'optimisation robuste.

## Prérequis

- Python ≥ 3.10
- [Docker](https://docs.docker.com/get-docker/) (build des images Tesseract)
- `pip install tesseract-core`
- GNU Make

## Installation et build

```bash
git clone https://github.com/lucasbertin06/ClimatCareRisk.git
cd ClimatCareRisk
pip install -e app -e components/shared_code

make build-c0     # build + tag les images fire_spread_torch et smoke_transport_cpp
make smoke-kernel # compile le kernel C++ (cmake) hors image, pour les tests locaux
```

## Reproduire les résultats

```bash
make tiny-direct    # E1 : simulation directe feu -> fumée (cas Tiny)
make tiny-gradient  # E2 : validation du gradient de bout en bout
make tiny-map       # E3 : problème inverse (MAP)
make test           # suite de tests complète
```

Portefeuille de résilience et frontière efficiente (E5/E6) :

```bash
python scripts/run_portfolio_experiment.py
```

Les résultats sont écrits dans `results/`.

## Structure du dépôt

app/climacare/ # orchestrateur : pipeline, inférence, UQ, finance, CLI  
components/tesseracts/ # composants Tesseract (fire_spread_torch, smoke_transport_cpp)  
components/shared_code/ # code Python partagé entre composants et orchestrateur  
src/ # génération de scénarios, modèle de perte, modèle sanitaire  
scripts/ # expériences (portefeuille, benchmarks)  
tests/ # suite de tests (pytest)  
docs/ # spécification mathématique  
results/ # sorties des expériences  
configs/ # configurations de scénarios (ex. tiny.yaml)

## Licence

Apache License 2.0 — voir [LICENSE](LICENSE).

## Statut

Prototype de recherche développé pour le Tesseract Hackathon 2026 (4–31 août 2026). Les résultats sanitaires, assurantiels et économiques présentés sont des hypothèses de démonstration : ils ne constituent ni une prévision opérationnelle, ni un avis clinique ou financier.