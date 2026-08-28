"""Configuration for the France wildfire pilot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from wildfire.schema import DataSource, GridSpec


@dataclass(frozen=True)
class WildfireConfig:
    """Reproducible defaults for the first Mediterranean pilot."""

    region_name: str = "france-mediterranean-pilot"
    departments: tuple[str, ...] = ("Var", "Bouches-du-Rhône", "Hérault")
    grid: GridSpec = field(default_factory=GridSpec)
    history_days: int = 14
    horizons_hours: tuple[int, ...] = (24, 48, 72)
    seed: int = 7
    data_dir: Path = Path("artifacts/wildfire-data")
    run_dir: Path = Path("artifacts/wildfire-runs")
    bdiff_url: str = "https://bdiff.agriculture.gouv.fr/incendies/zip"

    def sources(self) -> tuple[DataSource, ...]:
        """Return the public sources used by the pilot manifest."""
        return (
            DataSource(
                name="BDIFF",
                url=self.bdiff_url,
                license="Licence Ouverte / Open Licence 2.0",
                version="data.gouv.fr catalogue, updated 2025-12-13",
                notes="Official French wildfire events; catalogue is commune-level.",
            ),
            DataSource(
                name="Météo-France",
                url="https://portail-api.meteofrance.fr/",
                license="Météo-France public API terms",
                version="current public API",
                notes="Credentials and quotas are external to this repository.",
            ),
            DataSource(
                name="IGN BD Forêt V2",
                url="https://geoservices.ign.fr/telechargement-api",
                license="Licence Ouverte Etalab 2.0",
                version="BD Forêt V2",
                notes="Static forest and vegetation context.",
            ),
            DataSource(
                name="Santé publique France — chaleur",
                url="https://www.santepubliquefrance.fr/climat/fortes-chaleurs-canicule",
                license="Public health surveillance data terms",
                version="current surveillance publications",
                notes="Aggregate heat-health calibration and validation, not individual risk.",
            ),
            DataSource(
                name="INSEE population and age structure",
                url="https://catalogue-donnees.insee.fr/fr/accueil",
                license="INSEE public data terms",
                version="current population statistics",
                notes="Population and vulnerability proxies at an aggregate spatial level.",
            ),
        )


def load_config(path: Path | None = None) -> WildfireConfig:
    """Load a small YAML override file without requiring YAML at import time."""
    if path is None:
        return WildfireConfig()
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load a config file") from error
    values = yaml.safe_load(path.read_text()) or {}
    values = dict(values)
    if "grid" in values:
        values["grid"] = GridSpec(**values["grid"])
    for key in ("data_dir", "run_dir"):
        if key in values:
            values[key] = Path(values[key])
    for key in ("departments", "horizons_hours"):
        if key in values:
            values[key] = tuple(values[key])
    return WildfireConfig(**values)
