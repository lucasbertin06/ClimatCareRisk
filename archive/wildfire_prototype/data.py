"""Public-data ingestion and deterministic wildfire fixtures."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import jax.numpy as jnp

try:
    import certifi
except ImportError:
    certifi = None
import numpy as np

from wildfire.config import WildfireConfig
from wildfire.scenario import WildfireScenario, make_scenario
from wildfire.schema import DataManifest, FireEvent, parse_event_date

BDIFF_INTERMEDIATE_SHA256 = (
    "cdc78c3185ce918c8e87f9b2559197d641288e564c5a8b789cd796abdea298d4"
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ssl_context(url: str):
    """Build a strict CA context, repairing the known BDIFF chain omission."""
    import ssl

    if certifi is None:
        return ssl.create_default_context()
    ca_file = certifi.where()
    if urlsplit(url).hostname == "bdiff.agriculture.gouv.fr":
        intermediate = (
            Path(__file__).parents[2] / "certs" / "bdiff-harica-intermediate.pem"
        )
        if (
            intermediate.exists()
            and sha256_file(intermediate) == BDIFF_INTERMEDIATE_SHA256
        ):
            context = ssl.create_default_context(cafile=ca_file)
            context.load_verify_locations(cadata=intermediate.read_text())
            return context
    return ssl.create_default_context(cafile=ca_file)


def download_file(url: str, destination: Path, timeout: int = 60) -> Path:
    """Download one public resource atomically with strict TLS validation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "tesseract-wildfire-research/0.1"})
    with (
        urlopen(request, timeout=timeout, context=_ssl_context(url)) as response,
        temporary.open("wb") as handle,
    ):
        while block := response.read(1024 * 1024):
            handle.write(block)
    temporary.replace(destination)
    return destination


def extract_bdiff_csv(archive: Path, destination: Path) -> Path:
    """Extract the official CSV from a BDIFF ZIP export."""
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"BDIFF response is not a ZIP archive: {archive}")
    with zipfile.ZipFile(archive) as handle:
        members = [
            member
            for member in handle.namelist()
            if member.lower().endswith(".csv") and not member.startswith("__MACOSX/")
        ]
        if len(members) != 1:
            raise ValueError(f"expected one BDIFF CSV, found {len(members)}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(handle.read(members[0]))
    return destination


def _column(row: dict[str, str], *names: str) -> str | None:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered and lowered[name.lower()] not in (None, ""):
            return lowered[name.lower()]
    return None


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _event_from_row(row: dict[str, str], index: int) -> FireEvent | None:
    date_value = _column(
        row,
        "date",
        "date_incendie",
        "date du feu",
        "date_debut",
        "date de première alerte",
        "alerte",
        "date alerte",
    )
    if date_value is None:
        return None
    try:
        event_date = parse_event_date(date_value)
    except ValueError:
        return None
    latitude = _float(_column(row, "latitude", "lat", "y"))
    longitude = _float(_column(row, "longitude", "lon", "lng", "x"))
    area = _float(
        _column(
            row,
            "surface",
            "surface_brulee",
            "surface brûlée",
            "surface parcourue (m2)",
            "area",
        )
    )
    year = _column(row, "année", "annee")
    number = _column(row, "numéro", "numero")
    return FireEvent(
        event_id=_column(row, "id", "id_feu", "identifiant", "id incendie")
        or f"bdiff-{year or 'unknown'}-{number or index}",
        event_date=event_date,
        latitude=latitude,
        longitude=longitude,
        burned_area=max(area or 0.0, 0.0) / 10_000.0,
        source="BDIFF",
        department=_column(row, "departement", "département", "dept"),
        commune=_column(
            row, "commune", "libcommune", "nom commune", "nom de la commune"
        ),
        commune_code=_column(row, "code insee", "code_insee", "insee"),
    )


def parse_bdiff_csv(path: Path) -> list[FireEvent]:
    """Parse common BDIFF CSV or JSON exports while preserving provenance."""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return []
        return [
            event
            for index, row in enumerate(rows)
            if isinstance(row, dict)
            for event in [_event_from_row(row, index)]
            if event is not None
        ]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        lines = handle.readlines()
    header_index = next(
        index
        for index, line in enumerate(lines)
        if (
            "critères de sélection" not in line.lower()
            and (
                line.lower().startswith("année;")
                or line.lower().startswith("annee;")
                or "date alerte" in line.lower()
            )
        )
    )
    sample = "".join(lines[header_index : header_index + 3])
    dialect = csv.Sniffer().sniff(sample, delimiters=";,\t,")
    reader = csv.DictReader(lines[header_index:], dialect=dialect)
    return [
        event
        for index, row in enumerate(reader)
        for event in [_event_from_row(row, index)]
        if event is not None
    ]


def build_fixture(config: WildfireConfig, destination: Path | None = None) -> Path:
    """Write a compact NPZ fixture and provenance manifest."""
    scenario = make_scenario(
        seed=config.seed,
        history=config.history_days,
        size=config.grid.height,
        horizons=len(config.horizons_hours),
    )
    destination = destination or config.data_dir / "fixture.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=np.asarray(scenario.features),
        ignition=np.asarray(scenario.targets["ignition"]),
        growth=np.asarray(scenario.targets["growth"]),
        burned_area=np.asarray(scenario.targets["burned_area"]),
        fuel=np.asarray(scenario.fuel),
        slope=np.asarray(scenario.slope),
        wind=np.asarray(scenario.wind),
        population=np.asarray(scenario.population),
        vulnerable_population=np.asarray(scenario.vulnerable_population),
        temperature_history=np.asarray(scenario.temperature_history),
        relative_humidity_history=np.asarray(scenario.relative_humidity_history),
        heat_stress=np.asarray(scenario.heat_stress),
        heat_health_burden=np.asarray(scenario.heat_health_burden),
        health_baseline_rate=np.asarray(scenario.health_baseline_rate),
        ecological_cost=np.asarray(scenario.ecological_cost),
        intervention_cost=np.asarray(scenario.intervention_cost),
        hazard_weights=np.asarray(scenario.hazard_weights),
        hazard_bias=np.asarray(scenario.hazard_bias),
    )
    manifest = DataManifest(
        dataset_name="wildfire-synthetic-fixture",
        created_at=datetime.now(timezone.utc).isoformat(),
        observation_cutoff_time=datetime.now(timezone.utc).date().isoformat(),
        grid=config.grid,
        sources=list(config.sources()),
        files={destination.name: sha256_file(destination)},
        metadata={"synthetic": True, "region": scenario.region},
    )
    manifest.write(destination.with_name("manifest.json"))
    return destination


def load_fixture(path: Path) -> WildfireScenario:
    """Load the fixture into the canonical scenario object."""
    with np.load(path) as data:
        return WildfireScenario(
            features=jnp.asarray(data["features"]),
            targets={
                "ignition": jnp.asarray(data["ignition"]),
                "growth": jnp.asarray(data["growth"]),
                "burned_area": jnp.asarray(data["burned_area"]),
            },
            fuel=jnp.asarray(data["fuel"]),
            slope=jnp.asarray(data["slope"]),
            wind=jnp.asarray(data["wind"]),
            population=jnp.asarray(data["population"]),
            vulnerable_population=jnp.asarray(data["vulnerable_population"]),
            temperature_history=jnp.asarray(data["temperature_history"]),
            relative_humidity_history=jnp.asarray(data["relative_humidity_history"]),
            heat_stress=jnp.asarray(data["heat_stress"]),
            heat_health_burden=jnp.asarray(data["heat_health_burden"]),
            health_baseline_rate=jnp.asarray(data["health_baseline_rate"]),
            ecological_cost=jnp.asarray(data["ecological_cost"]),
            intervention_cost=jnp.asarray(data["intervention_cost"]),
            hazard_weights=jnp.asarray(data["hazard_weights"]),
            hazard_bias=jnp.asarray(data["hazard_bias"]),
            horizon_hours=jnp.array([24.0, 48.0, 72.0], dtype=jnp.float32),
            region="France Mediterranean fixture",
            crs="EPSG:2154",
        )


def load_event_json(path: Path) -> list[FireEvent]:
    """Load the repository's normalized event JSON without rescaling areas."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("normalized event JSON must be a list")
    events: list[FireEvent] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        try:
            event_date = parse_event_date(str(row["event_date"]))
            burned_area = max(float(row.get("burned_area") or 0.0), 0.0)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid normalized event row: {row!r}") from error
        events.append(
            FireEvent(
                event_id=str(row.get("event_id", f"normalized-{len(events)}")),
                event_date=event_date,
                latitude=_float(row.get("latitude")),
                longitude=_float(row.get("longitude")),
                burned_area=burned_area,
                source=str(row.get("source", "BDIFF")),
                department=row.get("department"),
                commune=row.get("commune"),
                commune_code=row.get("commune_code"),
            )
        )
    return events


def write_event_json(events: list[FireEvent], path: Path) -> None:
    """Write normalized fire events for downstream feature construction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "event_id": event.event_id,
                    "event_date": event.event_date.isoformat(),
                    "latitude": event.latitude,
                    "longitude": event.longitude,
                    "burned_area": event.burned_area,
                    "source": event.source,
                    "department": event.department,
                    "commune": event.commune,
                    "commune_code": event.commune_code,
                }
                for event in events
            ],
            indent=2,
        )
        + "\n"
    )
