"""Schemas and provenance records for the wildfire data pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataSource:
    """Immutable description of one external data source."""

    name: str
    url: str
    license: str
    version: str
    acquired_at: str | None = None
    sha256: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class GridSpec:
    """Common projected grid used by all wildfire features."""

    crs: str = "EPSG:2154"
    resolution_m: int = 1000
    width: int = 20
    height: int = 20
    x_min_m: float = 0.0
    y_min_m: float = 0.0
    # Optional WGS84 extent for real artifacts. Synthetic fixtures leave this
    # unset so the panel cannot place them on an arbitrary satellite location.
    bbox_lonlat: tuple[float, float, float, float] | None = None
    map_crs: str | None = None
    satellite_url: str | None = None
    satellite_attribution: str = "Institut national de l'information géographique et forestière (IGN)"


@dataclass
class DataManifest:
    """Versioned dataset manifest written beside every generated artifact."""

    dataset_name: str
    created_at: str
    observation_cutoff_time: str
    grid: GridSpec
    sources: list[DataSource] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest."""
        payload = asdict(self)
        payload["grid"] = asdict(self.grid)
        return payload

    def write(self, path: Path) -> None:
        """Write the manifest as formatted JSON."""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class FireEvent:
    """Normalized event record used to construct grid labels."""

    event_id: str
    event_date: date
    latitude: float | None
    longitude: float | None
    burned_area: float
    source: str
    department: str | None = None
    commune: str | None = None
    commune_code: str | None = None


def parse_event_date(value: str) -> date:
    """Parse common ISO and French date representations."""
    value = value.strip()
    for fmt in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
    ):
        try:
            return (
                datetime.strptime(value[:19], fmt).replace(tzinfo=timezone.utc).date()
            )
        except ValueError:
            continue
    raise ValueError(f"unsupported event date: {value!r}")
