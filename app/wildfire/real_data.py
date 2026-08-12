"""Real-data BDIFF spatialization without pretending commune centroids are ignitions."""

from __future__ import annotations

import hashlib
import json
import ssl
from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from wildfire.schema import DataManifest, FireEvent, GridSpec

DEFAULT_COMMUNE_URL = "https://geo.api.gouv.fr/communes?fields=code,nom,centre&format=json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_commune_centroids(
    destination: Path,
    url: str = DEFAULT_COMMUNE_URL,
    timeout: int = 120,
) -> Path:
    """Download official commune centroids with strict TLS and atomic writes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "tesseract-wildfire-research/0.1"})
    context = ssl.create_default_context()
    with urlopen(request, timeout=timeout, context=context) as response:
        temporary.write_bytes(response.read())
    temporary.replace(destination)
    return destination


def load_commune_centroids(path: Path) -> dict[str, tuple[float, float]]:
    """Load INSEE code -> (longitude, latitude) WGS84 centroid coordinates."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("commune centroid payload must be a JSON list")
    result: dict[str, tuple[float, float]] = {}
    for item in payload:
        if not isinstance(item, dict) or not item.get("code"):
            continue
        centre = item.get("centre")
        coordinates = centre.get("coordinates") if isinstance(centre, dict) else None
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue
        try:
            longitude, latitude = float(coordinates[0]), float(coordinates[1])
        except (TypeError, ValueError):
            continue
        if -180 <= longitude <= 180 and -90 <= latitude <= 90:
            result[str(item["code"]).zfill(5)] = (longitude, latitude)
    if not result:
        raise ValueError("no valid commune centroids found")
    return result


def _project_centroids(
    centroids: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Project WGS84 lon/lat to Lambert-93 using pyproj's explicit axis order."""
    try:
        from pyproj import Transformer
    except ImportError as error:
        raise RuntimeError("pyproj is required for real BDIFF spatialization") from error
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    result = {}
    for code, (longitude, latitude) in centroids.items():
        x, y = transformer.transform(longitude, latitude)
        result[code] = (float(x), float(y))
    return result


def build_grid(
    projected_centroids: dict[str, tuple[float, float]],
    resolution_m: int = 1000,
    padding_cells: int = 2,
) -> GridSpec:
    """Create a compact Lambert-93 grid around the selected pilot communes."""
    if not projected_centroids:
        raise ValueError("cannot build a grid without projected centroids")
    if resolution_m < 100 or padding_cells < 0:
        raise ValueError("invalid grid resolution or padding")
    xs = [value[0] for value in projected_centroids.values()]
    ys = [value[1] for value in projected_centroids.values()]
    x_min = np.floor(min(xs) / resolution_m - padding_cells) * resolution_m
    y_min = np.floor(min(ys) / resolution_m - padding_cells) * resolution_m
    x_max = np.ceil(max(xs) / resolution_m + padding_cells) * resolution_m
    y_max = np.ceil(max(ys) / resolution_m + padding_cells) * resolution_m
    width = round((x_max - x_min) / resolution_m) + 1
    height = round((y_max - y_min) / resolution_m) + 1
    if width * height > 250_000:
        raise ValueError("selected communes produce an unsafe grid; narrow the pilot")
    return GridSpec(
        crs="EPSG:2154",
        resolution_m=resolution_m,
        width=width,
        height=height,
        x_min_m=float(x_min),
        y_min_m=float(y_min),
    )


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def build_temporal_labels(
    events: Iterable[FireEvent],
    projected_centroids: dict[str, tuple[float, float]],
    grid: GridSpec,
    start: date,
    end: date,
    horizons_days: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    """Build leakage-safe daily labels from commune-centroid event proxies.

    A source event is assigned to its commune centroid, not treated as an exact
    ignition location. ``ignition`` means at least one event in the horizon;
    ``growth`` means at least 0.5 ha reported area in the horizon.
    """
    if start > end or not horizons_days or any(value < 1 for value in horizons_days):
        raise ValueError("invalid date range or horizon list")
    days = _date_range(start, end)
    max_horizon = max(horizons_days)
    shape = (len(days), grid.height, grid.width, len(horizons_days))
    event_list = list(events)
    event_count = np.zeros(shape, dtype=np.float32)
    burned_area = np.zeros(shape, dtype=np.float32)
    projected_events = 0
    for event in event_list:
        if not event.commune_code or event.commune_code not in projected_centroids:
            continue
        x_m, y_m = projected_centroids[event.commune_code]
        x = int(np.floor((x_m - grid.x_min_m) / grid.resolution_m))
        y = int(np.floor((y_m - grid.y_min_m) / grid.resolution_m))
        if not (0 <= x < grid.width and 0 <= y < grid.height):
            continue
        projected_events += 1
        for day_index, current_day in enumerate(days):
            delta = (event.event_date - current_day).days
            if not 0 <= delta < max_horizon:
                continue
            for horizon_index, horizon in enumerate(horizons_days):
                if delta < horizon:
                    event_count[day_index, y, x, horizon_index] += 1.0
                    burned_area[day_index, y, x, horizon_index] += max(
                        float(event.burned_area), 0.0
                    )
    return {
        "days": np.asarray([value.isoformat() for value in days]),
        "event_count": event_count,
        "ignition": (event_count > 0).astype(np.float32),
        "growth": (burned_area >= 0.5).astype(np.float32),
        "burned_area": burned_area,
        "projected_event_count": projected_events,
        "input_event_count": len(event_list),
        "max_horizon_days": max_horizon,
    }


def build_bdiff_dataset(
    events: list[FireEvent],
    centroids: dict[str, tuple[float, float]],
    output_dir: Path,
    departments: set[str],
    resolution_m: int = 1000,
    padding_cells: int = 2,
) -> dict[str, Any]:
    """Build and persist a real BDIFF spatial-label artifact."""
    department_aliases = {
        "var": "83",
        "bouches-du-rhône": "13",
        "bouches-du-rhone": "13",
        "hérault": "34",
        "herault": "34",
    }
    normalized_departments = {
        department_aliases.get(str(value).strip().lower(), str(value).strip())
        for value in departments
    }
    selected = [
        event
        for event in events
        if event.department in normalized_departments
        and event.commune_code in centroids
    ]
    if not selected:
        raise ValueError("no BDIFF events matched selected departments and centroids")
    projected = _project_centroids(centroids)
    selected_codes = {event.commune_code for event in selected if event.commune_code}
    selected_projected = {code: projected[code] for code in selected_codes}
    grid = build_grid(selected_projected, resolution_m, padding_cells)
    start = min(event.event_date for event in selected)
    end = max(event.event_date for event in selected)
    labels = build_temporal_labels(selected, selected_projected, grid, start, end)
    active_mask = np.zeros((grid.height, grid.width), dtype=np.float32)
    for x_m, y_m in selected_projected.values():
        x = int(np.floor((x_m - grid.x_min_m) / grid.resolution_m))
        y = int(np.floor((y_m - grid.y_min_m) / grid.resolution_m))
        if 0 <= x < grid.width and 0 <= y < grid.height:
            active_mask[y, x] = 1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "bdiff-spatial-labels.npz",
        days=labels["days"],
        event_count=labels["event_count"],
        ignition=labels["ignition"],
        growth=labels["growth"],
        burned_area=labels["burned_area"],
        active_mask=active_mask,
    )
    (output_dir / "selected-centroids-lambert93.json").write_text(
        json.dumps(
            {
                code: {"x_m": x, "y_m": y}
                for code, (x, y) in selected_projected.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest = DataManifest(
        dataset_name="wildfire-bdiff-spatial-labels",
        created_at=datetime.now(timezone.utc).date().isoformat(),
        observation_cutoff_time=end.isoformat(),
        grid=grid,
        files={
            "bdiff-spatial-labels.npz": _sha256(output_dir / "bdiff-spatial-labels.npz"),
            "selected-centroids-lambert93.json": _sha256(
                output_dir / "selected-centroids-lambert93.json"
            ),
        },
        metadata={
            "source": "BDIFF commune records + geo.api.gouv.fr commune centres",
            "source_license": "Licence Ouverte / Open Licence 2.0",
            "commune_centroid_proxy": True,
            "selected_departments": sorted(normalized_departments),
            "input_event_count": len(selected),
            "projected_event_count": labels["projected_event_count"],
            "unique_commune_count": len(selected_projected),
            "active_cell_count": int(active_mask.sum()),
            "date_start": start.isoformat(),
            "date_end": end.isoformat(),
            "label_horizons_days": [1, 2, 3],
            "notes": "Centroids are spatial proxies; not exact ignition coordinates.",
        },
    )
    manifest.write(output_dir / "bdiff-spatial-manifest.json")
    return manifest.as_dict()
