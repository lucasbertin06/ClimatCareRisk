"""Build a geographically real wildfire and heat-resilience scenario.

The builder deliberately keeps raw public inputs beside the derived arrays. Every
network response is checksummed and the resulting manifest distinguishes observed,
satellite-derived, modelled, and proxy variables.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import LineString, shape
from shapely.ops import transform as transform_geometry

from wildfire.scenario import WildfireScenario
from wildfire.schema import DataManifest, DataSource, GridSpec
from wildfire_shared.health import heat_health_forward

FEATURE_NAMES = (
    "fuel_proxy",
    "tree_cover",
    "shrub_grass_cover",
    "slope",
    "built_fraction",
    "temperature_max",
    "relative_dryness",
    "wind_speed",
    "precipitation_deficit",
    "soil_dryness",
    "historical_fire_density",
    "road_access",
)

WORLDCOVER_CLASSES = {
    "tree_fraction": 10,
    "shrub_fraction": 20,
    "grass_fraction": 30,
    "cropland_fraction": 40,
    "built_fraction": 50,
    "bare_fraction": 60,
    "water_fraction": 80,
    "wetland_fraction": 90,
}

METROPOLITAN_DEPARTMENTS = tuple(
    [f"{number:02d}" for number in range(1, 20)]
    + ["2A", "2B"]
    + [f"{number:02d}" for number in range(21, 96)]
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_real_config(path: Path) -> dict[str, Any]:
    """Load a regional configuration and resolve its local paths."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("data_dir", "run_dir", "raw_dir", "bdiff_events", "hazard_checkpoint"):
        payload[key] = Path(payload[key])
    payload["bbox_lonlat"] = tuple(float(value) for value in payload["bbox_lonlat"])
    return payload


def _download(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    method: str = "GET",
    data: dict[str, str] | None = None,
    timeout: int = 300,
) -> Path:
    """Download atomically, reusing immutable files already on disk."""
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with session.request(
        method,
        url,
        data=data,
        timeout=timeout,
        stream=True,
        headers={"User-Agent": "IGNIS-Tesseract-Hackathon/1.0"},
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    temporary.replace(destination)
    return destination


def _download_commune_contours(
    session: requests.Session,
    config: dict[str, Any],
    destination: Path,
) -> Path:
    """Fetch commune contours in department-sized requests and merge them.

    geo.api.gouv.fr deliberately rejects an unfiltered national GeoJSON
    request with contours.  Department fan-out is the documented API shape,
    keeps each response bounded, and lets us resume from the merged immutable
    file on subsequent scenario builds.
    """
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    selector = config.get("commune_departments")
    if selector == "all_metropolitan":
        departments = METROPOLITAN_DEPARTMENTS
    else:
        departments = tuple(str(value) for value in (selector or ()))
    template = str(config["sources"]["commune_geojson_template"])
    features: list[dict[str, Any]] = []
    for department in departments:
        url = template.format(department=department)
        response = session.get(
            url,
            timeout=180,
            headers={"User-Agent": "IGNIS-Tesseract-Hackathon/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        features.extend(payload.get("features", []))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _worldcover_tiles(bbox: tuple[float, float, float, float]) -> list[str]:
    west, south, east, north = bbox
    lat_starts = range(math.floor(south / 3) * 3, math.floor((north - 1e-8) / 3) * 3 + 1, 3)
    lon_starts = range(math.floor(west / 3) * 3, math.floor((east - 1e-8) / 3) * 3 + 1, 3)
    return [
        f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
        for lat in lat_starts
        for lon in lon_starts
    ]


def _srtm_tiles(bbox: tuple[float, float, float, float]) -> list[str]:
    west, south, east, north = bbox
    return [
        f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
        for lat in range(math.floor(south), math.floor(north - 1e-8) + 1)
        for lon in range(math.floor(west), math.floor(east - 1e-8) + 1)
    ]


def _overpass_query(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    return (
        "[out:json][timeout:240];("
        f'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track)$"]'
        f"({south},{west},{north},{east});"
        ");out geom;"
    )


def download_real_sources(config: dict[str, Any], *, refresh_weather: bool = False) -> dict[str, Path]:
    """Download the source files required by the configured regional scenario."""
    raw_dir: Path = config["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources = config["sources"]
    session = requests.Session()
    paths: dict[str, Path] = {}
    communes_filename = str(
        config.get("communes_filename", f"communes-{config.get('region_name', 'region')}.geojson")
    )
    communes_destination = raw_dir / communes_filename
    if config.get("commune_departments"):
        paths["communes"] = _download_commune_contours(
            session, config, communes_destination
        )
    else:
        paths["communes"] = _download(
            session, sources["commune_geojson"], communes_destination
        )
    paths["insee"] = _download(
        session, sources["insee_population_age"], raw_dir / "insee-pop1b-2021.zip"
    )
    for tile in _worldcover_tiles(config["bbox_lonlat"]):
        key = f"worldcover_{tile}"
        paths[key] = _download(
            session,
            sources["worldcover_template"].format(tile=tile),
            raw_dir / f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif",
        )
    for tile in _srtm_tiles(config["bbox_lonlat"]):
        compressed = raw_dir / f"{tile}.hgt.gz"
        paths[f"srtm_gz_{tile}"] = _download(
            session,
            sources["srtm_template"].format(lat_band=tile[:3], tile=tile),
            compressed,
        )
        expanded = raw_dir / f"{tile}.hgt"
        if not expanded.exists():
            with gzip.open(compressed, "rb") as source, expanded.open("wb") as target:
                shutil.copyfileobj(source, target)
        paths[f"srtm_{tile}"] = expanded
    roads = raw_dir / str(
        config.get("roads_filename", f"osm-roads-{config.get('region_name', 'region')}.json")
    )
    if config.get("download_roads", True):
        if not roads.exists():
            try:
                _download(
                    session,
                    sources["overpass"],
                    roads,
                    method="POST",
                    data={"data": _overpass_query(config["bbox_lonlat"])},
                    timeout=420,
                )
            except requests.RequestException:
                fallback = "https://overpass-api.de/api/interpreter"
                _download(
                    session,
                    fallback,
                    roads,
                    method="POST",
                    data={"data": _overpass_query(config["bbox_lonlat"])},
                    timeout=420,
                )
        paths["roads"] = roads
    weather = raw_dir / f"open-meteo-{config['forecast_date']}.json"
    if refresh_weather and weather.exists():
        weather.unlink()
    paths["weather"] = _download_weather(session, config, weather)
    return paths


def _sample_coordinates(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    west, south, east, north = config["bbox_lonlat"]
    lons = np.linspace(west, east, int(config["weather_sample_columns"]))
    lats = np.linspace(south, north, int(config["weather_sample_rows"]))
    return lats, lons


def _download_weather(
    session: requests.Session, config: dict[str, Any], destination: Path
) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    lats, lons = _sample_coordinates(config)
    locations = [(lat, lon) for lat in lats for lon in lons]
    forecast_date = date.fromisoformat(config["forecast_date"])
    params = {
        "latitude": ",".join(f"{lat:.5f}" for lat, _ in locations),
        "longitude": ",".join(f"{lon:.5f}" for _, lon in locations),
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,"
            "wind_direction_10m,wind_gusts_10m,soil_moisture_0_to_1cm"
        ),
        "start_date": (forecast_date - timedelta(days=int(config["history_days"]))).isoformat(),
        "end_date": (forecast_date + timedelta(days=3)).isoformat(),
        "timezone": "Europe/Paris",
    }
    response = session.get(
        config["sources"]["open_meteo_forecast"],
        params=params,
        timeout=180,
        headers={"User-Agent": "IGNIS-Tesseract-Hackathon/1.0"},
    )
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(response.json(), separators=(",", ":")) + "\n")
    return destination


def _projected_grid(config: dict[str, Any]) -> tuple[GridSpec, Any]:
    crs = str(config["grid"]["crs"])
    resolution = int(config["grid"]["resolution_m"])
    west, south, east, north = config["bbox_lonlat"]
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corners = [transformer.transform(x, y) for x in (west, east) for y in (south, north)]
    x_min = math.floor(min(point[0] for point in corners) / resolution) * resolution
    y_min = math.floor(min(point[1] for point in corners) / resolution) * resolution
    x_max = math.ceil(max(point[0] for point in corners) / resolution) * resolution
    y_max = math.ceil(max(point[1] for point in corners) / resolution) * resolution
    width = round((x_max - x_min) / resolution)
    height = round((y_max - y_min) / resolution)
    grid = GridSpec(
        crs=crs,
        resolution_m=resolution,
        width=width,
        height=height,
        x_min_m=x_min,
        y_min_m=y_min,
        bbox_lonlat=config["bbox_lonlat"],
        map_crs="EPSG:4326",
        satellite_url=config["sources"]["ign_satellite_tiles"],
        satellite_attribution="Orthophotos IGN Géoplateforme",
    )
    return grid, from_origin(x_min, y_max, resolution, resolution)


def _target_lonlat(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    columns = grid.x_min_m + (np.arange(grid.width) + 0.5) * grid.resolution_m
    rows = grid.y_min_m + (grid.height - np.arange(grid.height) - 0.5) * grid.resolution_m
    xx, yy = np.meshgrid(columns, rows)
    inverse = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    lon, lat = inverse.transform(xx, yy)
    return np.asarray(lon), np.asarray(lat)


def _worldcover_layers(paths: dict[str, Path], grid: GridSpec, transform: Any) -> dict[str, np.ndarray]:
    import rasterio
    from rasterio.windows import from_bounds

    layers = {
        name: np.zeros((grid.height, grid.width), dtype=np.float32)
        for name in WORLDCOVER_CLASSES
    }
    for key, path in paths.items():
        if not key.startswith("worldcover_"):
            continue
        with rasterio.open(path) as source:
            west, south, east, north = grid.bbox_lonlat or source.bounds
            left = max(west, source.bounds.left)
            bottom = max(south, source.bounds.bottom)
            right = min(east, source.bounds.right)
            top = min(north, source.bounds.top)
            if left >= right or bottom >= top:
                continue
            window = from_bounds(left, bottom, right, top, source.transform)
            window = window.round_offsets().round_lengths()
            # National runs use a 10 km analysis grid.  Reading an entire
            # 3-degree 10 m tile would allocate roughly a gigabyte per tile;
            # a 100 m intermediate keeps the source signal while bounding
            # memory and download-time decompression.  The class masks are
            # then area-averaged onto the analysis grid below.
            decimation = max(1, int(grid.resolution_m // 100))
            out_rows = max(1, math.ceil(window.height / decimation))
            out_columns = max(1, math.ceil(window.width / decimation))
            values = source.read(
                1,
                window=window,
                boundless=False,
                out_shape=(out_rows, out_columns),
                resampling=Resampling.nearest,
            )
            source_transform = source.window_transform(window) * source.transform.scale(
                window.width / out_columns, window.height / out_rows
            )
            for name, code in WORLDCOVER_CLASSES.items():
                destination = np.zeros((grid.height, grid.width), dtype=np.float32)
                reproject(
                    source=(values == code).astype(np.float32),
                    destination=destination,
                    src_transform=source_transform,
                    src_crs=source.crs,
                    dst_transform=transform,
                    dst_crs=grid.crs,
                    resampling=Resampling.average,
                )
                layers[name] += destination
    for values in layers.values():
        np.clip(values, 0.0, 1.0, out=values)
    return layers


def _terrain_layers(paths: dict[str, Path], grid: GridSpec, transform: Any) -> tuple[np.ndarray, np.ndarray]:
    import rasterio

    elevation_sum = np.zeros((grid.height, grid.width), dtype=np.float64)
    elevation_count = np.zeros((grid.height, grid.width), dtype=np.float64)
    for key, path in paths.items():
        if not key.startswith("srtm_") or key.startswith("srtm_gz_"):
            continue
        with rasterio.open(path) as source:
            destination = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=-32768,
                dst_transform=transform,
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            valid = np.isfinite(destination)
            elevation_sum[valid] += destination[valid]
            elevation_count[valid] += 1.0
    elevation = np.divide(
        elevation_sum,
        elevation_count,
        out=np.zeros_like(elevation_sum),
        where=elevation_count > 0,
    ).astype(np.float32)
    gradient_y, gradient_x = np.gradient(elevation, float(grid.resolution_m))
    slope_degrees = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y)))
    return elevation, np.clip(slope_degrees / 45.0, 0.0, 1.0).astype(np.float32)


def _insee_age(path: Path, department_code: str) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = {}
    with zipfile.ZipFile(path) as archive:
        name = next(item for item in archive.namelist() if item.lower().endswith(".csv"))
        with archive.open(name) as raw, io.TextIOWrapper(raw, encoding="utf-8") as text:
            reader = csv.DictReader(text, delimiter=";")
            for row in reader:
                code = row["CODGEO"]
                is_national = str(department_code).lower() in {"all", "france", "national"}
                if row["NIVGEO"] != "COM" or (
                    not is_national and not code.startswith(department_code)
                ):
                    continue
                try:
                    age = int(row["AGED100"])
                    count = float((row["NB"] or "").replace(",", "."))
                except (TypeError, ValueError):
                    # INSEE's national extract contains suppressed/empty
                    # cells for a small number of commune-age combinations.
                    # They carry no additive population mass and are ignored
                    # rather than poisoning the complete national raster.
                    continue
                values = aggregates.setdefault(code, {"total": 0.0, "age65": 0.0, "age75": 0.0})
                values["total"] += count
                if age >= 65:
                    values["age65"] += count
                if age >= 75:
                    values["age75"] += count
    return aggregates


def _population_layers(
    paths: dict[str, Path],
    config: dict[str, Any],
    grid: GridSpec,
    transform: Any,
    built_fraction: np.ndarray,
    road_access: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, tuple[float, float]]]:
    payload = json.loads(paths["communes"].read_text(encoding="utf-8"))
    age = _insee_age(paths["insee"], str(config["department_code"]))
    project = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform
    features = []
    commune_rows: list[dict[str, Any]] = []
    centers: dict[str, tuple[float, float]] = {}
    for index, feature in enumerate(payload.get("features", []), start=1):
        properties = feature["properties"]
        geometry_wgs = shape(feature["geometry"])
        geometry = transform_geometry(project, geometry_wgs)
        code = str(properties["code"])
        official_population = float(properties.get("population") or age.get(code, {}).get("total", 0.0))
        population_total = age.get(code, {}).get("total", official_population)
        ratio65 = age.get(code, {}).get("age65", 0.0) / max(population_total, 1.0)
        ratio75 = age.get(code, {}).get("age75", 0.0) / max(population_total, 1.0)
        centroid = geometry_wgs.representative_point()
        centers[code] = (float(centroid.x), float(centroid.y))
        features.append((geometry, index))
        commune_rows.append(
            {
                "index": index,
                "code": code,
                "name": str(properties["nom"]),
                "population": official_population,
                "ratio65": ratio65,
                "ratio75": ratio75,
            }
        )
    commune_index = rasterize(
        features,
        out_shape=(grid.height, grid.width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="int32",
    )
    population = np.zeros_like(built_fraction, dtype=np.float32)
    age65 = np.zeros_like(population)
    age75 = np.zeros_like(population)
    weights = 0.025 + 1.8 * built_fraction + 0.22 * road_access
    # At national 10 km resolution many communes share one cell.  A
    # last-feature-wins raster mask would silently discard their populations.
    # Aggregate coarse cells by representative-point assignment in that case;
    # keep the finer regional allocation for pilot grids where cells resolve
    # individual communes.
    if len(commune_rows) > grid.width * grid.height * 2:
        y_max = grid.y_min_m + grid.height * grid.resolution_m
        for item in commune_rows:
            lon, lat = centers[item["code"]]
            x, y = project(lon, lat)
            column = int((x - grid.x_min_m) // grid.resolution_m)
            row = int((y_max - y) // grid.resolution_m)
            if 0 <= row < grid.height and 0 <= column < grid.width:
                population[row, column] += item["population"]
                age65[row, column] += item["population"] * item["ratio65"]
                age75[row, column] += item["population"] * item["ratio75"]
    else:
        for item in commune_rows:
            mask = commune_index == item["index"]
            total_weight = float(weights[mask].sum())
            if total_weight <= 0:
                continue
            allocation = weights[mask] / total_weight
            population[mask] = allocation * item["population"]
            age65[mask] = population[mask] * item["ratio65"]
            age75[mask] = population[mask] * item["ratio75"]
    name_lookup = np.asarray(["outside"] + [item["name"] for item in commune_rows])
    code_lookup = np.asarray([""] + [item["code"] for item in commune_rows])
    names = name_lookup[np.clip(commune_index, 0, len(name_lookup) - 1)]
    codes = code_lookup[np.clip(commune_index, 0, len(code_lookup) - 1)]
    return (
        {
            "population": population,
            "population_65_plus": age65,
            "population_75_plus": age75,
            "dwellings": population / 2.15,
            "commune_index": commune_index,
            "commune_name": names,
            "commune_code": codes,
        },
        commune_rows,
        centers,
    )


def _road_layers(path: Path, grid: GridSpec, transform: Any) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    project = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    geometries = []
    for element in payload.get("elements", []):
        coordinates = [
            project.transform(float(point["lon"]), float(point["lat"]))
            for point in element.get("geometry", [])
        ]
        if len(coordinates) >= 2:
            geometries.append((LineString(coordinates).buffer(55.0), 1.0))
    road_mask = rasterize(
        geometries,
        out_shape=(grid.height, grid.width),
        transform=transform,
        fill=0.0,
        all_touched=True,
        dtype="float32",
    )
    distance_m = distance_transform_edt(road_mask < 0.5) * grid.resolution_m
    access = np.exp(-distance_m / 3500.0).astype(np.float32)
    return road_mask.astype(np.float32), access


def _daily_weather(
    path: Path,
    config: dict[str, Any],
    grid: GridSpec,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    lats, lons = _sample_coordinates(config)
    rows, columns = len(lats), len(lons)
    if len(payload) != rows * columns:
        raise ValueError("Open-Meteo location count does not match the sampling grid")
    times = [datetime.fromisoformat(value) for value in payload[0]["hourly"]["time"]]
    unique_dates = sorted({value.date() for value in times})
    variables = {
        "temperature_max": "temperature_2m",
        "humidity_min": "relative_humidity_2m",
        "wind_max": "wind_speed_10m",
        "precipitation": "precipitation",
        "soil_moisture": "soil_moisture_0_to_1cm",
    }
    samples = {
        name: np.zeros((len(unique_dates), rows, columns), dtype=np.float32)
        for name in variables
    }
    wind_components = []
    for location_index, location in enumerate(payload):
        row, column = divmod(location_index, columns)
        hourly = location["hourly"]
        for day_index, current_date in enumerate(unique_dates):
            indices = [index for index, value in enumerate(times) if value.date() == current_date]
            for output_name, source_name in variables.items():
                values = np.asarray([hourly[source_name][index] for index in indices], dtype=np.float32)
                values = values[np.isfinite(values)]
                if not values.size:
                    value = 0.0
                elif output_name == "temperature_max" or output_name == "wind_max":
                    value = float(values.max())
                elif output_name == "humidity_min":
                    value = float(values.min())
                elif output_name == "precipitation":
                    value = float(values.sum())
                else:
                    value = float(values.mean())
                samples[output_name][day_index, row, column] = value
        forecast_date = date.fromisoformat(config["forecast_date"])
        future = [
            index
            for index, value in enumerate(times)
            if forecast_date <= value.date() <= forecast_date + timedelta(days=3)
        ]
        speed = np.asarray([hourly["wind_speed_10m"][index] for index in future], dtype=np.float32)
        direction = np.radians(
            np.asarray([hourly["wind_direction_10m"][index] for index in future], dtype=np.float32)
        )
        wind_components.append((float(np.mean(speed * np.sin(direction))), float(np.mean(speed * np.cos(direction)))))
    lon_grid, lat_grid = _target_lonlat(grid)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    target_dates = unique_dates[-int(config["history_days"]):]
    date_indices = [unique_dates.index(value) for value in target_dates]
    interpolated: dict[str, np.ndarray] = {}
    for name, values in samples.items():
        layers = []
        for day_index in date_indices:
            interpolator = RegularGridInterpolator(
                (lats, lons), values[day_index], bounds_error=False, fill_value=None
            )
            layers.append(interpolator(points).reshape(grid.height, grid.width))
        interpolated[name] = np.asarray(layers, dtype=np.float32)
    wind_east, wind_north = np.mean(np.asarray(wind_components), axis=0)
    wind = np.asarray([wind_east / 30.0, wind_north / 30.0], dtype=np.float32)
    return interpolated, wind, [value.isoformat() for value in target_dates]


def _fire_density(
    events_path: Path,
    centers: dict[str, tuple[float, float]],
    grid: GridSpec,
) -> tuple[np.ndarray, int]:
    events = json.loads(events_path.read_text(encoding="utf-8"))
    project = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    counts = np.zeros((grid.height, grid.width), dtype=np.float32)
    selected = 0
    y_max = grid.y_min_m + grid.height * grid.resolution_m
    for event in events:
        code = str(event.get("commune_code") or "")
        if code not in centers:
            continue
        lon, lat = centers[code]
        x, y = project.transform(lon, lat)
        column = int((x - grid.x_min_m) // grid.resolution_m)
        row = int((y_max - y) // grid.resolution_m)
        if 0 <= row < grid.height and 0 <= column < grid.width:
            counts[row, column] += 1.0 + math.log1p(float(event.get("burned_area") or 0.0))
            selected += 1
    density = gaussian_filter(counts, sigma=1.25)
    density = np.log1p(density)
    if float(density.max()) > 0:
        density /= float(density.max())
    return density.astype(np.float32), selected


def _expert_hazard_parameters(channels: int, horizons: int = 3) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros((channels, horizons * 4), dtype=np.float32)
    bias = np.zeros((horizons * 4,), dtype=np.float32)
    ignition = np.asarray([0.95, 0.25, 0.20, 0.10, 0.18, 0.85, 0.85, 0.45, 0.35, 0.50, 0.75, 0.18])
    growth = np.asarray([1.10, 0.40, 0.45, 0.35, -0.10, 0.35, 0.55, 0.70, 0.20, 0.45, 0.25, 0.10])
    area = np.asarray([0.55, 0.25, 0.30, 0.20, -0.10, 0.20, 0.30, 0.30, 0.15, 0.20, 0.15, 0.05])
    for horizon in range(horizons):
        weights[:, horizon * 4] = ignition[:channels]
        weights[:, horizon * 4 + 1] = growth[:channels]
        weights[:, horizon * 4 + 2] = area[:channels]
        bias[horizon * 4 : horizon * 4 + 4] = (-7.0, -6.8, -0.8, -0.4)
    return weights, bias


def _scenario_features(
    land: dict[str, np.ndarray],
    slope: np.ndarray,
    weather: dict[str, np.ndarray],
    fire_density: np.ndarray,
    road_access: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shrub_grass = np.clip(land["shrub_fraction"] + land["grass_fraction"], 0.0, 1.0)
    fuel = np.clip(
        0.98 * land["tree_fraction"]
        + 0.82 * land["shrub_fraction"]
        + 0.62 * land["grass_fraction"]
        + 0.34 * land["cropland_fraction"]
        + 0.20 * land["wetland_fraction"]
        + 0.08 * land["bare_fraction"],
        0.0,
        1.0,
    ).astype(np.float32)
    dynamic = []
    for index in range(weather["temperature_max"].shape[0]):
        dynamic.append(
            np.stack(
                [
                    fuel,
                    land["tree_fraction"],
                    shrub_grass,
                    slope,
                    land["built_fraction"],
                    np.clip((weather["temperature_max"][index] - 18.0) / 24.0, 0.0, 1.0),
                    np.clip((70.0 - weather["humidity_min"][index]) / 60.0, 0.0, 1.0),
                    np.clip(weather["wind_max"][index] / 65.0, 0.0, 1.0),
                    np.exp(-weather["precipitation"][index] / 6.0),
                    np.clip((0.38 - weather["soil_moisture"][index]) / 0.33, 0.0, 1.0),
                    fire_density,
                    road_access,
                ],
                axis=-1,
            )
        )
    return np.asarray(dynamic, dtype=np.float32), fuel


def build_real_scenario(
    config_path: Path,
    *,
    refresh_weather: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Download, harmonize, and persist a regional real-data scenario."""
    config = load_real_config(config_path)
    paths = download_real_sources(config, refresh_weather=refresh_weather)
    grid, transform = _projected_grid(config)
    land = _worldcover_layers(paths, grid, transform)
    elevation, slope = _terrain_layers(paths, grid, transform)
    if "roads" in paths and paths["roads"].is_file():
        road_mask, road_access = _road_layers(paths["roads"], grid, transform)
    else:
        # A deliberately explicit national fallback: downloading one Overpass
        # response for the whole country is brittle and needlessly enormous.
        # Built-cover morphology provides a bounded access prior and is marked
        # as a proxy in the manifest; regional runs can opt into OSM roads.
        built = np.asarray(land["built_fraction"], dtype=np.float32)
        road_mask = (built > 0.05).astype(np.float32)
        road_access = np.clip(
            0.25 + 0.75 * gaussian_filter(built, sigma=1.0), 0.0, 1.0
        ).astype(np.float32)
    population, communes, centers = _population_layers(
        paths, config, grid, transform, land["built_fraction"], road_access
    )
    weather, wind, weather_dates = _daily_weather(paths["weather"], config, grid)
    fire_density, event_count = _fire_density(config["bdiff_events"], centers, grid)
    features, fuel = _scenario_features(land, slope, weather, fire_density, road_access)
    checkpoint: Path = config["hazard_checkpoint"]
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as values:
            hazard_weights = np.asarray(values["weights"], dtype=np.float32)
            hazard_bias = np.asarray(values["bias"], dtype=np.float32)
        parameter_source = str(checkpoint)
    else:
        hazard_weights, hazard_bias = _expert_hazard_parameters(features.shape[-1])
        parameter_source = "transparent expert prior; replace with the chronological BDIFF checkpoint"
    baseline_rate = np.asarray(3.0e-5, dtype=np.float32)
    health = heat_health_forward(
        np.asarray(weather["temperature_max"], dtype=np.float32),
        np.asarray(weather["humidity_min"], dtype=np.float32),
        np.asarray(population["population_65_plus"], dtype=np.float32),
        baseline_rate,
    )
    forest_fraction = np.clip(
        land["tree_fraction"] + 0.7 * land["shrub_fraction"] + 0.35 * land["grass_fraction"],
        0.0,
        1.0,
    ).astype(np.float32)
    ecological_cost = np.clip(
        0.75 * land["tree_fraction"]
        + 0.35 * land["shrub_fraction"]
        + 0.15 * land["wetland_fraction"],
        0.0,
        1.0,
    ).astype(np.float32)
    intervention_cost = (
        90.0 * (0.55 + 0.90 * slope + 0.45 * (1.0 - road_access))
    ).astype(np.float32)
    scenario_filename = str(
        config.get("scenario_filename", f"{config.get('region_name', 'region')}-scenario.npz")
    )
    output = config["data_dir"] / scenario_filename
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        ignition=np.zeros((*fuel.shape, 3), dtype=np.float32),
        growth=np.zeros((*fuel.shape, 3), dtype=np.float32),
        burned_area=np.zeros((*fuel.shape, 3), dtype=np.float32),
        fuel=fuel,
        forest_fraction=forest_fraction,
        tree_fraction=land["tree_fraction"],
        shrub_fraction=land["shrub_fraction"],
        grass_fraction=land["grass_fraction"],
        cropland_fraction=land["cropland_fraction"],
        built_fraction=land["built_fraction"],
        water_fraction=land["water_fraction"],
        elevation_m=elevation,
        slope=slope,
        road_mask=road_mask,
        road_access=road_access,
        historical_fire_density=fire_density,
        population=population["population"],
        vulnerable_population=population["population_65_plus"],
        population_75_plus=population["population_75_plus"],
        dwellings=population["dwellings"],
        commune_index=population["commune_index"],
        commune_name=population["commune_name"],
        commune_code=population["commune_code"],
        temperature_history=weather["temperature_max"],
        relative_humidity_history=weather["humidity_min"],
        heat_stress=np.asarray(health["heat_stress"], dtype=np.float32),
        heat_health_burden=np.asarray(health["expected_excess_burden"], dtype=np.float32),
        health_baseline_rate=baseline_rate,
        ecological_cost=ecological_cost,
        intervention_cost=intervention_cost,
        hazard_weights=hazard_weights,
        hazard_bias=hazard_bias,
        horizon_hours=np.asarray(config["horizons_hours"], dtype=np.float32),
        wind=wind,
        cell_area_hectares=np.asarray(
            float(config["grid"]["resolution_m"]) ** 2 / 10000.0,
            dtype=np.float32,
        ),
    )
    raw_sources = []
    for key, path in sorted(paths.items()):
        if key.startswith("srtm_gz_"):
            continue
        raw_sources.append(
            DataSource(
                name=key,
                url=str(config["sources"].get(key.split("_")[0], "see configuration")),
                license="See source-specific licence in metadata",
                version=path.name,
                acquired_at=datetime.now(timezone.utc).isoformat(),
                sha256=sha256_file(path),
            )
        )
    limitations = [
        "BDIFF locations are commune-level catalogue records, not ignition coordinates.",
        "WorldCover classes are land-cover observations, not direct fuel-moisture measurements.",
        "Dwellings are a population-and-built-cover allocation proxy, not cadastral buildings.",
        "Heat burden is an aggregate research index, not an individual medical prediction.",
    ]
    if float(population["population_65_plus"].sum()) <= 0.0:
        limitations.append(
            "The selected INSEE POP1B extract has no age rows for this territory; "
            "65+ heat burden is left at zero until a dedicated age source is supplied."
        )
    manifest = DataManifest(
        dataset_name=f"ignis-{config.get('region_name', 'regional')}-real-scenario",
        created_at=datetime.now(timezone.utc).isoformat(),
        observation_cutoff_time=config["forecast_date"],
        grid=grid,
        sources=raw_sources,
        files={output.name: sha256_file(output)},
        metadata={
            "feature_names": list(FEATURE_NAMES),
            "forecast_date": config["forecast_date"],
            "weather_dates": weather_dates,
            "bdiff_event_count": event_count,
            "commune_count": len(communes),
            "hazard_parameter_source": parameter_source,
            "variable_classes": {
                "observed_catalogue": ["BDIFF events", "INSEE population and age"],
                "satellite_derived": ["WorldCover land cover", "SRTM elevation"],
                "modelled": ["Open-Meteo weather", "differentiable hazard and spread"],
                "proxy": [
                    "fuel proxy",
                    "dwellings",
                    "aggregate heat-health burden",
                    "road access from built-cover morphology"
                    if "roads" not in paths
                    else "road access from OSM geometry",
                ],
            },
            "limitations": limitations,
        },
    )
    manifest_filename = str(
        config.get("manifest_filename", f"{config.get('region_name', 'region')}-manifest.json")
    )
    manifest_path = config["data_dir"] / manifest_filename
    manifest.write(manifest_path)
    metadata = manifest.as_dict()
    metadata["scenario_path"] = str(output)
    metadata["manifest_path"] = str(manifest_path)
    return output, metadata


def load_real_scenario(path: Path) -> tuple[WildfireScenario, dict[str, np.ndarray]]:
    """Load a built real-data NPZ into the shared planner scenario contract."""
    with np.load(path, allow_pickle=False) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    targets = {
        "ignition": arrays["ignition"],
        "growth": arrays["growth"],
        "burned_area": arrays["burned_area"],
    }
    scenario = WildfireScenario(
        features=arrays["features"],
        targets=targets,
        fuel=arrays["fuel"],
        slope=arrays["slope"],
        wind=arrays["wind"],
        population=arrays["population"],
        vulnerable_population=arrays["vulnerable_population"],
        temperature_history=arrays["temperature_history"],
        relative_humidity_history=arrays["relative_humidity_history"],
        heat_stress=arrays["heat_stress"],
        heat_health_burden=arrays["heat_health_burden"],
        health_baseline_rate=arrays["health_baseline_rate"],
        ecological_cost=arrays["ecological_cost"],
        intervention_cost=arrays["intervention_cost"],
        hazard_weights=arrays["hazard_weights"],
        hazard_bias=arrays["hazard_bias"],
        horizon_hours=arrays["horizon_hours"],
        region="France — real open-data scenario",
        crs="EPSG:2154",
        dwelling_density=arrays["dwellings"],
        forest_fraction=arrays["forest_fraction"],
        road_access=arrays["road_access"],
        cell_area_hectares=float(
            arrays.get("cell_area_hectares", np.asarray(400.0)).item()
        ),
    )
    return scenario, arrays
