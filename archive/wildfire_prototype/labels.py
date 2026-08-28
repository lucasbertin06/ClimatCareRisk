"""Leakage-safe wildfire target construction and temporal splits."""

from __future__ import annotations

from datetime import date, timedelta

import jax.numpy as jnp

from wildfire.schema import FireEvent, GridSpec


def event_grid_labels(
    events: list[FireEvent],
    days: list[date],
    grid: GridSpec,
    x_min: float = 0.0,
    y_min: float = 0.0,
) -> dict[str, jnp.ndarray]:
    """Rasterize events into daily multi-horizon labels.

    Coordinates must be in the same projected CRS and units as ``grid``. A
    label at day ``d`` only uses events in ``[d, d + horizon)``.
    """
    horizon_days = (1, 2, 3)
    shape = (len(days), grid.height, grid.width, len(horizon_days))
    ignition = jnp.zeros(shape, dtype=jnp.float32)
    growth = jnp.zeros(shape, dtype=jnp.float32)
    burned_area = jnp.zeros(shape, dtype=jnp.float32)
    for event in events:
        if event.latitude is None or event.longitude is None:
            continue
        x = int((event.longitude - x_min) // grid.resolution_m)
        y = int((event.latitude - y_min) // grid.resolution_m)
        if not (0 <= x < grid.width and 0 <= y < grid.height):
            continue
        for day_index, current_day in enumerate(days):
            delta = (event.event_date - current_day).days
            for horizon_index, horizon in enumerate(horizon_days):
                if 0 <= delta < horizon:
                    ignition = ignition.at[day_index, y, x, horizon_index].set(1.0)
                    growth = growth.at[day_index, y, x, horizon_index].set(1.0)
                    burned_area = burned_area.at[day_index, y, x, horizon_index].add(
                        event.burned_area
                    )
    return {"ignition": ignition, "growth": growth, "burned_area": burned_area}


def temporal_split(
    days: list[date],
    validation_start: date,
    test_start: date,
) -> dict[str, list[int]]:
    """Return chronological train/validation/test indices."""
    if validation_start >= test_start:
        raise ValueError("validation must start before test")
    result = {"train": [], "validation": [], "test": []}
    for index, current_day in enumerate(days):
        bucket = (
            "train"
            if current_day < validation_start
            else "validation"
            if current_day < test_start
            else "test"
        )
        result[bucket].append(index)
    return result


def available_context_days(target_day: date, history_days: int) -> list[date]:
    """Return only observations available before a prediction timestamp."""
    if history_days < 1:
        raise ValueError("history_days must be positive")
    return [
        target_day - timedelta(days=offset) for offset in range(history_days, 0, -1)
    ]


def assert_no_future_leakage(context_days: list[date], cutoff: date) -> None:
    """Raise if a feature context contains data after its cutoff."""
    if any(current_day >= cutoff for current_day in context_days):
        raise ValueError("feature context includes data at or after forecast cutoff")
