"""Forecast and intervention metrics without heavyweight dependencies."""

from __future__ import annotations

import jax.numpy as jnp


def brier_score(probability: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Return mean squared probabilistic forecast error."""
    return jnp.mean((probability - target) ** 2)


def binary_nll(probability: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Return numerically stable Bernoulli negative log likelihood."""
    probability = jnp.clip(probability, 1e-6, 1.0 - 1e-6)
    return -jnp.mean(
        target * jnp.log(probability) + (1.0 - target) * jnp.log1p(-probability)
    )


def expected_calibration_error(
    probability: jnp.ndarray, target: jnp.ndarray, bins: int = 10
) -> jnp.ndarray:
    """Estimate calibration error with equally spaced probability bins."""
    probability = probability.reshape(-1)
    target = target.reshape(-1)
    edges = jnp.linspace(0.0, 1.0, bins + 1)
    total = probability.shape[0]
    error = jnp.array(0.0, dtype=probability.dtype)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (probability >= lower) & (
            probability < upper if index < bins - 1 else probability <= upper
        )
        count = jnp.sum(selected)
        weight = count / max(total, 1)
        confidence = jnp.sum(jnp.where(selected, probability, 0.0)) / (count + 1e-6)
        accuracy = jnp.sum(jnp.where(selected, target, 0.0)) / (count + 1e-6)
        error = error + weight * jnp.abs(confidence - accuracy)
    return error


def average_precision(probability: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Compute average precision by sorting scores without sklearn."""
    scores = probability.reshape(-1)
    labels = (target.reshape(-1) > 0.5).astype(jnp.float32)
    order = jnp.argsort(scores)[::-1]
    sorted_labels = labels[order]
    cumulative = jnp.cumsum(sorted_labels)
    ranks = jnp.arange(1, labels.shape[0] + 1, dtype=jnp.float32)
    precision = cumulative / ranks
    positives = jnp.sum(sorted_labels)
    return jnp.sum(jnp.where(sorted_labels > 0, precision, 0.0)) / (positives + 1e-6)


def top_k_recall(
    probability: jnp.ndarray, target: jnp.ndarray, fraction: float = 0.01
) -> jnp.ndarray:
    """Return recall among the highest-risk fraction of grid cells."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    scores = probability.reshape(-1)
    labels = target.reshape(-1) > 0.5
    count = max(1, int(scores.shape[0] * fraction))
    top = jnp.argsort(scores)[-count:]
    true_positives = jnp.sum(labels[top])
    return true_positives / (jnp.sum(labels) + 1e-6)


def choose_threshold(
    probability: jnp.ndarray, target: jnp.ndarray, thresholds: int = 101
) -> jnp.ndarray:
    """Choose a validation threshold maximizing F2, favoring recall."""
    values = jnp.linspace(0.01, 0.99, thresholds)
    labels = target > 0.5
    def score(value: jnp.ndarray) -> jnp.ndarray:
        predicted = probability >= value
        true_positive = jnp.sum(predicted & labels)
        false_positive = jnp.sum(predicted & ~labels)
        false_negative = jnp.sum(~predicted & labels)
        precision = true_positive / (true_positive + false_positive + 1e-6)
        recall = true_positive / (true_positive + false_negative + 1e-6)
        return 5.0 * precision * recall / (4.0 * precision + recall + 1e-6)
    scores = jnp.stack([score(value) for value in values])
    return values[jnp.argmax(scores)]


def threshold_metrics(
    probability: jnp.ndarray, target: jnp.ndarray, threshold: jnp.ndarray
) -> dict[str, float | int]:
    """Report decision metrics at a threshold selected independently."""
    labels = target > 0.5
    predicted = probability >= threshold
    true_positive = jnp.sum(predicted & labels)
    false_positive = jnp.sum(predicted & ~labels)
    false_negative = jnp.sum(~predicted & labels)
    return {
        "threshold": float(threshold),
        "actual_positive_count": int(jnp.sum(labels)),
        "predicted_positive_count": int(jnp.sum(predicted)),
        "recall": float(true_positive / (true_positive + false_negative + 1e-6)),
        "precision": float(true_positive / (true_positive + false_positive + 1e-6)),
        "f2": float(
            5.0 * true_positive
            / (5.0 * true_positive + 4.0 * false_positive + false_negative + 1e-6)
        ),
    }


def intervention_summary(
    baseline_burned: jnp.ndarray,
    planned_burned: jnp.ndarray,
    baseline_exposure: jnp.ndarray,
    planned_exposure: jnp.ndarray,
    cost: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Summarize expected and population-weighted intervention benefit."""
    area_reduction = jnp.mean(baseline_burned - planned_burned)
    exposure_reduction = jnp.mean(baseline_exposure - planned_exposure)
    return {
        "burned_area_reduction": area_reduction,
        "exposed_population_reduction": exposure_reduction,
        "area_reduction_per_cost": area_reduction / (cost + 1e-6),
        "exposure_reduction_per_cost": exposure_reduction / (cost + 1e-6),
    }


def risk_quantiles(values: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Return compact uncertainty summaries over scenario or ensemble axes."""
    return {
        "p10": jnp.quantile(values, 0.1, axis=0),
        "p50": jnp.quantile(values, 0.5, axis=0),
        "p90": jnp.quantile(values, 0.9, axis=0),
        "mean": jnp.mean(values, axis=0),
    }


def evaluate_hazard(
    prediction: dict[str, jnp.ndarray], targets: dict[str, jnp.ndarray]
) -> dict[str, float]:
    """Compute headline calibration metrics for the first forecast horizon."""
    probability = prediction["ignition_probability"][..., 0]
    target = targets["ignition"][..., 0]
    return {
        "brier": float(brier_score(probability, target)),
        "nll": float(binary_nll(probability, target)),
        "ece": float(expected_calibration_error(probability, target)),
        "average_precision": float(average_precision(probability, target)),
        "top_1pct_recall": float(top_k_recall(probability, target, 0.01)),
    }
