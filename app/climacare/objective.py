"""Observations, likelihood and MAP objective of specification section 8.

The loss is a rank-zero JAX scalar built from the sensor predictions returned by
the composed pipeline plus the priors on the four physical parameters. Health
and finance stay out of it, exactly as decided in ADR-001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from climacare.config import PARAMETER_NAMES, TinyConfig

__all__ = [
    "PARAMETER_NAMES",
    "Observations",
    "data_misfit",
    "decode_free_parameters",
    "encode_free_parameters",
    "log_prior_penalty",
    "make_observations",
    "map_loss",
]


@dataclass(frozen=True)
class Observations:
    """Fixed synthetic dataset of specification section 5.5."""

    values: np.ndarray  # (N_t, S)
    mask: np.ndarray  # (N_t, S), one where observed
    noise_std: np.ndarray  # (S,)
    clean: np.ndarray  # noise-free predictions at the truth
    seed: int

    @property
    def count(self) -> int:
        """Return the number of unmasked observations."""
        return int(self.mask.sum())

    def summary(self) -> dict[str, object]:
        """Return a JSON-serialisable dataset summary."""
        return {
            "seed": self.seed,
            "shape": list(self.values.shape),
            "n_observed": self.count,
            "masked_out": int(self.mask.size - self.count),
            "noise_std": self.noise_std.tolist(),
            "max_clean_signal": float(np.max(np.abs(self.clean))),
            "signal_to_noise": float(
                np.max(np.abs(self.clean)) / float(np.min(self.noise_std))
            ),
        }


def make_observations(
    config: TinyConfig, clean_predictions: np.ndarray
) -> Observations:
    """Add the fixed noise realisation and the fixed observation mask.

    The generator is seeded once with the configuration seed, so the dataset and
    the mask are identical for every command and every loss evaluation.
    """
    clean = np.asarray(clean_predictions, dtype=np.float64)
    _n_steps, n_sensors = clean.shape
    if n_sensors != config.sensors.count:
        raise ValueError(
            f"predictions have {n_sensors} sensors, configuration declares "
            f"{config.sensors.count}"
        )
    rng = np.random.default_rng(config.seed)
    noise = rng.normal(size=clean.shape) * config.sensors.noise_std[None, :]
    values = clean + config.sensors.bias[None, :] + noise
    keep = rng.random(clean.shape) >= config.sensors.mask_fraction
    return Observations(
        values=values,
        mask=keep.astype(np.float64),
        noise_std=config.sensors.noise_std.copy(),
        clean=clean,
        seed=config.seed,
    )


def data_misfit(
    predictions: jax.Array, observations: Observations
) -> jax.Array:
    r"""Return :math:`\mathcal L_{data}` of specification section 8."""
    mask = jnp.asarray(observations.mask)
    values = jnp.asarray(observations.values)
    sigma = jnp.asarray(observations.noise_std)[None, :]
    residual = (predictions - values) / sigma
    normalisation = jnp.log(2.0 * math.pi * sigma**2)
    weighted = mask * (0.5 * residual**2 + 0.5 * normalisation)
    return jnp.sum(weighted) / jnp.maximum(jnp.sum(mask), 1.0)


def log_prior_penalty(theta: jax.Array, config: TinyConfig) -> jax.Array:
    r"""Return the prior contribution to :math:`\mathcal L_{MAP}`.

    The uniform priors on :math:`x_0, y_0` are constant inside their bounds and
    therefore contribute nothing to the gradient; the decoding of section 2.2
    keeps the optimiser inside those bounds.
    """
    priors = config.priors
    log_amplitude = theta[2]
    delta_phi = theta[3]
    amplitude_term = (log_amplitude - priors.log_amplitude_mean) ** 2 / (
        2.0 * priors.log_amplitude_std**2
    )
    angle_term = delta_phi**2 / (2.0 * priors.delta_phi_std**2)
    return amplitude_term + angle_term


def map_loss(
    theta: jax.Array,
    predictions: jax.Array,
    observations: Observations,
    config: TinyConfig,
) -> jax.Array:
    r"""Return the scalar :math:`\mathcal L_{MAP}` (negative log-posterior)."""
    return (
        data_misfit(predictions, observations) + log_prior_penalty(theta, config)
    ).reshape(())


# --------------------------------------------------------------------------- #
# Bound-respecting reparameterisation of section 2.2
# --------------------------------------------------------------------------- #
def decode_free_parameters(free: jax.Array, config: TinyConfig) -> jax.Array:
    r"""Map unconstrained :math:`z\in\mathbb R^4` to physical :math:`\theta`."""
    margin = config.priors.margin
    span = 1.0 - 2.0 * margin
    x0 = margin + span * jax.nn.sigmoid(free[0])
    y0 = margin + span * jax.nn.sigmoid(free[1])
    log_amplitude = (
        config.priors.log_amplitude_mean + config.priors.log_amplitude_std * free[2]
    )
    delta_phi = config.wind.delta_phi_max * jnp.tanh(free[3])
    return jnp.stack([x0, y0, log_amplitude, delta_phi])


def encode_free_parameters(theta: np.ndarray, config: TinyConfig) -> np.ndarray:
    """Invert :func:`decode_free_parameters` for a physical parameter vector."""
    margin = config.priors.margin
    span = 1.0 - 2.0 * margin
    values = np.asarray(theta, dtype=np.float64)
    fractions = np.clip((values[:2] - margin) / span, 1e-9, 1.0 - 1e-9)
    logits = np.log(fractions / (1.0 - fractions))
    amplitude = (
        values[2] - config.priors.log_amplitude_mean
    ) / config.priors.log_amplitude_std
    ratio = np.clip(values[3] / config.wind.delta_phi_max, -1.0 + 1e-9, 1.0 - 1e-9)
    return np.array(
        [logits[0], logits[1], amplitude, np.arctanh(ratio)], dtype=np.float64
    )
