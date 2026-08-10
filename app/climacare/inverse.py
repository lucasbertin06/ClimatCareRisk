"""Gradient check and MAP optimisation for the composed pipeline.

Both routines take a served :class:`~climacare.pipeline.TesseractPipeline` and
call the same scalar loss, so nothing here can silently bypass a container.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from climacare.config import PARAMETER_NAMES, TinyConfig
from climacare.objective import (
    Observations,
    decode_free_parameters,
    encode_free_parameters,
    map_loss,
)
from climacare.pipeline import TesseractPipeline

__all__ = [
    "GradientCheck",
    "MapResult",
    "gradient_check",
    "make_physical_loss",
    "run_map",
]


def make_physical_loss(
    pipeline: TesseractPipeline, observations: Observations
) -> Callable[[jax.Array], jax.Array]:
    r"""Return :math:`\theta \mapsto \mathcal L_{MAP}(\theta)`.

    The returned callable contains both ``apply_tesseract`` calls, so a single
    ``jax.value_and_grad`` drives the C++ adjoint and then the PyTorch VJP.
    """

    def loss(theta: jax.Array) -> jax.Array:
        predictions = pipeline.sensor_predictions(theta)
        return map_loss(theta, predictions, observations, pipeline.config)

    return loss


@dataclass
class GradientCheck:
    """Result of the centred finite-difference comparison of section 9."""

    theta: np.ndarray
    loss: float
    tesseract_gradient: np.ndarray
    finite_difference: dict[float, np.ndarray]
    relative_error: dict[float, np.ndarray]
    sign_agreement: dict[float, list[bool]]
    seconds_vjp: float
    seconds_finite_difference: float
    absolute_floor: float
    nominal_factor: float = 1.0

    @property
    def median_relative_error(self) -> float:
        """Return the median relative error at the nominal step."""
        return float(np.median(self.relative_error[self.nominal_factor]))

    @property
    def all_finite(self) -> bool:
        """Return whether every compared gradient value is finite."""
        return bool(
            np.all(np.isfinite(self.tesseract_gradient))
            and all(np.all(np.isfinite(value)) for value in self.finite_difference.values())
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable gradient-check summary."""
        return {
            "parameters": list(PARAMETER_NAMES),
            "theta": self.theta.tolist(),
            "loss": self.loss,
            "tesseract_gradient": self.tesseract_gradient.tolist(),
            "finite_difference": {
                f"{factor:g}": value.tolist()
                for factor, value in self.finite_difference.items()
            },
            "relative_error": {
                f"{factor:g}": value.tolist()
                for factor, value in self.relative_error.items()
            },
            "sign_agreement": {
                f"{factor:g}": value for factor, value in self.sign_agreement.items()
            },
            "median_relative_error": self.median_relative_error,
            "all_finite": self.all_finite,
            "timing_seconds": {
                "tesseract_vjp": self.seconds_vjp,
                "finite_differences": self.seconds_finite_difference,
            },
            "absolute_floor": self.absolute_floor,
        }


def gradient_check(
    pipeline: TesseractPipeline,
    observations: Observations,
    theta: np.ndarray | None = None,
) -> GradientCheck:
    """Compare the composed Tesseract gradient with centred differences.

    Args:
        pipeline: served pipeline holding both containers.
        observations: the fixed synthetic dataset.
        theta: evaluation point, defaulting to the configured starting point.

    Returns:
        The populated :class:`GradientCheck`.
    """
    config = pipeline.config
    settings = config.gradient_check
    point = (
        np.asarray(config.initial_guess, dtype=np.float64)
        if theta is None
        else np.asarray(theta, dtype=np.float64)
    )
    loss_fn = make_physical_loss(pipeline, observations)
    value_and_grad = jax.value_and_grad(loss_fn)

    started = time.perf_counter()
    loss_value, gradient = value_and_grad(jnp.asarray(point))
    seconds_vjp = time.perf_counter() - started
    gradient = np.asarray(gradient, dtype=np.float64)

    finite_difference: dict[float, np.ndarray] = {}
    relative_error: dict[float, np.ndarray] = {}
    sign_agreement: dict[float, list[bool]] = {}
    started = time.perf_counter()
    for factor in settings.step_factors:
        estimate = np.empty_like(point)
        for index in range(point.size):
            step = factor * settings.epsilon * max(1.0, abs(point[index]))
            forward = np.array(point)
            backward = np.array(point)
            forward[index] += step
            backward[index] -= step
            estimate[index] = float(
                loss_fn(jnp.asarray(forward)) - loss_fn(jnp.asarray(backward))
            ) / (2.0 * step)
        finite_difference[factor] = estimate
        denominator = np.maximum(1e-12, np.abs(gradient) + np.abs(estimate))
        relative_error[factor] = np.abs(gradient - estimate) / denominator
        sign_agreement[factor] = [
            bool(
                gradient[index] * estimate[index] > 0.0
                or (
                    abs(gradient[index]) < settings.absolute_floor
                    and abs(estimate[index]) < settings.absolute_floor
                )
            )
            for index in range(point.size)
        ]
    seconds_finite_difference = time.perf_counter() - started

    return GradientCheck(
        theta=point,
        loss=float(loss_value),
        tesseract_gradient=gradient,
        finite_difference=finite_difference,
        relative_error=relative_error,
        sign_agreement=sign_agreement,
        seconds_vjp=seconds_vjp,
        seconds_finite_difference=seconds_finite_difference,
        absolute_floor=settings.absolute_floor,
    )


@dataclass
class MapResult:
    """Trace and outcome of the MAP optimisation of specification section 8."""

    config: TinyConfig
    truth: np.ndarray
    initial: np.ndarray
    estimate: np.ndarray
    loss_history: list[float]
    position_error_history: list[float]
    parameter_history: list[list[float]]
    gradient_norm_history: list[float]
    seconds: float
    iterations: int = field(default=0)
    optimizer: str = field(default="lbfgs")
    loss_evaluations: int = field(default=0)

    @property
    def initial_position_error(self) -> float:
        """Return the ignition-position error before optimisation."""
        return self.position_error_history[0]

    @property
    def final_position_error(self) -> float:
        """Return the ignition-position error after optimisation."""
        return self.position_error_history[-1]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable MAP result summary."""
        return {
            "parameters": list(PARAMETER_NAMES),
            "optimizer": self.optimizer,
            "iterations": self.iterations,
            "loss_evaluations": self.loss_evaluations,
            "truth": dict(zip(PARAMETER_NAMES, self.truth.tolist(), strict=True)),
            "initial": dict(zip(PARAMETER_NAMES, self.initial.tolist(), strict=True)),
            "map_estimate": dict(
                zip(PARAMETER_NAMES, self.estimate.tolist(), strict=True)
            ),
            "absolute_error": dict(
                zip(
                    PARAMETER_NAMES,
                    np.abs(self.estimate - self.truth).tolist(),
                    strict=True,
                )
            ),
            "loss_history": self.loss_history,
            "loss_initial": self.loss_history[0],
            "loss_final": self.loss_history[-1],
            "loss_reduction": self.loss_history[0] - self.loss_history[-1],
            "gradient_norm_history": self.gradient_norm_history,
            "position_error_history": self.position_error_history,
            "position_error_initial": self.initial_position_error,
            "position_error_final": self.final_position_error,
            "parameter_history": self.parameter_history,
            "seconds": self.seconds,
        }


def _free_loss_and_grad(
    pipeline: TesseractPipeline, observations: Observations
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """Return ``z -> (loss, dloss/dz)`` for the bound-respecting coordinates."""
    physical_loss = make_physical_loss(pipeline, observations)
    config = pipeline.config

    def free_loss(free: jax.Array) -> jax.Array:
        return physical_loss(decode_free_parameters(free, config))

    return jax.value_and_grad(free_loss)


def run_map(
    pipeline: TesseractPipeline,
    observations: Observations,
    iterations: int | None = None,
    learning_rate: float | None = None,
    optimizer: str | None = None,
) -> MapResult:
    """Minimise the MAP loss through both containers.

    Args:
        pipeline: served pipeline holding both Tesseracts.
        observations: fixed synthetic dataset.
        iterations: optimiser iterations, defaulting to the configuration.
        learning_rate: Adam step size, ignored by ``lbfgs``.
        optimizer: ``"lbfgs"`` (default) or ``"adam"``.

    Returns:
        The populated :class:`MapResult`.

    Raises:
        ValueError: for an unknown optimiser name.

    The optimised objective is the negative log-posterior in physical
    coordinates; the sigmoid and tanh decoding of section 2.2 only enforces the
    bounds and adds no change-of-variable Jacobian term.
    """
    config = pipeline.config
    steps = config.iterations if iterations is None else int(iterations)
    name = (config.optimizer if optimizer is None else optimizer).lower()
    if name not in ("lbfgs", "adam"):
        raise ValueError(f"unknown MAP optimizer {name!r}, expected lbfgs or adam")

    value_and_grad = _free_loss_and_grad(pipeline, observations)
    free0 = encode_free_parameters(config.initial_guess, config)
    truth = np.asarray(config.truth, dtype=np.float64)

    trace = _Trace(config=config, truth=truth)
    started = time.perf_counter()
    if name == "lbfgs":
        final_free = _run_lbfgs(value_and_grad, free0, steps, trace)
    else:
        rate = config.learning_rate if learning_rate is None else float(learning_rate)
        final_free = _run_adam(value_and_grad, free0, steps, rate, trace)
    seconds = time.perf_counter() - started
    del final_free

    best = int(np.argmin(trace.loss))
    return MapResult(
        config=config,
        truth=truth,
        initial=np.asarray(config.initial_guess, dtype=np.float64),
        estimate=np.asarray(trace.parameters[best], dtype=np.float64),
        loss_history=trace.loss,
        position_error_history=trace.position_error,
        parameter_history=trace.parameters,
        gradient_norm_history=trace.gradient_norm,
        seconds=seconds,
        iterations=steps,
        optimizer=name,
        loss_evaluations=trace.evaluations,
    )


@dataclass
class _Trace:
    """Iteration trace shared by both optimisers."""

    config: TinyConfig
    truth: np.ndarray
    loss: list[float] = field(default_factory=list)
    position_error: list[float] = field(default_factory=list)
    parameters: list[list[float]] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    evaluations: int = 0

    def record(self, free: np.ndarray, loss: float, gradient: np.ndarray) -> None:
        theta = np.asarray(
            decode_free_parameters(jnp.asarray(free), self.config), dtype=np.float64
        )
        self.loss.append(float(loss))
        self.parameters.append(theta.tolist())
        self.gradient_norm.append(float(np.linalg.norm(gradient)))
        self.position_error.append(float(np.linalg.norm(theta[:2] - self.truth[:2])))


def _run_lbfgs(
    value_and_grad: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    free0: np.ndarray,
    steps: int,
    trace: _Trace,
) -> np.ndarray:
    """Run bounded-memory quasi-Newton descent on the unconstrained parameters."""
    from scipy.optimize import minimize

    cache: dict[str, Any] = {"free": None, "loss": None, "gradient": None}

    def objective(free: np.ndarray) -> tuple[float, np.ndarray]:
        loss, gradient = value_and_grad(jnp.asarray(free))
        trace.evaluations += 1
        cache["free"] = np.array(free, dtype=np.float64)
        cache["loss"] = float(loss)
        cache["gradient"] = np.asarray(gradient, dtype=np.float64)
        return cache["loss"], cache["gradient"]

    def callback(free: np.ndarray) -> None:
        # scipy calls this once per accepted iterate; the cached values belong to
        # the last evaluation, which is that iterate for L-BFGS-B.
        trace.record(np.asarray(free, dtype=np.float64), cache["loss"], cache["gradient"])

    loss0, gradient0 = objective(np.asarray(free0, dtype=np.float64))
    trace.record(np.asarray(free0, dtype=np.float64), loss0, gradient0)
    result = minimize(
        objective,
        np.asarray(free0, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": steps, "maxfun": 4 * steps, "ftol": 1e-12, "gtol": 1e-10},
    )
    final = np.asarray(result.x, dtype=np.float64)
    if not np.allclose(final, trace.parameters[-1], atol=0.0):
        loss, gradient = objective(final)
        trace.record(final, loss, gradient)
    return final


def _run_adam(
    value_and_grad: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    free0: np.ndarray,
    steps: int,
    rate: float,
    trace: _Trace,
) -> np.ndarray:
    """Run plain Adam on the unconstrained parameters."""
    free = jnp.asarray(free0)
    optimiser = optax.adam(rate)
    state = optimiser.init(free)
    for _ in range(steps + 1):
        loss, gradient = value_and_grad(free)
        trace.evaluations += 1
        trace.record(
            np.asarray(free, dtype=np.float64),
            float(loss),
            np.asarray(gradient, dtype=np.float64),
        )
        updates, state = optimiser.update(gradient, state)
        free = optax.apply_updates(free, updates)
    return np.asarray(free, dtype=np.float64)
