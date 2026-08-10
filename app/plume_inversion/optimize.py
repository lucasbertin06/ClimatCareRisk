"""Gradient-based source inversion."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from plume_inversion.objective import loss_hybrid
from plume_inversion.scenario import Scenario

Loss = Callable[[jax.Array, Scenario], jax.Array]


def initial_parameters(sensor_count: int) -> jax.Array:
    """Return a deliberately imperfect source and calibration guess."""
    return jnp.concatenate(
        [
            jnp.array([-0.35, 0.1, -0.2, 0.0], dtype=jnp.float32),
            jnp.zeros(sensor_count),
        ]
    )


def physical_parameters(raw: jax.Array) -> jax.Array:
    """Convert raw optimizer coordinates to source position, time, and rate."""
    return jnp.concatenate(
        [
            jax.nn.sigmoid(raw[:2]),
            jnp.array(
                [0.35 * jax.nn.sigmoid(raw[2]), jnp.exp(raw[3])],
                dtype=raw.dtype,
            ),
        ]
    )


def recover(
    scenario: Scenario,
    steps: int = 80,
    learning_rate: float = 0.04,
    loss_fn: Loss = loss_hybrid,
) -> dict[str, jax.Array]:
    """Optimize transformed source parameters through the composed pipeline."""
    parameters = initial_parameters(scenario.sensor_positions.shape[0])
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(parameters)

    def step(
        values: tuple[jax.Array, optax.OptState], _: int
    ) -> tuple[tuple[jax.Array, optax.OptState], jax.Array]:
        raw, opt_state = values
        loss, gradient = jax.value_and_grad(loss_fn)(raw, scenario)
        updates, next_state = optimizer.update(gradient, opt_state, raw)
        return (optax.apply_updates(raw, updates), next_state), loss

    (recovered, _), losses = jax.lax.scan(step, (parameters, state), jnp.arange(steps))
    return {"parameters": recovered, "losses": losses, "initial": parameters}
