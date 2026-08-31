from __future__ import annotations

import jax
import jax.numpy as jnp

from climacare.finance import smooth_plus

__all__ = ["expected_shortfall", "optimal_expected_shortfall", "value_at_risk_smooth"]


def expected_shortfall(losses, zeta, alpha=0.95, tau=10_000.0):
    tail = smooth_plus(losses - jnp.asarray(zeta), tau)
    scale = (1.0 - alpha) * losses.shape[0]
    return jnp.asarray(zeta) + jnp.sum(tail) / scale


def optimal_zeta(losses, alpha=0.95, tau=10_000.0, steps=10):
    zeta = jax.lax.stop_gradient(jnp.quantile(losses, alpha))
    scale = jax.lax.stop_gradient(jnp.maximum(jnp.std(losses), tau))
    gradient = jax.grad(lambda value: expected_shortfall(losses, value, alpha, tau))

    def body(_, value):
        return value - 0.5 * scale * gradient(value)

    return jax.lax.fori_loop(0, steps, body, zeta)


def optimal_expected_shortfall(losses, alpha=0.95, tau=10_000.0, steps=10):
    return expected_shortfall(losses, optimal_zeta(losses, alpha, tau, steps), alpha, tau)


def value_at_risk_smooth(losses, alpha=0.95, tau=10_000.0, steps=10):
    return optimal_zeta(losses, alpha, tau, steps)
