from __future__ import annotations

import jax
import jax.numpy as jnp

from climacare.finance import smooth_plus, conditional_value_at_risk

__all__ = ["expected_shortfall", "optimal_expected_shortfall", "value_at_risk_smooth"]

def expected_shortfall( # this is a CVaR optimisation based on Rockafellar-Uryasev
    losses : jax.Array,
    zeta : jax.Array | float, # zeta is the treshold found by "optimal_expected_shortfall"
    alpha : float = 0.95, # alpha is the confidence level (Ex : 0.95 -> we look at the 5% worst case scenario)
    tau : float = 10_000.0, ) -> jax.Array: # tau is the Smoothing bandwidth in currency units. Must match the scale of losses
    
    tail = smooth_plus(losses - jnp.asarray(zeta), tau)
    scale = (1.0 - alpha) * losses.shape[0]
    return jnp.asarray(zeta) + jnp.sum(tail) / scale

def optimal_expected_shortfall(
    losses: jax.Array,
    alpha: float = 0.95,
    tau: float = 10_000.0,
    steps: int = 200, ) -> jax.Array:
    
    # Minimizes expected shortfall over zeta (strictly convex optimization in zeta)
    
    zeta = jnp.mean(losses)
    grad_fn = jax.grad(lambda z : expected_shortfall(losses, z, alpha, tau))
    scale = jnp.maximum(jnp.std(losses), tau)
    
    for _ in range(steps):
        zeta = zeta - 0.5 * scale * grad_fn(zeta)
        
    return expected_shortfall(losses, zeta, alpha, tau)