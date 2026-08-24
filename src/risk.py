# In this code, we only put the function associated to risk calculation (VaR, CVaR...)

from __future__ import annotations

import jax
import jax.numpy as jnp

from climacare.finance import smooth_plus

__all__ = ["expected_shortfall", "optimal_expected_shortfall", "value_at_risk_smooth"]

def expected_shortfall( # this is a CVaR optimisation based on Rockafellar-Uryasev
    losses : jax.Array,
    zeta : jax.Array | float, # zeta is the treshold found by "optimal_expected_shortfall"
    alpha : float = 0.95, # alpha is the confidence level (Ex : 0.95 -> we look at the 5% worst case scenario)
    tau : float = 10_000.0, ) -> jax.Array: # tau is the Smoothing bandwidth in currency units. Must match the scale of losses
    
    tail = smooth_plus(losses - jnp.asarray(zeta), tau)
    scale = (1.0 - alpha) * losses.shape[0]
    return jnp.asarray(zeta) + jnp.sum(tail) / scale

def optimal_zeta( # finds the zeta* who minimize expected_shortfall(losses, zeta).
    losses : jax.Array,
    alpha : float = 0.95,
    tau : float = 10_000.0,
    steps : int = 10, ) -> jax.Array:

    # stop_gradient : jnp.quantile's subgradient is 0/0 (NaN) when many losses
    # are tied (e.g. all-zero after full insurance coverage). Same for std.
    # Both only size the refinement step - no gradient is needed through them;
    # the true CVaR gradient flows through expected_shortfall's smoothed hinge.
    zeta0 = jax.lax.stop_gradient(jnp.quantile(losses, alpha))
    scale = jax.lax.stop_gradient(jnp.maximum(jnp.std(losses), tau))
    grad_fn = jax.grad(lambda z: expected_shortfall(losses, z, alpha, tau))

    def body(_, zeta):
        return zeta - 0.5 * scale * grad_fn(zeta)

    return jax.lax.fori_loop(0, steps, body, zeta0)

def optimal_expected_shortfall(
    losses : jax.Array,
    alpha : float = 0.95,
    tau : float = 10_000.0,
    steps : int = 10, ) -> jax.Array:
    
    # Minimizes expected shortfall over zeta (strictly convex optimization in zeta)
    
    zeta = optimal_zeta(losses, alpha, tau, steps)
    return expected_shortfall(losses, zeta, alpha, tau)

def value_at_risk_smooth(
    losses : jax.Array,
    alpha : float = 0.95,
    tau : float = 10_000.0,
    steps : int = 10, ) -> jax.Array:

    return optimal_zeta(losses, alpha, tau, steps) # The optimal zeta* at convergence is mathematically the smoothed VaR
