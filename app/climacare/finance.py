r"""Physical loss, parametric insurance, smoothed CVaR and exact budgets.

Specification sections 11 to 13. Native JAX, downstream of the pipeline and
outside the MAP likelihood. Four properties are enforced by construction:

* insurance never appears in any equation of :math:`T`, :math:`F`, :math:`S`,
  :math:`c` or :math:`\Delta H`, so it cannot reduce a physical impact;
* the CVaR uses the Rockafellar-Uryasev form with a smoothed hinge and an
  explicit :math:`\zeta`; no scenario is ever sorted inside the gradient loop;
* the budget is satisfied exactly through a softmax, not by a penalty;
* the liquidity requirement is a smooth positive part of the uninsured loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "FinanceParams",
    "budget_allocation",
    "conditional_value_at_risk",
    "insurance_payout",
    "liquidity_requirement",
    "net_loss",
    "physical_loss",
    "robust_objective",
    "smooth_plus",
]


@dataclass(frozen=True)
class FinanceParams:
    """Synthetic financial coefficients."""

    health_cost: float = 1.0  # c_H
    burn_cost: float = 1.0  # c_B
    interruption_cost: float = 1.0  # c_D
    coverage: float = 1.0  # C_cover
    trigger: float = 0.5  # tau_trigger
    trigger_width: float = 0.05  # eps_trigger
    premium_rate: float = 0.12  # pi
    reserve: float = 0.2  # Reserve
    smoothing: float = 0.05  # tau
    alpha: float = 0.9
    risk_weight: float = 1.0  # gamma
    equity_weight: float = 0.0  # mu

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")
        if self.smoothing <= 0.0:
            raise ValueError(f"smoothing tau must be positive, got {self.smoothing}")
        if self.trigger_width <= 0.0:
            raise ValueError("trigger width must be positive")


def smooth_plus(value: jax.Array, tau: float) -> jax.Array:
    r"""Return :math:`\tau\log(1+e^{x/\tau})`, evaluated stably."""
    scaled = value / tau
    return tau * (jnp.maximum(scaled, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(scaled))))


def physical_loss(
    health_impacts: jax.Array,
    burned_area: jax.Array,
    interruption: jax.Array | float,
    params: FinanceParams,
) -> jax.Array:
    r"""Return :math:`L_{phys}` of section 11."""
    return (
        params.health_cost * jnp.sum(health_impacts)
        + params.burn_cost * burned_area
        + params.interruption_cost * jnp.asarray(interruption)
    )


def insurance_payout(
    event_index: jax.Array, insurance_level: jax.Array | float, params: FinanceParams
) -> jax.Array:
    r"""Return the smoothed parametric payout of section 11."""
    trigger = jax.nn.sigmoid(
        (event_index - params.trigger) / params.trigger_width
    )
    return jnp.asarray(insurance_level) * params.coverage * trigger


def premium(insurance_level: jax.Array | float, params: FinanceParams) -> jax.Array:
    r"""Return :math:`\mathrm{Premium} = \pi u C_{cover}`."""
    return params.premium_rate * jnp.asarray(insurance_level) * params.coverage


def net_loss(
    loss_physical: jax.Array,
    event_index: jax.Array,
    insurance_level: jax.Array | float,
    prevention_cost: jax.Array | float,
    filtering_cost: jax.Array | float,
    params: FinanceParams,
) -> jax.Array:
    r"""Return :math:`L_{net}` of section 11."""
    return (
        loss_physical
        - insurance_payout(event_index, insurance_level, params)
        + premium(insurance_level, params)
        + jnp.asarray(prevention_cost)
        + jnp.asarray(filtering_cost)
    )


def liquidity_requirement(
    loss_physical: jax.Array,
    event_index: jax.Array,
    insurance_level: jax.Array | float,
    params: FinanceParams,
) -> jax.Array:
    r"""Return :math:`L_{liquidity}` of section 11."""
    uncovered = (
        loss_physical
        - insurance_payout(event_index, insurance_level, params)
        - params.reserve
    )
    return smooth_plus(uncovered, params.smoothing)


def conditional_value_at_risk(
    losses: jax.Array, zeta: jax.Array | float, params: FinanceParams
) -> jax.Array:
    r"""Return the smoothed Rockafellar-Uryasev CVaR estimator of section 12.

    No sort is used, so the estimator stays differentiable under any permutation
    of the scenarios.
    """
    tail = smooth_plus(losses - jnp.asarray(zeta), params.smoothing)
    scale = (1.0 - params.alpha) * losses.shape[0]
    return jnp.asarray(zeta) + jnp.sum(tail) / scale


def optimal_cvar(losses: jax.Array, params: FinanceParams, steps: int = 200) -> jax.Array:
    r"""Minimise the CVaR estimator over :math:`\zeta` by bisection-free descent.

    The estimator is convex in :math:`\zeta`; a short gradient descent from the
    empirical mean is enough for the C0 invariant tests.
    """
    zeta = jnp.mean(losses)
    grad_fn = jax.grad(lambda value: conditional_value_at_risk(losses, value, params))
    scale = jnp.maximum(jnp.std(losses), params.smoothing)
    for _ in range(steps):
        zeta = zeta - 0.5 * scale * grad_fn(zeta)
    return conditional_value_at_risk(losses, zeta, params)


def budget_allocation(
    free: jax.Array, total_budget: float, scales: jax.Array
) -> tuple[jax.Array, jax.Array]:
    r"""Return ``(budget, intensity)`` of section 13.

    The softmax guarantees :math:`\sum_i \mathrm{budget}_i = B` to machine
    precision and :math:`0 \le u_i < 1`.
    """
    shares = jax.nn.softmax(free)
    budget = total_budget * shares
    intensity = 1.0 - jnp.exp(-budget / scales)
    return budget, intensity


def robust_objective(
    losses_physical: jax.Array,
    losses_net: jax.Array,
    investment_cost: jax.Array | float,
    zeta: jax.Array | float,
    params: FinanceParams,
    equity: jax.Array | float = 0.0,
) -> jax.Array:
    r"""Return :math:`J(u, \zeta)` of section 12.

    Only the function and its invariants are required in C0; the full portfolio
    optimisation is explicitly out of scope.
    """
    expected = jnp.mean(losses_physical) + jnp.asarray(investment_cost)
    risk = conditional_value_at_risk(losses_net, zeta, params)
    return expected + params.risk_weight * risk + params.equity_weight * jnp.asarray(equity)
