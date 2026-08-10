"""Differentiable physical sensor response models."""

from __future__ import annotations

from typing import Any


def sensor_apply_jax(inputs: dict[str, Any]) -> dict[str, Any]:
    """Apply the reference sensor response in JAX."""
    import jax
    import jax.numpy as jnp

    concentration = inputs["concentration"]
    gains = inputs["gains"]
    biases = inputs["biases"]
    time_constants = jnp.maximum(inputs["time_constants"], 1e-3)
    observations = inputs["observations"]
    dt = inputs["dt"]
    noise_scale = jnp.maximum(inputs["noise_scale"], 1e-4)
    alpha = jnp.exp(-dt / time_constants)

    def step(state: jax.Array, sample: jax.Array) -> tuple[jax.Array, jax.Array]:
        filtered = alpha * state + (1.0 - alpha) * sample
        return filtered, filtered

    initial = jnp.zeros(concentration.shape[1], dtype=concentration.dtype)
    _, filtered = jax.lax.scan(step, initial, concentration)
    predicted = gains[None, :] * filtered + biases[None, :]
    residual = (predicted - observations) / noise_scale
    nll = 0.5 * jnp.mean(
        residual**2 + 2.0 * jnp.log(noise_scale) + jnp.log(2.0 * jnp.pi)
    )
    return {"filtered": filtered, "predicted": predicted, "nll": nll}


def sensor_apply_torch(inputs: dict[str, Any]) -> dict[str, Any]:
    """Apply the physical sensor response with PyTorch autograd."""
    import torch

    concentration = inputs["concentration"]
    gains = inputs["gains"]
    biases = inputs["biases"]
    time_constants = torch.clamp(inputs["time_constants"], min=1e-3)
    observations = inputs["observations"]
    dt = inputs["dt"]
    noise_scale = torch.clamp(inputs["noise_scale"], min=1e-4)
    alpha = torch.exp(-dt / time_constants)
    state = torch.zeros_like(concentration[0])
    filtered_values = []
    for sample in concentration:
        state = alpha * state + (1.0 - alpha) * sample
        filtered_values.append(state)
    filtered = torch.stack(filtered_values)
    predicted = gains.unsqueeze(0) * filtered + biases.unsqueeze(0)
    residual = (predicted - observations) / noise_scale
    nll = 0.5 * torch.mean(
        residual**2
        + 2.0 * torch.log(noise_scale)
        + torch.log(torch.tensor(2.0 * torch.pi))
    )
    return {"filtered": filtered, "predicted": predicted, "nll": nll}
