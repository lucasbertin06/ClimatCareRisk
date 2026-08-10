"""End-to-end gradient across both container boundaries, specification section 9."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from climacare.inverse import gradient_check, make_physical_loss
from climacare.objective import (
    data_misfit,
    decode_free_parameters,
    encode_free_parameters,
    make_observations,
    map_loss,
)
from climacare.pipeline import TesseractPipeline
from conftest import requires_containers

MEDIAN_TARGET = 5.0e-2


@pytest.fixture(scope="module")
def observations(pipeline: TesseractPipeline) -> object:
    """Return the fixed synthetic dataset built at the truth."""
    clean = np.asarray(pipeline.sensor_predictions(jnp.asarray(pipeline.config.truth)))
    return make_observations(pipeline.config, clean)


@requires_containers
def test_loss_is_a_rank_zero_scalar(
    pipeline: TesseractPipeline, observations: object
) -> None:
    loss = make_physical_loss(pipeline, observations)
    value = loss(jnp.asarray(pipeline.config.initial_guess))
    assert value.shape == ()
    assert value.dtype == jnp.float64
    assert np.isfinite(float(value))


@requires_containers
def test_loss_is_minimal_at_the_truth(
    pipeline: TesseractPipeline, observations: object
) -> None:
    loss = make_physical_loss(pipeline, observations)
    at_truth = float(loss(jnp.asarray(pipeline.config.truth)))
    at_guess = float(loss(jnp.asarray(pipeline.config.initial_guess)))
    assert at_truth < at_guess


@requires_containers
def test_gradient_is_finite_and_nonzero(
    pipeline: TesseractPipeline, observations: object
) -> None:
    loss = make_physical_loss(pipeline, observations)
    gradient = jax.grad(loss)(jnp.asarray(pipeline.config.initial_guess))
    values = np.asarray(gradient)
    assert values.shape == (4,)
    assert np.all(np.isfinite(values))
    assert np.all(np.abs(values) > 0.0), (
        "every physical parameter must receive a non-zero gradient, including the "
        "wind correction shared by the two solvers"
    )


@requires_containers
def test_gradient_matches_centred_differences(
    pipeline: TesseractPipeline, observations: object
) -> None:
    check = gradient_check(pipeline, observations)
    assert check.all_finite
    assert check.median_relative_error < MEDIAN_TARGET
    assert all(check.sign_agreement[1.0])
    # Step sensitivity: the comparison must not depend on the finite-difference
    # step, which would signal a truncation or cancellation artefact.
    for factor in pipeline.config.gradient_check.step_factors:
        assert float(np.median(check.relative_error[factor])) < MEDIAN_TARGET


@requires_containers
def test_wind_gradient_sums_both_paths(
    pipeline: TesseractPipeline, observations: object
) -> None:
    r"""The :math:`\delta_\phi` derivative must combine the fire and smoke paths.

    Freezing the smoke wind at the evaluation point removes the direct transport
    contribution; the remaining derivative therefore has to differ from the
    fully coupled one.
    """
    config = pipeline.config
    theta0 = jnp.asarray(config.initial_guess)
    frozen_wind = config.wind.smoke_speed * jnp.asarray(
        [
            np.cos(config.wind.phi_base + float(config.initial_guess[3])),
            np.sin(config.wind.phi_base + float(config.initial_guess[3])),
        ]
    )

    def partial_loss(theta: jax.Array) -> jax.Array:
        from tesseract_jax import apply_tesseract

        fire = apply_tesseract(pipeline.fire_client, pipeline.fire_inputs(theta, 1))
        inputs = pipeline.smoke_inputs(theta, fire["smoke_source"], 1)
        inputs["wind"] = frozen_wind
        smoke = apply_tesseract(pipeline.smoke_client, inputs)
        return map_loss(theta, smoke["sensor_concentration"], observations, config)

    full = float(jax.grad(make_physical_loss(pipeline, observations))(theta0)[3])
    fire_only = float(jax.grad(partial_loss)(theta0)[3])
    assert np.isfinite(fire_only)
    assert abs(full - fire_only) > 1e-6 * max(1.0, abs(full))


@requires_containers
def test_chain_rule_holds_through_the_reparameterisation(
    pipeline: TesseractPipeline, observations: object
) -> None:
    config = pipeline.config
    physical = make_physical_loss(pipeline, observations)
    free0 = jnp.asarray(encode_free_parameters(config.initial_guess, config))

    def free_loss(free: jax.Array) -> jax.Array:
        return physical(decode_free_parameters(free, config))

    grad_free = np.asarray(jax.grad(free_loss)(free0))
    grad_theta = np.asarray(jax.grad(physical)(decode_free_parameters(free0, config)))
    jacobian = np.asarray(jax.jacobian(decode_free_parameters)(free0, config))
    assert np.allclose(grad_free, jacobian.T @ grad_theta, rtol=1e-9, atol=1e-12)


@requires_containers
def test_decoding_recovers_the_physical_parameters(pipeline: TesseractPipeline) -> None:
    config = pipeline.config
    for vector in (config.truth, config.initial_guess):
        free = encode_free_parameters(vector, config)
        recovered = np.asarray(decode_free_parameters(jnp.asarray(free), config))
        assert np.allclose(recovered, vector, atol=1e-9)


def test_data_misfit_is_zero_free_of_noise(tiny_config: object) -> None:
    """A perfect fit leaves only the constant normalisation term."""
    clean = np.full((4, tiny_config.sensors.count), 0.5)
    observations = make_observations(tiny_config, clean)
    perfect = jnp.asarray(observations.values)
    residual_free = float(data_misfit(perfect, observations))
    expected = float(
        0.5 * np.log(2 * np.pi * tiny_config.sensors.noise_std[0] ** 2)
    )
    assert abs(residual_free - expected) < 1e-12
