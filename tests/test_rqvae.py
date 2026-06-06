"""Unit tests for the RQ-VAE Flax module."""

import jax
import jax.numpy as jnp
import numpy as np
from models.rqvae import RQVAE


def test_rqvae_shapes():
    batch_size = 4
    embedding_dim = 64
    latent_dim = 16
    num_levels = 3
    num_codes = 32

    # Instantiate model
    model = RQVAE(
        latent_dim=latent_dim,
        num_levels=num_levels,
        num_codes=num_codes,
        embedding_dim=embedding_dim,
    )

    # Initialize model variables
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((batch_size, embedding_dim))
    variables = model.init(key, dummy_input)

    # Forward pass
    output = model.apply(variables, dummy_input)

    assert output["x_recon"].shape == (batch_size, embedding_dim)
    assert output["indices"].shape == (batch_size, num_levels)
    assert output["codebook_loss"].shape == ()
    assert output["commitment_loss"].shape == ()
    assert jnp.all(output["indices"] >= 0)
    assert jnp.all(output["indices"] < num_codes)


def test_rqvae_encode_equivalence():
    batch_size = 8
    embedding_dim = 128
    latent_dim = 32
    num_levels = 4
    num_codes = 64

    model = RQVAE(
        latent_dim=latent_dim,
        num_levels=num_levels,
        num_codes=num_codes,
        embedding_dim=embedding_dim,
    )

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, embedding_dim))

    # Initialize variables
    init_key, apply_key = jax.random.split(key)
    variables = model.init(init_key, x)

    # Run forward pass and encode pass
    output = model.apply(variables, x)
    encoded_indices = model.apply(variables, x, method=model.encode)

    # Check equivalence
    np.testing.assert_array_equal(output["indices"], encoded_indices)


def test_rqvae_gradient_flow_and_jit():
    batch_size = 4
    embedding_dim = 32
    latent_dim = 8
    num_levels = 2
    num_codes = 16

    model = RQVAE(
        latent_dim=latent_dim,
        num_levels=num_levels,
        num_codes=num_codes,
        embedding_dim=embedding_dim,
    )

    key = jax.random.PRNGKey(101)
    x = jax.random.normal(key, (batch_size, embedding_dim))

    variables = model.init(key, x)
    params = variables["params"]

    # Define loss function that is differentiable w.r.t parameters
    def loss_fn(p):
        outputs = model.apply({"params": p}, x)
        recon_loss = jnp.mean((outputs["x_recon"] - x) ** 2)
        total_loss = (
            recon_loss
            + outputs["codebook_loss"]
            + model.commitment_weight * outputs["commitment_loss"]
        )
        return total_loss

    # Check JIT compilation safety
    jit_loss_fn = jax.jit(loss_fn)
    loss_val = jit_loss_fn(params)
    assert not jnp.isnan(loss_val)

    # Check differentiability and gradient flow
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params)

    # Verify gradients are computed for encoder, decoder and codebooks
    assert "encoder" in grads
    assert "decoder" in grads
    assert "codebooks" in grads

    # Ensure encoder gradients are non-zero (proving STE works)
    encoder_grads = grads["encoder"]["kernel"]
    assert jnp.any(encoder_grads != 0.0)
    assert not jnp.any(jnp.isnan(encoder_grads))
