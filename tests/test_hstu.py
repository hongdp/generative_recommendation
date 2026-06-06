import jax
import jax.numpy as jnp
import numpy as np

from generative_recommendation.models.hstu import HSTUBlock, HSTUModel, log_bucket


def test_log_bucket():
    # Test log bucketing functionality
    dt = jnp.array([0.0, 1.0, 3.0, 7.0, 15.0])
    buckets = log_bucket(dt, num_buckets=8)
    # log2(1) = 0 -> bucket 0
    # log2(2) = 1 -> bucket 1
    # log2(4) = 2 -> bucket 2
    # log2(8) = 3 -> bucket 3
    # log2(16) = 4 -> bucket 4
    np.testing.assert_array_equal(np.array(buckets), np.array([0, 1, 2, 3, 4]))


def test_hstu_block_shapes():
    # Test basic forward shapes
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 10, 64))  # batch_size=2, seq_len=10, embed_dim=64
    timestamps = jnp.array([[1000, 1001, 1005, 1006, 1010, 1015, 1020, 1025, 1030, 1040],
                            [2000, 2005, 2006, 2010, 2012, 2015, 2020, 2030, 2040, 2050]])

    block = HSTUBlock(attention_dim=32, linear_dim=128, num_heads=2)
    params = block.init(key, x, timestamps=timestamps)

    out = block.apply(params, x, timestamps=timestamps)
    assert out.shape == (2, 10, 64)


def test_hstu_block_causality():
    # Test that future elements do not affect past elements (causality check)
    key = jax.random.PRNGKey(42)
    x1 = jax.random.normal(key, (1, 5, 32))
    
    # Modify the last element to make x2
    x2 = x1.at[0, 4, :].set(x1[0, 4, :] + 10.0)

    block = HSTUBlock(attention_dim=16, linear_dim=64, num_heads=2)
    params = block.init(key, x1)

    out1 = block.apply(params, x1, deterministic=True)
    out2 = block.apply(params, x2, deterministic=True)

    # Outputs for indices 0 to 3 should be identical
    np.testing.assert_allclose(
        np.array(out1[0, :4, :]),
        np.array(out2[0, :4, :]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_hstu_model_end_to_end():
    # Test complete HSTUModel behavior
    key = jax.random.PRNGKey(123)
    item_seq = jnp.array([[1, 5, 2, 0, 0], [4, 3, 1, 2, 5]])  # batch=2, seq_len=5
    timestamps = jnp.array([[100, 105, 110, 110, 110], [200, 201, 205, 210, 220]])

    model = HSTUModel(
        num_items=10,
        embedding_dim=32,
        num_blocks=2,
        num_heads=2,
        attention_dim=16,
        linear_dim=64,
        max_sequence_len=10,
    )
    
    # Initialize variables
    init_key, dropout_key = jax.random.split(key)
    variables = model.init(init_key, item_seq, timestamps=timestamps)

    # Forward pass
    logits = model.apply(
        variables,
        item_seq,
        timestamps=timestamps,
        deterministic=True,
    )
    # Output shape should be [batch, seq_len, num_items + 1] -> [2, 5, 11]
    assert logits.shape == (2, 5, 11)


def test_hstu_jit_and_grads():
    # Test JIT compilation safety and gradient flow
    key = jax.random.PRNGKey(999)
    item_seq = jnp.array([[1, 2, 3, 4, 5]])
    targets = jnp.array([[2, 3, 4, 5, 1]])

    model = HSTUModel(
        num_items=5,
        embedding_dim=16,
        num_blocks=1,
        num_heads=2,
        attention_dim=8,
        linear_dim=32,
        max_sequence_len=5,
    )
    variables = model.init(key, item_seq)

    # Define a simple loss function
    def loss_fn(params):
        # Forward pass
        logits = model.apply(
            {"params": params},
            item_seq,
            deterministic=True,
        )
        # Compute simple cross entropy loss
        one_hot_targets = jax.nn.one_hot(targets, num_classes=6)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(one_hot_targets * log_probs)
        return loss

    # Test JIT compatibility of the loss function
    jitted_loss = jax.jit(loss_fn)
    loss_val = jitted_loss(variables["params"])
    assert loss_val > 0.0

    # Test gradient computation
    grads = jax.grad(loss_fn)(variables["params"])
    
    # Assert gradients exist and are not NaN/inf
    for v in jax.tree_util.tree_leaves(grads):
        assert not jnp.isnan(v).any()
        assert not jnp.isinf(v).any()

