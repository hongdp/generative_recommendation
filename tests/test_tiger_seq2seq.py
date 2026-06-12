"""Unit tests for the TIGER sequence-to-sequence model."""

import jax
import jax.numpy as jnp
from models.tiger_seq2seq import TIGERSeq2SeqModel


def test_tiger_seq2seq_shapes():
    batch_size = 4
    enc_len = 15   # 3 * L (e.g. 5 items * 3 levels)
    dec_len = 3    # next item tokens (c1, c2, c3)
    vocab_size = 3 * 256 + 2  # 770

    model = TIGERSeq2SeqModel(
        vocab_size=vocab_size,
        embedding_dim=64,
        num_blocks=2,
        num_heads=2,
        attention_dim=32,
        linear_dim=128,
        max_encoder_len=32,
        max_decoder_len=8,
    )

    key = jax.random.PRNGKey(0)
    dummy_enc_input = jnp.ones((batch_size, enc_len), dtype=jnp.int32)
    dummy_dec_input = jnp.ones((batch_size, dec_len), dtype=jnp.int32)

    # Initialize variables using __call__
    variables = model.init(key, dummy_enc_input, dummy_dec_input)
    params = variables["params"]

    # Test training forward pass
    logits = model.apply(variables, dummy_enc_input, dummy_dec_input)
    assert logits.shape == (batch_size, dec_len, vocab_size)

    # Test encode method
    encoder_outputs = model.apply(
        {"params": params},
        dummy_enc_input,
        method=model.encode,
        deterministic=True,
    )
    assert encoder_outputs.shape == (batch_size, enc_len, 64)

    # Test decode_step method
    dec_step_logits = model.apply(
        {"params": params},
        dummy_dec_input,
        encoder_outputs,
        dummy_enc_input,
        method=model.decode_step,
        deterministic=True,
    )
    assert dec_step_logits.shape == (batch_size, vocab_size)
