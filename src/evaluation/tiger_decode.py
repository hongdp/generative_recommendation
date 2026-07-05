"""Shared beam-search decoders for TIGER-family generative recommenders.

Both the decoder-only and encoder-decoder (seq2seq) TIGER scripts previously
carried near-identical ~60-line copies of the 3-level Semantic-ID beam search.
They are centralized here and parametrized by codebook size ``K`` so the level
token ranges are derived rather than hard-coded to ``K = 256``.

Level token ranges (see ``models.tiger_tokenization``)::

    level 1 logits slice: [1,      K + 1)
    level 2 logits slice: [K + 1,  2K + 1)
    level 3 logits slice: [2K + 1, 3K + 1)

Each decoder returns ``(c1, c2, c3)`` arrays of shape ``[batch, B]`` holding the
top-B raw 0-based codes ordered by descending cumulative log-probability, ready
to look up in ``build_semantic_id_to_item``.
"""

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Encoder-decoder (seq2seq) family
# ---------------------------------------------------------------------------
def make_seq2seq_predictors(model):
    """Builds jitted encoder/decoder-step closures for a seq2seq TIGER model.

    The model must expose ``encode`` and ``decode_step`` methods. Returns a tuple
    ``(predict_enc, predict_dec_step, predict_dec_step_beams)``.
    """

    @jax.jit
    def predict_enc(params, encoder_tokens):
        return model.apply(
            {"params": params}, encoder_tokens,
            method=model.encode, deterministic=True,
        )

    @jax.jit
    def predict_dec_step(params, decoder_tokens, encoder_outputs, encoder_tokens):
        return model.apply(
            {"params": params}, decoder_tokens, encoder_outputs, encoder_tokens,
            method=model.decode_step, deterministic=True,
        )

    @jax.jit
    def predict_dec_step_beams(params, decoder_tokens_beams, encoder_outputs, encoder_tokens):
        # Vectorize the decoder step across the beam dimension (axis 1).
        vmap_fn = jax.vmap(
            lambda dec: model.apply(
                {"params": params}, dec, encoder_outputs, encoder_tokens,
                method=model.decode_step, deterministic=True,
            ),
            in_axes=1, out_axes=1,
        )
        return vmap_fn(decoder_tokens_beams)

    return predict_enc, predict_dec_step, predict_dec_step_beams


def beam_search_decode_seq2seq(params, batch_enc_in, predictors, start_token, K, B=10):
    """Top-B 3-level beam search for encoder-decoder TIGER models."""
    predict_enc, predict_dec_step, predict_dec_step_beams = predictors
    batch_size = len(batch_enc_in)
    rows = jnp.arange(batch_size)[:, None]

    encoder_outputs = predict_enc(params, batch_enc_in)

    # Step 1: level-1 token.
    dec_in1 = jnp.ones((batch_size, 1), dtype=jnp.int32) * start_token
    logits1 = predict_dec_step(params, dec_in1, encoder_outputs, batch_enc_in)
    log_probs1 = jax.nn.log_softmax(logits1[:, 1 : K + 1], axis=-1)
    top_probs, top_indices = jax.lax.top_k(log_probs1, k=B)  # [batch, B]
    top_tokens1 = top_indices + 1

    # Step 2: level-2 token (batched across beams).
    dec_in2_start = jnp.ones((batch_size, B, 1), dtype=jnp.int32) * start_token
    dec_in2 = jnp.concatenate([dec_in2_start, top_tokens1[:, :, None]], axis=-1)
    logits2 = predict_dec_step_beams(params, dec_in2, encoder_outputs, batch_enc_in)
    log_probs2 = jax.nn.log_softmax(logits2[:, :, K + 1 : 2 * K + 1], axis=-1)
    cum_probs2 = (top_probs[:, :, None] + log_probs2).reshape(batch_size, -1)
    top_probs2, top_flat_indices2 = jax.lax.top_k(cum_probs2, k=B)
    beam_idx2 = top_flat_indices2 // K
    c2 = top_flat_indices2 % K
    c1 = top_indices[rows, beam_idx2]
    top_tokens1_expanded = c1 + 1
    top_tokens2 = c2 + K + 1

    # Step 3: level-3 token.
    dec_in3_start = jnp.ones((batch_size, B, 1), dtype=jnp.int32) * start_token
    dec_in3 = jnp.concatenate(
        [dec_in3_start, top_tokens1_expanded[:, :, None], top_tokens2[:, :, None]],
        axis=-1,
    )
    logits3 = predict_dec_step_beams(params, dec_in3, encoder_outputs, batch_enc_in)
    log_probs3 = jax.nn.log_softmax(logits3[:, :, 2 * K + 1 : 3 * K + 1], axis=-1)
    cum_probs3 = (top_probs2[:, :, None] + log_probs3).reshape(batch_size, -1)
    _, top_flat_indices3 = jax.lax.top_k(cum_probs3, k=B)
    beam_idx3 = top_flat_indices3 // K
    c3 = top_flat_indices3 % K
    c2_final = c2[rows, beam_idx3]
    c1_final = c1[rows, beam_idx3]

    return np.array(c1_final), np.array(c2_final), np.array(c3)


# ---------------------------------------------------------------------------
# Decoder-only family
# ---------------------------------------------------------------------------
def make_decoder_only_predictor(model):
    """Builds the jitted next-token closure for a decoder-only TIGER model."""

    @jax.jit
    def predict_next_token(params, current_tokens):
        logits = model.apply({"params": params}, current_tokens, deterministic=True)
        return logits[:, -1, :]

    return predict_next_token


def beam_search_decode_decoder_only(params, batch_inputs, predict_next_token, K, B=10):
    """Top-B 3-level beam search for decoder-only TIGER models."""
    batch_size = len(batch_inputs)
    rows = np.arange(batch_size)[:, None]

    # Step 1: level-1 token.
    logits1 = predict_next_token(params, batch_inputs)
    log_probs1 = jax.nn.log_softmax(logits1[:, 1 : K + 1], axis=-1)
    top_probs, top_indices = jax.lax.top_k(log_probs1, k=B)
    top_tokens1 = top_indices + 1

    # Step 2: level-2 token (batched across beams by replicating context).
    context2 = np.repeat(batch_inputs, B, axis=0)
    context2 = np.concatenate([context2, np.array(top_tokens1).reshape(-1, 1)], axis=-1)
    logits2 = predict_next_token(params, context2)
    log_probs2 = jax.nn.log_softmax(logits2[:, K + 1 : 2 * K + 1], axis=-1).reshape(batch_size, B, K)
    cum_probs2 = (top_probs[:, :, None] + log_probs2).reshape(batch_size, -1)
    top_probs2, top_flat_indices2 = jax.lax.top_k(cum_probs2, k=B)
    beam_idx2 = top_flat_indices2 // K
    c2 = top_flat_indices2 % K
    c1 = top_indices[rows, beam_idx2]
    top_tokens1_expanded = c1 + 1
    top_tokens2 = c2 + K + 1

    # Step 3: level-3 token.
    context3 = np.repeat(batch_inputs, B, axis=0)
    context3 = np.concatenate(
        [context3, np.array(top_tokens1_expanded).reshape(-1, 1), np.array(top_tokens2).reshape(-1, 1)],
        axis=-1,
    )
    logits3 = predict_next_token(params, context3)
    log_probs3 = jax.nn.log_softmax(logits3[:, 2 * K + 1 : 3 * K + 1], axis=-1).reshape(batch_size, B, K)
    cum_probs3 = (top_probs2[:, :, None] + log_probs3).reshape(batch_size, -1)
    _, top_flat_indices3 = jax.lax.top_k(cum_probs3, k=B)
    beam_idx3 = top_flat_indices3 // K
    c3 = top_flat_indices3 % K
    c2_final = c2[rows, beam_idx3]
    c1_final = c1[rows, beam_idx3]

    return np.array(c1_final), np.array(c2_final), np.array(c3)
