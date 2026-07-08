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


def beam_search_decode_seq2seq(params, batch_enc_in, predictors, start_token, K, B=10, num_levels=3):
    """Top-B L-level beam search for encoder-decoder TIGER models.

    Returns a tuple of ``num_levels`` arrays, each ``[batch, B]``, holding the
    0-based codes of the top-B paths ordered by cumulative log-probability.
    """
    predict_enc, predict_dec_step, predict_dec_step_beams = predictors
    batch_size = len(batch_enc_in)
    rows = jnp.arange(batch_size)[:, None]
    L = num_levels

    encoder_outputs = predict_enc(params, batch_enc_in)

    # Level 1: expand from the bare start token.
    W = min(B, K)
    dec_in = jnp.ones((batch_size, 1), dtype=jnp.int32) * start_token
    logits = predict_dec_step(params, dec_in, encoder_outputs, batch_enc_in)
    log_probs = jax.nn.log_softmax(logits[:, 1 : K + 1], axis=-1)
    cum_probs, top_indices = jax.lax.top_k(log_probs, k=W)  # [batch, W]
    beams = [top_indices]  # per-level 0-based codes of live beams

    # Levels 2..L: decoder inputs [start, tok_1, ..., tok_lvl] per beam.
    for lvl in range(1, L):
        dec_start = jnp.ones((batch_size, W, 1), dtype=jnp.int32) * start_token
        prefix = [(beams[j] + j * K + 1)[:, :, None] for j in range(lvl)]
        dec_in = jnp.concatenate([dec_start] + prefix, axis=-1)  # [batch, W, lvl+1]
        logits = predict_dec_step_beams(params, dec_in, encoder_outputs, batch_enc_in)
        lo = lvl * K + 1
        log_probs = jax.nn.log_softmax(logits[:, :, lo : lo + K], axis=-1)  # [batch, W, K]
        flat = (cum_probs[:, :, None] + log_probs).reshape(batch_size, -1)
        W = min(B, W * K)
        cum_probs, top_flat = jax.lax.top_k(flat, k=W)
        beam_idx = top_flat // K
        new_code = top_flat % K
        beams = [b[rows, beam_idx] for b in beams] + [new_code]

    return tuple(np.array(b) for b in beams)


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


def beam_search_decode_decoder_only(params, batch_inputs, predict_next_token, K, B=10, num_levels=3):
    """Top-B L-level beam search for decoder-only TIGER models.

    Returns a tuple of ``num_levels`` arrays, each ``[batch, B]``, holding the
    0-based codes of the top-B paths ordered by cumulative log-probability.
    """
    batch_size = len(batch_inputs)
    rows = np.arange(batch_size)[:, None]
    L = num_levels

    # Level 1: expand from the raw context. The live width W is capped by the
    # number of expandable paths (K at level 1, W*K afterwards).
    W = min(B, K)
    logits = predict_next_token(params, batch_inputs)
    log_probs = jax.nn.log_softmax(logits[:, 1 : K + 1], axis=-1)
    cum_probs, top_indices = jax.lax.top_k(log_probs, k=W)  # [batch, W]
    beams = [np.array(top_indices)]  # per-level 0-based codes of live beams

    # Levels 2..L: replicate context with the beam prefix appended, expand, re-rank.
    for lvl in range(1, L):
        context = np.repeat(batch_inputs, W, axis=0)
        prefix_tokens = [
            (beams[j] + j * K + 1).reshape(-1, 1) for j in range(lvl)
        ]
        context = np.concatenate([context] + prefix_tokens, axis=-1)
        logits = predict_next_token(params, context)
        lo = lvl * K + 1
        log_probs = jax.nn.log_softmax(logits[:, lo : lo + K], axis=-1).reshape(batch_size, W, K)
        flat = (cum_probs[:, :, None] + log_probs).reshape(batch_size, -1)
        W = min(B, W * K)
        cum_probs, top_flat = jax.lax.top_k(flat, k=W)  # [batch, W]
        top_flat = np.array(top_flat)
        beam_idx = top_flat // K
        new_code = top_flat % K
        beams = [b[rows, beam_idx] for b in beams] + [new_code]

    return tuple(np.array(b) for b in beams)
