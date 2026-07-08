"""Shared TIGER semantic-ID tokenization utilities.

These helpers convert item-index sequences into the level-shifted Semantic-ID
token space used by every TIGER-family model. Centralizing them removes the
per-script copies of ``sequence_to_tiger_tokens`` / ``preprocess_*`` that had
drifted apart between the decoder-only and encoder-decoder training scripts.

Token layout for a codebook of size ``K`` and ``L`` levels (default ``L = 3``)::

    pad      -> 0
    level i  -> [i*K + 1, (i+1)*K]   (stored code + i*K + 1), i = 0..L-1
    start    -> L*K + 1              (== vocab_size - 1)

with ``vocab_size = L * K + 2``. Stored Semantic IDs in the JSON files are the
raw 0-based codes ``[c1, ..., cL]`` with each ``ci in [0, K)``.
"""

import hashlib
import json

import numpy as np


def load_semantic_ids(path):
    """Loads a Semantic-ID JSON file into ``{item_index: [c1, c2, c3]}``.

    Keys are coerced to ``int`` (JSON object keys are always strings).
    """
    with open(path, "r") as f:
        return {int(k): v for k, v in json.load(f).items()}


def semantic_ids_hash(semantic_ids):
    """Returns a stable short hash of a Semantic-ID assignment.

    Used to bind a checkpoint to the exact code assignment it was trained on so
    that reloading against a regenerated (e.g. re-clustered) ID file fails loudly
    instead of silently decoding into the wrong code space.
    """
    items = sorted((int(k), [int(c) for c in v]) for k, v in semantic_ids.items())
    payload = json.dumps(items, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_semantic_id_to_item(semantic_ids):
    """Inverts ``{item: [c1,c2,c3]}`` into ``{(c1,c2,c3): item}`` for decoding.

    On collisions (multiple items sharing one Semantic ID) the last writer wins,
    making the shadowed items unreachable at decode time. Prefer
    ``build_semantic_id_to_items`` for collision-aware decoding.
    """
    return {tuple(v): k for k, v in semantic_ids.items()}


def build_semantic_id_to_items(semantic_ids, item_priors=None):
    """Inverts ``{item: codes}`` into ``{codes: [items...]}``, collision-aware.

    Every item sharing a Semantic ID is kept. Each list is sorted by descending
    ``item_priors`` (e.g. training-set frequency, approximating P(item | path)),
    with ascending item id as the deterministic tie-breaker. The padding item 0
    is excluded.

    Args:
        semantic_ids: mapping item id -> list of level codes.
        item_priors: optional mapping item id -> prior score (missing = 0).

    Returns:
        Dict mapping code tuple -> list of item ids (never empty).
    """
    priors = item_priors or {}
    mapping = {}
    for item, codes in semantic_ids.items():
        if item == 0:
            continue
        mapping.setdefault(tuple(codes), []).append(item)
    for path, items in mapping.items():
        items.sort(key=lambda it: (-priors.get(it, 0), it))
    return mapping


def sequence_to_encoder_tokens(item_seq, semantic_ids, K, num_levels=3):
    """Flattens item sequences into level-shifted encoder tokens (no start token).

    Used by the encoder-decoder (seq2seq) TIGER family. Output shape
    ``[batch, L * max_len]``; padding items stay all-zero and real items are
    left-padded so the most recent interactions sit at the end of the sequence.
    """
    batch_size = len(item_seq)
    max_len = item_seq.shape[1]
    L = num_levels
    encoder_inputs = np.zeros((batch_size, L * max_len), dtype=np.int32)

    for i in range(batch_size):
        seq = item_seq[i]
        non_pad_indices = np.where(seq != 0)[0]
        num_pad = max_len - len(non_pad_indices)
        for idx, pos in enumerate(non_pad_indices):
            codes = semantic_ids[int(seq[pos])]
            write_pos = L * num_pad + L * idx
            for lvl in range(L):
                encoder_inputs[i, write_pos + lvl] = codes[lvl] + lvl * K + 1

    return encoder_inputs


def sequence_to_decoder_only_tokens(item_seq, semantic_ids, K, start_token, num_levels=3):
    """Flattens item sequences into decoder-only tokens with a leading start token.

    Used by the decoder-only TIGER family. Output shape ``[batch, L * max_len + 1]``.
    """
    batch_size = len(item_seq)
    max_len = item_seq.shape[1]
    L = num_levels
    flat_tokens = np.zeros((batch_size, L * max_len + 1), dtype=np.int32)
    flat_tokens[:, 0] = start_token

    for i in range(batch_size):
        seq = item_seq[i]
        non_pad_indices = np.where(seq != 0)[0]
        num_pad = max_len - len(non_pad_indices)
        for idx, pos in enumerate(non_pad_indices):
            codes = semantic_ids[int(seq[pos])]
            write_pos = 1 + L * num_pad + L * idx
            for lvl in range(L):
                flat_tokens[i, write_pos + lvl] = codes[lvl] + lvl * K + 1

    return flat_tokens


def preprocess_seq2seq_training_data(inputs, targets, semantic_ids, K, start_token, num_levels=3):
    """Builds (encoder_inputs, decoder_inputs, decoder_targets) for seq2seq training.

    decoder_inputs  = [start, c1+1, ..., c(L-1)+(L-2)K+1]   (length L)
    decoder_targets = [c1+1, ..., cL+(L-1)K+1]              (length L)
    """
    L = num_levels
    encoder_inputs = sequence_to_encoder_tokens(inputs, semantic_ids, K, num_levels=L)

    batch_size = len(inputs)
    decoder_inputs = np.zeros((batch_size, L), dtype=np.int32)
    decoder_targets = np.zeros((batch_size, L), dtype=np.int32)
    decoder_inputs[:, 0] = start_token

    for i in range(batch_size):
        codes = semantic_ids[int(targets[i])]
        for lvl in range(L):
            tok = codes[lvl] + lvl * K + 1
            decoder_targets[i, lvl] = tok
            if lvl + 1 < L:
                decoder_inputs[i, lvl + 1] = tok

    return encoder_inputs, decoder_inputs, decoder_targets


def preprocess_decoder_only_training_data(inputs, targets, semantic_ids, K, start_token, num_levels=3):
    """Builds shifted (inputs, targets) token streams for decoder-only training.

    The full sequence is ``[start, <history tokens>, <target item tokens>]`` of
    length ``L * max_len + L + 1``; inputs are the first ``N-1`` tokens and targets
    are the last ``N-1`` (teacher forcing).
    """
    batch_size = len(inputs)
    max_len = inputs.shape[1]
    L = num_levels
    flat_tokens = np.zeros((batch_size, L * max_len + L + 1), dtype=np.int32)
    flat_tokens[:, 0] = start_token

    for i in range(batch_size):
        seq = inputs[i]
        non_pad_indices = np.where(seq != 0)[0]
        num_pad = max_len - len(non_pad_indices)
        for idx, pos in enumerate(non_pad_indices):
            codes = semantic_ids[int(seq[pos])]
            write_pos = 1 + L * num_pad + L * idx
            for lvl in range(L):
                flat_tokens[i, write_pos + lvl] = codes[lvl] + lvl * K + 1

        codes = semantic_ids[int(targets[i])]
        write_pos = 1 + L * max_len
        for lvl in range(L):
            flat_tokens[i, write_pos + lvl] = codes[lvl] + lvl * K + 1

    return flat_tokens[:, :-1], flat_tokens[:, 1:]
