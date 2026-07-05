"""Shared TIGER semantic-ID tokenization utilities.

These helpers convert item-index sequences into the level-shifted Semantic-ID
token space used by every TIGER-family model. Centralizing them removes the
per-script copies of ``sequence_to_tiger_tokens`` / ``preprocess_*`` that had
drifted apart between the decoder-only and encoder-decoder training scripts.

Token layout for a codebook of size ``K`` and ``C = 3`` levels::

    pad   -> 0
    c1    -> [1,        K]       (stored code + 1)
    c2    -> [K + 1,    2K]      (stored code + K + 1)
    c3    -> [2K + 1,   3K]      (stored code + 2K + 1)
    start -> 3K + 1              (== vocab_size - 1)

with ``vocab_size = 3 * K + 2``. Stored Semantic IDs in the JSON files are the
raw 0-based codes ``[c1, c2, c3]`` with each ``ci in [0, K)``.
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
    """Inverts ``{item: [c1,c2,c3]}`` into ``{(c1,c2,c3): item}`` for decoding."""
    return {tuple(v): k for k, v in semantic_ids.items()}


def sequence_to_encoder_tokens(item_seq, semantic_ids, K):
    """Flattens item sequences into level-shifted encoder tokens (no start token).

    Used by the encoder-decoder (seq2seq) TIGER family. Output shape
    ``[batch, 3 * max_len]``; padding items stay ``[0, 0, 0]`` and real items are
    left-padded so the most recent interactions sit at the end of the sequence.
    """
    batch_size = len(item_seq)
    max_len = item_seq.shape[1]
    encoder_inputs = np.zeros((batch_size, 3 * max_len), dtype=np.int32)

    for i in range(batch_size):
        seq = item_seq[i]
        non_pad_indices = np.where(seq != 0)[0]
        num_pad = max_len - len(non_pad_indices)
        for idx, pos in enumerate(non_pad_indices):
            c1, c2, c3 = semantic_ids[int(seq[pos])]
            write_pos = 3 * num_pad + 3 * idx
            encoder_inputs[i, write_pos] = c1 + 1
            encoder_inputs[i, write_pos + 1] = c2 + K + 1
            encoder_inputs[i, write_pos + 2] = c3 + 2 * K + 1

    return encoder_inputs


def sequence_to_decoder_only_tokens(item_seq, semantic_ids, K, start_token):
    """Flattens item sequences into decoder-only tokens with a leading start token.

    Used by the decoder-only TIGER family. Output shape ``[batch, 3 * max_len + 1]``.
    """
    batch_size = len(item_seq)
    max_len = item_seq.shape[1]
    flat_tokens = np.zeros((batch_size, 3 * max_len + 1), dtype=np.int32)
    flat_tokens[:, 0] = start_token

    for i in range(batch_size):
        seq = item_seq[i]
        non_pad_indices = np.where(seq != 0)[0]
        num_pad = max_len - len(non_pad_indices)
        for idx, pos in enumerate(non_pad_indices):
            c1, c2, c3 = semantic_ids[int(seq[pos])]
            write_pos = 1 + 3 * num_pad + 3 * idx
            flat_tokens[i, write_pos] = c1 + 1
            flat_tokens[i, write_pos + 1] = c2 + K + 1
            flat_tokens[i, write_pos + 2] = c3 + 2 * K + 1

    return flat_tokens


def preprocess_seq2seq_training_data(inputs, targets, semantic_ids, K, start_token):
    """Builds (encoder_inputs, decoder_inputs, decoder_targets) for seq2seq training.

    decoder_inputs  = [start, c1+1, c2+K+1]
    decoder_targets = [c1+1,  c2+K+1, c3+2K+1]
    """
    encoder_inputs = sequence_to_encoder_tokens(inputs, semantic_ids, K)

    batch_size = len(inputs)
    decoder_inputs = np.zeros((batch_size, 3), dtype=np.int32)
    decoder_targets = np.zeros((batch_size, 3), dtype=np.int32)
    decoder_inputs[:, 0] = start_token

    for i in range(batch_size):
        c1, c2, c3 = semantic_ids[int(targets[i])]
        decoder_inputs[i, 1] = c1 + 1
        decoder_inputs[i, 2] = c2 + K + 1
        decoder_targets[i, 0] = c1 + 1
        decoder_targets[i, 1] = c2 + K + 1
        decoder_targets[i, 2] = c3 + 2 * K + 1

    return encoder_inputs, decoder_inputs, decoder_targets


def preprocess_decoder_only_training_data(inputs, targets, semantic_ids, K, start_token):
    """Builds shifted (inputs, targets) token streams for decoder-only training.

    The full sequence is ``[start, <history tokens>, <target item tokens>]`` of
    length ``3 * max_len + 4``; inputs are the first ``N-1`` tokens and targets are
    the last ``N-1`` (teacher forcing).
    """
    batch_size = len(inputs)
    max_len = inputs.shape[1]
    flat_tokens = np.zeros((batch_size, 3 * max_len + 4), dtype=np.int32)
    flat_tokens[:, 0] = start_token

    for i in range(batch_size):
        seq = inputs[i]
        non_pad_indices = np.where(seq != 0)[0]
        num_pad = max_len - len(non_pad_indices)
        for idx, pos in enumerate(non_pad_indices):
            c1, c2, c3 = semantic_ids[int(seq[pos])]
            write_pos = 1 + 3 * num_pad + 3 * idx
            flat_tokens[i, write_pos] = c1 + 1
            flat_tokens[i, write_pos + 1] = c2 + K + 1
            flat_tokens[i, write_pos + 2] = c3 + 2 * K + 1

        c1, c2, c3 = semantic_ids[int(targets[i])]
        write_pos = 1 + 3 * max_len
        flat_tokens[i, write_pos] = c1 + 1
        flat_tokens[i, write_pos + 1] = c2 + K + 1
        flat_tokens[i, write_pos + 2] = c3 + 2 * K + 1

    return flat_tokens[:, :-1], flat_tokens[:, 1:]
