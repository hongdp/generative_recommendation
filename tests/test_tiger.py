"""Unit tests for the TIGER sequence model and tokenization mapping."""

import jax
import jax.numpy as jnp
import numpy as np
from models.tiger_model import TIGERModel


def test_tiger_tokenization_mapping():
    # Mock Semantic IDs database
    # Level codes in [0, 255]
    mock_semantic_ids = {
        0: [100, 200, 50],   # padding item (usually mapped to 0, 0, 0 under TIGER, let's test both)
        1: [10, 20, 30],
        2: [15, 25, 35],
    }
    K = 256
    vocab_size = 3 * K + 2
    start_token = vocab_size - 1

    # Simple helper mapping a sequence of items to flattened TIGER tokens
    def sequence_to_tiger_tokens(item_seq, semantic_ids, K, start_token):
        tokens = [start_token]
        for item in item_seq:
            if item == 0:
                # Padding items map to padding tokens [0, 0, 0]
                tokens.extend([0, 0, 0])
            else:
                c1, c2, c3 = semantic_ids[item]
                tokens.append(c1 + 1)
                tokens.append(c2 + K + 1)
                tokens.append(c3 + 2 * K + 1)
        return tokens

    # Test mapping
    item_seq = [0, 1, 2]  # pad, item 1, item 2
    tokens = sequence_to_tiger_tokens(item_seq, mock_semantic_ids, K, start_token)
    
    # Expected output:
    # pad item: [0, 0, 0]
    # item 1: [10 + 1, 20 + 256 + 1, 30 + 512 + 1] -> [11, 277, 543]
    # item 2: [15 + 1, 25 + 256 + 1, 35 + 512 + 1] -> [16, 282, 548]
    # start token: start_token
    expected_tokens = [start_token, 0, 0, 0, 11, 277, 543, 16, 282, 548]
    assert tokens == expected_tokens


def test_tiger_model_shapes():
    batch_size = 4
    seq_len = 16  # flat token sequence length
    vocab_size = 3 * 256 + 2  # 770

    model = TIGERModel(
        vocab_size=vocab_size,
        embedding_dim=64,
        num_blocks=2,
        num_heads=2,
        attention_dim=32,
        linear_dim=128,
        max_sequence_len=32,
    )

    key = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    variables = model.init(key, dummy_input)

    logits = model.apply(variables, dummy_input)
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_tiger_autoregressive_step():
    batch_size = 2
    seq_len = 5  # e.g., start token + 1 item [start, c1, c2, c3, pad]
    vocab_size = 3 * 256 + 2  # 770
    K = 256
    start_token = vocab_size - 1

    model = TIGERModel(
        vocab_size=vocab_size,
        embedding_dim=32,
        num_blocks=1,
        num_heads=2,
        attention_dim=16,
        linear_dim=64,
        max_sequence_len=10,
    )

    key = jax.random.PRNGKey(42)
    # Token seq: shape [batch, seq_len]
    token_seq = jnp.array([
        [start_token, 11, 277, 543, 0],
        [start_token, 16, 282, 548, 0]
    ], dtype=jnp.int32)

    variables = model.init(key, token_seq)
    params = variables["params"]

    # Autoregressive generation simulation for next-item predictions (predicting level 1 token c1)
    # We feed inputs up to sequence position 4 (i.e. first 4 tokens: start, c1, c2, c3)
    input_tokens = token_seq[:, :4]  # [batch, 4]
    
    # Forward pass
    logits = model.apply({"params": params}, input_tokens, deterministic=True)
    # Check shape: [batch, 4, vocab_size]
    assert logits.shape == (batch_size, 4, vocab_size)

    # Predict the 5th token (target item's c1) from the output at the last position (index 3)
    last_position_logits = logits[:, -1, :]  # [batch, vocab_size]
    
    # We restrict predictions to Level 1 vocabulary: [1, 256]
    level1_logits = last_position_logits[:, 1 : 257]
    predicted_level1_codes = jnp.argmax(level1_logits, axis=-1) + 1  # Map argmin in [0, 255] back to level-1 token ids [1, 256]

    assert predicted_level1_codes.shape == (batch_size,)
    assert jnp.all(predicted_level1_codes >= 1)
    assert jnp.all(predicted_level1_codes <= 256)
