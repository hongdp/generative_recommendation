import jax
import jax.numpy as jnp
import numpy as np

from src.models.tiger_joint import TIGERJointModel


def test_tiger_joint_model_shapes():
    num_items = 100
    codebook_size = 256
    embedding_dim = 64
    batch_size = 4
    seq_len = 10

    # Create dummy Semantic ID table [num_items + 1, 3]
    # SIDs are in range [0, codebook_size - 1]
    c_table = np.random.randint(0, codebook_size, size=(num_items + 1, 3))
    # padding item 0 has SIDs [0, 0, 0]
    c_table[0] = [0, 0, 0]
    c_table = jnp.array(c_table)

    model = TIGERJointModel(
        num_items=num_items,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        num_blocks=2,
        num_heads=2,
        attention_dim=32,
        linear_dim=128,
        max_sequence_len=seq_len
    )

    rng = jax.random.PRNGKey(0)
    # create dummy sequences
    item_seq = jax.random.randint(rng, (batch_size, seq_len), 0, num_items + 1)

    # initialize variables
    variables = model.init(rng, item_seq, c_table, deterministic=True)

    # apply
    (logits_c1, logits_c2, logits_c3), logits_item, h = model.apply(variables, item_seq, c_table, deterministic=True)

    # Check shapes
    assert logits_c1.shape == (batch_size, seq_len, codebook_size)
    assert logits_c2.shape == (batch_size, seq_len, codebook_size)
    assert logits_c3.shape == (batch_size, seq_len, codebook_size)
    assert logits_item.shape == (batch_size, seq_len, num_items + 1)
    assert h.shape == (batch_size, seq_len, embedding_dim)
