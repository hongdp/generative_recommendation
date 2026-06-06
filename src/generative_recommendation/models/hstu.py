"""Hierarchical Sequential Transduction Unit (HSTU) architecture.

Replicates the HSTU model from Meta AI's "Actions Speak Louder than Words" paper
(ICML 2024) in JAX/Flax.
"""

from typing import Optional
import flax.linen as nn
import jax
import jax.numpy as jnp


def log_bucket(dt: jnp.ndarray, num_buckets: int = 64, max_val: float = 1e8) -> jnp.ndarray:
    """Logarithmically buckets time differences for temporal relative attention bias.

    Args:
        dt: time differences tensor.
        num_buckets: number of buckets.
        max_val: maximum value for clipping time differences.

    Returns:
        Integer bucket indices.
    """
    dt = jnp.clip(dt, 0.0, max_val)
    # Use log2(1 + dt) to compute buckets
    buckets = jnp.floor(jnp.log2(dt + 1.0))
    return jnp.clip(buckets, 0, num_buckets - 1).astype(jnp.int32)


class HSTUBlock(nn.Module):
    """Hierarchical Sequential Transduction Unit (HSTU) block."""

    attention_dim: int = 128
    linear_dim: int = 512
    num_heads: int = 4
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1
    enable_relative_attention_bias: bool = True
    max_sequence_len: int = 50
    num_temp_buckets: int = 64

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        timestamps: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Applies the HSTU block.

        Args:
            x: input tensor, shape [batch, seq_len, embedding_dim].
            timestamps: optional timestamp tensor, shape [batch, seq_len].
            deterministic: if True, disables dropout.

        Returns:
            output tensor, shape [batch, seq_len, embedding_dim].
        """
        batch_size, seq_len, embedding_dim = x.shape
        head_dim_a = self.attention_dim // self.num_heads
        head_dim_v = self.linear_dim // self.num_heads

        # 1. Projections
        # Q, K shape: [batch, seq_len, num_heads, head_dim_a]
        # V, U shape: [batch, seq_len, num_heads, head_dim_v]
        q = nn.Dense(self.attention_dim, name="q_proj")(x)
        k = nn.Dense(self.attention_dim, name="k_proj")(x)
        v = nn.Dense(self.linear_dim, name="v_proj")(x)
        u = nn.Dense(self.linear_dim, name="u_proj")(x)

        q = q.reshape((batch_size, seq_len, self.num_heads, head_dim_a))
        k = k.reshape((batch_size, seq_len, self.num_heads, head_dim_a))
        v = v.reshape((batch_size, seq_len, self.num_heads, head_dim_v))
        u = u.reshape((batch_size, seq_len, self.num_heads, head_dim_v))

        # 2. Attention scores
        # scores shape: [batch, num_heads, seq_len, seq_len]
        scores = jnp.einsum("bihd,bjhd->bhij", q, k) / jnp.sqrt(head_dim_a)

        # 3. Add Relative Attention Bias (RAB)
        if self.enable_relative_attention_bias:
            # Positional relative bias
            # rel_pos_bias shape: [num_heads, max_sequence_len]
            rel_pos_bias = self.param(
                "rel_pos_bias",
                nn.initializers.zeros,
                (self.num_heads, self.max_sequence_len),
            )
            # Compute relative distances for causal mapping: diffs[i, j] = i - j
            indices = jnp.arange(seq_len)
            diffs = indices[:, None] - indices[None, :]
            # Clip diffs to [0, max_sequence_len - 1]
            diffs = jnp.clip(diffs, 0, self.max_sequence_len - 1)
            pos_bias = rel_pos_bias[:, diffs]  # [num_heads, seq_len, seq_len]
            scores = scores + pos_bias[None, :, :, :]

            # Temporal bias
            if timestamps is not None:
                # time_diffs shape: [batch, seq_len, seq_len]
                time_diffs = timestamps[:, :, None] - timestamps[:, None, :]
                buckets = log_bucket(time_diffs, self.num_temp_buckets)

                # temporal_bias_table shape: [num_temp_buckets, num_heads]
                temporal_bias_table = self.param(
                    "temporal_bias_table",
                    nn.initializers.zeros,
                    (self.num_temp_buckets, self.num_heads),
                )
                # Lookup temporal bias -> [batch, seq_len, seq_len, num_heads]
                temp_bias = temporal_bias_table[buckets]
                # Transpose to match scores shape -> [batch, num_heads, seq_len, seq_len]
                temp_bias = jnp.transpose(temp_bias, (0, 3, 1, 2))
                scores = scores + temp_bias

        # 4. Pointwise Attention (SiLU) and Causal Masking
        # A shape: [batch, num_heads, seq_len, seq_len]
        A = jax.nn.silu(scores)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        A = A * causal_mask[None, None, :, :]

        # Attention dropout
        A = nn.Dropout(rate=self.attn_dropout_rate)(A, deterministic=deterministic)

        # 5. Spatial Aggregation
        # Z shape: [batch, seq_len, num_heads, head_dim_v]
        Z = jnp.einsum("bhij,bjhd->bihd", A, v)
        # Reshape to [batch, seq_len, linear_dim]
        Z = Z.reshape((batch_size, seq_len, self.linear_dim))
        u = u.reshape((batch_size, seq_len, self.linear_dim))

        # 6. Gated Pointwise Transformation
        Z_norm = nn.LayerNorm()(Z)
        Z_gated = Z_norm * u
        Z_gated = nn.Dropout(rate=self.linear_dropout_rate)(Z_gated, deterministic=deterministic)

        # Project back to embedding dimension
        output = nn.Dense(embedding_dim, name="out_proj")(Z_gated)

        # Residual connection
        return output + x


class HSTUModel(nn.Module):
    """Sequential recommendation model utilizing HSTU encoder stack and weight-tied predictions."""

    num_items: int
    embedding_dim: int = 256
    num_blocks: int = 2
    num_heads: int = 4
    attention_dim: int = 128
    linear_dim: int = 512
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1
    enable_relative_attention_bias: bool = True
    max_sequence_len: int = 50
    num_temp_buckets: int = 64

    @nn.compact
    def __call__(
        self,
        item_seq: jnp.ndarray,
        timestamps: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Applies HSTU sequential recommendation model.

        Args:
            item_seq: batch of item sequences, shape [batch, seq_len]. 0 is used for padding.
            timestamps: optional timestamps corresponding to sequences, shape [batch, seq_len].
            deterministic: if True, disables dropout.

        Returns:
            logits over all items for next-item prediction, shape [batch, seq_len, num_items + 1].
        """
        # Embed input items (0 is reserved for padding)
        embed_layer = nn.Embed(
            num_embeddings=self.num_items + 1,
            features=self.embedding_dim,
            name="item_embedding",
        )
        x = embed_layer(item_seq)

        # Apply stack of HSTU blocks
        for i in range(self.num_blocks):
            x = HSTUBlock(
                attention_dim=self.attention_dim,
                linear_dim=self.linear_dim,
                num_heads=self.num_heads,
                attn_dropout_rate=self.attn_dropout_rate,
                linear_dropout_rate=self.linear_dropout_rate,
                enable_relative_attention_bias=self.enable_relative_attention_bias,
                max_sequence_len=self.max_sequence_len,
                num_temp_buckets=self.num_temp_buckets,
                name=f"hstu_block_{i}",
            )(x, timestamps=timestamps, deterministic=deterministic)

        # Weight-tied projection: dot product with the embedding weights
        shared_weights = embed_layer.variables["params"]["embedding"]  # [num_items + 1, embedding_dim]
        # Compute scores for all items
        logits = jnp.dot(x, shared_weights.T)  # [batch, seq_len, num_items + 1]

        return logits
