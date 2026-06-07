"""Standard causal Transformer architecture for sequential recommendation.

Replicates standard decoder-only Transformer (SASRec style) in JAX/Flax.
"""

from typing import Optional
import flax.linen as nn
import jax.numpy as jnp


class TransformerBlock(nn.Module):
    """Standard causal Transformer block (SASRec style)."""

    num_heads: int = 4
    attention_dim: int = 128
    linear_dim: int = 512
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Applies the Transformer block.

        Args:
            x: input tensor, shape [batch, seq_len, embedding_dim].
            deterministic: if True, disables dropout.

        Returns:
            output tensor, shape [batch, seq_len, embedding_dim].
        """
        batch_size, seq_len, embedding_dim = x.shape

        # 1. Multi-Head Attention with Causal Masking (Pre-LN style)
        x_norm = nn.LayerNorm(name="attn_ln")(x)

        # Causal mask: [1, 1, seq_len, seq_len]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        causal_mask = causal_mask[None, None, :, :]

        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attention_dim,
            dropout_rate=self.attn_dropout_rate,
            name="self_attention",
        )(x_norm, mask=causal_mask, deterministic=deterministic)

        # Residual connection
        x = x + attn_out

        # 2. Feed-Forward Network (Pre-LN style)
        x_norm2 = nn.LayerNorm(name="ffn_ln")(x)

        ffn_out = nn.Dense(self.linear_dim, name="ffn_dense_1")(x_norm2)
        ffn_out = nn.gelu(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(ffn_out, deterministic=deterministic)
        ffn_out = nn.Dense(embedding_dim, name="ffn_dense_2")(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(ffn_out, deterministic=deterministic)

        # Residual connection
        return x + ffn_out


class TransformerModel(nn.Module):
    """Sequential recommendation model utilizing standard Transformer block stack."""

    num_items: int
    embedding_dim: int = 256
    num_blocks: int = 2
    num_heads: int = 4
    attention_dim: int = 128
    linear_dim: int = 512
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1
    max_sequence_len: int = 50

    @nn.compact
    def __call__(
        self,
        item_seq: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Applies Transformer sequential recommendation model.

        Args:
            item_seq: batch of item sequences, shape [batch, seq_len]. 0 is used for padding.
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

        # Learned causal position embeddings
        seq_len = item_seq.shape[1]
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.max_sequence_len, self.embedding_dim),
        )
        # Slice position embeddings up to seq_len
        x = x + pos_embedding[None, :seq_len, :]

        x = nn.Dropout(rate=self.linear_dropout_rate)(x, deterministic=deterministic)

        # Apply stack of Transformer blocks
        for i in range(self.num_blocks):
            x = TransformerBlock(
                num_heads=self.num_heads,
                attention_dim=self.attention_dim,
                linear_dim=self.linear_dim,
                attn_dropout_rate=self.attn_dropout_rate,
                linear_dropout_rate=self.linear_dropout_rate,
                name=f"transformer_block_{i}",
            )(x, deterministic=deterministic)

        # Weight-tied projection: dot product with the embedding weights
        shared_weights = embed_layer.variables["params"]["embedding"]  # [num_items + 1, embedding_dim]
        logits = jnp.dot(x, shared_weights.T)  # [batch, seq_len, num_items + 1]

        return logits
