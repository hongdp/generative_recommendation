"""TIGER sequence model using HSTU block backbone for next-token prediction over Semantic IDs."""

import jax.numpy as jnp
import flax.linen as nn
from models.hstu import HSTUBlock


class TIGERModel(nn.Module):
    """TIGER model utilizing HSTU blocks to perform next-token prediction.

    Attributes:
        vocab_size: Total vocabulary size (C * K + 2: codewords, pad, and start tokens).
        embedding_dim: Embedding dimension.
        num_blocks: Number of HSTU blocks.
        num_heads: Number of attention heads.
        attention_dim: Attention projection dimension.
        linear_dim: Gated pointwise transformation dimension.
        attn_dropout_rate: Attention dropout rate.
        linear_dropout_rate: Linear projection dropout rate.
        enable_relative_attention_bias: Whether to use Relative Attention Bias.
        max_sequence_len: Maximum length of the flattened token sequence (e.g., C * L + 1).
    """
    vocab_size: int
    embedding_dim: int = 256
    num_blocks: int = 2
    num_heads: int = 4
    attention_dim: int = 128
    linear_dim: int = 512
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1
    enable_relative_attention_bias: bool = True
    max_sequence_len: int = 151  # 3 * 50 + 1 (start token + 50 items * 3 levels)

    @nn.compact
    def __call__(
        self,
        token_seq: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Applies the TIGER sequence model to input token sequences.

        Args:
            token_seq: Batch of token sequences, shape [batch, seq_len].
            deterministic: If True, disables dropout.

        Returns:
            logits: Output logits over vocabulary, shape [batch, seq_len, vocab_size].
        """
        # 1. Embed input tokens (0 is padding, vocab_size - 1 is start token)
        embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_dim,
            name="token_embedding",
        )
        x = embed_layer(token_seq)

        # 2. Apply stack of HSTU blocks (without timestamps)
        for i in range(self.num_blocks):
            x = HSTUBlock(
                attention_dim=self.attention_dim,
                linear_dim=self.linear_dim,
                num_heads=self.num_heads,
                attn_dropout_rate=self.attn_dropout_rate,
                linear_dropout_rate=self.linear_dropout_rate,
                enable_relative_attention_bias=self.enable_relative_attention_bias,
                max_sequence_len=self.max_sequence_len,
                name=f"hstu_block_{i}",
            )(x, timestamps=None, deterministic=deterministic)

        # 3. Weight-tied projection: dot product with token embedding weights
        shared_weights = embed_layer.variables["params"]["embedding"]  # [vocab_size, embedding_dim]
        logits = jnp.dot(x, shared_weights.T)  # [batch, seq_len, vocab_size]

        return logits
