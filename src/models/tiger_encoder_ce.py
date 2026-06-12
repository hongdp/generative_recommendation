"""TIGER encoder + item embedding CE prediction head.

Uses the TIGER Seq2Seq encoder (bidirectional attention over semantic ID tokens)
to encode user history, then predicts the next item via dot-product with a
learnable item embedding table.
"""

import jax.numpy as jnp
import flax.linen as nn

from models.tiger_seq2seq import TIGERSeq2SeqModel


class TIGEREncoderCEModel(nn.Module):
    """TIGER encoder + item embedding CE prediction head."""
    num_items: int
    vocab_size: int
    embedding_dim: int = 384
    num_blocks: int = 4
    num_heads: int = 6
    attention_dim: int = 384
    linear_dim: int = 1024
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1
    max_encoder_len: int = 64

    @nn.compact
    def __call__(
        self,
        encoder_tokens: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Forward pass: encode history → pool → project → dot product with item embeddings.

        Args:
            encoder_tokens: Input historical tokens, shape [batch, enc_len].
            deterministic: If True, disables dropout.

        Returns:
            logits: shape [batch, num_items + 1] (index 0 = padding).
        """
        # 1. TIGER encoder
        encoder = TIGERSeq2SeqModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            attention_dim=self.attention_dim,
            linear_dim=self.linear_dim,
            max_encoder_len=self.max_encoder_len,
            max_decoder_len=4,
            attn_dropout_rate=self.attn_dropout_rate,
            linear_dropout_rate=self.linear_dropout_rate,
            name="tiger_encoder",
        )
        enc_out = encoder.encode(encoder_tokens, deterministic=deterministic)
        # enc_out: [batch, enc_len, embedding_dim]

        # 2. Mean pooling over non-padding positions
        mask = (encoder_tokens != 0).astype(jnp.float32)  # [batch, enc_len]
        mask_sum = jnp.sum(mask, axis=-1, keepdims=True)  # [batch, 1]
        mask_sum = jnp.maximum(mask_sum, 1.0)
        pooled = jnp.sum(enc_out * mask[:, :, None], axis=1) / mask_sum  # [batch, embedding_dim]

        # 3. Projection head
        h = nn.Dense(self.embedding_dim, name="proj_1")(pooled)
        h = nn.gelu(h)
        h = nn.Dense(self.embedding_dim, name="proj_2")(h)

        # 4. Item embedding table
        item_embeddings = nn.Embed(
            num_embeddings=self.num_items + 1,
            features=self.embedding_dim,
            name="item_embedding",
        )(jnp.arange(self.num_items + 1))
        # item_embeddings: [num_items + 1, embedding_dim]

        # 5. Dot product logits
        logits = h @ item_embeddings.T  # [batch, num_items + 1]

        return logits
