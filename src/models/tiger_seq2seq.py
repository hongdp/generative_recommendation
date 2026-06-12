"""TIGER sequence-to-sequence model using T5 style encoder-decoder architecture in Flax.

Uses bidirectional attention in encoder for historical sequences and causal self-attention
with cross-attention in decoder to autoregressively predict next item Semantic IDs.
"""

import jax.numpy as jnp
import flax.linen as nn


class EncoderBlock(nn.Module):
    """Transformer Encoder block with bidirectional self-attention."""

    num_heads: int
    attention_dim: int
    linear_dim: int
    attn_dropout_rate: float
    linear_dropout_rate: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # Pre-LN Bidirectional Self-Attention
        x_norm = nn.LayerNorm(name="attn_ln")(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attention_dim,
            dropout_rate=self.attn_dropout_rate,
            name="self_attention",
        )(x_norm, mask=mask, deterministic=deterministic)
        x = x + attn_out

        # Pre-LN FFN
        x_norm2 = nn.LayerNorm(name="ffn_ln")(x)
        ffn_out = nn.Dense(self.linear_dim, name="ffn_dense_1")(x_norm2)
        ffn_out = nn.gelu(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(ffn_out, deterministic=deterministic)
        ffn_out = nn.Dense(x.shape[-1], name="ffn_dense_2")(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(ffn_out, deterministic=deterministic)
        return x + ffn_out


class DecoderBlock(nn.Module):
    """Transformer Decoder block with causal self-attention and cross-attention."""

    num_heads: int
    attention_dim: int
    linear_dim: int
    attn_dropout_rate: float
    linear_dropout_rate: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        encoder_outputs: jnp.ndarray,
        self_attn_mask: jnp.ndarray = None,
        cross_attn_mask: jnp.ndarray = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # 1. Pre-LN Causal Self-Attention
        x_norm = nn.LayerNorm(name="self_attn_ln")(x)
        self_attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attention_dim,
            dropout_rate=self.attn_dropout_rate,
            name="self_attention",
        )(x_norm, mask=self_attn_mask, deterministic=deterministic)
        x = x + self_attn_out

        # 2. Pre-LN Cross-Attention
        x_norm2 = nn.LayerNorm(name="cross_attn_ln")(x)
        cross_attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attention_dim,
            dropout_rate=self.attn_dropout_rate,
            name="cross_attention",
        )(x_norm2, encoder_outputs, mask=cross_attn_mask, deterministic=deterministic)
        x = x + cross_attn_out

        # 3. Pre-LN FFN
        x_norm3 = nn.LayerNorm(name="ffn_ln")(x)
        ffn_out = nn.Dense(self.linear_dim, name="ffn_dense_1")(x_norm3)
        ffn_out = nn.gelu(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(ffn_out, deterministic=deterministic)
        ffn_out = nn.Dense(x.shape[-1], name="ffn_dense_2")(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(ffn_out, deterministic=deterministic)
        return x + ffn_out


class TIGERSeq2SeqModel(nn.Module):
    """TIGER Sequence-to-Sequence Model with T5-style Encoder-Decoder architecture."""

    vocab_size: int
    embedding_dim: int = 384
    num_blocks: int = 4
    num_heads: int = 6
    attention_dim: int = 384
    linear_dim: int = 1024
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1
    max_encoder_len: int = 64  # Max 3 * L items
    max_decoder_len: int = 4   # Start token + 3 semantic ID tokens

    def setup(self):
        # Declare submodules and parameters in setup() to share them across methods
        self.embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_dim,
            name="token_embedding",
        )
        self.encoder_blocks = [
            EncoderBlock(
                num_heads=self.num_heads,
                attention_dim=self.attention_dim,
                linear_dim=self.linear_dim,
                attn_dropout_rate=self.attn_dropout_rate,
                linear_dropout_rate=self.linear_dropout_rate,
                name=f"encoder_block_{i}",
            ) for i in range(self.num_blocks)
        ]
        self.decoder_blocks = [
            DecoderBlock(
                num_heads=self.num_heads,
                attention_dim=self.attention_dim,
                linear_dim=self.linear_dim,
                attn_dropout_rate=self.attn_dropout_rate,
                linear_dropout_rate=self.linear_dropout_rate,
                name=f"decoder_block_{i}",
            ) for i in range(self.num_blocks)
        ]
        self.enc_pos = self.param(
            "enc_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.max_encoder_len, self.embedding_dim),
        )
        self.dec_pos = self.param(
            "dec_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.max_decoder_len, self.embedding_dim),
        )

    def __call__(
        self,
        encoder_tokens: jnp.ndarray,
        decoder_tokens: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Forward pass for Seq2Seq teacher-forced training.

        Args:
            encoder_tokens: Input historical tokens, shape [batch, enc_len].
            decoder_tokens: Target prefix tokens, shape [batch, dec_len].
            deterministic: If True, disables dropout.

        Returns:
            logits: Output logits over vocab, shape [batch, dec_len, vocab_size].
        """
        # 1. Encoder Forward
        encoder_outputs = self.encode(encoder_tokens, deterministic=deterministic)

        # 2. Decoder Forward
        dec_emb = self.embed_layer(decoder_tokens)

        # Causal position embedding for decoder
        dec_seq_len = decoder_tokens.shape[1]
        dec_emb = dec_emb + self.dec_pos[None, :dec_seq_len, :]

        # Decoder masks
        causal_mask = jnp.tril(jnp.ones((dec_seq_len, dec_seq_len), dtype=jnp.bool_))
        causal_mask = causal_mask[None, None, :, :]
        cross_mask = (encoder_tokens != 0)[:, None, None, :]

        y = dec_emb
        for i in range(self.num_blocks):
            y = self.decoder_blocks[i](
                y,
                encoder_outputs,
                self_attn_mask=causal_mask,
                cross_attn_mask=cross_mask,
                deterministic=deterministic,
            )

        # Output weight-tied projection
        shared_weights = self.embed_layer.variables["params"]["embedding"]
        logits = jnp.dot(y, shared_weights.T)

        return logits

    def encode(
        self,
        encoder_tokens: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Encodes user historical sequences.

        Args:
            encoder_tokens: shape [batch, enc_len].
            deterministic: If True, disables dropout.

        Returns:
            encoder_outputs: shape [batch, enc_len, embedding_dim].
        """
        enc_emb = self.embed_layer(encoder_tokens)

        # Encoder position embedding
        enc_seq_len = encoder_tokens.shape[1]
        enc_emb = enc_emb + self.enc_pos[None, :enc_seq_len, :]

        # Bidirectional self-attention mask (ignore pad=0)
        enc_mask = (encoder_tokens != 0)[:, None, None, :]

        x = enc_emb
        for i in range(self.num_blocks):
            x = self.encoder_blocks[i](x, mask=enc_mask, deterministic=deterministic)

        return x

    def decode_step(
        self,
        decoder_tokens: jnp.ndarray,
        encoder_outputs: jnp.ndarray,
        encoder_tokens: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Decoder step for single next-token prediction during beam search.

        Args:
            decoder_tokens: Current generated target tokens, shape [batch, dec_len].
            encoder_outputs: Precomputed encoder representations, shape [batch, enc_len, embedding_dim].
            encoder_tokens: Raw encoder input tokens (to compute cross-attention mask), shape [batch, enc_len].
            deterministic: If True, disables dropout.

        Returns:
            logits: Output logits over vocab at the last sequence position, shape [batch, vocab_size].
        """
        dec_emb = self.embed_layer(decoder_tokens)

        dec_seq_len = decoder_tokens.shape[1]
        dec_emb = dec_emb + self.dec_pos[None, :dec_seq_len, :]

        # Masks
        causal_mask = jnp.tril(jnp.ones((dec_seq_len, dec_seq_len), dtype=jnp.bool_))
        causal_mask = causal_mask[None, None, :, :]
        cross_mask = (encoder_tokens != 0)[:, None, None, :]

        y = dec_emb
        for i in range(self.num_blocks):
            y = self.decoder_blocks[i](
                y,
                encoder_outputs,
                self_attn_mask=causal_mask,
                cross_attn_mask=cross_mask,
                deterministic=deterministic,
            )

        shared_weights = self.embed_layer.variables["params"]["embedding"]
        # Return logits only at the last decoder position
        logits = jnp.dot(y[:, -1, :], shared_weights.T)

        return logits
