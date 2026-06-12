"""TIGER and RQVAE Joint Training Sequence-to-Sequence Model."""

from typing import Dict, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from .tiger_cot import EncoderBlock, DecoderBlock


class TIGERJointModel(nn.Module):
    """TIGER Seq2Seq Model with RQVAE continuous reconstruction and quantization."""

    vocab_size: int
    embedding_dim: int = 384
    target_dim: int = 768
    latent_dim: int = 256
    num_levels: int = 3
    num_codes: int = 256
    num_blocks: int = 4
    num_heads: int = 6
    attention_dim: int = 384
    linear_dim: int = 1024
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1
    max_encoder_len: int = 64
    max_decoder_len: int = 4

    def setup(self):
        # 1. TIGER Seq2Seq Components
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

        # 2. RQVAE Components
        self.latent_proj = nn.Dense(self.latent_dim, name="latent_proj")
        
        self.codebooks = self.param(
            "codebooks",
            nn.initializers.variance_scaling(1.0, "fan_avg", "uniform"),
            (self.num_levels, self.num_codes, self.latent_dim)
        )
        self.decoder = nn.Dense(self.target_dim, name="decoder")

    def __call__(
        self,
        encoder_tokens: jnp.ndarray,
        decoder_tokens: jnp.ndarray,
        deterministic: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass for joint Seq2Seq and RQVAE training.

        Args:
            encoder_tokens: Input historical tokens, shape [batch, enc_len].
            decoder_tokens: Target prefix tokens, shape [batch, dec_len].
            deterministic: If True, disables dropout.

        Returns:
            A dictionary containing logits, reconstructed item embedding, and losses.
        """
        # --- 1. TIGER Encoder-Decoder Forward ---
        encoder_outputs = self.encode(encoder_tokens, deterministic=deterministic)

        dec_emb = self.embed_layer(decoder_tokens)
        dec_seq_len = decoder_tokens.shape[1]
        dec_emb = dec_emb + self.dec_pos[None, :dec_seq_len, :]

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

        # Output discrete token logits
        shared_weights = self.embed_layer.variables["params"]["embedding"]
        logits = jnp.dot(y, shared_weights.T)

        # --- 2. RQVAE Forward using TIGER outputs as residuals ---
        # The sequence length for semantic IDs should be at least num_levels
        # We take the first `num_levels` steps of the decoder output
        h_seq = self.latent_proj(y[:, :self.num_levels, :]) # [batch, num_levels, latent_dim]
        
        quantized_sum = jnp.zeros_like(h_seq[:, 0, :])
        all_indices = []
        codebook_losses = 0.0
        commitment_losses = 0.0

        for c in range(self.num_levels):
            h_c = h_seq[:, c, :] # [batch, latent_dim]
            codebook_c = self.codebooks[c] # [num_codes, latent_dim]

            # Compute Euclidean distances
            h_sq = jnp.sum(h_c ** 2, axis=-1, keepdims=True)
            code_sq = jnp.sum(codebook_c ** 2, axis=-1)[None, :]
            dot_product = jnp.matmul(h_c, codebook_c.T)
            distances = h_sq + code_sq - 2 * dot_product

            # Find closest code
            indices = jnp.argmin(distances, axis=-1)
            all_indices.append(indices)
            e_c = codebook_c[indices] # [batch, latent_dim]

            # STE
            e_c_ste = h_c + jax.lax.stop_gradient(e_c - h_c)
            quantized_sum = quantized_sum + e_c_ste

            # Level losses
            level_codebook_loss = jnp.mean((jax.lax.stop_gradient(h_c) - e_c) ** 2)
            level_commitment_loss = jnp.mean((h_c - jax.lax.stop_gradient(e_c)) ** 2)

            codebook_losses += level_codebook_loss
            commitment_losses += level_commitment_loss

        # Decode quantized sum back to target item reconstruction space
        x_recon = self.decoder(quantized_sum)

        return {
            "logits": logits,
            "x_recon": x_recon,
            "indices": jnp.stack(all_indices, axis=-1),
            "codebook_loss": codebook_losses,
            "commitment_loss": commitment_losses,
        }

    def encode(
        self,
        encoder_tokens: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        enc_emb = self.embed_layer(encoder_tokens)
        enc_seq_len = encoder_tokens.shape[1]
        enc_emb = enc_emb + self.enc_pos[None, :enc_seq_len, :]

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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Decoder step for beam search.

        Returns:
            logits: shape [batch, vocab_size]
            h_latent: shape [batch, latent_dim]
        """
        dec_emb = self.embed_layer(decoder_tokens)
        dec_seq_len = decoder_tokens.shape[1]
        dec_emb = dec_emb + self.dec_pos[None, :dec_seq_len, :]

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
        last_y = y[:, -1, :]
        logits = jnp.dot(last_y, shared_weights.T)
        h_latent = self.latent_proj(last_y)

        return logits, h_latent
