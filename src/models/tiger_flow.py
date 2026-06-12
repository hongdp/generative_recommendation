"""TIGER Flow: DiT-style Flow Matching model for generative recommendation.

Applies the Diffusion Transformer paradigm to item recommendation:
  Stage 1 (N-step denoising): A Transformer denoises z_T -> z_0 in continuous
    latent space, conditioned on user interaction history via cross-attention.
  Stage 2 (one-shot retrieval): ANN lookup maps the denoised z_0 to the
    nearest item ID — analogous to the VAE Decoder in image generation.

No discrete tokens, codebooks, or Semantic IDs are used during training.
"""

import jax.numpy as jnp
import flax.linen as nn

from .tiger_cot import EncoderBlock


# =============================================================================
# VAE Components (learned latent space for flow matching)
# =============================================================================


class ItemVAEEncoder(nn.Module):
    """Maps frozen item embeddings to a KL-regularized latent space.

    Analogous to the VAE Encoder in Stable Diffusion: it transforms raw item
    representations (Sentence-T5 embeddings) into a latent z₀ ~ N(0, I) that
    is properly calibrated for flow matching / diffusion.

    The KL regularization solves the scale mismatch problem:
      - Sentence-T5 outputs are L2-normalized (all items at cosine_sim ~0.84)
      - Flow matching noise ε ~ N(0, I) has much larger scale
      - KL forces the encoder to produce z₀ ~ N(0, I), matching the noise scale
    """
    latent_dim: int = 256
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, e):
        """Encode item embedding to latent distribution parameters.

        Args:
            e: Frozen item embedding, shape [..., input_dim] (e.g., 768).

        Returns:
            mu: Mean of q(z|e), shape [..., latent_dim].
            log_var: Log-variance of q(z|e), shape [..., latent_dim].
        """
        h = nn.Dense(self.hidden_dim)(e)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.gelu(h)
        mu = nn.Dense(self.latent_dim)(h)
        log_var = nn.Dense(self.latent_dim)(h)
        return mu, log_var


class ItemVAEDecoder(nn.Module):
    """Reconstructs item embeddings from the latent space.

    Ensures the latent space preserves enough item identity information.
    Analogous to the VAE Decoder in image diffusion (pixel reconstruction),
    but here we reconstruct the Sentence-T5 embedding.
    """
    output_dim: int = 768
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, z):
        """Decode latent vector to reconstructed item embedding.

        Args:
            z: Latent vector, shape [..., latent_dim].

        Returns:
            e_hat: Reconstructed embedding, shape [..., output_dim].
        """
        h = nn.Dense(self.hidden_dim)(z)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.gelu(h)
        e_hat = nn.Dense(self.output_dim)(h)
        return e_hat


class FlowHead(nn.Module):
    """Lightweight MLP denoiser for flow matching, conditioned on user representation.

    Takes the user representation h_user from any backbone (e.g., HSTU) and denoises
    z_t → z_0 in the item embedding space. Much simpler than full DiT since z_t is
    a single vector (not a sequence).

    Input:  concat(h_user, z_t, t_emb) → MLP → v_hat
    """
    hidden_dim: int = 512
    output_dim: int = 256  # = item embedding dim

    @nn.compact
    def __call__(self, h_user, z_t, t):
        """Predict velocity v = dz/dt for flow matching.

        Args:
            h_user: User representation from backbone, shape [batch, user_dim].
            z_t: Noisy item embedding at timestep t, shape [batch, output_dim].
            t: Timestep in [0, 1], shape [batch].

        Returns:
            v_hat: Predicted velocity, shape [batch, output_dim].
        """
        t_emb = sinusoidal_embedding(t, self.hidden_dim)

        h = jnp.concatenate([h_user, z_t, t_emb], axis=-1)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.gelu(h)
        h = nn.LayerNorm()(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.gelu(h)
        h = nn.LayerNorm()(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.gelu(h)
        v = nn.Dense(self.output_dim)(h)
        return v


def sinusoidal_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Sinusoidal timestep embedding (same as DDPM / DiT).

    Args:
        t: Timestep values in [0, 1], shape [batch].
        dim: Embedding dimension.

    Returns:
        Sinusoidal embedding, shape [batch, dim].
    """
    half_dim = dim // 2
    freq = jnp.exp(jnp.arange(half_dim) * -(jnp.log(10000.0) / (half_dim - 1)))
    args = t[:, None] * freq[None, :]  # [batch, half_dim]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
    return emb


class FlowDecoderBlock(nn.Module):
    """Transformer Decoder block with AdaLN-Zero timestep conditioning (DiT-style).

    Structure: self-attention + cross-attention + FFN, same as tiger_cot.DecoderBlock,
    but every LayerNorm is replaced by Adaptive LayerNorm:
        AdaLN(x) = LN(x) * (1 + scale) + shift
    and every residual connection is gated:
        x = x + gate * SubLayer(AdaLN(x))

    The scale, shift, and gate parameters are projected from the timestep embedding.
    All gates are zero-initialized so that at init the block is an identity function.

    Attributes:
        num_heads: Number of attention heads.
        attention_dim: QKV projection dimension.
        linear_dim: FFN hidden dimension.
        attn_dropout_rate: Dropout rate for attention.
        linear_dropout_rate: Dropout rate for FFN.
    """

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
        t_emb: jnp.ndarray,
        self_attn_mask: jnp.ndarray = None,
        cross_attn_mask: jnp.ndarray = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Decoder input, shape [batch, seq_len, dim].
            encoder_outputs: Encoder outputs, shape [batch, enc_len, dim].
            t_emb: Timestep embedding, shape [batch, dim].
            self_attn_mask: Causal self-attention mask.
            cross_attn_mask: Cross-attention mask.
            deterministic: If True, disables dropout.

        Returns:
            Updated decoder hidden state, shape [batch, seq_len, dim].
        """
        dim = x.shape[-1]

        # Project t_emb → AdaLN params for 3 sub-layers: (scale, shift, gate) × 3
        adaln_params = nn.Dense(
            9 * dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="adaln_proj",
        )(t_emb)  # [batch, 9 * dim]
        adaln_params = adaln_params.reshape(-1, 9, dim)

        s1, sh1, g1 = adaln_params[:, 0], adaln_params[:, 1], adaln_params[:, 2]
        s2, sh2, g2 = adaln_params[:, 3], adaln_params[:, 4], adaln_params[:, 5]
        s3, sh3, g3 = adaln_params[:, 6], adaln_params[:, 7], adaln_params[:, 8]

        # --- Sub-layer 1: Self-attention with AdaLN-Zero ---
        x_norm = nn.LayerNorm(name="self_attn_ln")(x)
        x_norm = x_norm * (1 + s1[:, None, :]) + sh1[:, None, :]
        self_attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attention_dim,
            dropout_rate=self.attn_dropout_rate,
            name="self_attention",
        )(x_norm, mask=self_attn_mask, deterministic=deterministic)
        x = x + g1[:, None, :] * self_attn_out

        # --- Sub-layer 2: Cross-attention with AdaLN-Zero ---
        x_norm2 = nn.LayerNorm(name="cross_attn_ln")(x)
        x_norm2 = x_norm2 * (1 + s2[:, None, :]) + sh2[:, None, :]
        cross_attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attention_dim,
            dropout_rate=self.attn_dropout_rate,
            name="cross_attention",
        )(x_norm2, encoder_outputs, mask=cross_attn_mask, deterministic=deterministic)
        x = x + g2[:, None, :] * cross_attn_out

        # --- Sub-layer 3: FFN with AdaLN-Zero ---
        x_norm3 = nn.LayerNorm(name="ffn_ln")(x)
        x_norm3 = x_norm3 * (1 + s3[:, None, :]) + sh3[:, None, :]
        ffn_out = nn.Dense(self.linear_dim, name="ffn_dense_1")(x_norm3)
        ffn_out = nn.gelu(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(
            ffn_out, deterministic=deterministic
        )
        ffn_out = nn.Dense(dim, name="ffn_dense_2")(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(
            ffn_out, deterministic=deterministic
        )
        x = x + g3[:, None, :] * ffn_out

        return x


class TIGERFlowModel(nn.Module):
    """TIGER Flow: DiT-style Transformer for item embedding prediction via flow matching.

    Two-stage design mirroring Diffusion Transformer image generation:
      Stage 1: Transformer iteratively denoises z_T → z_0 in latent space,
               conditioned on user history via cross-attention and timestep
               via AdaLN-Zero.
      Stage 2: Denoised z_0 is mapped to nearest item via ANN lookup
               (handled externally, not part of the model).

    Attributes:
        embedding_dim: Transformer hidden dimension.
        latent_dim: PCA-reduced item latent space dimension.
        num_blocks: Number of encoder/decoder Transformer blocks.
        num_heads: Number of attention heads.
        attention_dim: QKV projection dimension.
        linear_dim: FFN hidden dimension.
        attn_dropout_rate: Attention dropout rate.
        linear_dropout_rate: FFN dropout rate.
        max_encoder_len: Maximum user history sequence length.
    """

    embedding_dim: int = 384
    latent_dim: int = 256
    num_blocks: int = 4
    num_heads: int = 6
    attention_dim: int = 384
    linear_dim: int = 1024
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1
    max_encoder_len: int = 20

    def setup(self):
        # --- Encoder: user history ---
        self.item_proj = nn.Dense(self.embedding_dim, name="item_proj")
        self.encoder_blocks = [
            EncoderBlock(
                num_heads=self.num_heads,
                attention_dim=self.attention_dim,
                linear_dim=self.linear_dim,
                attn_dropout_rate=self.attn_dropout_rate,
                linear_dropout_rate=self.linear_dropout_rate,
                name=f"encoder_block_{i}",
            )
            for i in range(self.num_blocks)
        ]
        self.enc_pos = self.param(
            "enc_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.max_encoder_len, self.embedding_dim),
        )

        # --- Timestep embedding: sinusoidal → MLP ---
        self.time_dense1 = nn.Dense(self.embedding_dim, name="time_dense1")
        self.time_dense2 = nn.Dense(self.embedding_dim, name="time_dense2")

        # --- Decoder / Denoiser ---
        self.z_in_proj = nn.Dense(self.embedding_dim, name="z_in_proj")
        self.decoder_blocks = [
            FlowDecoderBlock(
                num_heads=self.num_heads,
                attention_dim=self.attention_dim,
                linear_dim=self.linear_dim,
                attn_dropout_rate=self.attn_dropout_rate,
                linear_dropout_rate=self.linear_dropout_rate,
                name=f"decoder_block_{i}",
            )
            for i in range(self.num_blocks)
        ]
        self.final_ln = nn.LayerNorm(name="final_ln")
        self.v_out_proj = nn.Dense(self.latent_dim, name="v_out_proj")

    def __call__(
        self,
        encoder_latents: jnp.ndarray,
        encoder_mask: jnp.ndarray,
        z_t: jnp.ndarray,
        t: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Full forward pass: encode user history + predict velocity at timestep t.

        Args:
            encoder_latents: [batch, max_len, latent_dim] PCA-projected item embeddings.
            encoder_mask: [batch, max_len] boolean mask (True/1 = valid item).
            z_t: [batch, latent_dim] noisy target latent at timestep t.
            t: [batch] timestep values in [0, 1].
            deterministic: If True, disables dropout.

        Returns:
            v_hat: [batch, latent_dim] predicted velocity field.
        """
        enc_out = self.encode(encoder_latents, encoder_mask, deterministic)
        return self.predict_velocity(enc_out, encoder_mask, z_t, t, deterministic)

    def encode(
        self,
        encoder_latents: jnp.ndarray,
        encoder_mask: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Encode user interaction history (called once, reused across denoising steps).

        Args:
            encoder_latents: [batch, max_len, latent_dim] PCA-projected item embeddings.
            encoder_mask: [batch, max_len] boolean mask.
            deterministic: If True, disables dropout.

        Returns:
            Encoder outputs, shape [batch, max_len, embedding_dim].
        """
        x = self.item_proj(encoder_latents)  # [batch, max_len, embedding_dim]
        seq_len = encoder_latents.shape[1]
        x = x + self.enc_pos[None, :seq_len, :]

        enc_attn_mask = encoder_mask[:, None, None, :]  # [batch, 1, 1, max_len]

        for block in self.encoder_blocks:
            x = block(x, mask=enc_attn_mask, deterministic=deterministic)

        return x

    def predict_velocity(
        self,
        enc_out: jnp.ndarray,
        encoder_mask: jnp.ndarray,
        z_t: jnp.ndarray,
        t: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Predict velocity field given pre-encoded user history and noisy latent.

        This is one denoising step. For N-step inference, call this N times
        with Euler integration: z_{t-dt} = z_t - dt * v_hat.

        Args:
            enc_out: [batch, max_len, embedding_dim] pre-computed encoder outputs.
            encoder_mask: [batch, max_len] boolean mask.
            z_t: [batch, latent_dim] noisy latent.
            t: [batch] current timestep in [0, 1].
            deterministic: If True, disables dropout.

        Returns:
            v_hat: [batch, latent_dim] predicted velocity.
        """
        # Timestep embedding
        t_emb = sinusoidal_embedding(t, self.embedding_dim)  # [batch, embedding_dim]
        t_emb = self.time_dense1(t_emb)
        t_emb = nn.gelu(t_emb)
        t_emb = self.time_dense2(t_emb)  # [batch, embedding_dim]

        # Project noisy latent → single-token decoder input
        z_emb = self.z_in_proj(z_t)[:, None, :]  # [batch, 1, embedding_dim]

        # Cross-attention mask for encoder outputs
        cross_mask = encoder_mask[:, None, None, :]  # [batch, 1, 1, max_len]

        # Decoder blocks with AdaLN-Zero
        y = z_emb
        for block in self.decoder_blocks:
            y = block(
                y,
                enc_out,
                t_emb,
                cross_attn_mask=cross_mask,
                deterministic=deterministic,
            )

        # Output projection → velocity
        y = self.final_ln(y[:, 0, :])  # [batch, embedding_dim]
        v_hat = self.v_out_proj(y)  # [batch, latent_dim]
        return v_hat
