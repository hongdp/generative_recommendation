"""Residual Quantized Variational Autoencoder (RQ-VAE) implementation in Flax."""

from typing import Dict, Sequence, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn


class _MLP(nn.Module):
    """ReLU MLP: hidden layers (with optional dropout) followed by a linear output."""

    hidden_dims: Sequence[int]
    out_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        for h in self.hidden_dims:
            x = nn.relu(nn.Dense(h)(x))
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return nn.Dense(self.out_dim)(x)


class RQVAE(nn.Module):
    """Residual Quantized Variational Autoencoder (RQ-VAE) for hierarchical Semantic ID generation.

    Attributes:
        latent_dim: Dimension of the quantization latent space.
        num_levels: Number of residual quantization levels (C).
        num_codes: Codebook vocabulary size per level (K).
        embedding_dim: Dimension of the input/output item text embeddings.
        commitment_weight: Weight of commitment loss.
        hidden_dims: Optional MLP hidden sizes for the encoder (LIGER: [768, 512, 256]).
            The decoder mirrors them in reverse. Empty = original single-Dense behavior.
        dropout_rate: Dropout inside the MLP hidden layers (LIGER: 0.1).
    """
    latent_dim: int
    num_levels: int
    num_codes: int
    embedding_dim: int
    commitment_weight: float = 0.25
    hidden_dims: Tuple[int, ...] = ()
    dropout_rate: float = 0.0

    def setup(self):
        # Codebooks shape: [num_levels, num_codes, latent_dim]
        # Using standard uniform variance scaling initializer
        self.codebooks = self.param(
            "codebooks",
            nn.initializers.variance_scaling(1.0, "fan_avg", "uniform"),
            (self.num_levels, self.num_codes, self.latent_dim)
        )
        if self.hidden_dims:
            self.encoder = _MLP(self.hidden_dims, self.latent_dim,
                                self.dropout_rate, name="encoder")
            self.decoder = _MLP(tuple(reversed(self.hidden_dims)), self.embedding_dim,
                                self.dropout_rate, name="decoder")
        else:
            self.encoder = nn.Dense(self.latent_dim, name="encoder")
            self.decoder = nn.Dense(self.embedding_dim, name="decoder")

    def _encode_z(self, x, deterministic: bool = True):
        if self.hidden_dims:
            return self.encoder(x, deterministic=deterministic)
        return self.encoder(x)

    def _decode_z(self, z, deterministic: bool = True):
        if self.hidden_dims:
            return self.decoder(z, deterministic=deterministic)
        return self.decoder(z)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> Dict[str, jnp.ndarray]:
        """Runs the autoencoder forwarding pass (encoding, quantization, decoding).

        Args:
            x: Input embeddings, shape (batch_size, embedding_dim).

        Returns:
            A dictionary containing:
              - 'x_recon': Reconstructed embeddings, shape (batch_size, embedding_dim).
              - 'indices': Quantized indices, shape (batch_size, num_levels).
              - 'codebook_loss': Codebook quantization loss (mean scalar).
              - 'commitment_loss': Encoder commitment loss (mean scalar).
        """
        # 1. Project input to latent space
        z = self._encode_z(x, deterministic=deterministic)  # [batch_size, latent_dim]

        # 2. RVQ Quantization loop
        residuals = z
        quantized_sum = jnp.zeros_like(z)
        all_indices = []
        codebook_losses = 0.0
        commitment_losses = 0.0

        for c in range(self.num_levels):
            codebook_c = self.codebooks[c]  # [num_codes, latent_dim]

            # Compute Euclidean distances: ||residuals - codebook_c||^2
            res_sq = jnp.sum(residuals ** 2, axis=-1, keepdims=True)  # [batch_size, 1]
            code_sq = jnp.sum(codebook_c ** 2, axis=-1)[None, :]  # [1, num_codes]
            dot_product = jnp.matmul(residuals, codebook_c.T)  # [batch_size, num_codes]
            distances = res_sq + code_sq - 2 * dot_product  # [batch_size, num_codes]

            # Argmin to find closest codebook vector
            indices = jnp.argmin(distances, axis=-1)  # [batch_size]
            all_indices.append(indices)

            e_c = codebook_c[indices]  # [batch_size, latent_dim]

            # Straight-Through Estimator (STE)
            # Copy gradients from quantization output directly back to residuals input
            e_c_ste = residuals + jax.lax.stop_gradient(e_c - residuals)
            quantized_sum = quantized_sum + e_c_ste

            # Level quantization loss
            level_codebook_loss = jnp.mean((jax.lax.stop_gradient(residuals) - e_c) ** 2)
            level_commitment_loss = jnp.mean((residuals - jax.lax.stop_gradient(e_c)) ** 2)

            codebook_losses += level_codebook_loss
            commitment_losses += level_commitment_loss

            # Update residual for the next level
            residuals = residuals - e_c_ste

        # 3. Decode quantized sum back to reconstruction space
        x_recon = self._decode_z(quantized_sum, deterministic=deterministic)  # [batch_size, embedding_dim]
        indices_array = jnp.stack(all_indices, axis=-1)  # [batch_size, num_levels]

        return {
            "x_recon": x_recon,
            "indices": indices_array,
            "codebook_loss": codebook_losses,
            "commitment_loss": commitment_losses,
        }

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encodes inputs to quantized discrete index tuples only.

        Args:
            x: Input embeddings, shape (batch_size, embedding_dim).

        Returns:
            indices: Quantized index tuples, shape (batch_size, num_levels).
        """
        z = self._encode_z(x, deterministic=True)
        residuals = z
        all_indices = []

        for c in range(self.num_levels):
            codebook_c = self.codebooks[c]
            res_sq = jnp.sum(residuals ** 2, axis=-1, keepdims=True)
            code_sq = jnp.sum(codebook_c ** 2, axis=-1)[None, :]
            dot_product = jnp.matmul(residuals, codebook_c.T)
            distances = res_sq + code_sq - 2 * dot_product

            indices = jnp.argmin(distances, axis=-1)
            all_indices.append(indices)

            e_c = codebook_c[indices]
            residuals = residuals - e_c

        return jnp.stack(all_indices, axis=-1)
