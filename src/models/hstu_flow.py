"""HSTU Flow Matching Models.

Contains various HSTU-based Flow Matching model implementations including:
- VAE Encoders/Decoders for latent space mapping.
- HSTUFlowModel (standard continuous representation prediction).
- HSTUFlowCEModel (Flow + Cross-Entropy joint training).
- HSTUIDFlowModel (Semantic ID / T5 continuous embeddings flow).
"""

import jax.numpy as jnp
import flax.linen as nn

from models.hstu import HSTUBlock
from models.tiger_flow import FlowHead


class VAEEncoder(nn.Module):
    latent_dim: int
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mu, logvar


class VAEDecoder(nn.Module):
    output_dim: int
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(self.hidden_dim)(z)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        out = nn.Dense(self.output_dim)(x)
        return out


class HSTUFlowModel(nn.Module):
    """Base HSTU backbone + FlowHead.
    
    Predicts velocity given user history and noisy target z_t.
    """
    embedding_dim: int = 256
    num_blocks: int = 4
    num_heads: int = 4
    attention_dim: int = 128
    linear_dim: int = 512
    flow_hidden_dim: int = 512
    attn_dropout_rate: float = 0.2
    linear_dropout_rate: float = 0.2
    max_sequence_len: int = 20

    @nn.compact
    def __call__(self, user_embs, z_t, t, deterministic=True):
        """Forward pass.

        Args:
            user_embs: Item embeddings sequence [batch, seq_len, emb_dim]
            z_t: Noisy target embedding [batch, emb_dim]
            t: Timestep [batch]
        """
        x = user_embs

        for i in range(self.num_blocks):
            x = HSTUBlock(
                attention_dim=self.attention_dim,
                linear_dim=self.linear_dim,
                num_heads=self.num_heads,
                attn_dropout_rate=self.attn_dropout_rate,
                linear_dropout_rate=self.linear_dropout_rate,
                enable_relative_attention_bias=True,
                max_sequence_len=self.max_sequence_len,
                name=f"hstu_block_{i}",
            )(x, deterministic=deterministic)

        h_user = x[:, -1, :]  # [batch, emb_dim]

        v_hat = FlowHead(
            hidden_dim=self.flow_hidden_dim,
            output_dim=self.embedding_dim,
            name="flow_head",
        )(h_user, z_t, t)

        return v_hat


class HSTUFlowCEModel(nn.Module):
    """HSTU backbone + FlowHead with shared output for Flow + CE.

    The flow head's output z_0_hat is used for both losses.
    """
    num_items: int
    embedding_dim: int = 256
    num_blocks: int = 4
    num_heads: int = 4
    attention_dim: int = 128
    linear_dim: int = 512
    flow_hidden_dim: int = 512
    attn_dropout_rate: float = 0.2
    linear_dropout_rate: float = 0.2
    max_sequence_len: int = 20

    @nn.compact
    def __call__(self, item_seq, z_t, t, deterministic=True):
        """Forward pass.

        Args:
            item_seq: Item ID sequence [batch, seq_len]
            z_t: Noisy target embedding [batch, emb_dim]
            t: Timestep [batch]

        Returns:
            v_hat: Flow velocity prediction [batch, emb_dim]
            emb_table: Embedding matrix [num_items+1, emb_dim]
        """
        # Shared embedding table
        embed_layer = nn.Embed(
            num_embeddings=self.num_items + 1,
            features=self.embedding_dim,
            name="item_embedding",
        )
        x = embed_layer(item_seq)  # [batch, seq_len, emb_dim]

        # HSTU encoder blocks
        for i in range(self.num_blocks):
            x = HSTUBlock(
                attention_dim=self.attention_dim,
                linear_dim=self.linear_dim,
                num_heads=self.num_heads,
                attn_dropout_rate=self.attn_dropout_rate,
                linear_dropout_rate=self.linear_dropout_rate,
                enable_relative_attention_bias=True,
                max_sequence_len=self.max_sequence_len,
                name=f"hstu_block_{i}",
            )(x, deterministic=deterministic)

        # User representation = last position
        h_user = x[:, -1, :]  # [batch, emb_dim]

        # Flow velocity prediction
        v_hat = FlowHead(
            hidden_dim=self.flow_hidden_dim,
            output_dim=self.embedding_dim,
            name="flow_head",
        )(h_user, z_t, t)

        emb_table = embed_layer.variables["params"]["embedding"]
        return v_hat, emb_table


class HSTUIDFlowModel(nn.Module):
    vocab_size: int
    latent_dim: int
    num_blocks: int
    num_heads: int
    attention_dim: int
    linear_dim: int
    max_sequence_len: int
    attn_dropout_rate: float = 0.2
    linear_dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x_seq, z_t=None, t=None, deterministic=False):
        # x_seq is [batch, seq_len] of item IDs
        embed_layer = nn.Embed(num_embeddings=self.vocab_size, features=self.latent_dim, name="item_embedding")
        x = embed_layer(x_seq)

        # Apply HSTU blocks
        for i in range(self.num_blocks):
            x = HSTUBlock(
                num_heads=self.num_heads,
                attention_dim=self.attention_dim,
                linear_dim=self.linear_dim,
                attn_dropout_rate=self.attn_dropout_rate,
                linear_dropout_rate=self.linear_dropout_rate,
                max_sequence_len=self.max_sequence_len,
                name=f"hstu_block_{i}",
            )(x, deterministic=deterministic)

        h_user = x[:, -1, :]  # [batch, latent_dim]

        if z_t is not None and t is not None:
            v_hat = FlowHead(
                hidden_dim=self.latent_dim * 2,
                output_dim=self.latent_dim,
                name="flow_head",
            )(h_user, z_t, t)
            return v_hat, embed_layer.variables["params"]["embedding"]
        else:
            return h_user, embed_layer.variables["params"]["embedding"]
