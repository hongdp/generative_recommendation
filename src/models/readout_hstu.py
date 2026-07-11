"""HSTU variant with explicit attention masks and logical position IDs.

Built for the dedicated-readout-token experiment: the same backbone must run

  (a) baseline  — causal attention over the item sequence, readout at item
      positions (the residual stream serves both as KV source and as the
      user-summary vector), and
  (b) treatment — the sequence is extended with per-fork ``<begin>`` branch
      tokens that are mutually invisible, each seeing only the context prefix
      up to its anchor position; readout happens exclusively at ``<begin>``.

Two deltas vs. :class:`models.hstu.HSTUBlock`:

1. The causal ``tril`` mask is replaced by a caller-supplied boolean mask
   [batch, N, N] (anchor-function masks for branch tokens, §3.3 of the design).
2. The relative attention bias is indexed by *logical* position differences
   (``pos[q] - pos[k]``) instead of physical ``arange`` differences, so a
   branch token anchored at context position t reproduces exactly the distance
   profile of a ``<begin>`` appended to a length-(t+1) sequence (§3.4).

The model returns final hidden states plus the output item-embedding table
(two-tower scoring happens in the training loss / eval code, enabling sampled
softmax rather than materializing full logits per position).
"""

from typing import Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp


class MaskedHSTUBlock(nn.Module):
    """HSTU block with external visibility mask and logical-position bias."""

    attention_dim: int = 128
    linear_dim: int = 512
    num_heads: int = 4
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1
    num_rel_buckets: int = 64

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,          # [batch, N, d]
        mask: jnp.ndarray,       # [batch, N, N] bool, True = visible
        rel_idx: jnp.ndarray,    # [N, N] int32, bucketed logical distances
        deterministic: bool = True,
    ) -> jnp.ndarray:
        batch_size, seq_len, embedding_dim = x.shape
        head_dim_a = self.attention_dim // self.num_heads
        head_dim_v = self.linear_dim // self.num_heads

        q = nn.Dense(self.attention_dim, name="q_proj")(x)
        k = nn.Dense(self.attention_dim, name="k_proj")(x)
        v = nn.Dense(self.linear_dim, name="v_proj")(x)
        u = nn.Dense(self.linear_dim, name="u_proj")(x)

        q = q.reshape((batch_size, seq_len, self.num_heads, head_dim_a))
        k = k.reshape((batch_size, seq_len, self.num_heads, head_dim_a))
        v = v.reshape((batch_size, seq_len, self.num_heads, head_dim_v))

        scores = jnp.einsum("bihd,bjhd->bhij", q, k) / jnp.sqrt(head_dim_a)

        rel_pos_bias = self.param(
            "rel_pos_bias", nn.initializers.zeros, (self.num_heads, self.num_rel_buckets)
        )
        scores = scores + rel_pos_bias[:, rel_idx][None, :, :, :]

        A = jax.nn.silu(scores)
        A = A * mask[:, None, :, :]
        A = nn.Dropout(rate=self.attn_dropout_rate)(A, deterministic=deterministic)

        Z = jnp.einsum("bhij,bjhd->bihd", A, v)
        Z = Z.reshape((batch_size, seq_len, self.linear_dim))
        u = u.reshape((batch_size, seq_len, self.linear_dim))

        Z_norm = nn.LayerNorm()(Z)
        Z_gated = Z_norm * u
        Z_gated = nn.Dropout(rate=self.linear_dropout_rate)(Z_gated, deterministic=deterministic)

        output = nn.Dense(embedding_dim, name="out_proj")(Z_gated)
        return output + x


class ReadoutHSTUModel(nn.Module):
    """Two-tower sequential model: HSTU encoder + learnable output item table.

    Vocabulary: 0 = padding, 1..num_items = items, num_items + 1 = ``<begin>``.
    Returns (hidden_states [batch, N, d], out_table [num_items + 1, d]); the
    caller scores readout positions against ``out_table`` by dot product.
    """

    num_items: int
    embedding_dim: int = 256
    num_blocks: int = 4
    num_heads: int = 4
    attention_dim: int = 128
    linear_dim: int = 512
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1
    num_rel_buckets: int = 64
    tie_output: bool = False
    feat_zero_init: bool = False

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,     # [batch, N]
        mask: jnp.ndarray,       # [batch, N, N] bool
        rel_idx: jnp.ndarray,    # [N, N] int32
        deterministic: bool = True,
        feats: jnp.ndarray = None,       # optional [batch, N, F] request-side features
        feat_mask: jnp.ndarray = None,   # [batch, N, 1], 1 where feats apply (branch rows)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        embed_layer = nn.Embed(
            num_embeddings=self.num_items + 2,
            features=self.embedding_dim,
            name="item_embedding",
        )
        x = embed_layer(tokens)
        if feats is not None:
            # Request-side conditioning of readout tokens (design doc §10):
            # a learned projection added onto the <begin> input embeddings only.
            # Zero init makes step 0 exactly the unconditioned baseline (the
            # same symmetry-breaking logic as alpha_init in the reinject arm).
            init = nn.initializers.zeros if self.feat_zero_init else nn.linear.default_kernel_init
            x = x + nn.Dense(self.embedding_dim, name="feat_proj", kernel_init=init)(feats) * feat_mask

        for i in range(self.num_blocks):
            x = MaskedHSTUBlock(
                attention_dim=self.attention_dim,
                linear_dim=self.linear_dim,
                num_heads=self.num_heads,
                attn_dropout_rate=self.attn_dropout_rate,
                linear_dropout_rate=self.linear_dropout_rate,
                num_rel_buckets=self.num_rel_buckets,
                name=f"hstu_block_{i}",
            )(x, mask, rel_idx, deterministic=deterministic)

        if self.tie_output:
            out_table = embed_layer.variables["params"]["embedding"][: self.num_items + 1]
        else:
            out_table = self.param(
                "out_embedding",
                nn.initializers.normal(stddev=0.02),
                (self.num_items + 1, self.embedding_dim),
            )
        return x, out_table
