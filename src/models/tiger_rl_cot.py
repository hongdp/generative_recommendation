"""TIGER RL-CoT Sequence-to-Sequence Model with Gumbel-Softmax discrete routing."""

import jax
import jax.numpy as jnp
import flax.linen as nn

def gumbel_softmax(logits, temperature, prng_key, hard=False):
    gumbels = jax.random.gumbel(prng_key, shape=logits.shape)
    y_soft = jax.nn.softmax((logits + gumbels) / temperature)
    
    if hard:
        # Straight-through estimator
        index = jnp.argmax(y_soft, axis=-1)
        y_hard = jax.nn.one_hot(index, logits.shape[-1])
        y = jax.lax.stop_gradient(y_hard - y_soft) + y_soft
    else:
        y = y_soft
    return y


class EncoderBlock(nn.Module):
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
        x_norm = nn.LayerNorm(name="attn_ln")(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attention_dim,
            dropout_rate=self.attn_dropout_rate,
            name="self_attention",
        )(x_norm, mask=mask, deterministic=deterministic)
        x = x + attn_out

        x_norm2 = nn.LayerNorm(name="ffn_ln")(x)
        ffn_out = nn.Dense(self.linear_dim, name="ffn_dense_1")(x_norm2)
        ffn_out = nn.gelu(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(ffn_out, deterministic=deterministic)
        ffn_out = nn.Dense(x.shape[-1], name="ffn_dense_2")(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(ffn_out, deterministic=deterministic)
        return x + ffn_out


class DecoderBlock(nn.Module):
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
        x_norm = nn.LayerNorm(name="self_attn_ln")(x)
        self_attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attention_dim,
            dropout_rate=self.attn_dropout_rate,
            name="self_attention",
        )(x_norm, mask=self_attn_mask, deterministic=deterministic)
        x = x + self_attn_out

        x_norm2 = nn.LayerNorm(name="cross_attn_ln")(x)
        cross_attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.attention_dim,
            dropout_rate=self.attn_dropout_rate,
            name="cross_attention",
        )(x_norm2, encoder_outputs, mask=cross_attn_mask, deterministic=deterministic)
        x = x + cross_attn_out

        x_norm3 = nn.LayerNorm(name="ffn_ln")(x)
        ffn_out = nn.Dense(self.linear_dim, name="ffn_dense_1")(x_norm3)
        ffn_out = nn.gelu(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(ffn_out, deterministic=deterministic)
        ffn_out = nn.Dense(x.shape[-1], name="ffn_dense_2")(ffn_out)
        ffn_out = nn.Dropout(rate=self.linear_dropout_rate)(ffn_out, deterministic=deterministic)
        return x + ffn_out


class TIGERRLCoTModel(nn.Module):
    num_items: int
    vocab_size: int = 770
    embedding_dim: int = 384
    num_blocks: int = 4
    num_heads: int = 6
    attention_dim: int = 384
    linear_dim: int = 1024
    attn_dropout_rate: float = 0.1
    linear_dropout_rate: float = 0.1
    max_encoder_len: int = 20
    max_decoder_len: int = 4

    def setup(self):
        self.item_embedding = nn.Embed(
            num_embeddings=self.num_items + 1,
            features=self.embedding_dim,
            name="item_embedding",
        )
        self.token_embedding = nn.Embed(
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
        self.e_out_proj = nn.Dense(768, name="e_out_proj")

    def __call__(
        self,
        encoder_tokens: jnp.ndarray,
        deterministic: bool = True,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> tuple:
        # 1. Encoder Forward
        encoder_outputs = self.encode(encoder_tokens, deterministic=deterministic)
        enc_mask = (encoder_tokens != 0)[:, None, None, :]

        # 2. Autoregressive Rollout using Gumbel-Softmax
        batch_size = encoder_tokens.shape[0]
        start_token_id = self.vocab_size - 1
        
        emb_0 = self.token_embedding(jnp.full((batch_size,), start_token_id, dtype=jnp.int32))
        dec_emb_seq = emb_0[:, None, :]
        
        def run_decoder(dec_embs):
            seq_len = dec_embs.shape[1]
            y = dec_embs + self.dec_pos[None, :seq_len, :]
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            causal_mask = causal_mask[None, None, :, :]
            
            for i in range(self.num_blocks):
                y = self.decoder_blocks[i](
                    y,
                    encoder_outputs,
                    self_attn_mask=causal_mask,
                    cross_attn_mask=enc_mask,
                    deterministic=deterministic,
                )
            return y
            
        shared_weights = self.token_embedding.variables["params"]["embedding"]
        gumbel_rng = self.make_rng("gumbel")
        
        # Step 1: Predict c1
        y1 = run_decoder(dec_emb_seq)
        logits1 = jnp.dot(y1[:, -1, :], shared_weights.T)
        c1_logits = logits1[:, 1:257]
        gumbel_rng, step_rng = jax.random.split(gumbel_rng)
        c1_soft = gumbel_softmax(c1_logits, temperature, step_rng, hard=hard)
        c1_emb = jnp.dot(c1_soft, shared_weights[1:257])
        
        # Step 2: Predict c2
        dec_emb_seq = jnp.concatenate([dec_emb_seq, c1_emb[:, None, :]], axis=1)
        y2 = run_decoder(dec_emb_seq)
        logits2 = jnp.dot(y2[:, -1, :], shared_weights.T)
        c2_logits = logits2[:, 257:513]
        gumbel_rng, step_rng = jax.random.split(gumbel_rng)
        c2_soft = gumbel_softmax(c2_logits, temperature, step_rng, hard=hard)
        c2_emb = jnp.dot(c2_soft, shared_weights[257:513])
        
        # Step 3: Predict c3
        dec_emb_seq = jnp.concatenate([dec_emb_seq, c2_emb[:, None, :]], axis=1)
        y3 = run_decoder(dec_emb_seq)
        logits3 = jnp.dot(y3[:, -1, :], shared_weights.T)
        c3_logits = logits3[:, 513:769]
        gumbel_rng, step_rng = jax.random.split(gumbel_rng)
        c3_soft = gumbel_softmax(c3_logits, temperature, step_rng, hard=hard)
        c3_emb = jnp.dot(c3_soft, shared_weights[513:769])
        
        # Step 4: Predict e_out
        dec_emb_seq = jnp.concatenate([dec_emb_seq, c3_emb[:, None, :]], axis=1)
        y4 = run_decoder(dec_emb_seq)
        e_out = self.e_out_proj(y4[:, -1, :])
        
        return e_out, (c1_soft, c2_soft, c3_soft)

    def encode(self, encoder_tokens: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        enc_emb = self.item_embedding(encoder_tokens)
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
    ) -> tuple:
        dec_emb = self.token_embedding(decoder_tokens)
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

        shared_weights = self.token_embedding.variables["params"]["embedding"]
        last_y = y[:, -1, :]
        logits = jnp.dot(last_y, shared_weights.T)
        e_outs = self.e_out_proj(last_y)

        return logits, e_outs
