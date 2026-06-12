import os
import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from torch.utils.tensorboard import SummaryWriter

from datasets import MovieLensDataLoader, AmazonDataLoader, SteamDataLoader
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
            return v_hat
        return h_user

def main():
    parser = argparse.ArgumentParser(description="HSTU + Fixed T5 + VAE Compression")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tb_log_dir", type=str, default="./data/tensorboard/hstu_t5_vae")
    parser.add_argument("--dataset", type=str, default="steam")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--linear_dim", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--lambda_kl", type=float, default=0.1)
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--z_anchor_path", type=str, default="./data/steam_t5_embeddings.npy")
    args = parser.parse_args()

    # Load dataset
    if args.dataset == "steam":
        loader = SteamDataLoader(data_dir="./data")
    else:
        raise ValueError("Use steam")
    
    max_len = 20
    train_dataset = loader.get_split("train", max_len=max_len, format_type="index")
    train_in, train_tar = train_dataset.to_numpy()

    val_dataset = loader.get_split("val", max_len=max_len, format_type="index")
    val_in, val_tar = val_dataset.to_numpy()

    test_dataset = loader.get_split("test", max_len=max_len, format_type="index")
    test_in, test_tar = test_dataset.to_numpy()
    
    # Load Z_frozen
    if args.z_anchor_path == "./data/steam_t5_embeddings.npy" and args.dataset != "steam":
        args.z_anchor_path = f"./data/{args.dataset}_t5_embeddings.npy"
    Z_frozen = np.load(args.z_anchor_path).astype(np.float32)
    embed_dim_raw = Z_frozen.shape[1]
    
    Z_frozen_jnp = jax.device_put(jnp.array(Z_frozen))

    model = HSTUIDFlowModel(
        vocab_size=loader.num_items + 1,
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        max_sequence_len=max_len,
        attn_dropout_rate=args.dropout_rate,
        linear_dropout_rate=args.dropout_rate,
    )
    vae_enc = VAEEncoder(latent_dim=args.latent_dim)
    vae_dec = VAEDecoder(output_dim=embed_dim_raw)

    key, enc_key, dec_key = jax.random.split(jax.random.PRNGKey(42), 3)
    dummy_seq = jnp.zeros((1, max_len), dtype=jnp.int32)
    dummy_z = jnp.zeros((1, args.latent_dim))
    dummy_t = jnp.zeros((1,))
    
    params = model.init(key, dummy_seq, dummy_z, dummy_t)["params"]
    params = flax.core.unfreeze(params)
    params["vae_enc"] = vae_enc.init(enc_key, jnp.zeros((1, embed_dim_raw)))["params"]
    params["vae_dec"] = vae_dec.init(dec_key, jnp.zeros((1, args.latent_dim)))["params"]
    params = flax.core.freeze(params)

    optimizer = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_in, batch_frozen_tar, rng_key):
        def loss_fn(p):
            noise_rng, t_rng, reparam_rng, dropout_rng = jax.random.split(rng_key, 4)
            batch_size = batch_frozen_tar.shape[0]

            # VAE Encode
            mu, logvar = vae_enc.apply({"params": p["vae_enc"]}, batch_frozen_tar)
            std = jnp.exp(0.5 * logvar)
            eps_reparam = jax.random.normal(reparam_rng, mu.shape)
            z_latent_target = mu + std * eps_reparam

            kl_loss = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1))

            # Flow Matching
            epsilon = jax.random.normal(noise_rng, z_latent_target.shape)
            t = jax.random.uniform(t_rng, (batch_size,), minval=0.01, maxval=1.0)
            t_expand = t[:, None]
            
            z_t = (1 - t_expand) * z_latent_target + t_expand * epsilon
            v_target = epsilon - z_latent_target

            v_hat = model.apply(
                {"params": p}, batch_in, z_t, t,
                rngs={"dropout": dropout_rng},
                deterministic=False,
            )
            flow_loss = jnp.mean(jnp.sum((v_hat - v_target) ** 2, axis=-1))

            # VAE Decode & Recon
            e_hat = vae_dec.apply({"params": p["vae_dec"]}, z_latent_target)
            rec_loss = jnp.mean(jnp.sum((e_hat - batch_frozen_tar) ** 2, axis=-1))

            total_loss = flow_loss + args.lambda_kl * kl_loss + args.lambda_rec * rec_loss
            return total_loss, (flow_loss, kl_loss, rec_loss)

        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        flow_loss, kl_loss, rec_loss = aux
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, total_loss, flow_loss, kl_loss, rec_loss

    @jax.jit
    def get_h_user(params, batch_in):
        return model.apply({"params": params}, batch_in, deterministic=True)

    @jax.jit
    def predict_velocity(params, h_user, z_t, t):
        v_hat = FlowHead(
            hidden_dim=args.latent_dim * 2,
            output_dim=args.latent_dim,
            name="flow_head",
        ).apply({"params": params["flow_head"]}, h_user, z_t, t)
        return v_hat

    @jax.jit
    def denoise_n_steps(params, h_user, z_T, num_steps):
        dt = 1.0 / num_steps
        z = z_T
        def body_fun(i, z_val):
            t_val = 1.0 - i * dt
            t_batch = jnp.full((z_val.shape[0],), t_val)
            v_hat = predict_velocity(params, h_user, z_val, t_batch)
            return z_val - dt * v_hat
        return jax.lax.fori_loop(0, num_steps, body_fun, z)

    def evaluate_flow(params, eval_in, eval_tar, num_steps):
        batch_size = args.batch_size
        num_samples = len(eval_in)
        item_sq = jnp.sum(Z_frozen[1:] ** 2, axis=-1)

        all_ranks = []
        from tqdm import tqdm
        for i in tqdm(range(0, num_samples, batch_size), desc=f"Eval {num_steps} steps", leave=False):
            batch_in = eval_in[i:i+batch_size]
            batch_tar = eval_tar[i:i+batch_size]
            actual_bs = len(batch_tar)

            batch_in_jnp = jnp.array(batch_in)
            h_user = get_h_user(params, batch_in_jnp)

            h_user_rep = jnp.repeat(h_user, args.num_seeds, axis=0)
            noise_rng = jax.random.PRNGKey(i)
            z_T = jax.random.normal(noise_rng, (actual_bs * args.num_seeds, args.latent_dim))
            z_hat_latent = denoise_n_steps(params, h_user_rep, z_T, num_steps)
            
            z_hat = vae_dec.apply({"params": params["vae_dec"]}, z_hat_latent)

            z_hat_sq = jnp.sum(z_hat ** 2, axis=-1, keepdims=True)
            dot = z_hat @ Z_frozen_jnp[1:].T
            dists = z_hat_sq + item_sq[None, :] - 2 * dot

            top20_items = jnp.argsort(dists, axis=-1)[:, :20] + 1
            top20_items = np.array(top20_items).reshape(actual_bs, args.num_seeds, 20)

            for j in range(actual_bs):
                target_item = batch_tar[j]
                user_preds = top20_items[j].flatten()
                _, unique_indices = np.unique(user_preds, return_index=True)
                unique_preds = user_preds[np.sort(unique_indices)][:20]
                
                try:
                    rank = np.where(unique_preds == target_item)[0][0] + 1
                except IndexError:
                    rank = 0
                all_ranks.append(rank)

        hits_10 = sum(1 for r in all_ranks if 0 < r <= 10)
        ndcg_10 = sum(1.0 / np.log2(r + 1) for r in all_ranks if 0 < r <= 10)
        return {"HR@10": hits_10 / num_samples, "NDCG@10": ndcg_10 / num_samples}

    writer = SummaryWriter(args.tb_log_dir)
    print("Starting training...")

    best_hr = 0
    patience_cnt = 0
    global_step = 0
    epoch_rng = jax.random.PRNGKey(100)

    for epoch in range(1, args.epochs + 1):
        shuffled_idx = np.random.permutation(len(train_in))
        shuffled_in = train_in[shuffled_idx]
        shuffled_tar = train_tar[shuffled_idx]

        epoch_flow, epoch_kl, epoch_rec = 0.0, 0.0, 0.0
        start_time = time.time()
        num_batches = 0

        for i in range(0, len(train_in), args.batch_size):
            batch_in = shuffled_in[i:i+args.batch_size]
            batch_tar = shuffled_tar[i:i+args.batch_size]
            
            batch_in_jnp = jnp.array(batch_in)
            batch_frozen_tar = Z_frozen_jnp[batch_tar]

            epoch_rng, step_rng = jax.random.split(epoch_rng)
            params, opt_state, t_loss, f_loss, k_loss, r_loss = train_step(
                params, opt_state, batch_in_jnp, batch_frozen_tar, step_rng
            )

            epoch_flow += f_loss
            epoch_kl += k_loss
            epoch_rec += r_loss
            num_batches += 1
            global_step += 1

            if global_step % 10 == 0:
                writer.add_scalar("Loss/flow", float(f_loss), global_step)
                writer.add_scalar("Loss/kl", float(k_loss), global_step)
                writer.add_scalar("Loss/recon", float(r_loss), global_step)
                writer.add_scalar("Loss/total", float(t_loss), global_step)

        elapsed = time.time() - start_time
        avg_flow = epoch_flow / num_batches
        avg_kl = epoch_kl / num_batches
        avg_rec = epoch_rec / num_batches
        print(f"Epoch {epoch:02d}/{args.epochs} | Flow: {avg_flow:.2f}  KL: {avg_kl:.4f}  Recon: {avg_rec:.4f} | Time: {elapsed:.1f}s")

        if epoch % 5 == 0 or epoch == args.epochs:
            print("Evaluating...")
            for steps in [1, 3, 5, 10]:
                val_results = evaluate_flow(params, val_in, val_tar, num_steps=steps)
                hr, ndcg = val_results["HR@10"], val_results["NDCG@10"]
                print(f"  steps={steps:2d} | NDCG@10: {ndcg:.5f} | HR@10: {hr:.5f}")

            # Update best using 10 steps (last computed)
            if hr > best_hr:
                best_hr = hr
                patience_cnt = 0
                print(">>> New best! Saving...")
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print("Early stopping!")
                    break

if __name__ == "__main__":
    main()
