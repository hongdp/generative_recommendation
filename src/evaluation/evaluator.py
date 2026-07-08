"""Evaluation runner for generative recommendation systems.

Provides a unified Evaluator class with three evaluation paradigms:
  1. Index-based: Full score matrix → rank computation (HSTU, SASRec, etc.)
  2. Generative-discrete: Beam search → Semantic ID → Item ID mapping (TIGER)
  3. Generative-continuous: Flow denoising → ANN retrieval (Flow Matching)

All methods return a consistent Dict[str, float] with HR@K, NDCG@K, MRR,
plus paradigm-specific extras (Valid@1, Valid@Beam, Diversity).
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import jax.numpy as jnp

from tqdm import tqdm

from evaluation.metrics import (
    compute_ranks,
    compute_ranks_from_predictions,
    calculate_metrics_from_ranks,
)


class Evaluator:
    """Orchestrates sequential recommendation evaluation across all model types."""

    def __init__(self, k_list: Sequence[int] = (1, 5, 10, 20)):
        """Initializes the evaluator.

        Args:
            k_list: list of K values to compute HR@K and NDCG@K.
        """
        self.k_list = sorted(list(k_list))
        self.max_k = self.k_list[-1]

    # =========================================================================
    # 1. Index-Based Evaluation (HSTU, SASRec, etc.)
    # =========================================================================
    def evaluate_index_based(
        self,
        predict_fn: Callable[[np.ndarray], Union[np.ndarray, jnp.ndarray]],
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Evaluates an index-based recommendation model.

        The model produces a full score matrix [batch, num_items]. We compute
        ranks by comparing the target item's score against all others.

        Args:
            predict_fn: A function that takes a batch of input sequences
              (shape: [batch, seq_len]) and returns prediction scores for all
              items (shape: [batch, num_items]).
            inputs: input sequences, shape (num_samples, seq_len).
            targets: target item indices, shape (num_samples,).
            batch_size: batch size for batching predictions.

        Returns:
            A dictionary containing metrics (HR@K, NDCG@K, MRR).
        """
        num_samples = len(inputs)
        all_ranks = []

        for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating", leave=False):
            batch_inputs = inputs[i : i + batch_size]
            batch_targets = targets[i : i + batch_size]

            batch_scores = predict_fn(batch_inputs)
            batch_ranks = compute_ranks(batch_scores, batch_targets)

            if isinstance(batch_ranks, jnp.ndarray):
                batch_ranks = np.array(batch_ranks)
            all_ranks.extend(batch_ranks)

        all_ranks = np.array(all_ranks)
        return calculate_metrics_from_ranks(all_ranks, self.k_list)

    # =========================================================================
    # 2. Generative-Discrete Evaluation (TIGER beam search)
    # =========================================================================
    def evaluate_generative_discrete(
        self,
        decode_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        semantic_id_to_item: Dict[tuple, int],
        inputs: np.ndarray,
        targets: np.ndarray,
        beam_size: int = 20,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Evaluates a generative model that decodes discrete Semantic IDs via beam search.

        The decode_fn produces top-B Semantic ID paths (c1, ..., cL) which are
        mapped back to item IDs via semantic_id_to_item. Invalid paths map to 0.

        Args:
            decode_fn: A function that takes a batch of encoded input tokens
              (shape: [batch, seq_len]) and returns a tuple of L arrays (one per
              Semantic-ID level), each of shape [batch, beam_size].
            semantic_id_to_item: mapping from (c1, ..., cL) tuple to item ID.
            inputs: encoded input tokens, shape (num_samples, token_len).
            targets: target item indices, shape (num_samples,).
            beam_size: number of beams in beam search.
            batch_size: batch size for decoding.

        Returns:
            A dictionary containing metrics (HR@K, NDCG@K, MRR, Valid@1, Valid@Beam).
        """
        num_samples = len(inputs)
        all_ranks = []

        total_paths = 0
        valid_paths = 0
        total_top1_paths = 0
        valid_top1_paths = 0

        for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating", leave=False):
            batch_in = inputs[i : i + batch_size]
            batch_tar = targets[i : i + batch_size]

            level_codes = decode_fn(batch_in)  # tuple of L arrays, each [batch, B]

            batch_predictions = []
            for j in range(len(batch_in)):
                sample_preds = []
                for b in range(beam_size):
                    path = tuple(int(codes[j, b]) for codes in level_codes)
                    is_valid = path in semantic_id_to_item
                    mapped = semantic_id_to_item.get(path, 0)
                    if isinstance(mapped, (list, tuple)):
                        # Collision-aware mapping: every item sharing this Semantic
                        # ID enters the ranking here, pre-sorted by prior (see
                        # build_semantic_id_to_items). Later beams are pushed down
                        # accordingly, which is the honest ranking semantics.
                        sample_preds.extend(mapped)
                    else:
                        sample_preds.append(mapped)

                    total_paths += 1
                    if is_valid:
                        valid_paths += 1
                    if b == 0:
                        total_top1_paths += 1
                        if is_valid:
                            valid_top1_paths += 1

                batch_predictions.append(sample_preds)

            batch_ranks = compute_ranks_from_predictions(batch_predictions, batch_tar)
            all_ranks.extend(batch_ranks)

        ranks = np.array(all_ranks)
        results = calculate_metrics_from_ranks(ranks, self.k_list)
        results["Valid@1"] = float(valid_top1_paths) / max(total_top1_paths, 1)
        results["Valid@Beam"] = float(valid_paths) / max(total_paths, 1)
        return results

    # =========================================================================
    # 3. Generative-Continuous Evaluation (Flow Matching + ANN)
    # =========================================================================
    def evaluate_generative_continuous(
        self,
        generate_fn: Callable[[np.ndarray, jnp.ndarray], jnp.ndarray],
        embedding_table: Union[np.ndarray, jnp.ndarray],
        inputs: np.ndarray,
        targets: np.ndarray,
        num_seeds: int = 20,
        batch_size: int = 256,
        metric: str = "l2",
        desc: str = "Evaluating",
    ) -> Dict[str, float]:
        """Evaluates a generative model that produces continuous embeddings via flow denoising.

        The generate_fn takes a batch of input sequences and a noise tensor,
        then returns denoised embeddings. We retrieve the nearest items via
        L2 distance or cosine similarity.

        Args:
            generate_fn: A function (batch_in, z_noise) -> z_hat where:
              - batch_in: input sequences, shape [batch, seq_len]
              - z_noise: noise tensor, shape [batch * num_seeds, embed_dim]
              Returns z_hat of shape [batch * num_seeds, embed_dim].
            embedding_table: item embedding table of shape [num_items + 1, embed_dim].
              Index 0 is padding and is excluded from retrieval.
            inputs: input sequences, shape (num_samples, seq_len).
            targets: target item indices, shape (num_samples,).
            num_seeds: number of noise seeds per user for multi-seed sampling.
            batch_size: batch size for generation.
            metric: distance metric, either "l2" or "cosine".
            desc: description for tqdm progress bar.

        Returns:
            A dictionary containing metrics (HR@K, NDCG@K, MRR, Diversity).
        """
        num_samples = len(inputs)
        embed_dim = embedding_table.shape[-1]

        # Precompute embedding table properties (exclude padding at index 0)
        emb_table = jnp.array(embedding_table[1:]) if not isinstance(embedding_table, jnp.ndarray) else embedding_table[1:]
        if metric == "l2":
            item_sq = jnp.sum(emb_table ** 2, axis=-1)
        elif metric == "cosine":
            emb_norms = jnp.linalg.norm(emb_table, axis=-1, keepdims=True) + 1e-8
            emb_normed = emb_table / emb_norms
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'l2' or 'cosine'.")

        all_ranks = []
        total_unique = 0
        total_predictions = 0

        for i in tqdm(range(0, num_samples, batch_size), desc=desc, leave=False):
            batch_in = inputs[i : i + batch_size]
            batch_tar = targets[i : i + batch_size]
            actual_bs = len(batch_tar)

            # Sample noise
            noise_rng_key = i  # deterministic seed per batch for reproducibility
            import jax
            z_T = jax.random.normal(
                jax.random.PRNGKey(noise_rng_key),
                (actual_bs * num_seeds, embed_dim),
            )

            # Generate denoised embeddings
            z_hat = generate_fn(batch_in, z_T)

            if metric == "l2":
                # L2 distance: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
                z_hat_sq = jnp.sum(z_hat ** 2, axis=-1, keepdims=True)
                dot = z_hat @ emb_table.T
                dists = z_hat_sq + item_sq[None, :] - 2 * dot

                top1_items = jnp.argmin(dists, axis=-1) + 1
                top1_scores = -jnp.min(dists, axis=-1)  # negate so higher = better
            else:  # cosine
                z_hat_norm = z_hat / (jnp.linalg.norm(z_hat, axis=-1, keepdims=True) + 1e-8)
                scores = z_hat_norm @ emb_normed.T
                top1_items = jnp.argmax(scores, axis=-1) + 1
                top1_scores = jnp.max(scores, axis=-1)

            top1_items = np.array(top1_items).reshape(actual_bs, num_seeds)
            top1_scores = np.array(top1_scores).reshape(actual_bs, num_seeds)

            # Build prediction lists sorted by confidence (highest score first)
            batch_predictions = []
            for j in range(actual_bs):
                sorted_indices = np.argsort(-top1_scores[j])
                preds = [int(top1_items[j, idx]) for idx in sorted_indices]
                batch_predictions.append(preds)
                total_unique += len(set(preds))
                total_predictions += len(preds)

            batch_ranks = compute_ranks_from_predictions(batch_predictions, batch_tar)
            all_ranks.extend(batch_ranks)

        ranks = np.array(all_ranks)
        results = calculate_metrics_from_ranks(ranks, self.k_list)
        results["Diversity"] = total_unique / max(total_predictions, 1)
        return results
