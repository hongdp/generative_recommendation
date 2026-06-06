"""Evaluation runner for generative recommendation systems.

Coordinates the evaluation loop for index-based and text-based models.
"""

from typing import Callable, Dict, List, Sequence, Union
import numpy as np
import jax.numpy as jnp

from evaluation.metrics import (
    compute_ranks,
    compute_text_ranks,
    hit_rate_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
)


class Evaluator:
    """Orchestrates sequential recommendation evaluation for index-based and text-based models."""

    def __init__(self, k_list: Sequence[int] = (1, 5, 10)):
        """Initializes the evaluator.

        Args:
            k_list: list of K values to compute HR@K and NDCG@K.
        """
        self.k_list = sorted(list(k_list))
        self.max_k = self.k_list[-1]

    def evaluate_index_based(
        self,
        predict_fn: Callable[[np.ndarray], Union[np.ndarray, jnp.ndarray]],
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Evaluates an index-based recommendation model.

        Args:
            predict_fn: A function that takes a batch of input sequences (shape: [batch, seq_len])
              and returns prediction scores for all items (shape: [batch, num_items]).
            inputs: input sequences, shape (num_samples, seq_len).
            targets: target item indices, shape (num_samples,).
            batch_size: batch size for batching predictions.

        Returns:
            A dictionary containing metrics (e.g., HR@1, NDCG@5, MRR).
        """
        num_samples = len(inputs)
        all_ranks = []

        for i in range(0, num_samples, batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_targets = targets[i : i + batch_size]

            # Get scores from predictions
            batch_scores = predict_fn(batch_inputs)

            # Compute ranks for this batch
            batch_ranks = compute_ranks(batch_scores, batch_targets)
            
            # Convert to numpy array for aggregation
            if isinstance(batch_ranks, jnp.ndarray):
                batch_ranks = np.array(batch_ranks)
            all_ranks.extend(batch_ranks)

        all_ranks = np.array(all_ranks)
        
        # Calculate metrics
        results = {}
        for k in self.k_list:
            results[f"HR@{k}"] = hit_rate_at_k(all_ranks, k)
            results[f"NDCG@{k}"] = ndcg_at_k(all_ranks, k)
        results["MRR"] = mean_reciprocal_rank(all_ranks)

        return results

    def evaluate_generative_text(
        self,
        predict_fn: Callable[[Sequence[Sequence[str]]], Sequence[Sequence[str]]],
        inputs: Sequence[Sequence[str]],
        targets: Sequence[str],
        batch_size: int = 256,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """Evaluates a generative text-based recommendation model.

        Args:
            predict_fn: A function that takes a batch of text sequence histories (list of list of strings)
              and returns a list of top-N generated candidate strings (list of list of strings).
            inputs: list of input text histories.
            targets: list of target text items.
            batch_size: batch size for predictions.
            normalize: whether to normalize text during matching.

        Returns:
            A dictionary containing metrics (e.g., HR@1, NDCG@5, MRR).
        """
        num_samples = len(inputs)
        all_ranks = []

        for i in range(0, num_samples, batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_targets = targets[i : i + batch_size]

            # Get generated text recommendations (e.g., size: batch_size x top_k)
            batch_predictions = predict_fn(batch_inputs)

            # Compute text-based ranks
            batch_ranks = compute_text_ranks(
                batch_predictions, batch_targets, normalize=normalize
            )
            all_ranks.extend(batch_ranks)

        all_ranks = np.array(all_ranks)

        # Calculate metrics
        results = {}
        for k in self.k_list:
            results[f"HR@{k}"] = hit_rate_at_k(all_ranks, k)
            results[f"NDCG@{k}"] = ndcg_at_k(all_ranks, k)
        results["MRR"] = mean_reciprocal_rank(all_ranks)

        return results
