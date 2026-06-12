"""Evaluation metrics for generative recommendation systems.

Includes Hit Rate @ K (HR@K), Normalized Discounted Cumulative Gain @ K (NDCG@K),
and Mean Reciprocal Rank (MRR). Supports both index-based and text-based predictions.
"""

import jax.numpy as jnp
import numpy as np
from typing import Union, Sequence, Dict


def compute_ranks(
    scores: Union[np.ndarray, jnp.ndarray],
    targets: Union[np.ndarray, jnp.ndarray],
) -> Union[np.ndarray, jnp.ndarray]:
    """Computes the 1-based rank of the target items given scores for each item/candidate.

    Expected ranking is used to handle ties fairly (items with identical scores
    to the target share the rank mass uniformly).

    Args:
        scores: array of shape (batch_size, num_items) containing predictions.
        targets: array of shape (batch_size,) containing target item indices.

    Returns:
        ranks: array of shape (batch_size,) containing 1-based ranks.
    """
    # Select JAX or NumPy depending on input type
    xp = jnp if isinstance(scores, jnp.ndarray) or isinstance(targets, jnp.ndarray) else np

    # Extract target scores
    batch_size = scores.shape[0]
    batch_indices = xp.arange(batch_size)
    
    # target_scores shape: (batch_size, 1)
    target_scores = scores[batch_indices, targets][:, xp.newaxis]

    # Expected ranking is used to handle ties fairly
    # Rank = (number of scores strictly greater) + (number of equal scores + 1) / 2
    num_greater = xp.sum(scores > target_scores, axis=1)
    num_equal = xp.sum(scores == target_scores, axis=1)
    
    ranks = num_greater + (num_equal + 1.0) / 2.0
    return ranks


def hit_rate_at_k(
    ranks: Union[np.ndarray, jnp.ndarray],
    k: int,
) -> float:
    """Computes Hit Rate at K (HR@K).

    Args:
        ranks: array of shape (batch_size,) containing 1-based target ranks.
        k: Cut-off value.

    Returns:
        HR@K score.
    """
    xp = jnp if isinstance(ranks, jnp.ndarray) else np
    hits = ranks <= k
    return float(xp.mean(hits))


def ndcg_at_k(
    ranks: Union[np.ndarray, jnp.ndarray],
    k: int,
) -> float:
    """Computes Normalized Discounted Cumulative Gain at K (NDCG@K) for single targets.

    For a single target, NDCG is 1.0 / log2(rank + 1) if rank <= K, else 0.

    Args:
        ranks: array of shape (batch_size,) containing 1-based target ranks.
        k: Cut-off value.

    Returns:
        NDCG@K score.
    """
    xp = jnp if isinstance(ranks, jnp.ndarray) else np

    # Calculate log2(rank + 1)
    discounts = xp.log2(ranks + 1.0)
    ndcg = xp.where(ranks <= k, 1.0 / discounts, 0.0)
    return float(xp.mean(ndcg))


def mean_reciprocal_rank(
    ranks: Union[np.ndarray, jnp.ndarray],
) -> float:
    """Computes Mean Reciprocal Rank (MRR).

    Args:
        ranks: array of shape (batch_size,) containing 1-based target ranks.

    Returns:
        MRR score.
    """
    xp = jnp if isinstance(ranks, jnp.ndarray) else np
    mrr = 1.0 / ranks.astype(xp.float32)
    return float(xp.mean(mrr))


def compute_text_ranks(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[str],
    normalize: bool = True,
) -> np.ndarray:
    """Computes 1-based ranks for text-based predictions.

    Args:
        predictions: sequence of sequences, shape (batch_size, top_k), where each
          element is a predicted title.
        targets: sequence of target strings, shape (batch_size,).
        normalize: if True, normalizes strings (lowercase, alphanumeric characters only)
          before matching.

    Returns:
        ranks: numpy array of shape (batch_size,) containing 1-based ranks. If not found,
          rank is set to a large number (e.g. 999999) to indicate no match.
    """
    batch_size = len(targets)
    ranks = np.zeros(batch_size, dtype=np.int32)

    def _normalize(s: str) -> str:
        if not normalize:
            return s
        return "".join(c for c in s.lower() if c.isalnum())

    for i in range(batch_size):
        target_norm = _normalize(targets[i])
        found_idx = -1
        for idx, pred in enumerate(predictions[i]):
            if _normalize(pred) == target_norm:
                found_idx = idx
                break

        if found_idx != -1:
            ranks[i] = found_idx + 1
        else:
            ranks[i] = 999999  # Not found

    return ranks


def compute_ranks_from_predictions(
    predictions: Sequence[Sequence[Union[int, str]]],
    targets: Sequence[Union[int, str]],
) -> np.ndarray:
    """Computes 1-based target ranks from top-N predictions list.

    Args:
        predictions: sequence of sequences, shape (batch_size, top_k), where each
          element is a predicted item ID or string.
        targets: sequence of target items, shape (batch_size,).

    Returns:
        ranks: numpy array of shape (batch_size,) containing 1-based ranks. If not found,
          rank is set to a large number (e.g. 999999) to indicate no match.
    """
    batch_size = len(targets)
    ranks = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        target = targets[i]
        found_idx = -1
        for idx, pred in enumerate(predictions[i]):
            if pred == target:
                found_idx = idx
                break

        if found_idx != -1:
            ranks[i] = found_idx + 1
        else:
            ranks[i] = 999999  # Not found

    return ranks


def calculate_metrics_from_ranks(
    ranks: np.ndarray,
    k_list: Sequence[int] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """Computes all recommendation metrics given 1-based target ranks.

    Args:
        ranks: array of shape (batch_size,) containing 1-based target ranks.
        k_list: list of K values to compute HR@K and NDCG@K.

    Returns:
        A dictionary containing HR@K, NDCG@K, and MRR.
    """
    results = {}
    for k in k_list:
        results[f"HR@{k}"] = hit_rate_at_k(ranks, k)
        results[f"NDCG@{k}"] = ndcg_at_k(ranks, k)
    results["MRR"] = mean_reciprocal_rank(ranks)
    return results
