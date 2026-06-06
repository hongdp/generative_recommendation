import numpy as np
import jax.numpy as jnp
from generative_recommendation.evaluation.metrics import (
    compute_ranks,
    hit_rate_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    compute_text_ranks,
)


def test_compute_ranks_numpy():
    # scores: shape (2, 4)
    # targets: shape (2,)
    scores = np.array([
        [0.1, 0.8, 0.2, 0.4],  # target 2 (score 0.2), should be ranked 3rd (0.8, 0.4 are larger)
        [0.9, 0.2, 0.3, 0.1]   # target 0 (score 0.9), should be ranked 1st
    ])
    targets = np.array([2, 0])

    ranks = compute_ranks(scores, targets)
    np.testing.assert_array_equal(ranks, np.array([3, 1]))


def test_compute_ranks_jax():
    scores = jnp.array([
        [0.1, 0.8, 0.2, 0.4],
        [0.9, 0.2, 0.3, 0.1]
    ])
    targets = jnp.array([2, 0])

    ranks = compute_ranks(scores, targets)
    np.testing.assert_array_equal(ranks, np.array([3, 1]))


def test_metrics_calculation():
    # ranks: 1, 3, 5, 12
    ranks = np.array([1, 3, 5, 12])

    # HR@5: 1, 3, 5 are <= 5 (3 out of 4) -> 0.75
    # HR@1: 1 is <= 1 (1 out of 4) -> 0.25
    assert hit_rate_at_k(ranks, k=5) == 0.75
    assert hit_rate_at_k(ranks, k=1) == 0.25
    assert hit_rate_at_k(ranks, k=10) == 0.75

    # NDCG@5:
    # rank 1: 1 / log2(2) = 1.0
    # rank 3: 1 / log2(4) = 0.5
    # rank 5: 1 / log2(6) = 0.38685
    # rank 12: 0 (since > 5)
    # mean: (1.0 + 0.5 + 0.38685) / 4 = 1.88685 / 4 = 0.47171
    expected_ndcg = (1.0 + 0.5 + 1.0 / np.log2(6.0)) / 4.0
    assert np.isclose(ndcg_at_k(ranks, k=5), expected_ndcg)

    # MRR: (1/1 + 1/3 + 1/5 + 1/12) / 4 = 1.61666 / 4 = 0.40416
    expected_mrr = (1.0 / 1.0 + 1.0 / 3.0 + 1.0 / 5.0 + 1.0 / 12.0) / 4.0
    assert np.isclose(mean_reciprocal_rank(ranks), expected_mrr)


def test_compute_text_ranks():
    predictions = [
        ["Toy Story", "Jumanji", "Grumpier Old Men"],
        ["Father of the Bride", "Heat", "GoldenEye"]
    ]
    targets = ["jumanji ", "goldeneye"]

    ranks = compute_text_ranks(predictions, targets, normalize=True)
    np.testing.assert_array_equal(ranks, np.array([2, 3]))

    # Test not found
    targets_not_found = ["jumanji", "Toy Story 2"]
    ranks_not_found = compute_text_ranks(predictions, targets_not_found, normalize=True)
    np.testing.assert_array_equal(ranks_not_found, np.array([2, 999999]))
