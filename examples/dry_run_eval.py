"""Dry-run script to verify the evaluation framework on MovieLens-100K."""

import os
import numpy as np
from generative_recommendation.datasets.movielens import MovieLensDataLoader
from generative_recommendation.evaluation.evaluator import Evaluator


def main():
    print("--- Starting Dry Run for Evaluation Framework ---")

    # 1. Initialize data loader (downloads ML-100K if not present)
    data_dir = "./data"
    print(f"Loading MovieLens-100K dataset from {data_dir}...")
    loader = MovieLensDataLoader(dataset_name="ml-100k", data_dir=data_dir, min_rating=0)
    print(f"Dataset stats: Users = {loader.num_users}, Items = {loader.num_items}")

    # 2. Get test splits
    print("Generating train, validation, and test splits...")
    test_dataset = loader.get_split("test", max_len=10, format_type="index")
    inputs, targets = test_dataset.to_numpy()
    print(f"Test split: {len(targets)} samples")

    # 3. Create a dummy model prediction function for index-based evaluation
    # This dummy model assigns random scores to all items
    np.random.seed(42)
    def dummy_index_predict_fn(batch_inputs):
        batch_size = len(batch_inputs)
        # return random scores for all items (remember items are 1-indexed, size is loader.num_items + 1)
        return np.random.rand(batch_size, loader.num_items + 1)

    # 4. Run index-based evaluation
    print("Running index-based evaluation with dummy model...")
    evaluator = Evaluator(k_list=[1, 5, 10])
    index_results = evaluator.evaluate_index_based(
        dummy_index_predict_fn, inputs, targets, batch_size=128
    )

    print("\n--- Index-based Evaluation Results ---")
    for metric, score in index_results.items():
        print(f"{metric}: {score:.5f}")

    # 5. Get text-based test split
    print("\nGenerating text-based test split...")
    text_inputs, text_targets = loader.get_split("test", max_len=10, format_type="text")

    # 6. Create a dummy model prediction function for text-based evaluation
    # This dummy model generates the top-10 most popular movie titles
    # Find most popular movies in raw data
    item_counts = {}
    for hist in loader.user_history.values():
        for item in hist:
            item_counts[item] = item_counts.get(item, 0) + 1
    sorted_items = sorted(item_counts.keys(), key=lambda x: item_counts[x], reverse=True)
    top_titles = [loader.token_to_title[item] for item in sorted_items[:10]]

    print(f"Top 5 most popular movies: {top_titles[:5]}")

    def dummy_text_predict_fn(batch_inputs):
        batch_size = len(batch_inputs)
        # predict the top_titles for all inputs
        return [top_titles for _ in range(batch_size)]

    # 7. Run text-based evaluation
    print("Running text-based evaluation using Most Popular items...")
    text_results = evaluator.evaluate_generative_text(
        dummy_text_predict_fn, text_inputs, text_targets, batch_size=128, normalize=True
    )

    print("\n--- Text-based (Most Popular) Evaluation Results ---")
    for metric, score in text_results.items():
        print(f"{metric}: {score:.5f}")

    print("\n--- Dry Run Completed Successfully! ---")


if __name__ == "__main__":
    main()
