from unittest.mock import patch
import numpy as np

from datasets.base import pad_or_truncate, build_sequence_data
from datasets.movielens import MovieLensDataLoader
from datasets.amazon import AmazonDataLoader


def test_pad_or_truncate():
    # Test padding
    seq = [1, 2]
    padded = pad_or_truncate(seq, max_len=5, pad_val=0)
    assert padded == [0, 0, 0, 1, 2]

    # Test truncation
    seq_long = [1, 2, 3, 4, 5, 6]
    truncated = pad_or_truncate(seq_long, max_len=4, pad_val=0)
    assert truncated == [3, 4, 5, 6]

    # Test exact length
    seq_exact = [1, 2, 3]
    exact = pad_or_truncate(seq_exact, max_len=3, pad_val=0)
    assert exact == [1, 2, 3]


def test_build_sequence_data():
    # user history:
    # user 1: [10, 20, 30, 40]
    # user 2: [10, 30]
    user_history = {
        1: [10, 20, 30, 40],
        2: [10, 30],
    }

    # Test test split (leave-one-out, target is last)
    # user 1 input: [10, 20, 30], target: 40
    # user 2 input: [10], target: 30
    inputs, targets = build_sequence_data(user_history, max_len=3, split="test", pad_val=0)
    assert inputs == [[10, 20, 30], [0, 0, 10]]
    assert targets == [40, 30]

    # Test val split (leave-one-out, target is second to last)
    # user 1 input: [10, 20], target: 30
    # user 2 has length 2, which is < 3, so skipped
    inputs_val, targets_val = build_sequence_data(user_history, max_len=3, split="val", pad_val=0)
    assert inputs_val == [[0, 10, 20]]
    assert targets_val == [30]

    # Test train split (prefixes up to second-to-last item)
    # For user 1 ([10, 20, 30, 40]):
    # - t=1: input [10], target 20
    # - t=2: input [10, 20], target 30
    # For user 2 ([10, 30]):
    # - no prefixes because length is 2, range(1, 1) is empty.
    inputs_train, targets_train = build_sequence_data(user_history, max_len=3, split="train", pad_val=0)
    assert inputs_train == [[0, 0, 10]]
    assert targets_train == [20]


# Mock return data for MovieLens
mock_ml_interactions = [
    {"user": 1, "item": 101, "rating": 5, "timestamp": 1000},
    {"user": 1, "item": 102, "rating": 4, "timestamp": 1001},
    {"user": 1, "item": 103, "rating": 3, "timestamp": 1002},
    {"user": 2, "item": 101, "rating": 5, "timestamp": 1000},
    {"user": 2, "item": 103, "rating": 2, "timestamp": 1001},
]
mock_ml_titles = {101: "Toy Story", 102: "Jumanji", 103: "Grumpier Old Men"}


@patch("datasets.movielens.download_and_extract_movielens")
@patch("datasets.movielens.parse_movielens_data")
def test_movielens_loader(mock_parse, mock_download):
    mock_download.return_value = "/dummy/ml-100k"
    mock_parse.return_value = (mock_ml_interactions, mock_ml_titles)

    loader = MovieLensDataLoader(dataset_name="ml-100k", min_rating=0)

    # Check contiguous IDs mapping
    assert loader.num_users == 2
    assert loader.num_items == 3

    # Check test split (index format)
    test_ds = loader.get_split("test", max_len=5, format_type="index")
    # user 1 (mapped 1): item 101, 102, 103 -> mapped item IDs: 1, 2, 3
    # user 2 (mapped 2): item 101, 103 -> mapped item IDs: 1, 3
    # user 1 input: [1, 2], target 3
    # user 2 input: [1], target 3
    inputs, targets = test_ds.to_numpy()
    np.testing.assert_array_equal(inputs, np.array([[0, 0, 0, 1, 2], [0, 0, 0, 0, 1]]))
    np.testing.assert_array_equal(targets, np.array([3, 3]))

    # Check test split (text format)
    text_inputs, text_targets = loader.get_split("test", max_len=5, format_type="text")
    # user 1: inputs ["<pad>", "<pad>", "<pad>", "Toy Story", "Jumanji"], target "Grumpier Old Men"
    assert text_inputs[0] == ["<pad>", "<pad>", "<pad>", "Toy Story", "Jumanji"]
    assert text_targets[0] == "Grumpier Old Men"


# Mock return data for Amazon
mock_amzn_interactions = [
    {"user": "u1", "item": "asin1", "rating": 5.0, "timestamp": 1000},
    {"user": "u1", "item": "asin2", "rating": 4.0, "timestamp": 1001},
    {"user": "u1", "item": "asin3", "rating": 3.0, "timestamp": 1002},
    {"user": "u2", "item": "asin1", "rating": 5.0, "timestamp": 1000},
    {"user": "u2", "item": "asin3", "rating": 2.0, "timestamp": 1001},
]
mock_amzn_metadata = {"asin1": "Shampoo", "asin2": "Conditioner", "asin3": "Lipstick"}


@patch("datasets.amazon.download_amazon_files")
@patch("datasets.amazon.parse_amazon_reviews")
@patch("datasets.amazon.parse_amazon_metadata")
def test_amazon_loader(mock_parse_meta, mock_parse_rev, mock_download):
    mock_download.return_value = ("/dummy/reviews.json.gz", "/dummy/meta.json.gz")
    mock_parse_rev.return_value = mock_amzn_interactions
    mock_parse_meta.return_value = mock_amzn_metadata

    loader = AmazonDataLoader(category="beauty", load_metadata=True)

    assert loader.num_users == 2
    assert loader.num_items == 3

    # Check test split in text format
    text_inputs, text_targets = loader.get_split("test", max_len=3, format_type="text")
    assert text_inputs[0] == ["<pad>", "Shampoo", "Conditioner"]
    assert text_targets[0] == "Lipstick"
