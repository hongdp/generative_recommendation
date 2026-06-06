"""MovieLens dataset downloader and preprocessor.

Supports MovieLens-100K (ml-100k) and MovieLens-1M (ml-1m) datasets.
"""

import os
import urllib.request
import zipfile
from typing import Dict, List, Tuple, Union

from datasets.base import SequenceDataset, build_sequence_data

# Dataset download URLs
URL_100K = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
URL_1M = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_and_extract_movielens(dataset_name: str, data_dir: str = "./data") -> str:
    """Downloads and extracts the specified MovieLens dataset.

    Args:
        dataset_name: Either 'ml-100k' or 'ml-1m'.
        data_dir: Directory where the dataset will be saved and extracted.

    Returns:
        The path to the extracted directory containing the data files.
    """
    os.makedirs(data_dir, exist_ok=True)
    if dataset_name == "ml-100k":
        url = URL_100K
        expected_dir = os.path.join(data_dir, "ml-100k")
    elif dataset_name == "ml-1m":
        url = URL_1M
        expected_dir = os.path.join(data_dir, "ml-1m")
    else:
        raise ValueError(f"Unknown MovieLens dataset name: {dataset_name}. Choose 'ml-100k' or 'ml-1m'.")

    zip_path = os.path.join(data_dir, f"{dataset_name}.zip")

    # Download if not present
    if not os.path.exists(expected_dir):
        if not os.path.exists(zip_path):
            print(f"Downloading {url} to {zip_path}...")
            urllib.request.urlretrieve(url, zip_path)

        print(f"Extracting {zip_path} to {data_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    return expected_dir


def parse_movielens_data(
    dataset_name: str, extracted_dir: str
) -> Tuple[List[Dict[str, int]], Dict[int, str]]:
    """Parses the ratings and movies files from MovieLens.

    Args:
        dataset_name: Either 'ml-100k' or 'ml-1m'.
        extracted_dir: Path to the directory containing the dataset files.

    Returns:
        A tuple of (interactions, item_to_title):
          - interactions: List of dicts, each with keys 'user', 'item', 'rating', 'timestamp'.
          - item_to_title: Mapping from raw item ID (int) to movie title (str).
    """
    interactions = []
    item_to_title = {}

    if dataset_name == "ml-100k":
        # Parse items (movies)
        item_path = os.path.join(extracted_dir, "u.item")
        with open(item_path, "r", encoding="ISO-8859-1") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    item_id = int(parts[0])
                    title = parts[1]
                    item_to_title[item_id] = title

        # Parse ratings
        data_path = os.path.join(extracted_dir, "u.data")
        with open(data_path, "r", encoding="ISO-8859-1") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 4:
                    user_id = int(parts[0])
                    item_id = int(parts[1])
                    rating = int(parts[2])
                    timestamp = int(parts[3])
                    interactions.append(
                        {
                            "user": user_id,
                            "item": item_id,
                            "rating": rating,
                            "timestamp": timestamp,
                        }
                    )

    elif dataset_name == "ml-1m":
        # Parse items (movies)
        item_path = os.path.join(extracted_dir, "movies.dat")
        with open(item_path, "r", encoding="ISO-8859-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) >= 2:
                    item_id = int(parts[0])
                    title = parts[1]
                    item_to_title[item_id] = title

        # Parse ratings
        data_path = os.path.join(extracted_dir, "ratings.dat")
        with open(data_path, "r", encoding="ISO-8859-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) == 4:
                    user_id = int(parts[0])
                    item_id = int(parts[1])
                    rating = int(parts[2])
                    timestamp = int(parts[3])
                    interactions.append(
                        {
                            "user": user_id,
                            "item": item_id,
                            "rating": rating,
                            "timestamp": timestamp,
                        }
                    )

    return interactions, item_to_title


class MovieLensDataLoader:
    """Preprocesses and provides SequenceDatasets for MovieLens sequential recommendation."""

    def __init__(
        self,
        dataset_name: str = "ml-100k",
        data_dir: str = "./data",
        min_rating: int = 0,
    ):
        """Initializes the loader, downloads data, parses files, and prepares item mappings.

        Args:
            dataset_name: Either 'ml-100k' or 'ml-1m'.
            data_dir: Root directory for downloading datasets.
            min_rating: If > 0, interactions with ratings below this are discarded.
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.min_rating = min_rating

        # Download & parse raw files
        extracted_dir = download_and_extract_movielens(dataset_name, data_dir)
        raw_interactions, self.raw_item_to_title = parse_movielens_data(dataset_name, extracted_dir)

        # Filter interactions by rating if requested
        if min_rating > 0:
            raw_interactions = [x for x in raw_interactions if x["rating"] >= min_rating]

        # Map original users/items to contiguous IDs (1-indexed, 0 is padding)
        unique_users = sorted(list(set(x["user"] for x in raw_interactions)))
        unique_items = sorted(list(set(x["item"] for x in raw_interactions)))

        self.user_to_id = {user: i + 1 for i, user in enumerate(unique_users)}
        self.item_to_id = {item: i + 1 for i, item in enumerate(unique_items)}
        self.id_to_item = {i + 1: item for i, item in enumerate(unique_items)}

        self.num_users = len(self.user_to_id)
        self.num_items = len(self.item_to_id)

        # Create token ID to title mapping
        self.token_to_title = {
            tok_id: self.raw_item_to_title.get(raw_id, f"Movie_{raw_id}")
            for tok_id, raw_id in self.id_to_item.items()
        }
        self.token_to_title[0] = "<pad>"  # Padding token

        # Group interactions by mapped user ID, sorted by timestamp
        self.user_history = {}
        for x in raw_interactions:
            user_id = self.user_to_id[x["user"]]
            item_id = self.item_to_id[x["item"]]
            timestamp = x["timestamp"]
            
            if user_id not in self.user_history:
                self.user_history[user_id] = []
            self.user_history[user_id].append((timestamp, item_id))

        # Sort histories chronologically and extract item IDs
        for user_id in self.user_history:
            self.user_history[user_id].sort(key=lambda x: x[0])
            self.user_history[user_id] = [item_id for _, item_id in self.user_history[user_id]]

    def get_split(
        self, split: str, max_len: int = 50, format_type: str = "index"
    ) -> Union[SequenceDataset, Tuple[List[List[str]], List[str]]]:
        """Generates sequence inputs and targets for a specific split.

        Args:
            split: Split type ('train', 'val', or 'test').
            max_len: Maximum historical context length.
            format_type: Output format. Either 'index' (returns SequenceDataset with item tokens)
              or 'text' (returns inputs and targets as list of lists of movie title strings).

        Returns:
            If format_type is 'index':
              A SequenceDataset containing integer token ID lists and target IDs.
            If format_type is 'text':
              A tuple of (inputs, targets) where inputs is list of lists of strings,
              and targets is a list of strings (movie titles).
        """
        inputs, targets = build_sequence_data(self.user_history, max_len, split, pad_val=0)

        if format_type == "index":
            return SequenceDataset(inputs, targets)
        elif format_type == "text":
            # Map sequence token IDs to title strings
            text_inputs = [
                [self.token_to_title[tok] for tok in seq]
                for seq in inputs
            ]
            text_targets = [self.token_to_title[tok] for tok in targets]
            return text_inputs, text_targets
        else:
            raise ValueError(f"Unknown format_type: {format_type}. Choose 'index' or 'text'.")
