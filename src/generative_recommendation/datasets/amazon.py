"""Amazon Reviews dataset downloader and preprocessor.

Supports standard SNAP 5-core datasets: Beauty, Toys and Games, Sports and Outdoors.
"""

import ast
import gzip
import json
import os
import urllib.request
from typing import Dict, List, Tuple, Union

from generative_recommendation.datasets.base import SequenceDataset, build_sequence_data

# Mapping of user-friendly names to SNAP category names
CATEGORY_MAP = {
    "beauty": "Beauty",
    "toys": "Toys_and_Games",
    "sports": "Sports_and_Outdoors",
}

BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"


def download_amazon_files(
    category_key: str, load_metadata: bool = True, data_dir: str = "./data"
) -> Tuple[str, Union[str, None]]:
    """Downloads Amazon review and metadata files from SNAP.

    Args:
        category_key: Dataset name ('beauty', 'toys', or 'sports').
        load_metadata: Whether to download product metadata.
        data_dir: Root data directory.

    Returns:
        A tuple of (reviews_file_path, metadata_file_path).
    """
    category_name = CATEGORY_MAP.get(category_key.lower())
    if not category_name:
        raise ValueError(
            f"Unknown Amazon category: {category_key}. Choose from: {list(CATEGORY_MAP.keys())}"
        )

    amazon_dir = os.path.join(data_dir, "amazon", category_name)
    os.makedirs(amazon_dir, exist_ok=True)

    reviews_filename = f"reviews_{category_name}_5.json.gz"
    reviews_url = f"{BASE_URL}{reviews_filename}"
    reviews_path = os.path.join(amazon_dir, reviews_filename)

    # Download reviews
    if not os.path.exists(reviews_path):
        print(f"Downloading {reviews_url} to {reviews_path}...")
        urllib.request.urlretrieve(reviews_url, reviews_path)

    metadata_path = None
    if load_metadata:
        metadata_filename = f"meta_{category_name}.json.gz"
        metadata_url = f"{BASE_URL}{metadata_filename}"
        metadata_path = os.path.join(amazon_dir, metadata_filename)

        # Download metadata
        if not os.path.exists(metadata_path):
            print(f"Downloading {metadata_url} to {metadata_path}...")
            try:
                urllib.request.urlretrieve(metadata_url, metadata_path)
            except Exception as e:
                print(f"Failed to download metadata: {e}. Falling back to ASINs.")
                metadata_path = None

    return reviews_path, metadata_path


def parse_amazon_reviews(file_path: str) -> List[Dict[str, Union[str, int, float]]]:
    """Parses Amazon reviews .json.gz file.

    Each line is a JSON object representing a review.

    Args:
        file_path: Path to the gzipped reviews JSON file.

    Returns:
        List of interactions.
    """
    interactions = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                interactions.append(
                    {
                        "user": data["reviewerID"],
                        "item": data["asin"],
                        "rating": float(data["overall"]),
                        "timestamp": int(data["unixReviewTime"]),
                    }
                )
            except Exception:
                # Fallback or log if single lines are malformed
                continue
    return sorted(interactions, key=lambda x: x["timestamp"])


def parse_amazon_metadata(file_path: str) -> Dict[str, str]:
    """Parses Amazon product metadata .json.gz file.

    Note that SNAP metadata files are frequently formatted as raw Python dict string representation
    rather than strictly compliant JSON.

    Args:
        file_path: Path to the gzipped metadata file.

    Returns:
        Mapping from product ASIN to title string.
    """
    asin_to_title = {}
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            # Try parsing as standard JSON first, fallback to ast.literal_eval if needed
            try:
                data = json.loads(line)
            except Exception:
                try:
                    data = ast.literal_eval(line)
                except Exception:
                    continue

            if isinstance(data, dict) and "asin" in data:
                asin = data["asin"]
                title = data.get("title", f"Product_{asin}")
                asin_to_title[asin] = title
    return asin_to_title


class AmazonDataLoader:
    """Preprocesses and provides SequenceDatasets for SNAP Amazon Reviews."""

    def __init__(
        self,
        category: str = "beauty",
        data_dir: str = "./data",
        min_rating: float = 0.0,
        load_metadata: bool = True,
    ):
        """Initializes the loader, downloads data, and prepares item mappings.

        Args:
            category: Subset name ('beauty', 'toys', or 'sports').
            data_dir: Root directory for downloading datasets.
            min_rating: Discard reviews with ratings below this threshold.
            load_metadata: Whether to download and parse product titles.
        """
        self.category = category
        self.data_dir = data_dir
        self.min_rating = min_rating

        # Download & parse raw files
        reviews_path, metadata_path = download_amazon_files(category, load_metadata, data_dir)
        raw_interactions = parse_amazon_reviews(reviews_path)

        if metadata_path:
            self.raw_item_to_title = parse_amazon_metadata(metadata_path)
        else:
            self.raw_item_to_title = {}

        # Filter interactions by rating
        if min_rating > 0.0:
            raw_interactions = [x for x in raw_interactions if x["rating"] >= min_rating]

        # Map original users/items (strings) to contiguous IDs (1-indexed, 0 is padding)
        unique_users = sorted(list(set(x["user"] for x in raw_interactions)))
        unique_items = sorted(list(set(x["item"] for x in raw_interactions)))

        self.user_to_id = {user: i + 1 for i, user in enumerate(unique_users)}
        self.item_to_id = {item: i + 1 for i, item in enumerate(unique_items)}
        self.id_to_item = {i + 1: item for i, item in enumerate(unique_items)}

        self.num_users = len(self.user_to_id)
        self.num_items = len(self.item_to_id)

        # Create token ID to title/text mapping
        self.token_to_title = {
            tok_id: self.raw_item_to_title.get(asin, f"Product_{asin}")
            for tok_id, asin in self.id_to_item.items()
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
              or 'text' (returns inputs and targets as list of lists of product title strings).

        Returns:
            If format_type is 'index':
              A SequenceDataset containing integer token ID lists and target IDs.
            If format_type is 'text':
              A tuple of (inputs, targets) where inputs is list of lists of strings,
              and targets is a list of strings (product titles).
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
