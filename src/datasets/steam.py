"""Steam Reviews dataset downloader and preprocessor.

Supports downloading and parsing W.C. Kang's UCSD Steam datasets and performing iterative 5-core filtering.
"""

import ast
import gzip
import json
import os
import urllib.request
from typing import Dict, List, Tuple, Union

from datasets.base import SequenceDataset, build_sequence_data

REVIEWS_URL = "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz"
GAMES_URL = "http://cseweb.ucsd.edu/~wckang/steam_games.json.gz"


def download_steam_files(data_dir: str = "./data") -> Tuple[str, str]:
    """Downloads Steam reviews and games files from UCSD.

    Args:
        data_dir: Root data directory.

    Returns:
        A tuple of (reviews_file_path, games_file_path).
    """
    steam_dir = os.path.join(data_dir, "steam")
    os.makedirs(steam_dir, exist_ok=True)

    reviews_path = os.path.join(steam_dir, "steam_reviews.json.gz")
    if not os.path.exists(reviews_path):
        print(f"Downloading Steam reviews from {REVIEWS_URL} to {reviews_path}...")
        urllib.request.urlretrieve(REVIEWS_URL, reviews_path)

    games_path = os.path.join(steam_dir, "steam_games.json.gz")
    if not os.path.exists(games_path):
        print(f"Downloading Steam games metadata from {GAMES_URL} to {games_path}...")
        urllib.request.urlretrieve(GAMES_URL, games_path)

    return reviews_path, games_path


def parse_steam_reviews(file_path: str) -> List[Dict[str, Union[str, float, int]]]:
    """Parses Steam reviews .json.gz file.

    Each line in W.C. Kang's Steam reviews file is typically a Python dict represented as a string.

    Args:
        file_path: Path to the gzipped reviews file.

    Returns:
        List of raw interactions.
    """
    interactions = []
    # Count lines to show progress if needed
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                # Direct ast.literal_eval parsing for Python dict literal formatting
                data = ast.literal_eval(line)
            except Exception:
                continue
            
            if isinstance(data, dict) and "username" in data and "product_id" in data:
                # Get date for sorting. In W.C. Kang's file, date is like '2017-05-15'
                # If date is missing, we fall back to using the line number as chronological order
                date_str = data.get("date", "")
                if not date_str:
                    date_str = f"0000-00-00_{line_num:08d}"
                
                interactions.append(
                    {
                        "user": data["username"],
                        "item": data["product_id"],
                        "date": date_str,
                        # Keep playtime hours if present
                        "hours": float(data.get("hours", 0.0))
                    }
                )
                
    # Sort interactions chronologically by date
    # Standard format 'YYYY-MM-DD' allows correct lexicographical sorting
    return sorted(interactions, key=lambda x: x["date"])


def parse_steam_games(file_path: str) -> Dict[str, str]:
    """Parses Steam games metadata .json.gz file.

    W.C. Kang's steam games file is a JSON where each line contains game information,
    or the entire file is a dictionary where keys are app_id and values are metadata.

    Args:
        file_path: Path to the gzipped games file.

    Returns:
        Mapping from appID string to game title string.
    """
    id_to_title = {}
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        try:
            content = f.read()
            # Try to load as one single JSON dict
            data_dict = json.loads(content)
            for app_id, details in data_dict.items():
                if isinstance(details, dict):
                    title = details.get("name") or details.get("title") or f"Game_{app_id}"
                    id_to_title[str(app_id)] = title
        except Exception:
            # Fallback line-by-line parsing if formatted as JSON Lines
            f.seek(0)
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    try:
                        data = ast.literal_eval(line)
                    except Exception:
                        continue
                
                if isinstance(data, dict):
                    # W.C. Kang's metadata may have 'app_id' or keys inside a dict
                    app_id = data.get("app_id") or data.get("id") or data.get("asin")
                    title = data.get("name") or data.get("title") or data.get("publisher")
                    if app_id and title:
                        id_to_title[str(app_id)] = title

    return id_to_title


def filter_5_core(
    interactions: List[Dict[str, Union[str, float, int]]], k: int = 5
) -> List[Dict[str, Union[str, float, int]]]:
    """Iteratively filters interactions until all users and items have at least k interactions.

    Args:
        interactions: List of interactions.
        k: Minimum number of interactions.

    Returns:
        Filtered list of interactions.
    """
    curr_interactions = list(interactions)
    while True:
        user_counts = {}
        item_counts = {}
        for x in curr_interactions:
            user_counts[x["user"]] = user_counts.get(x["user"], 0) + 1
            item_counts[x["item"]] = item_counts.get(x["item"], 0) + 1

        filtered = [
            x for x in curr_interactions
            if user_counts[x["user"]] >= k and item_counts[x["item"]] >= k
        ]

        if len(filtered) == len(curr_interactions):
            break
        curr_interactions = filtered

    return curr_interactions


class SteamDataLoader:
    """Preprocesses and provides SequenceDatasets for the Steam Reviews dataset."""

    def __init__(
        self,
        data_dir: str = "./data",
        min_rating: float = 0.0,  # Included for compatibility with movielens/amazon loaders
        load_metadata: bool = True,
    ):
        """Initializes the loader, downloads data, and prepares item mappings.

        Args:
            data_dir: Root directory for downloading datasets.
            min_rating: Unused, kept for API compatibility.
            load_metadata: Whether to download and parse game metadata.
        """
        self.data_dir = data_dir
        import pickle

        cache_path = os.path.join(data_dir, "steam", "steam_cache.pkl")
        if os.path.exists(cache_path):
            print(f"Loading preprocessed Steam dataset from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            self.raw_item_to_title = cache_data["raw_item_to_title"]
            self.user_to_id = cache_data["user_to_id"]
            self.item_to_id = cache_data["item_to_id"]
            self.id_to_item = cache_data["id_to_item"]
            self.num_users = cache_data["num_users"]
            self.num_items = cache_data["num_items"]
            self.token_to_title = cache_data["token_to_title"]
            self.user_history = cache_data["user_history"]
            if "user_timestamps" in cache_data:
                self.user_timestamps = cache_data["user_timestamps"]
                return
            # Older cache without timestamps: fall through to a full rebuild
            # (deterministic pipeline -> identical mappings/history, now + timestamps).
            print("Cache lacks user_timestamps; rebuilding Steam dataset...")

        # Download & parse raw files
        reviews_path, games_path = download_steam_files(data_dir)
        raw_interactions = parse_steam_reviews(reviews_path)

        if load_metadata:
            self.raw_item_to_title = parse_steam_games(games_path)
        else:
            self.raw_item_to_title = {}

        # Apply iterative 5-core filtering (standard preprocessing in papers)
        print("Applying 5-core filtering on Steam dataset...")
        raw_interactions = filter_5_core(raw_interactions, k=5)
        print(f"Post 5-core interactions: {len(raw_interactions)}")

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
            tok_id: self.raw_item_to_title.get(app_id, f"Game_{app_id}")
            for tok_id, app_id in self.id_to_item.items()
        }
        self.token_to_title[0] = "<pad>"  # Padding token

        # Group interactions by mapped user ID, sorted by timestamp (date).
        # user_timestamps holds days-since-epoch aligned with user_history
        # (-1 where the raw record had no date and sorted by line number).
        from datetime import date as _date
        def _date_to_days(s):
            try:
                y, m, d = int(s[0:4]), int(s[5:7]), int(s[8:10])
                return (_date(y, m, d) - _date(1970, 1, 1)).days
            except Exception:
                return -1
        self.user_history = {}
        self.user_timestamps = {}
        for x in raw_interactions:
            user_id = self.user_to_id[x["user"]]
            item_id = self.item_to_id[x["item"]]

            if user_id not in self.user_history:
                self.user_history[user_id] = []
                self.user_timestamps[user_id] = []
            self.user_history[user_id].append(item_id)
            self.user_timestamps[user_id].append(_date_to_days(x["date"]))

        # Save to cache file
        cache_data = {
            "raw_item_to_title": self.raw_item_to_title,
            "user_to_id": self.user_to_id,
            "item_to_id": self.item_to_id,
            "id_to_item": self.id_to_item,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "token_to_title": self.token_to_title,
            "user_history": self.user_history,
            "user_timestamps": self.user_timestamps,
        }
        print(f"Saving preprocessed Steam dataset to cache: {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

    def get_split(
        self, split: str, max_len: int = 20, format_type: str = "index"
    ) -> Union[SequenceDataset, Tuple[List[List[str]], List[str]]]:
        """Generates sequence inputs and targets for a specific split.

        Args:
            split: Split type ('train', 'val', or 'test').
            max_len: Maximum historical context length (default 20 for LIGER paper compatibility).
            format_type: Output format. Either 'index' (returns SequenceDataset with item tokens)
              or 'text' (returns inputs and targets as list of lists of game title strings).

        Returns:
            If format_type is 'index':
              A SequenceDataset containing integer token ID lists and target IDs.
            If format_type is 'text':
              A tuple of (inputs, targets) where inputs is list of lists of strings,
              and targets is a list of strings (game titles).
        """
        inputs, targets = build_sequence_data(self.user_history, max_len, split, pad_val=0)

        if format_type == "index":
            return SequenceDataset(inputs, targets)
        elif format_type == "text":
            text_inputs = [
                [self.token_to_title[tok] for tok in seq]
                for seq in inputs
            ]
            text_targets = [self.token_to_title[tok] for tok in targets]
            return text_inputs, text_targets
        else:
            raise ValueError(f"Unknown format_type: {format_type}. Choose 'index' or 'text'.")
