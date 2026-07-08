"""Base dataset classes and utilities for sequential recommendation."""

from typing import Any, Dict, List, Tuple, Union
import numpy as np


def pad_or_truncate(
    sequence: List[Any],
    max_len: int,
    pad_val: Any = 0,
) -> List[Any]:
    """Pads or truncates a sequence to a fixed maximum length.

    Args:
        sequence: The list of items to pad/truncate.
        max_len: The target sequence length.
        pad_val: The value to use for padding.

    Returns:
        A new list of length `max_len`.
    """
    if len(sequence) >= max_len:
        return list(sequence[-max_len:])
    else:
        return [pad_val] * (max_len - len(sequence)) + list(sequence)


def build_sequence_data(
    user_history: Dict[Union[int, str], List[Any]],
    max_len: int,
    split: str,
    pad_val: Any = 0,
) -> Tuple[List[List[Any]], List[Any]]:
    """Builds historical sequences and targets for a specific dataset split.

    Uses the chronological leave-one-out split protocol:
    - 'test': input is user history except last item; target is last item.
    - 'val': input is user history except last two items; target is second-to-last item.
    - 'train': inputs are prefixes up to the second-to-last item; targets are the next items.

    Args:
        user_history: Mapping from user ID to their chronological list of interacted items.
        max_len: Maximum length of the history sequence.
        split: The dataset split ('train', 'val', or 'test').
        pad_val: Padding value used for short sequences.

    Returns:
        A tuple of (inputs, targets):
          - inputs: List of sequences, each of length `max_len`.
          - targets: List of target items.
    """
    inputs = []
    targets = []

    for user_id, seq in user_history.items():
        n = len(seq)
        if n < 2:
            continue

        if split == "test":
            inputs.append(pad_or_truncate(seq[:-1], max_len, pad_val))
            targets.append(seq[-1])
        elif split == "val":
            if n < 3:
                continue
            inputs.append(pad_or_truncate(seq[:-2], max_len, pad_val))
            targets.append(seq[-2])
        elif split == "train":
            # Generate prefix sequences
            for t in range(1, n - 2):
                inputs.append(pad_or_truncate(seq[:t], max_len, pad_val))
                targets.append(seq[t])

    return inputs, targets


def build_sequence_user_ids(user_history, split):
    """Returns the user id of every sample emitted by build_sequence_data.

    Iterates user_history in the same order and applies the same per-split
    emission rules, so index i here corresponds to sample i of
    build_sequence_data(user_history, ..., split).
    """
    user_ids = []
    for user_id, seq in user_history.items():
        n = len(seq)
        if n < 2:
            continue
        if split == "test":
            user_ids.append(user_id)
        elif split == "val":
            if n < 3:
                continue
            user_ids.append(user_id)
        elif split == "train":
            user_ids.extend([user_id] * max(0, n - 3))
    return user_ids


class SequenceDataset:
    """Wrapper class for sequential recommendation inputs and targets."""

    def __init__(self, inputs: List[List[Any]], targets: List[Any]):
        """Initializes the dataset.

        Args:
            inputs: List of input sequence lists.
            targets: List of target items.
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Converts inputs and targets to NumPy arrays (useful for index-based models).

        Returns:
            A tuple of (inputs_array, targets_array) of type int32.
        """
        return np.array(self.inputs, dtype=np.int32), np.array(self.targets, dtype=np.int32)
