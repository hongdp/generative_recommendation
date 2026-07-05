"""Shared training utilities for recommendation model training scripts.

Provides reusable components for early stopping, checkpointing, metric logging,
and result documentation that are common across all training scripts.
"""

import json
import os
import time
from typing import Dict, Optional, Sequence, Set

import flax.serialization
import numpy as np


class EarlyStopper:
    """Tracks best validation metrics and triggers early stopping.

    Improvement is detected when ANY tracked metric improves, excluding
    metrics in the `exclude_metrics` set (e.g., 'Valid@1', 'Valid@Beam', 'Diversity').
    """

    def __init__(self, patience: int = 5, exclude_metrics: Optional[Set[str]] = None):
        """Initializes the early stopper.

        Args:
            patience: number of epochs without improvement before stopping.
            exclude_metrics: set of metric names to exclude from improvement checking.
        """
        self.patience = patience
        self.exclude_metrics = exclude_metrics or {"Valid@1", "Valid@Beam", "Diversity"}
        self.best_metrics: Dict[str, float] = {}
        self.patience_counter = 0
        self.best_params = None

    def check(self, metrics: Dict[str, float], params) -> bool:
        """Checks if any metric improved.

        Args:
            metrics: dictionary of metric name → value.
            params: current model parameters (saved if improved).

        Returns:
            True if improved, False otherwise.
        """
        improved = False
        for metric, score in metrics.items():
            if metric in self.exclude_metrics:
                continue
            if metric not in self.best_metrics or score > self.best_metrics[metric]:
                self.best_metrics[metric] = score
                improved = True

        if improved:
            self.best_params = params
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return improved

    @property
    def should_stop(self) -> bool:
        """Returns True if patience has been exhausted."""
        return self.patience_counter >= self.patience

    def get_best(self, metric: str, default: float = -1.0) -> float:
        """Returns the best observed value for a metric."""
        return self.best_metrics.get(metric, default)


def save_checkpoint(
    params,
    opt_state,
    epoch: int,
    best_val_ndcg: float,
    checkpoint_dir: str,
    filename: str = "best_checkpoint.msgpack",
    semantic_ids_hash: Optional[str] = None,
) -> str:
    """Saves a Flax checkpoint to disk.

    Args:
        params: model parameters.
        opt_state: optimizer state.
        epoch: current epoch number.
        best_val_ndcg: best validation NDCG@10 so far.
        checkpoint_dir: directory to save the checkpoint.
        filename: checkpoint filename.
        semantic_ids_hash: optional hash of the Semantic-ID assignment the model
            is being trained on. When provided, a ``<filename>.meta.json`` sidecar
            is written so that reloading against a mismatched (e.g. regenerated)
            Semantic-ID file can be detected instead of silently decoding into the
            wrong code space.

    Returns:
        The full path to the saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    checkpoint_state = {
        "params": params,
        "opt_state": opt_state,
        "epoch": epoch,
        "best_val_ndcg": best_val_ndcg,
    }
    with open(checkpoint_path, "wb") as f:
        f.write(flax.serialization.to_bytes(checkpoint_state))
    if semantic_ids_hash is not None:
        with open(checkpoint_path + ".meta.json", "w") as f:
            json.dump({"semantic_ids_hash": semantic_ids_hash, "epoch": epoch}, f)
    return checkpoint_path


def read_checkpoint_meta(checkpoint_path: str) -> dict:
    """Reads the ``<checkpoint>.meta.json`` sidecar, or ``{}`` if absent."""
    meta_path = checkpoint_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


def verify_semantic_ids_hash(checkpoint_path: str, current_hash: str) -> None:
    """Fails loudly if a checkpoint was trained on a different Semantic-ID set.

    Raises:
        ValueError: if the checkpoint's recorded hash differs from ``current_hash``.
    """
    meta = read_checkpoint_meta(checkpoint_path)
    stored = meta.get("semantic_ids_hash")
    if stored is None:
        print(
            f"WARNING: no Semantic-ID hash recorded for {checkpoint_path}; cannot "
            "verify that the checkpoint matches the current Semantic-ID file. "
            "Results are only trustworthy if the same IDs were used for training."
        )
        return
    if stored != current_hash:
        raise ValueError(
            f"Semantic-ID mismatch for {checkpoint_path}: checkpoint was trained on "
            f"IDs with hash {stored} but the current Semantic-ID file hashes to "
            f"{current_hash}. Decoding would land in the wrong code space (all "
            "predictions invalid). Regenerate/point to the matching Semantic-ID file."
        )


def assert_decode_validity(results: Dict[str, float], min_valid_beam: float = 0.01) -> None:
    """Guards against silently reporting all-zero metrics from a broken decode.

    A near-zero ``Valid@Beam`` means essentially every decoded Semantic-ID path
    fails to map to a real item — usually a Semantic-ID / checkpoint mismatch — so
    the reported ranking metrics are meaningless. Raise instead of logging zeros.
    """
    valid_beam = results.get("Valid@Beam")
    if valid_beam is not None and valid_beam < min_valid_beam:
        raise ValueError(
            f"Decode validity too low (Valid@Beam={valid_beam:.5f} < {min_valid_beam}). "
            "Almost no decoded Semantic IDs map to real items — the checkpoint and the "
            "Semantic-ID file are almost certainly mismatched. Refusing to report "
            "meaningless all-zero metrics."
        )


def load_checkpoint(
    checkpoint_path: str,
    params_template,
    opt_state_template,
) -> dict:
    """Loads a Flax checkpoint from disk.

    Args:
        checkpoint_path: path to the checkpoint file.
        params_template: a template params pytree for deserialization.
        opt_state_template: a template opt_state pytree for deserialization.

    Returns:
        A dictionary with keys: 'params', 'opt_state', 'epoch', 'best_val_ndcg'.
    """
    state_template = {
        "params": params_template,
        "opt_state": opt_state_template,
        "epoch": 0,
        "best_val_ndcg": 0.0,
    }
    with open(checkpoint_path, "rb") as f:
        return flax.serialization.from_bytes(state_template, f.read())


def log_epoch_metrics(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    elapsed: float,
    val_metrics: Optional[Dict[str, float]] = None,
    extra_losses: Optional[Dict[str, float]] = None,
    writer=None,
    global_step: int = 0,
):
    """Prints formatted epoch summary and writes to TensorBoard.

    Args:
        epoch: current epoch number.
        total_epochs: total number of epochs.
        train_loss: average training loss for this epoch.
        elapsed: wall clock time for this epoch in seconds.
        val_metrics: optional validation metrics dictionary.
        extra_losses: optional dict of extra loss components to print (e.g., KL, Recon).
        writer: optional TensorBoard SummaryWriter.
        global_step: global step for TensorBoard.
    """
    # Build training loss line
    parts = [f"Epoch {epoch:02d}/{total_epochs} | Loss: {train_loss:.4f}"]
    if extra_losses:
        for name, val in extra_losses.items():
            parts.append(f"{name}: {val:.4f}")
    parts.append(f"Time: {elapsed:.1f}s")
    print(" | ".join(parts))

    if writer is not None:
        writer.add_scalar("Loss/train_epoch", train_loss, global_step)
        if extra_losses:
            for name, val in extra_losses.items():
                writer.add_scalar(f"Loss/{name}", val, global_step)

    # Print and log validation metrics
    if val_metrics is not None:
        core_metrics = ["NDCG@10", "HR@10", "MRR"]
        core_parts = []
        for m in core_metrics:
            if m in val_metrics:
                core_parts.append(f"{m}: {val_metrics[m]:.5f}")
        print(f"--- Validation @ Epoch {epoch} | " + " | ".join(core_parts))

        if writer is not None:
            for metric, score in val_metrics.items():
                writer.add_scalar(f"Val/{metric}", score, global_step)


def log_results_to_markdown(
    model_desc: str,
    dataset: str,
    test_results: Dict[str, float],
    best_val_ndcg: float,
    log_path: str = "experiment_results.md",
):
    """Appends experiment results to a markdown file.

    Args:
        model_desc: model description string.
        dataset: dataset name.
        test_results: test evaluation results dictionary.
        best_val_ndcg: best validation NDCG@10 achieved.
        log_path: path to the results markdown file.
    """
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Build results row with available metrics
    metrics = []
    for k in [5, 10, 20]:
        hr_key = f"HR@{k}"
        ndcg_key = f"NDCG@{k}"
        if hr_key in test_results:
            metrics.append(f"{test_results[hr_key]:.5f}")
        else:
            metrics.append("N/A")
        if ndcg_key in test_results:
            metrics.append(f"{test_results[ndcg_key]:.5f}")
        else:
            metrics.append("N/A")

    mrr = f"{test_results['MRR']:.5f}" if "MRR" in test_results else "N/A"
    metrics_str = " | ".join(metrics)

    results_row = (
        f"| {date_str} | {model_desc} on {dataset.upper()} | Local | "
        f"{metrics_str} | {mrr} | "
        f"Best Val NDCG@10={best_val_ndcg:.5f} |"
    )

    with open(log_path, "a") as f:
        f.write(results_row + "\n")
    print(f"\nResults written to {log_path}")
