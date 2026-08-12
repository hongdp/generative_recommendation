"""KuaiRand single-target sequence loading + readout attention layout helpers.

Shared by examples/train_kuairand_readout.py, examples/eval_kuairand_deepk.py,
and examples/train_kuairand_flow.py (previously three drifting copies).

The cache npz comes from examples/prepare_kuairand.py: ragged per-user watch
sequences under the production vocab policy (top-N by last-train-day watches),
with a GTS test flag per event.
"""

import numpy as np
import jax.numpy as jnp


class KuaiRandSequences:
    """Single-target sampling view over the ragged cache.

    Split convention (matches the readout A/B): train = non-test events with
    >=1 history event; val = first test day; test = remaining test days.
    Eval context may include earlier test-window events (serving semantics).
    """

    def __init__(self, path, max_len):
        d = np.load(path)
        self.video_ids = d["video_ids"]
        self.time_ms = d["time_ms"]
        self.is_test = d["is_test"]
        self.offsets = d["offsets"]
        self.max_len = max_len
        counts = np.diff(self.offsets)
        self.event_start = np.repeat(self.offsets[:-1], counts)
        has_history = (np.arange(len(self.video_ids)) - self.event_start) >= 1
        self.train_pool = np.where(~self.is_test & has_history)[0]
        test_days = np.unique(self.day(self.time_ms[self.is_test]))
        self.val_day, self.test_days = test_days[0], test_days[1:]
        ev_day = self.day(self.time_ms)
        self.val_pool = np.where(self.is_test & has_history & (ev_day == self.val_day))[0]
        self.test_pool = np.where(self.is_test & has_history & (ev_day > self.val_day))[0]

    @staticmethod
    def day(ms):
        return (ms // 86_400_000).astype(np.int64)

    def gather(self, event_idx):
        """Left-padded histories [B, max_len] + targets [B] for event indices.

        Vectorized: the window for target e is events [e-L, e) clipped at the
        user's sequence start — contiguous, so invalid slots are naturally the
        left-pad positions.
        """
        e = np.asarray(event_idx)
        L = self.max_len
        cols = e[:, None] + np.arange(-L, 0)[None, :]
        valid = cols >= self.event_start[e][:, None]
        x = np.where(valid, self.video_ids[np.clip(cols, 0, None)], 0).astype(np.int32)
        return x, self.video_ids[e].astype(np.int32)


def make_layout(max_len, readout, num_rel_buckets=64):
    """(anchor, rel_idx) for the single-target item/begin readout layouts."""
    ctx = np.arange(max_len)
    if readout == "item":
        anchor, pos = ctx, ctx
    else:  # one branch token anchored at the last context position
        anchor = np.concatenate([ctx, [max_len - 1]])
        pos = np.concatenate([ctx, [max_len]])
    rel = pos[:, None] - pos[None, :]
    rel_idx = np.clip(rel, 0, num_rel_buckets - 1).astype(np.int32)
    return jnp.array(anchor, dtype=jnp.int32), jnp.array(rel_idx)


def make_mask(x, anchor, max_len):
    """M[q,k] = (k is context and k <= anchor[q] and x[k] != 0) or k == q."""
    n = anchor.shape[0]
    k_ids = jnp.arange(n)
    valid_key = jnp.concatenate(
        [x != 0, jnp.zeros((x.shape[0], n - max_len), dtype=bool)], axis=1
    )
    reach = k_ids[None, :] <= anchor[:, None]
    mask = ((k_ids < max_len)[None, :] & reach)[None, :, :] & valid_key[:, None, :]
    return mask | jnp.eye(n, dtype=bool)[None, :, :]
