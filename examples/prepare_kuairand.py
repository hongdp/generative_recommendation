"""Materializes a training-ready cache from the KuaiRand-27K (or 1K) raw logs.

Replicates the production vocab policy: the item vocabulary is the top-N videos
by watch count on the LAST TRAIN DAY (not all-train frequency — empirically
+8.8pp test-target coverage at N=100K on 27K). Watches are is_click=1 events
from the standard logs; history and targets are restricted to the vocab
(out-of-vocab events are dropped, matching a production model whose task is
defined over its vocab).

Output: data/kuairand/kuairand{27k,1k}_top{N}.npz with ragged per-user
sequences (sorted by time), remapped video ids in [1, N] (0 = padding):
  user_ids      [U]      original user ids
  offsets       [U+1]    user u's events are values[offsets[u]:offsets[u+1]]
  video_ids     [E]      remapped, int32
  time_ms       [E]      int64
  is_test       [E]      bool (event falls in the last TEST_DAYS calendar days)
plus vocab_video_ids [N] (original ids, rank order) for feature joins later.

Run: python examples/prepare_kuairand.py --version 27k --vocab_size 100000
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

USECOLS = ["user_id", "video_id", "time_ms", "is_click", "date"]
DTYPES = {"user_id": np.int32, "video_id": np.int32, "time_ms": np.int64,
          "is_click": np.int8, "date": np.int32}


def load_watches(data_dir):
    logs = sorted(glob.glob(os.path.join(data_dir, "log_standard_*.csv")))
    assert logs, f"no standard logs found in {data_dir}"
    chunks = []
    for f in logs:
        for chunk in pd.read_csv(f, usecols=USECOLS, dtype=DTYPES, chunksize=5_000_000):
            chunks.append(chunk[chunk.is_click == 1].drop(columns="is_click"))
        print(f"  loaded {os.path.basename(f)}")
    watches = pd.concat(chunks, ignore_index=True)
    print(f"  {len(watches):,} watches")
    return watches


def main():
    parser = argparse.ArgumentParser(description="Build KuaiRand sequence cache with production-style vocab.")
    parser.add_argument("--data_dir", type=str, default="./data/kuairand")
    parser.add_argument("--version", type=str, default="27k", choices=["27k", "1k"])
    parser.add_argument("--vocab_size", type=int, default=100_000)
    parser.add_argument("--test_days", type=int, default=3)
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, f"KuaiRand-{'27K' if args.version == '27k' else '1K'}", "data")
    print(f"Loading watches from {raw_dir} ...")
    watches = load_watches(raw_dir)

    dates = np.sort(watches.date.unique())
    test_dates = set(dates[-args.test_days:].tolist())
    last_train_day = dates[-args.test_days - 1]

    # Production vocab policy: top-N by last-train-day watch count.
    lastday_freq = watches[watches.date == last_train_day].video_id.value_counts()
    vocab = lastday_freq.head(args.vocab_size)
    print(f"vocab: top {len(vocab):,} of {len(lastday_freq):,} videos watched on day {last_train_day}")

    remap = pd.Series(np.arange(1, len(vocab) + 1, dtype=np.int32), index=vocab.index)
    watches["vid"] = watches.video_id.map(remap)
    kept = watches.dropna(subset=["vid"]).copy()
    kept["vid"] = kept.vid.astype(np.int32)
    kept["is_test"] = kept.date.isin(test_dates)
    print(f"in-vocab watches: {len(kept):,} ({len(kept)/len(watches):.1%}) | "
          f"train {int((~kept.is_test).sum()):,} / test {int(kept.is_test.sum()):,}")

    kept = kept.sort_values(["user_id", "time_ms"], kind="stable")
    user_ids, counts = np.unique(kept.user_id.to_numpy(), return_counts=True)
    offsets = np.zeros(len(user_ids) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    out = os.path.join(args.data_dir, f"kuairand{args.version}_top{args.vocab_size}.npz")
    np.savez_compressed(
        out,
        user_ids=user_ids.astype(np.int32),
        offsets=offsets,
        video_ids=kept.vid.to_numpy(),
        time_ms=kept.time_ms.to_numpy(),
        is_test=kept.is_test.to_numpy(),
        vocab_video_ids=vocab.index.to_numpy().astype(np.int64),
    )
    meta = {
        "version": args.version, "vocab_size": int(len(vocab)),
        "vocab_policy": f"top-{args.vocab_size} by watch count on last train day {int(last_train_day)}",
        "test_days": args.test_days, "test_dates": sorted(int(d) for d in test_dates),
        "num_users": int(len(user_ids)), "num_events": int(len(kept)),
        "train_events": int((~kept.is_test).sum()), "test_events": int(kept.is_test.sum()),
        "median_seq_len": float(np.median(counts)),
    }
    with open(out.replace(".npz", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out}\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
