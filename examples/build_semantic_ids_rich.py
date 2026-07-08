"""Builds Semantic IDs from *rich* Amazon item text (title + brand + category + price).

Experiment context (2026-07-04): our Amazon TIGER runs sit ~2x below the published
TIGER numbers while Steam is fine. The mismatch bug and under-training were ruled
out empirically, leaving Semantic-ID quality as the prime suspect: the baseline IDs
encode *title-only* text with MiniLM, whereas the TIGER paper encodes
title+price+brand+category with Sentence-T5.

This script produces two ID sets for a controlled A/B/C comparison:
  B (treatment): rich item text -> Sentence-T5 -> seeded RQ-KMeans
  C (control):   title-only     -> MiniLM      -> seeded RQ-KMeans
B vs C isolates {text richness + encoder} with the quantizer held fixed.

The script runs in three stages because the repo's ``src/datasets`` package shadows
the HuggingFace ``datasets`` library that sentence-transformers imports internally —
the two cannot coexist in one interpreter:

  PYTHONPATH=src python examples/build_semantic_ids_rich.py --stage texts  --dataset beauty
  python              examples/build_semantic_ids_rich.py --stage encode --dataset beauty --device cuda
  python              examples/build_semantic_ids_rich.py --stage ids    --dataset beauty

Outputs:
  data/semantic_ids_rich_{category}.json           (arm B)
  data/semantic_ids_kmeans_seeded_{category}.json  (arm C)
plus codebook-utilization / collision statistics for each.
"""

import argparse
import ast
import gzip
import json
import os

import numpy as np

CATEGORY_DIRS = {"beauty": "Beauty", "sports": "Sports_and_Outdoors", "toys": "Toys_and_Games"}


# ---------------------------------------------------------------------------
# Stage 1: build item texts (repo imports — run with PYTHONPATH=src)
# ---------------------------------------------------------------------------
def parse_rich_metadata(file_path):
    """Parses Amazon metadata keeping title, brand, categories, and price."""
    rich = {}
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
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
            if isinstance(data, dict) and "asin" in data:
                rich[data["asin"]] = {
                    "title": data.get("title"),
                    "brand": data.get("brand"),
                    "categories": data.get("categories"),
                    "price": data.get("price"),
                    "description": data.get("description"),
                }
    return rich


def build_rich_text(meta, with_description=False):
    """Formats one item's metadata into the TIGER-paper content-field text."""
    parts = []
    if meta.get("title"):
        parts.append(f"Title: {meta['title']}.")
    if meta.get("brand"):
        parts.append(f"Brand: {meta['brand']}.")
    cats = meta.get("categories")
    if cats:
        if isinstance(cats, str):
            try:
                cats = ast.literal_eval(cats)
            except Exception:
                cats = None
        if cats and isinstance(cats, list) and cats[0]:
            parts.append("Category: " + " > ".join(str(c) for c in cats[0]) + ".")
    price = meta.get("price")
    if price is not None:
        parts.append(f"Price: ${price}.")
    if with_description and meta.get("description"):
        desc = str(meta["description"]).strip()[:300]
        parts.append(f"Description: {desc}")
    return " ".join(parts) if parts else "Unknown product."


def stage_texts(args):
    from datasets import AmazonDataLoader  # repo package (needs PYTHONPATH=src)

    loader = AmazonDataLoader(category=args.dataset, data_dir=args.data_dir, min_rating=0)
    print(f"Items: {loader.num_items}")

    cat_dir = CATEGORY_DIRS[args.dataset]
    meta_path = os.path.join(args.data_dir, "amazon", cat_dir, f"meta_{cat_dir}.json.gz")
    rich_meta = parse_rich_metadata(meta_path)
    print(f"Parsed rich metadata for {len(rich_meta)} ASINs")

    rich_texts, title_texts = [], []
    coverage = {"title": 0, "brand": 0, "categories": 0, "price": 0}
    for tok in range(1, loader.num_items + 1):
        asin = loader.id_to_item[tok]
        meta = rich_meta.get(asin, {})
        for k in coverage:
            if meta.get(k):
                coverage[k] += 1
        rich_texts.append(build_rich_text(meta, with_description=args.with_description) if meta else f"Product_{asin}")
        title_texts.append(loader.token_to_title.get(tok, f"Product_{asin}"))
    print("Field coverage over catalog:", {k: f"{v / loader.num_items:.1%}" for k, v in coverage.items()})
    print("Sample rich text:", rich_texts[0][:200])

    out = os.path.join(args.data_dir, f"item_texts{args.suffix}_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump({"rich": rich_texts, "title": title_texts}, f)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Stage 2: encode texts (HF libs only — run WITHOUT PYTHONPATH=src)
# ---------------------------------------------------------------------------
def minilm_mean_pool_encode(texts, device, batch_size=256):
    """Replicates datasets.embeddings.extract_movie_embeddings exactly
    (MiniLM + mean pooling, no normalization) without importing the repo package."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    out = []
    from tqdm import tqdm
    for start in tqdm(range(0, len(texts), batch_size), desc="MiniLM"):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)


def stage_encode(args):
    with open(os.path.join(args.data_dir, f"item_texts{args.suffix}_{args.dataset}.json")) as f:
        texts = json.load(f)
    rich_texts, title_texts = texts["rich"], texts["title"]
    n = len(rich_texts)

    # Arm B: rich text with Sentence-T5 (the TIGER paper's encoder).
    from sentence_transformers import SentenceTransformer
    print(f"[Arm B] Encoding {n} rich texts with {args.rich_encoder} on {args.device} (bf16={args.bf16})...")
    model_kwargs = {}
    if args.bf16:
        import torch
        model_kwargs = {"torch_dtype": torch.bfloat16}
    st = SentenceTransformer(args.rich_encoder, device=args.device, model_kwargs=model_kwargs)
    if args.encode_max_seq > 0:
        st.max_seq_length = args.encode_max_seq
    emb = st.encode(rich_texts, batch_size=args.encode_batch, show_progress_bar=True, convert_to_numpy=True)
    rich = np.zeros((n + 1, emb.shape[1]), dtype=np.float32)
    rich[1:] = emb
    np.save(os.path.join(args.data_dir, f"item_emb_rich{args.suffix}_{args.dataset}.npy"), rich)
    print(f"[Arm B] embeddings {rich.shape} saved")

    if args.skip_control:
        return
    # Arm C: title-only with MiniLM mean pooling (matches the existing baseline pipeline).
    print(f"[Arm C] Encoding {n} title texts with MiniLM on {args.device}...")
    emb_c = minilm_mean_pool_encode(title_texts, args.device)
    ctrl = np.zeros((n + 1, emb_c.shape[1]), dtype=np.float32)
    ctrl[1:] = emb_c
    np.save(os.path.join(args.data_dir, f"item_emb_title_{args.dataset}.npy"), ctrl)
    print(f"[Arm C] embeddings {ctrl.shape} saved")


# ---------------------------------------------------------------------------
# Stage 3: quantize to Semantic IDs (numpy only)
# ---------------------------------------------------------------------------
def kmeans(X, K, max_iter=30, tol=1e-4):
    """NumPy K-Means (same implementation as examples/train_rqkmeans.py)."""
    N, _ = X.shape
    centroids = X[np.random.choice(N, K, replace=False)].copy()
    labels = np.zeros(N, dtype=np.int64)
    for _ in range(max_iter):
        d = np.sum(X**2, 1, keepdims=True) + np.sum(centroids**2, 1, keepdims=True).T - 2 * X @ centroids.T
        labels = np.argmin(d, axis=1)
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            members = X[labels == k]
            new_centroids[k] = members.mean(0) if len(members) else X[np.random.choice(N)]
        if np.max(np.abs(centroids - new_centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids, labels


def rq_kmeans_ids(embeddings, num_levels, num_codes, seed):
    """Seeded sequential RQ-KMeans over item embeddings (index 0 = padding)."""
    np.random.seed(seed)
    train_x = embeddings[1:].copy()
    residuals = train_x.copy()
    all_labels, all_centroids = [], []
    for level in range(num_levels):
        print(f"  Level {level + 1} clustering...")
        centroids, labels = kmeans(residuals, num_codes)
        residuals = residuals - centroids[labels]
        all_labels.append(labels)
        all_centroids.append(centroids)

    indices = np.stack(all_labels, axis=1)
    recon = sum(all_centroids[l][indices[:, l]] for l in range(num_levels))
    mse = float(np.mean((train_x - recon) ** 2))

    ids = {0: [0] * num_levels}
    for i in range(len(indices)):
        ids[i + 1] = [int(c) for c in indices[i]]
    return ids, mse


def id_stats(semantic_ids, num_codes, num_levels):
    items = {k: v for k, v in semantic_ids.items() if k != 0}
    arr = np.array(list(items.values()))
    util = [len(np.unique(arr[:, l])) / num_codes for l in range(num_levels)]
    unique_triples = len(set(map(tuple, arr.tolist())))
    return util, 1.0 - unique_triples / len(arr), unique_triples


def stage_ids(args):
    for arm, emb_file, out_file in [
        ("B(rich+T5)", f"item_emb_rich_{args.dataset}.npy", f"semantic_ids_rich_{args.dataset}.json"),
        ("C(title+MiniLM)", f"item_emb_title_{args.dataset}.npy", f"semantic_ids_kmeans_seeded_{args.dataset}.json"),
    ]:
        emb = np.load(os.path.join(args.data_dir, emb_file))
        print(f"\n[{arm}] RQ-KMeans on {emb.shape}...")
        ids, mse = rq_kmeans_ids(emb, args.num_levels, args.num_codes, args.seed)
        util, coll, uniq = id_stats(ids, args.num_codes, args.num_levels)
        n_items = emb.shape[0] - 1
        print(f"[{arm}] recon MSE={mse:.5f} | codebook util={[f'{u:.1%}' for u in util]} | "
              f"unique triples={uniq}/{n_items} | collision rate={coll:.2%}")
        out_path = os.path.join(args.data_dir, out_file)
        with open(out_path, "w") as f:
            json.dump({str(k): v for k, v in ids.items()}, f)
        print(f"[{arm}] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build rich-text Semantic IDs for Amazon datasets.")
    parser.add_argument("--stage", type=str, required=True, choices=["texts", "encode", "ids"])
    parser.add_argument("--dataset", type=str, default="beauty", choices=list(CATEGORY_DIRS.keys()))
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_levels", type=int, default=3)
    parser.add_argument("--num_codes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rich_encoder", type=str, default="sentence-transformers/sentence-t5-base")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--suffix", type=str, default="", help="Filename suffix for text/embedding outputs (e.g. \"_desc\").")
    parser.add_argument("--with_description", action="store_true", help="Include the description field in rich text.")
    parser.add_argument("--bf16", action="store_true", help="Load the encoder in bfloat16 (needed to fit sentence-t5-xxl in 16GB).")
    parser.add_argument("--encode_batch", type=int, default=64, help="Encoder batch size (use 8 for t5-xxl on 16GB).")
    parser.add_argument("--encode_max_seq", type=int, default=0, help="If >0, cap the encoder max_seq_length (our texts are ~60 tokens; 128 saves memory).")
    parser.add_argument("--skip_control", action="store_true", help="Skip the MiniLM title-only control encoding.")
    args = parser.parse_args()

    {"texts": stage_texts, "encode": stage_encode, "ids": stage_ids}[args.stage](args)


if __name__ == "__main__":
    main()
