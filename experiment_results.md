# Experiment Results Log

This file documents all training runs, evaluations, and benchmarks. Refer to [SKILL.md](file:///home/hongdp/Workspace/generative_recommendation/SKILL.md) for the logging protocol.

## MovieLens-100K (ML-100K)

| Date | Configuration | Hardware | HR_5 | NDCG_5 | HR_10 | NDCG_10 | HR_20 | NDCG_20 | MRR | Notes |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| 2026-06-06 | ML-100K Baseline (Random) | Local (CPU) | - | - | 0.00848 | 0.00409 | - | - | 0.00580 | Random recommendation baseline |
| 2026-06-06 | ML-100K Baseline (Most Popular) | Local (CPU) | - | - | 0.04984 | 0.02303 | - | - | 0.01490 | Most Popular recommendation baseline |
| 2026-06-06 | HSTUModel next-item pred (5 epochs) on ML-100K | Local (CPU) | 0.06681 | 0.03692 | 0.13786 | 0.05943 | - | - | 0.05283 | Converged and ranking metrics improved significantly (+0.05024 MRR) |

## MovieLens-1M (ML-1M)

| Date | Configuration | Hardware | HR_5 | NDCG_5 | HR_10 | NDCG_10 | HR_20 | NDCG_20 | MRR | Notes |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| 2026-06-06 | Full HSTUModel (4 blocks, embed=256) on ML-1M | Local (GeForce RTX 4080) | 0.13212 | 0.08459 | 0.20414 | 0.10770 | - | - | 0.09371 | ⚠️ SUSPECT: Best Val NDCG_10=-1.00000 is an uninitialized placeholder — best-checkpoint selection never fired, so these test metrics are likely the last epoch, NOT the best-val checkpoint. Superseded by the row below (same config, valid best-val=0.13163). Do not cite. |
| 2026-06-06 | Full HSTUModel (4 blocks, embed=256) on ML-1M | Local (GeForce RTX 4080) | 0.14818 | 0.09544 | 0.23725 | 0.12411 | - | - | 0.10549 | Best Val NDCG_10=0.13163; Test metrics replicated |
| 2026-06-06 | Full HSTUModel (4 blocks, paused Epoch 14) on ML-1M | Local (GeForce RTX 4080) | 0.19023 | 0.12933 | 0.27384 | 0.15601 | - | - | 0.13502 | Paused run at Epoch 14 (Best Val Epoch 12). Strong convergence. |
| 2026-06-06 | Full TIGERModel (4 blocks, embed=256) on ML-1M | Local (GeForce RTX 4080) | 0.17169 | 0.11905 | 0.24156 | 0.14169 | - | - | 0.11112 | Best Val NDCG_10=0.15646; Test metrics replicated |
| 2026-06-06 | Full TIGERModel (4 blocks, embed=256) on ML-1M | Local (GeForce RTX 4080) | 0.15348 | 0.10610 | 0.20927 | 0.12417 | - | - | 0.09804 | Best Val NDCG_10=0.13230; Test metrics replicated |

## Amazon Beauty (BEAUTY)

| Date | Configuration | Hardware | HR_5 | NDCG_5 | HR_10 | NDCG_10 | HR_20 | NDCG_20 | MRR | Notes |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| 2026-06-06 | Full HSTU (blocks=4, embed=256) on BEAUTY | Local (GeForce RTX 4080) | 0.04507 | 0.03208 | 0.06479 | 0.03844 | 0.09270 | 0.04543 | 0.03534 | Replication on beauty matching LIGER paper evaluation (Best Val NDCG_10=0.05068) |
| 2026-06-06 | Full TIGER (VAE) (4 blocks, embed=256) on BEAUTY | Local (GeForce RTX 4080) | 0.02455 | 0.01671 | 0.03792 | 0.02101 | 0.05809 | 0.02607 | 0.01727 | Replication on beauty matching LIGER paper evaluation |
| 2026-06-06 | Full TIGER (VAE) (4 blocks, embed=256) on BEAUTY | Local (GeForce RTX 4080) | 0.02357 | 0.01597 | 0.03564 | 0.01980 | 0.05424 | 0.02450 | 0.01633 | Replication on beauty matching LIGER paper evaluation (Best Val NDCG_10=0.02776) |
| 2026-06-06 | Full TIGER (K-Means) (4 blocks, embed=256) on BEAUTY | Local (GeForce RTX 4080) | 0.02513 | 0.01645 | 0.03957 | 0.02112 | 0.05858 | 0.02590 | 0.01683 | Replication on beauty matching LIGER paper evaluation (Best Val NDCG_10=0.02982) |
| 2026-07-04 | [Arm A] TIGER (VAE title-only IDs) (blocks=4, embed=384) on BEAUTY | Local (GeForce RTX 4080) | 0.02263 | 0.01600 | 0.03363 | 0.01955 | 0.04785 | 0.02312 | 0.01625 | Gap-diagnosis control: same title-only MiniLM RQ-VAE IDs as the 2026-06-06 rows but bigger model (384 vs 256) + 30 epochs (early stop @25, patience 10). Test ≈ unchanged vs old runs → under-training/model-size RULED OUT as the cause of the Amazon gap. Best Val NDCG@10=0.02952 (val peaked at epoch 2). |
| 2026-07-04 | [Arm B] TIGER (RICH-text IDs: title+brand+category+price → Sentence-T5-base → seeded RQ-KMeans) (blocks=4, embed=384) on BEAUTY | Local (GeForce RTX 4080) | 0.02522 | 0.01754 | 0.03886 | 0.02192 | 0.05621 | 0.02630 | 0.01802 | ✅ Rich-text Semantic IDs: +12.1% NDCG@10 / +15.6% HR@10 over Arm A; best beauty TIGER row to date. ID collision rate 7.3% vs 14.1% (title-only). Early stop @15. Best Val NDCG@10=0.02991. Remaining gap to LIGER-paper TIGER (NDCG@10=0.0322): missing dedup token, Sentence-T5-XXL, and their 6-layer/128-dim/dropout-0.2 recipe. |
| 2026-07-04 | [Arm C] TIGER (title-only MiniLM → seeded RQ-KMeans) (blocks=4, embed=384) on BEAUTY | Local (GeForce RTX 4080) | 0.02200 | 0.01473 | 0.03264 | 0.01814 | 0.04865 | 0.02216 | 0.01482 | Quantizer-isolation control: same quantizer as Arm B, same text/encoder as Arm A. B vs C = +20.8% NDCG@10 from rich text+encoder alone; A vs C = +7.8% for RQ-VAE over RQ-KMeans. Early stop @16. Best Val NDCG@10=0.02772. See "Semantic-ID Quality A/B/C/D Experiment" section below. |
| 2026-07-04 | [Arm D] TIGER (RANDOM IDs, seeded uniform, no semantics) (blocks=4, embed=384) on BEAUTY | Local (GeForce RTX 4080) | 0.01905 | 0.01384 | 0.02661 | 0.01629 | 0.03519 | 0.01846 | 0.01375 | No-semantics floor: uniform random (c1,c2,c3), collision rate 0.04% (best reachability of all arms). Still learns substantially via co-occurrence memorization. Semantic value over random: title-only +11.4%, rich-text +34.6% NDCG@10. Note HR@1 (0.00854) ≈ Arm A (0.00908) but HR@20 −37% vs B → semantics mainly buys deep-rank recall/generalization, not top-1 memorization. Early stop @16. Best Val NDCG@10=0.02481. |

## Amazon Sports (SPORTS)

| Date | Configuration | Hardware | HR_5 | NDCG_5 | HR_10 | NDCG_10 | HR_20 | NDCG_20 | MRR | Notes |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| 2026-06-06 | Full HSTU (blocks=4, embed=256) on SPORTS | Local (GeForce RTX 4080) | 0.02568 | 0.01812 | 0.03731 | 0.02187 | 0.05174 | 0.02549 | 0.02035 | Replication on sports matching LIGER paper evaluation (Best Val NDCG_10=0.02898) |
| 2026-06-06 | Full TIGER (VAE) (4 blocks, embed=256) on SPORTS | Local (GeForce RTX 4080) | 0.01264 | 0.00828 | 0.02056 | 0.01081 | 0.03163 | 0.01361 | 0.00865 | Replication on sports matching LIGER paper evaluation (Best Val NDCG_10=0.01508) |
| 2026-06-06 | Full TIGER (K-Means) (4 blocks, embed=256) on SPORTS | Local (GeForce RTX 4080) | 0.01208 | 0.00777 | 0.02174 | 0.01089 | 0.03349 | 0.01385 | 0.00846 | Replication on sports matching LIGER paper evaluation (Best Val NDCG_10=0.01375) |

## Amazon Toys (TOYS)

| Date | Configuration | Hardware | HR_5 | NDCG_5 | HR_10 | NDCG_10 | HR_20 | NDCG_20 | MRR | Notes |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| 2026-06-06 | Full HSTU (blocks=4, embed=256) on TOYS | Local (GeForce RTX 4080) | 0.04708 | 0.03442 | 0.06238 | 0.03933 | 0.08289 | 0.04452 | 0.03643 | Replication on toys matching LIGER paper evaluation (Best Val NDCG_10=0.04994) |
| 2026-06-06 | Full TIGER (VAE) (4 blocks, embed=256) on TOYS | Local (GeForce RTX 4080) | 0.01927 | 0.01236 | 0.02885 | 0.01545 | 0.04137 | 0.01859 | 0.01223 | Replication on toys matching LIGER paper evaluation (Best Val NDCG_10=0.01852) |
| 2026-06-06 | Full TIGER (K-Means) (4 blocks, embed=256) on TOYS | Local (GeForce RTX 4080) | 0.02169 | 0.01447 | 0.03379 | 0.01839 | 0.05244 | 0.02305 | 0.01499 | Replication on toys matching LIGER paper evaluation (Best Val NDCG_10=0.02065) |

## Steam (STEAM)

| Date | Configuration | Hardware | HR_5 | NDCG_5 | HR_10 | NDCG_10 | HR_20 | NDCG_20 | MRR | Notes |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| 2026-06-06 | Full HSTU (blocks=4, embed=256) on STEAM | Local (GeForce RTX 4080) | 0.17071 | 0.14849 | 0.20537 | 0.15963 | 0.25475 | 0.17205 | 0.15487 | Replication on steam matching LIGER paper evaluation (Best Val NDCG_10=0.18403) |
| 2026-06-07 | Full HSTU (blocks=4, embed=256) on STEAM | Local (GeForce RTX 4080) | 0.17285 | 0.15049 | 0.20735 | 0.16159 | 0.25647 | 0.17395 | 0.15671 | Replication on steam matching LIGER paper evaluation (Best Val NDCG_10=0.18701) |
| 2026-06-07 | Full TIGER (VAE) (4 blocks, embed=256) on STEAM | Local (GeForce RTX 4080) | 0.15523 | 0.13832 | 0.18185 | 0.14688 | 0.21612 | 0.15555 | 0.13867 | Replication on steam matching LIGER paper evaluation (Best Val NDCG_10=0.16971) |
| 2026-06-07 | Full TIGER (K-Means) (4 blocks, embed=256) on STEAM | Local (GeForce RTX 4080) | 0.13924 | 0.12362 | 0.16457 | 0.13176 | 0.19809 | 0.14021 | 0.12414 | Replication on steam matching LIGER paper evaluation (Best Val NDCG_10=0.15211) |
| 2026-06-07 | Full TIGER (VAE) (blocks=4, embed=384) on STEAM | Local (GeForce RTX 4080) | 0.15810 | 0.13978 | 0.18636 | 0.14886 | 0.22213 | 0.15789 | 0.13996 | Replication on steam matching TIGER paper evaluation (Best Val NDCG_10=0.17238) |
| 2026-06-07 | Full TIGER (K-Means) (blocks=4, embed=384) on STEAM | Local (GeForce RTX 4080) | 0.14024 | 0.12427 | 0.16521 | 0.13229 | 0.19851 | 0.14069 | 0.12461 | Replication on steam matching TIGER paper evaluation (Best Val NDCG_10=0.15363) |
| 2026-06-07 | TIGER Joint V2 (End-to-End Indexing) on STEAM | Local | 0.18432 | 0.10905 | 0.18737 | 0.11001 | 0.22052 | 0.11880 | 0.08833 | ⚠️ EXPERIMENTAL / SUSPECT — do not compare against replication rows. Joint token+item soft-reconstruction, frozen Z anchoring. Anomalous metric shape: HR gain over ranks 6–10 (+0.003) ≪ gain over ranks 11–20 (+0.033), which is backwards for a healthy ranker. HR_5 (0.184) exceeds HSTU (0.171) yet NDCG_5 (0.109) is far below HSTU (0.148) and MRR is only 0.088 (NDCG_5/HR_5=0.59 vs HSTU 0.87) → hits cluster at deep ranks inside top-5, pointing at a beam-ordering / eval bug rather than a genuine gain. |
| 2026-06-07 | Full TIGER (Seq2Seq) (blocks=4, embed=384) on STEAM | Local (GeForce RTX 4080) | 0.06426 | 0.06042 | 0.06783 | 0.06158 | 0.06995 | 0.06213 | 0.05979 | ❌ INVALID — DO NOT CITE. Superseded by the 2026-07-03 row below. Root cause found 2026-07-03: the checkpoint was decoded against a *mismatched* Semantic-ID assignment (unseeded/regenerated IDs, no checkpoint↔ID binding), so nearly every decoded path was invalid (Valid@Beam≈0) and metrics collapsed. NOT a model or beam-search defect. |
| 2026-07-03 | Full TIGER (Seq2Seq) (blocks=4, embed=384) on STEAM | Local (GeForce RTX 4080) | 0.16417 | 0.14404 | 0.19431 | 0.15373 | 0.23235 | 0.16334 | 0.14404 | ✅ CORRECTED re-run (30 epochs) with consistent RQ-VAE Semantic IDs + fixed pipeline (checkpoint↔ID hash binding, best-ckpt eval, decode-validity guard). Valid@1=1.0, Valid@Beam=0.99998; HR grows normally with K (0.164→0.194→0.232). Now competitive with standard TIGER (HR_5=0.158) and just below HSTU (0.173). Best Val NDCG_10=0.17724. |
| 2026-06-07 | TIGER RL-CoT (End-to-End Latent Routing) on STEAM | Local (GeForce RTX 4080) | 0.00042 | 0.00036 | 0.00050 | 0.00039 | 0.00063 | 0.00042 | 0.00036 | ❌ FAILED RUN — training collapsed. Metrics ≈ random (Steam ~10k items → random HR_10≈0.001). Best Val NDCG_10=0.00066. Not a replication; kept only as a record of the failed experiment. |

## Known Issues & Caveats (added 2026-07-02 during results review)

- **⚠️ Amazon TIGER underperforms published TIGER by ~2×.** On Beauty/Sports/Toys, our HSTU rows nearly match the *published TIGER* numbers (e.g. Beauty HSTU 0.04507/0.03208/0.06479/0.03844 ≈ TIGER paper's TIGER), while our own TIGER (VAE **and** K-Means) lands at roughly half (Beauty ~0.0246/0.0167/0.0379/0.0210). This gap is consistent across all three Amazon datasets. On Steam the TIGER/HSTU gap is only ~10% (expected), so the problem is specific to the sparse Amazon setting.
  - **Diagnosed 2026-07-04: NOT the Semantic-ID/checkpoint mismatch.** `--eval_only` on `tiger_beauty_checkpoints/best_checkpoint.msgpack` (arch from param shapes: embed=256, heads=4, attn=128, linear=512) gives `Valid@Beam=0.99637` and reproduces the 2026-06-06 table row *exactly* (HR_5=0.02357 … MRR=0.01633). The checkpoint and `semantic_ids_beauty.json` are consistent; the gap is a genuine performance issue.
  - **Prime suspect: severe under-training.** The best-val checkpoint is from **epoch 5** (~2,500 gradient steps at bs=256 on 131k samples) and the model is smaller (embed=256) than the Steam replication (embed=384, 30 epochs). Likely early-stopping fired too early on the sparse dataset. Secondary suspects: Semantic-ID quality (MiniLM title-only embeddings vs the paper's richer item text with Sentence-T5) and missing user token.
- **Suspect / failed runs** (see ⚠️/❌ tags above): ML-1M HSTU line with Best Val=-1.0, STEAM TIGER Joint V2, STEAM TIGER RL-CoT. (The STEAM TIGER Seq2Seq row is now RESOLVED — see below.)
- **Beam width.** All TIGER-family decoders use `beam_size=20`, so HR@20 is capped by beam recall by construction (standard TIGER practice, not a bug).

## Resolution: Seq2Seq "degeneration" was a Semantic-ID / checkpoint mismatch (2026-07-03)

The STEAM TIGER Seq2Seq degeneration (0.062, HR flat across K) was **not** a model or beam-search defect. Root cause: TIGER-family checkpoints had **no binding to the Semantic-ID assignment** they were trained on, and RQ-KMeans IDs were generated non-deterministically (unseeded). Reloading a checkpoint against a regenerated ID file decoded into the wrong code space — `Valid@Beam≈0`, all metrics ≈0 — and eval reported the zeros silently.

Fixes landed:
- Shared `src/models/tiger_tokenization.py` + `src/evaluation/tiger_decode.py` (dedup + single source of truth for tokenization/beam search; byte-equivalence verified against the originals).
- `save_checkpoint` writes a `<ckpt>.meta.json` Semantic-ID **hash**; `verify_semantic_ids_hash` checks it on load; `assert_decode_validity` **raises** when `Valid@Beam≈0` instead of logging zeros.
- Eval loads the **best** checkpoint (not latest); resumed runs restore true best-val params.
- RQ-KMeans seeded (`--seed`, default 42); grain loaders use `worker_count=0` (fixes a child-process crash).

Re-running Seq2Seq on STEAM with the consistent RQ-VAE IDs + fixed pipeline gives HR_5=0.164 / NDCG_10=0.154 / HR_20=0.232, Valid@Beam=0.99998 — competitive with standard TIGER. See the 2026-07-03 row above.

> Note: the same checkpoint↔ID mismatch class may also affect the Amazon TIGER 2× gap and other TIGER-family rows — those checkpoints predate the hash binding, so they cannot be verified from disk and should be re-run before being cited.

## Semantic-ID Quality A/B/C/D Experiment on Beauty (2026-07-04)

Four-arm controlled comparison, all with identical model (blocks=4, embed=384), schedule (30 epochs, patience 10), and eval; only the Semantic-ID source varies:

| Arm | Item text | Encoder | Quantizer | ID collision | Test NDCG@10 | Test HR@10 | vs D |
|:--|:--|:--|:--|--:|--:|--:|:--|
| D | — (no semantics) | — | seeded uniform random | 0.04% | 0.01629 | 0.02661 | — |
| C | title only | MiniLM | seeded RQ-KMeans | 14.1% | 0.01814 | 0.03264 | +11.4% |
| A | title only | MiniLM | RQ-VAE | — | 0.01955 | 0.03363 | +20.0% |
| B | title+brand+category+price | Sentence-T5-base | seeded RQ-KMeans | 7.3% | **0.02192** | **0.03886** | **+34.6%** |

Decomposition:
- **D (random floor): TIGER retains ~74% of its title-only performance with meaningless IDs** — most beauty performance comes from the discrete hierarchical ID structure + co-occurrence memorization, not semantics. Note D has the *best* reachability (0.04% collisions), so the semantic gains below are conservative lower bounds.
- **B vs C (quantizer held fixed): +20.8% NDCG@10 from rich text + stronger encoder.** This is the causal effect of Semantic-ID content quality. Relative to the random floor, rich-text semantics is worth +34.6% vs title-only's +11.4% — i.e. rich text **triples** the semantic contribution.
- A vs C: RQ-VAE is ~+7.8% over RQ-KMeans at equal text/encoder — combining rich text with RQ-VAE is the obvious next stack (est. ~0.0235).
- Rank-profile insight: D matches A at HR@1 (0.0085 vs 0.0091) but loses badly at HR@20 (0.0352 vs B's 0.0562, −37%) → semantics buys deep-rank recall/generalization; top-1 hits are mostly memorization.
- Remaining gap to LIGER's TIGER (0.0322): dedup token (they have 0% collisions), Sentence-T5-XXL (11B vs our 110M), their 6-layer/128-dim/dropout-0.2 recipe (our val peaks by epoch ~3 → overfitting), and description field in item text.

### Val→Test gap diagnostic (Beauty −30-35% vs Steam −13%), 2026-07-04

All arms drop sharply from val to test on Beauty (e.g. Arm B HR@10: 0.0527 val peak → 0.0389 test) but only ~13% on Steam. Diagnosed by bucketing Arm B's HR@10 by the target item's training-set frequency:

| Target train-freq | Val share → Test share | Val HR@10 | Test HR@10 |
|:--|:--|--:|--:|
| <5 | 2.1% → 4.3% | 0.000 | 0.001 |
| 5–20 | 13.7% → 16.7% | 0.009 | 0.006 |
| 20–50 | 22.3% → 24.4% | 0.017 | 0.015 |
| 50–100 | 18.6% → 17.7% | 0.029 | 0.028 |
| 100–500 | 32.5% → 28.6% | 0.067 | 0.062 |
| 500+ | 10.8% → 8.3% | 0.174 | 0.141 |

Counterfactual decomposition (val per-bucket HR × test bucket shares = 0.0444):
- **54% of the drop = composition shift.** Test targets (each user's chronologically last item) skew newer/rarer: freq<5 share doubles, freq<20 goes 15.9%→21.0%. The HR-vs-frequency curve is extremely steep (HR ≈ 0 below freq 5 vs 0.17 at freq 500+), so a small leftward shift in composition costs a lot.
- **46% = within-bucket decline**, from (a) model-selection bias (best epoch picked on val inflates val) and (b) temporal drift (equal-frequency test targets are more recent; their co-occurrence evidence is staler).
- **Why Steam doesn't show this:** median test-target train-freq is 16,182 (vs Beauty's 58); 99.6% of Steam targets sit deep in the saturated flat part of the frequency curve, and the val/test popularity ratio (1.30×, identical on both datasets) shifts nothing there.
- **Actionable target:** Beauty's ceiling is set by tail-item learnability — freq<20 targets (21% of test) have HR ≈ 0 even with rich-text IDs, i.e. semantic generalization is not yet reaching the tail. Improving tail transfer (better IDs, dedup token, user token) matters more than squeezing the head.
