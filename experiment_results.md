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

| 2026-07-08 | 🏆 **Winning stack on STEAM** (rich text title+genres+tags+specs+price+publisher → t5-XXL bf16 → MLP RQ-VAE → dedup L4; enc-dec 6+6 128d dropout 0.2, cosine 3e-4) | 0.17347 | 0.15103 | 0.20771 | 0.16203 | 0.25447 | 0.17382 | 0.15139 | ✅✅✅ **Generalization confirmed on dense data: +6.9% HR@10 over our prior seq2seq (0.19431), +9.4% over LIGER-paper steam TIGER (0.18980; we are at 109% of their number), beats the random-ID ceiling (0.19966) AND edges out HSTU (0.20735/0.16159) — first TIGER-family win over dense retrieval on Steam.** ID stats: 2.42% collisions, full codebooks, Valid@Beam 0.99998. |

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
## Semantic-ID Levels × Codebook Grid on Beauty (2026-07-06/07)

All runs: rich-text Sentence-T5 embeddings + seeded RQ-KMeans IDs, identical model (blocks=4, embed=384) and schedule (30 epochs, patience 10). Baseline is the Arm-B row above (L3×K256 = 0.02192 NDCG@10).

| Config | ID space | Collision rate | Test HR@5 | HR@10 | HR@20 | NDCG@10 | vs baseline |
|:--|--:|--:|--:|--:|--:|--:|:--|
| L2×K256 | 65K | 40.7% | 0.01704 | 0.02737 | 0.04132 | 0.01482 | −32% |
| L2×K512 | 262K | 24.9% | 0.01936 | 0.02951 | 0.04494 | 0.01594 | −27% |
| L3×K128 | 2.1M | 10.0% | 0.02428 | 0.03448 | 0.05040 | 0.01974 | −10% |
| L4×K64 | 16.7M | 5.1% | 0.02482 | 0.03622 | 0.05232 | 0.02028 | −7.5% |
| **L3×K256 (baseline)** | 16.7M | 7.3% | 0.02522 | 0.03886 | 0.05621 | **0.02192** | — |
| L3×K256 + level loss weights [2,1,0.5] | 16.7M | 7.3% | 0.02129 | 0.03255 | 0.04727 | 0.01861 | **−15%** |

Findings:
1. **Two regimes.** At high collision rates (L2 configs) reachability dominates: 41% unreachable items puts L2×K256 (0.01482) *below the random-ID floor* (0.01629). At low collision rates the **level penalty** appears: L4×K64 has *fewer* collisions than baseline (5.1% vs 7.3%) yet loses by 7.5% — each extra level is another sequential beam commitment (see the beam-attrition diagnostic), and K=64 is too coarse per level. **L3×K256 sits near the joint optimum**; the TIGER paper's choice is well-supported.
2. **Front-weighted level loss HURTS: −15% NDCG@10**, and Valid@Beam drops 0.975 → 0.892. Empirical close-out of the "weight earlier levels for generalization" hypothesis: CE weighting cannot add information at L1 (its difficulty is intrinsic next-cluster uncertainty — L1 teacher-forced acc is frequency-independent at ~6%), while down-weighting L3 degrades the memorization deeper levels need, producing more invalid paths. Consistent with the earlier probe where even exhaustive L1 beam widening gained nothing.

### Collision-aware decoding probe (expand tied items by popularity, 2026-07-07)

Decode-side alternative to the dedup token: map each decoded path to *all* items sharing that Semantic ID, ordered by train-frequency prior (tie-break: item id). Re-evaluated every checkpoint, decoding once and ranking under both mappings:

| Config (collision rate) | NDCG@10 single → collide+ | Verdict |
|:--|:--|:--|
| L2×K256 (40.7%) | 0.01482 → 0.01273 | **−14%: backfires** — fat paths dilute ranks; popularity is a poor P(item\|path) when clusters are large |
| L2×K512 (24.9%) | 0.01594 → 0.01638 | +2.8% |
| L3×K128 (10.0%) | 0.01974 → 0.01923 | −2.6% |
| L3×K256 rich (7.3%) | 0.02192 → 0.02159 | −1.5% (HR@20 +4.4%) |
| L4×K64 (5.1%) | 0.02028 → 0.02030 | flat |
| L3×K256 title/VAE (arm A) | 0.01955 → 0.02025 | +3.6% |
| weighted arm | 0.01861 → 0.01981 | +6.4% (partially repairs its sloppy L3) |
| random IDs (0.04%) | 0.01629 → 0.01629 | sanity check: identical ✓ |

Conclusion: collision expansion is **not a free win** — reachability gains are mostly offset (or overwhelmed, at high collision rates) by rank dilution from fat paths. The static popularity prior is too weak an approximation of P(item|path). The proper fix remains the **dedup token**, which gives every item a unique, *learned* suffix instead of a fixed prior.
## Close-the-gap iterations toward LIGER's TIGER (HR@10=0.0601), 2026-07-07

| Date | Config | Test HR@5 | HR@10 | HR@20 | NDCG@10 | Val peak NDCG@10 | Notes |
|:--|:--|--:|--:|--:|--:|--:|:--|
| 2026-07-07 | Dedup token (rich KMeans IDs + 4th disambiguation level, collisions 7.3%→0%) | 0.02535 | 0.03792 | 0.05362 | 0.02173 | 0.03337 | **Val +12% but test FLAT vs baseline 0.02192.** Val plateaus at epoch 3; loose any-metric best-checkpoint selection drifted to epoch 26 (23 epochs of overfitting). Valid@Beam 0.936 (4th level adds some invalid paths). Lesson: overfitting + val-selection noise is now the binding constraint — regularization (LIGER recipe) needed before ID gains can transfer to test. |
| 2026-07-07 | RQ-VAE(std-emb) rich IDs + dedup (collisions 2.68%→0%), standard arch | 0.02772 | 0.04190 | 0.06216 | 0.02409 | — | ✅ **+7.8% HR@10 / +9.9% NDCG@10 over baseline.** RQ-VAE required per-dim standardization of the L2-normalized T5 embeddings (raw scale collapses the codebook to 3/5/9 codes, 98.9% collisions; standardized: 138/256/256 codes, 2.68%). Valid@Beam 0.993 — VAE's cleaner clusters make the dedup level far easier than KMeans' (0.936). Early stop @14. |
| 2026-07-07 | **LIGER recipe** (6 blocks × embed 128, heads 4, FFN 1024, dropout 0.2) on rich KMeans+dedup IDs | 0.03130 | **0.04744** | 0.07038 | **0.02643** | — | ✅✅ **+22% HR@10 / +20.6% NDCG@10 over baseline — regularization was the binding constraint.** Trains productively to epoch 38 (vs val-peak-at-3 with the fat 4×384 model); unlocks the dedup gains that tested flat under the old recipe. Params ~1.3M vs 5.8M. Valid@Beam 0.989. |
| 2026-07-07 | Combo: LIGER recipe + RQ-VAE+dedup IDs | 0.03211 | **0.04820** | 0.07070 | 0.02623 | 0.03781 | Current best HR@10, but only +1.6% over the KMeans+dedup recipe run (NDCG@10 −0.8%) — **the RQ-VAE and regularization gains do not compose**; under strong regularization the quantizer difference is largely absorbed. Early stop @32. Valid@Beam 0.996. Remaining to LIGER's 0.0601: −22%. |
| 2026-07-07 | Text upgrade: +description field, sentence-t5-LARGE encoder (RQ-VAE+dedup, LIGER recipe) | 0.02848 | 0.04324 | 0.06609 | 0.02431 | 0.03803 | ❌ **−10% vs combo (0.04820)** — description text + larger encoder makes embeddings more diffuse (pre-dedup collisions 2.68%→5.27%), degrading cluster/behavior alignment. Text-side levers exhausted; title+brand+category+price with t5-base remains the best ID recipe. |
| 2026-07-07 | +User token (2000 hash buckets, combo config) | 0.01963 | 0.03135 | 0.04856 | 0.01699 | 0.02604 | ❌ **−35% vs combo** — fragments sparse data (~11 users/bucket of signal); val peaked at epoch 10 then declined (run interrupted at 14, best ckpt evaluated). User token is a NEGATIVE on 5-core Beauty. |
| 2026-07-07 | LIGER-exact long run (cosine 3e-4, warmup 10k, wd 0.035, ~200k-step budget, attention 384=64x6, eval/10ep) | 0.02969 | 0.04561 | 0.06649 | 0.02548 | 0.03501 | ❌ **Training budget ruled out**: early stop @180 epochs, val flat 0.034-0.035 from epoch 90 on — model saturates ~100 epochs; 10x budget and 3x attention width land slightly BELOW combo (0.04820). |
| 2026-07-07 | **Enc-dec (T5-style 6+6, embed 128, attn 384=64x6, dropout 0.2) + RQ-VAE+dedup IDs, cosine 3e-4** | 0.03327 | **0.04972** | 0.07374 | **0.02804** | 0.04062 | ✅ **NEW BEST — architecture switch works (+3.2% HR@10 / +6.9% NDCG@10 over combo).** Best val at epoch 40 (0.04062, highest of campaign) then declines; run killed at 90, best ckpt evaluated. Now at 82.7% of LIGER's HR@10 (0.0601) and 87.1% of their NDCG@10 (0.0322). Valid@Beam 0.989. |
| 2026-07-07 | 🏆 **FINAL: Enc-dec + sentence-t5-XXL IDs (bf16 local) + RQ-VAE + dedup, cosine 100ep/warmup 5k** | 0.03398 | **0.05263** | 0.07651 | **0.02920** | 0.04278 | ✅✅ **Campaign best: 87.6% of LIGER's HR@10 (0.0601), 90.7% of their NDCG@10 (0.0322).** XXL embeddings (encoder-only 4.8B fits 16GB in bf16) add +5.9% HR@10 over t5-base IDs; shortened cosine aligns LR decay with the observed ~35-epoch convergence window (val peak 0.04278 @35, best val of campaign). Valid@Beam 0.990. Early stop @85, epoch-35 weights. |

| 2026-07-08 | 🏆 **MLP RQ-VAE IDs (LIGER arch: hidden 768→512→256→128, dropout 0.1, 8000ep) + enc-dec + short cosine** | 0.03720 | **0.05558** | 0.08273 | **0.03078** | 0.04542 | ✅✅✅ **New campaign best — 92.5% of LIGER's HR@10, 95.6% of their NDCG@10.** The MLP encoder lifts L1 codebook utilization 54%→99% (fixes the density-concentration dead-code problem the linear encoder could not), directly attacking the diagnosed L1 bottleneck. +5.6% HR@10 over the linear-VAE XXL run. Val peak 0.04542 @ep30 (campaign high); run stopped at 46 on the declining tail, epoch-30 ckpt evaluated. |

| 2026-07-08 | **Category-direct (lookup) SIDs**: L1=depth-3 taxonomy (46), L2=leaf-in-parent, L3=residual-KMeans (category-mean removed), L4=dedup — ZERO training | 0.03483 | 0.05505 | 0.08358 | 0.02994 | — | ✅ **Within 1-3% of the campaign-best learned quantizer (HR@20 even +1%), at zero training cost.** Answers "why not use labels directly": taxonomy prefixes carry most of the generalization value (human categories ≈ distilled behavioral structure); 55% 3-level collisions are absorbed by the frequency-ordered dedup level. Early stop @70. Valid@Beam 0.996. |

| 2026-07-08 | **MGCL SIDs** (UniSID-style: 3 linear heads over XXL emb, per-level SupCon [depth-3 cat / leaf cat / instance-discrimination views] + usage-entropy bonus, argmax codes) | 0.03331 | 0.04807 | 0.06739 | 0.02808 | — | Learned taxonomy-contrastive IDs: excellent ID stats (3.5% collisions, L1 99% usage, dedup groups ≤5) and the **best HR@1 of the campaign (0.01306)**, but −14% HR@10 / −9% NDCG@10 vs MLP RQ-VAE. Sharp instance-separated codes trade deep-rank recall for top-rank precision. Early stop @75. |

### SID-construction three-way ablation on Beauty (2026-07-08)

Same downstream everything (enc-dec 6+6 128d, dropout 0.2, cosine, dedup 4th level); only the 3 semantic levels differ:

| Family | Method | Train cost | 3-lvl collisions | HR@1 | HR@10 | NDCG@10 |
|:--|:--|:--|--:|--:|--:|--:|
| RQ (residual quantization) | MLP RQ-VAE on XXL emb | 8000-ep VAE | 4.3% | 0.01225 | **0.05558** | **0.03078** |
| Lookup (taxonomy-direct) | depth-3 cat / leaf / residual-KMeans | **zero** | 55.5% | 0.01181 | 0.05505 | 0.02994 |
| Contrastive (UniSID-style) | MGCL 3 heads, argmax codes | 2000 steps | **3.5%** | **0.01306** | 0.04807 | 0.02808 |

Takeaways:
1. **Zero-training taxonomy lookup reaches 97-99% of the best learned quantizer** — with a curated category tree, SID sophistication buys very little; human taxonomy ≈ pre-distilled behavioral structure. High raw collisions (55%) are absorbed by the frequency-ordered dedup level.
2. **MGCL trades recall for precision**: instance-discrimination L3 makes codes maximally separated (best HR@1, near-zero collisions) but the sharper partition loses the smooth prefix-neighborhoods that deep-rank recall rides on (−14% HR@10).
3. RQ's residual structure remains the best HR@10/NDCG@10 — its levels factorize variance (coarse→fine of the SAME geometry), which beam search exploits better than parallel independently-supervised heads.

### Campaign summary: closing the Beauty TIGER gap (2026-07-07)

Goal: reproduce LIGER-paper TIGER on Beauty (HR@10=0.0601, NDCG@10=0.0322). Result: **0.03886 -> 0.05263 HR@10 (+35.4%)**, reaching 87.6%/90.7% of the target.

**Effective levers (stacked, in order of adoption):**
1. Rich-text Semantic IDs (title+brand+category+price -> Sentence-T5) — +12% (earlier A/B/C/D experiment)
2. RQ-VAE quantizer (per-dim standardization of the L2-normalized T5 embeddings is REQUIRED — raw scale collapses the codebook to 3/5/9 codes) + dedup 4th token (collisions -> 0)
3. LIGER regularization recipe (d_model 128, dropout 0.2) — +22%; overfitting was the binding constraint masking all ID-side gains
4. T5-style encoder-decoder architecture (6+6, d_kv 64x6=384) — +3-7% over decoder-only
5. sentence-t5-XXL item embeddings (bf16, fits 16GB locally) — +5.9%
6. Cosine schedule matched to the actual convergence window (100 epochs, warmup 5k) instead of the paper's 200k-step budget

**Ruled out empirically (9 hypotheses):** under-training/model size (10x budget: flat), user-id token (-35%; LIGER's own config sets include_user_id=False), description field (-10%; LIGER also excludes it), t5-large IDs (-10%), front-weighted level loss (-15%), beam widening (0), decode-side collision expansion (backfires at high collision rates), 3x attention width alone (0), LIGER's RQ-VAE hyperparams on our linear-encoder RQ-VAE (worse — theirs is an MLP encoder).

**Cross-dataset validation (2026-07-08):** the winning stack transferred to STEAM (dense regime) at full strength — HR@10 0.20771 / NDCG@10 0.16203, exceeding the LIGER paper's steam TIGER by 9.4%/7.8% and beating HSTU. The improvements are general, not sparse-data-specific; even the earlier "random IDs match semantic IDs on Steam" finding is overturned by the upgraded ID pipeline (+4% over the random-ID ceiling).

**Remaining -12% gap, attributed to:** (a) LIGER's "extensive hyperparameter search" (their words), (b) their MLP RQ-VAE encoder (hidden [768,512,256], 8000 epochs, batch 2048) vs our linear encoder, (c) possible eval-protocol deltas (they exclude cold-start items from the in-set metric; ~0.6% of our test targets are cold). The per-lever causal chain above is fully reproducible from the checkpoints + hash-bound Semantic-ID files in data/.
| 2026-07-08 | Readout A/B arm=item (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03600 | 0.02532 | 0.04896 | 0.02947 | 0.06551 | 0.03362 | 0.02685 | Best val NDCG@10=0.04011 (epoch 25); leak self/rand=0.325/-0.001, readout leak=0.325, table align=0.017 |
| 2026-07-08 | Readout A/B arm=begin (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03412 | 0.02365 | 0.04937 | 0.02855 | 0.07012 | 0.03377 | 0.02617 | Best val NDCG@10=0.04215 (epoch 31); leak self/rand=0.150/-0.001, readout leak=0.032, table align=0.003 |
| 2026-07-08 | Readout A/B arm=item (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.16669 | 0.14594 | 0.19939 | 0.15644 | 0.24581 | 0.16811 | 0.15213 | Best val NDCG@10=0.18150 (epoch 12); leak self/rand=0.583/0.001, readout leak=0.583, table align=0.091 |
| 2026-07-08 | Readout A/B arm=begin (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.16821 | 0.14674 | 0.20205 | 0.15760 | 0.25086 | 0.16988 | 0.15310 | Best val NDCG@10=0.18268 (epoch 12); leak self/rand=0.350/0.002, readout leak=0.054, table align=0.006 |
| 2026-07-08 | Readout A/B arm=item (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03609 | 0.02629 | 0.04870 | 0.03033 | 0.06596 | 0.03466 | 0.02819 | Best val NDCG@10=0.04023 (epoch 12); leak self/rand=0.297/-0.001, readout leak=0.297, table align=0.017 |
| 2026-07-08 | Readout A/B arm=begin (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03492 | 0.02447 | 0.05049 | 0.02951 | 0.07123 | 0.03470 | 0.02706 | Best val NDCG@10=0.04284 (epoch 18); leak self/rand=0.124/0.000, readout leak=0.020, table align=0.002 |
| 2026-07-08 | Readout A/B arm=item (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03416 | 0.02466 | 0.04955 | 0.02960 | 0.06967 | 0.03467 | 0.02726 | Best val NDCG@10=0.04014 (epoch 12); leak self/rand=0.299/0.000, readout leak=0.299, table align=0.018 |
| 2026-07-08 | Readout A/B arm=begin (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03461 | 0.02422 | 0.04905 | 0.02889 | 0.06895 | 0.03387 | 0.02654 | Best val NDCG@10=0.04177 (epoch 34); leak self/rand=0.136/0.000, readout leak=0.031, table align=0.004 |
| 2026-07-08 | Readout A/B arm=item (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.16716 | 0.14642 | 0.20001 | 0.15697 | 0.24670 | 0.16870 | 0.15260 | Best val NDCG@10=0.18160 (epoch 11); leak self/rand=0.600/0.001, readout leak=0.600, table align=0.095 |
| 2026-07-08 | Readout A/B arm=begin (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.16841 | 0.14686 | 0.20233 | 0.15775 | 0.25073 | 0.16991 | 0.15313 | Best val NDCG@10=0.18250 (epoch 11); leak self/rand=0.365/0.002, readout leak=0.040, table align=0.017 |
| 2026-07-08 | Readout A/B arm=item (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.16817 | 0.14688 | 0.20094 | 0.15740 | 0.24701 | 0.16897 | 0.15287 | Best val NDCG@10=0.18201 (epoch 12); leak self/rand=0.583/-0.002, readout leak=0.583, table align=0.091 |
| 2026-07-08 | Readout A/B arm=begin (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.16915 | 0.14729 | 0.20301 | 0.15817 | 0.25026 | 0.17005 | 0.15339 | Best val NDCG@10=0.18278 (epoch 12); leak self/rand=0.413/0.000, readout leak=0.063, table align=0.008 |

## Dedicated readout-token A/B (2026-07-08) — summary

12 raw rows above (2 arms × 2 datasets × 3 seeds 42/123/7). Full write-up with setup, §-by-§ implementation mapping, equivalence proofs, diagnostics, and interpretation lives in `dedicated_readout_token_design.md` §13.

- **Question**: does moving the retrieval readout off the last item's residual stream onto a dedicated `<begin>` token (anchor-masked, logical-position rel-bias, K isolated forks per sequence) help a shallow two-tower sampled-softmax model?
- **Steam (dense)**: begin arm wins **all 7 test metrics in all 3 seeds** (HR@10 0.20246±0.0004 vs 0.20011±0.0006, +1.2%, ~4σ). 
- **Beauty (sparse)**: recall/precision trade — HR@20 +4.6%, HR@10 +1.2%, val NDCG@10 +5.2% (all seeds), but NDCG@5 −5.2%, MRR −3.1%. The removed identity residue doubles as a useful last-item-similarity prior on sparse data.
- **Mechanism confirmed**: baseline readout vector carries cos 0.31 (Beauty) / 0.59 (Steam) of the current item's input embedding (random ≈ 0); `<begin>` cuts it ~11× to 0.03/0.05. Item arm learns in/out table alignment 0.09 on Steam (leakage → logit bias regime per design doc §9.2-2) — exactly where it loses.
- **Cost**: +3% (Beauty) / +25% (Steam) wall-clock per epoch; identical supervision by construction.
- **Caveat**: this trainer (multi-position + sampled softmax M=1024 + untied towers) is a new baseline; absolute numbers are below the legacy full-softmax HSTU rows — compare arms, not families.
- Code: `src/models/readout_hstu.py`, `examples/train_readout_hstu.py`; §4 train/serve equivalence verified numerically (4/4 checks at 1e-5).
| 2026-07-08 | Readout A/B arm=begin_reinjscalar (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03251 | 0.02277 | 0.04865 | 0.02797 | 0.06936 | 0.03315 | 0.02558 | Best val NDCG@10=0.04218 (epoch 30); leak self/rand=0.145/-0.001, readout leak=0.026, table align=0.003, alpha=-0.5836 |
| 2026-07-08 | Readout A/B arm=begin_reinjscalar (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03564 | 0.02439 | 0.05040 | 0.02914 | 0.06953 | 0.03395 | 0.02642 | Best val NDCG@10=0.04319 (epoch 20); leak self/rand=0.119/-0.000, readout leak=0.029, table align=0.003, alpha=1.5327 |
| 2026-07-08 | Readout A/B arm=begin_reinjscalar (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03501 | 0.02420 | 0.05133 | 0.02946 | 0.07217 | 0.03468 | 0.02685 | Best val NDCG@10=0.04237 (epoch 25); leak self/rand=0.127/-0.002, readout leak=0.041, table align=0.004, alpha=1.8045 |
| 2026-07-08 | Readout A/B arm=begin_reinjscalar (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.16852 | 0.14738 | 0.20209 | 0.15815 | 0.25118 | 0.17049 | 0.15392 | Best val NDCG@10=0.18267 (epoch 5); leak self/rand=0.300/0.003, readout leak=-0.006, table align=-0.049, alpha=-3.6236 |
| 2026-07-08 | Readout A/B arm=begin_reinjscalar (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.16637 | 0.14630 | 0.20019 | 0.15716 | 0.24847 | 0.16929 | 0.15319 | Best val NDCG@10=0.18191 (epoch 5); leak self/rand=0.342/-0.001, readout leak=0.038, table align=0.085, alpha=4.8811 |
| 2026-07-08 | Readout A/B arm=begin_reinjscalar (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.16850 | 0.14774 | 0.20266 | 0.15870 | 0.25221 | 0.17116 | 0.15454 | Best val NDCG@10=0.18303 (epoch 5); leak self/rand=0.362/0.002, readout leak=0.040, table align=0.092, alpha=5.4809 |
| 2026-07-08 | Readout A/B arm=begin_reinjscalar (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03416 | 0.02315 | 0.04995 | 0.02822 | 0.07079 | 0.03347 | 0.02566 | Best val NDCG@10=0.04253 (epoch 21); leak self/rand=0.146/0.000, readout leak=0.079, table align=0.008, alpha=2.3180 |

### (c′) Gated transition-prior re-injection + stratified follow-ups (2026-07-08)

7 runs above tagged `begin_reinjscalar` (3 seeds × 2 datasets + 1 alpha-init rerun). Full analysis in `dedicated_readout_token_design.md` §13.8–13.9. TLDR:
- **Beauty (α>0 basin)**: ALL-slice HR@5 parity with item arm (0.03533 vs 0.03542) while keeping/extending the complement advantage (bigram_out HR@5 +107% vs item); val NDCG@10 0.04319 = campaign best. α does NOT recover the bigram slice (memorized transitions need the prior inside joint training) — it acts as a soft similarity prior on the long tail.
- **Steam**: metric-neutral, but converges 2.4× faster (best epoch 12→5, all seeds); α large in all seeds with effective term α·align consistently positive — α measures routing, not data need.
- **α sign is non-identifiable** (joint flip with embedding geometry); on Beauty the positive basin is strictly better — use `--alpha_init 0.1`.
- Next ablation suggested: dedicated transition table e_trans (untied FPMC factor) to recover bigram memorization without re-contaminating the readout.
| 2026-07-09 | Readout A/B arm=begin_time (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03497 | 0.02387 | 0.04972 | 0.02862 | 0.06788 | 0.03319 | 0.02571 | Best val NDCG@10=0.03888 (epoch 141); leak self/rand=0.305/0.002, readout leak=-0.023, table align=-0.001 |
| 2026-07-09 | Readout A/B arm=begin_time (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.02920 | 0.01800 | 0.04807 | 0.02408 | 0.07392 | 0.03059 | 0.02137 | Best val NDCG@10=0.03311 (epoch 51); leak self/rand=0.277/-0.002, readout leak=0.001, table align=0.001 |
| 2026-07-09 | Readout A/B arm=begin_time (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03618 | 0.02419 | 0.05527 | 0.03030 | 0.07647 | 0.03563 | 0.02664 | Best val NDCG@10=0.04237 (epoch 89); leak self/rand=0.177/-0.001, readout leak=0.016, table align=0.002 |
| 2026-07-09 | Readout A/B arm=begin_time (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.21541 | 0.17800 | 0.26309 | 0.19335 | 0.32478 | 0.20888 | 0.18248 | Best val NDCG@10=0.21911 (epoch 11); leak self/rand=0.631/0.001, readout leak=-0.043, table align=0.008 |
| 2026-07-09 | Readout A/B arm=begin_time (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.21476 | 0.17822 | 0.26217 | 0.19347 | 0.32417 | 0.20908 | 0.18292 | Best val NDCG@10=0.21943 (epoch 13); leak self/rand=0.667/0.001, readout leak=-0.044, table align=0.007 |
| 2026-07-09 | Readout A/B arm=begin_time (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.21257 | 0.17525 | 0.25957 | 0.19037 | 0.32133 | 0.20592 | 0.17961 | Best val NDCG@10=0.21832 (epoch 14); leak self/rand=0.650/0.001, readout leak=-0.039, table align=0.007 |
| 2026-07-09 | Readout A/B arm=begin_timez (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03796 | 0.02585 | 0.05706 | 0.03201 | 0.07937 | 0.03761 | 0.02848 | Best val NDCG@10=0.04515 (epoch 41); leak self/rand=0.142/-0.000, readout leak=0.036, table align=0.005 |
| 2026-07-09 | Readout A/B arm=begin_timez (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03770 | 0.02583 | 0.05411 | 0.03110 | 0.08000 | 0.03762 | 0.02854 | Best val NDCG@10=0.04502 (epoch 21); leak self/rand=0.127/-0.000, readout leak=0.017, table align=0.002 |
| 2026-07-09 | Readout A/B arm=begin_timez (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03940 | 0.02718 | 0.05697 | 0.03282 | 0.08004 | 0.03860 | 0.02957 | Best val NDCG@10=0.04694 (epoch 37); leak self/rand=0.111/-0.001, readout leak=0.030, table align=0.005 |

### Request-time conditioning of <begin> (2026-07-09) — summary

Rows tagged `begin_time` / `begin_timez` (zero-init projection). Full analysis in design doc §13.11.
- **Steam**: HR@10 0.26161 / NDCG@10 0.19240 (3 seeds) = **+29%/+22% over begin baseline, +26% over all prior repo models**. Shuffle test: shuffled time collapses to 0.162 (below no-time 0.202) → genuine clock conditioning, not artifact.
- **Beauty**: default-init unstable (val below baseline); **zero-init projection → uniform new best on all 7 metrics** (HR@10 0.05605, NDCG@10 0.03198, HR@5 0.03835 — finally beats the item arm's top-rank precision without any transition machinery).
- Rules: zero-init additive conditioning pathways; add feature-missing robustness before production (zero-feature input destroys the readout).
| 2026-07-11 | Readout A/B arm=begin_timelate (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.19715 | 0.16493 | 0.24072 | 0.17896 | 0.29884 | 0.19358 | 0.17017 | Best val NDCG@10=0.20733 (epoch 12); leak self/rand=0.412/0.002, readout leak=0.064, table align=0.007 |
| 2026-07-11 | Readout A/B arm=begin_timelate (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.19833 | 0.16578 | 0.24252 | 0.17999 | 0.30161 | 0.19485 | 0.17105 | Best val NDCG@10=0.20693 (epoch 13); leak self/rand=0.462/0.000, readout leak=0.037, table align=0.013 |

### Late-fusion ablation of request-time conditioning (2026-07-11) — summary

Rows tagged `begin_timelate` (2 seeds, Steam). Question: does time need to condition the *aggregation* (input mode: features added onto `<begin>` before the transformer), or is the §13.11 gain just a separable time prior (late mode: residual MLP `h + MLP([h; t])` with zero-init output layer on the finished readout vector)? Full analysis in design doc §13.12.
- **Steam late fusion: HR@10 0.24162 / NDCG@10 0.17948 (2 seeds, tight spread)** — clearly above the no-time begin baseline (0.20246) but clearly below input mode (0.26161). Late fusion recovers **~66% of the input-mode gain**.
- Reading: about two-thirds of the request-time benefit is a post-hoc time prior/transform on a fixed user summary; the remaining third requires time *inside* the transformer, i.e. time genuinely changes what the readout aggregates. Input mode (`--time_features`, without `--time_mode late`) stays the recommended configuration.
- Not yet run: the `--late_w2_std` / `--late_no_residual` factorization cells, and the Beauty late arm (one launch aborted at epoch 2, no result row).
| 2026-07-11 | Readout A/B arm=begin_timelate (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.19691 | 0.16452 | 0.24042 | 0.17852 | 0.29866 | 0.19317 | 0.16964 | Best val NDCG@10=0.20652 (epoch 15); leak self/rand=0.466/0.001, readout leak=0.058, table align=0.008 |
| 2026-07-11 | Readout A/B arm=begin_timelate (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03966 | 0.02676 | 0.05844 | 0.03277 | 0.08170 | 0.03859 | 0.02928 | Best val NDCG@10=0.04648 (epoch 29); leak self/rand=0.154/0.001, readout leak=0.042, table align=0.003 |
| 2026-07-11 | Readout A/B arm=begin_timelate (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.04118 | 0.02829 | 0.05822 | 0.03380 | 0.08036 | 0.03938 | 0.03040 | Best val NDCG@10=0.04713 (epoch 26); leak self/rand=0.102/-0.000, readout leak=0.024, table align=0.002 |
| 2026-07-11 | Readout A/B arm=begin_timelate (HSTU 4x256, sampled-softmax M=1024, untied) on BEAUTY | Local (GeForce RTX 4080) | 0.03953 | 0.02727 | 0.05880 | 0.03347 | 0.08308 | 0.03962 | 0.03027 | Best val NDCG@10=0.04541 (epoch 21); leak self/rand=0.128/-0.000, readout leak=0.025, table align=0.003 |
| 2026-07-11 | Readout A/B arm=begin_timelatenr (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.19285 | 0.16089 | 0.23623 | 0.17484 | 0.29415 | 0.18941 | 0.16613 | Best val NDCG@10=0.20307 (epoch 12); leak self/rand=0.694/0.002, readout leak=0.013, table align=0.001 |
| 2026-07-11 | Readout A/B arm=begin_timelatenr (HSTU 4x256, sampled-softmax M=1024, untied) on STEAM | Local (GeForce RTX 4080) | 0.19408 | 0.16242 | 0.23802 | 0.17655 | 0.29660 | 0.19131 | 0.16796 | Best val NDCG@10=0.20370 (epoch 13); leak self/rand=0.710/0.002, readout leak=0.030, table align=-0.003 |
