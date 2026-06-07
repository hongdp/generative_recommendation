# Experiment Results Log

This file documents all training runs, evaluations, and benchmarks. Refer to [SKILL.md](file:///home/hongdp/Workspace/generative_recommendation/SKILL.md) for the logging protocol.

## Trials Log

| Date | Configuration | Hardware | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | MRR | Notes |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| 2026-06-06 | MovieLens-100K Baseline (Random) | Local (CPU) | - | - | 0.00848 | 0.00409 | - | - | 0.00580 | Random recommendation baseline |
| 2026-06-06 | MovieLens-100K Baseline (Most Popular) | Local (CPU) | - | - | 0.04984 | 0.02303 | - | - | 0.01490 | Most Popular recommendation baseline |
| 2026-06-06 | HSTUModel next-item pred (5 epochs) on ML-100K | Local (CPU) | 0.06681 | 0.03692 | 0.13786 | 0.05943 | - | - | 0.05283 | Converged and ranking metrics improved significantly (+0.05024 MRR) |
| 2026-06-06 | Full HSTUModel (4 blocks, embed=256) on ML-1M | Local (GeForce RTX 4080) | 0.13212 | 0.08459 | 0.20414 | 0.10770 | - | - | 0.09371 | Best Val NDCG@10=-1.00000; Test metrics replicated |
| 2026-06-06 | Full HSTUModel (4 blocks, embed=256) on ML-1M | Local (GeForce RTX 4080) | 0.14818 | 0.09544 | 0.23725 | 0.12411 | - | - | 0.10549 | Best Val NDCG@10=0.13163; Test metrics replicated |
| 2026-06-06 | Full HSTUModel (4 blocks, paused Epoch 14) on ML-1M | Local (GeForce RTX 4080) | 0.19023 | 0.12933 | 0.27384 | 0.15601 | - | - | 0.13502 | Paused run at Epoch 14 (Best Val Epoch 12). Strong convergence. |
| 2026-06-06 | Full TIGERModel (4 blocks, embed=256) on ML-1M | Local (GeForce RTX 4080) | 0.17169 | 0.11905 | 0.24156 | 0.14169 | - | - | 0.11112 | Best Val NDCG@10=0.15646; Test metrics replicated |
| 2026-06-06 | Full TIGERModel (4 blocks, embed=256) on ML-1M | Local (GeForce RTX 4080) | 0.15348 | 0.10610 | 0.20927 | 0.12417 | - | - | 0.09804 | Best Val NDCG@10=0.13230; Test metrics replicated |
| 2026-06-06 | Full TIGER (VAE) (4 blocks, embed=256) on BEAUTY | Local (GeForce RTX 4080) | 0.02455 | 0.01671 | 0.03792 | 0.02101 | 0.05809 | 0.02607 | 0.01727 | Replication on beauty matching LIGER paper evaluation |
| 2026-06-06 | Full HSTU (blocks=4, embed=256) on BEAUTY | Local (GeForce RTX 4080) | 0.04507 | 0.03208 | 0.06479 | 0.03844 | 0.09270 | 0.04543 | 0.03534 | Replication on beauty matching LIGER paper evaluation (Best Val NDCG@10=0.05068) |
| 2026-06-06 | Full TIGER (VAE) (4 blocks, embed=256) on BEAUTY | Local (GeForce RTX 4080) | 0.02357 | 0.01597 | 0.03564 | 0.01980 | 0.05424 | 0.02450 | 0.01633 | Replication on beauty matching LIGER paper evaluation (Best Val NDCG@10=0.02776) |
| 2026-06-06 | Full TIGER (K-Means) (4 blocks, embed=256) on BEAUTY | Local (GeForce RTX 4080) | 0.02513 | 0.01645 | 0.03957 | 0.02112 | 0.05858 | 0.02590 | 0.01683 | Replication on beauty matching LIGER paper evaluation (Best Val NDCG@10=0.02982) |
| 2026-06-06 | Full HSTU (blocks=4, embed=256) on SPORTS | Local (GeForce RTX 4080) | 0.02568 | 0.01812 | 0.03731 | 0.02187 | 0.05174 | 0.02549 | 0.02035 | Replication on sports matching LIGER paper evaluation (Best Val NDCG@10=0.02898) |
| 2026-06-06 | Full TIGER (VAE) (4 blocks, embed=256) on SPORTS | Local (GeForce RTX 4080) | 0.01264 | 0.00828 | 0.02056 | 0.01081 | 0.03163 | 0.01361 | 0.00865 | Replication on sports matching LIGER paper evaluation (Best Val NDCG@10=0.01508) |
| 2026-06-06 | Full TIGER (K-Means) (4 blocks, embed=256) on SPORTS | Local (GeForce RTX 4080) | 0.01208 | 0.00777 | 0.02174 | 0.01089 | 0.03349 | 0.01385 | 0.00846 | Replication on sports matching LIGER paper evaluation (Best Val NDCG@10=0.01375) |
| 2026-06-06 | Full HSTU (blocks=4, embed=256) on TOYS | Local (GeForce RTX 4080) | 0.04708 | 0.03442 | 0.06238 | 0.03933 | 0.08289 | 0.04452 | 0.03643 | Replication on toys matching LIGER paper evaluation (Best Val NDCG@10=0.04994) |
| 2026-06-06 | Full TIGER (VAE) (4 blocks, embed=256) on TOYS | Local (GeForce RTX 4080) | 0.01927 | 0.01236 | 0.02885 | 0.01545 | 0.04137 | 0.01859 | 0.01223 | Replication on toys matching LIGER paper evaluation (Best Val NDCG@10=0.01852) |
| 2026-06-06 | Full TIGER (K-Means) (4 blocks, embed=256) on TOYS | Local (GeForce RTX 4080) | 0.02169 | 0.01447 | 0.03379 | 0.01839 | 0.05244 | 0.02305 | 0.01499 | Replication on toys matching LIGER paper evaluation (Best Val NDCG@10=0.02065) |
| 2026-06-06 | Full HSTU (blocks=4, embed=256) on STEAM | Local (GeForce RTX 4080) | 0.17071 | 0.14849 | 0.20537 | 0.15963 | 0.25475 | 0.17205 | 0.15487 | Replication on steam matching LIGER paper evaluation (Best Val NDCG@10=0.18403) |

| 2026-06-07 | Full TIGER (VAE) (4 blocks, embed=256) on STEAM | Local (GeForce RTX 4080) | 0.15523 | 0.13832 | 0.18185 | 0.14688 | 0.21612 | 0.15555 | 0.13867 | Replication on steam matching LIGER paper evaluation (Best Val NDCG@10=0.16971) |
| 2026-06-07 | Full TIGER (K-Means) (4 blocks, embed=256) on STEAM | Local (GeForce RTX 4080) | 0.13924 | 0.12362 | 0.16457 | 0.13176 | 0.19809 | 0.14021 | 0.12414 | Replication on steam matching LIGER paper evaluation (Best Val NDCG@10=0.15211) |
| 2026-06-07 | Full HSTU (blocks=4, embed=256) on STEAM | Local (GeForce RTX 4080) | 0.17285 | 0.15049 | 0.20735 | 0.16159 | 0.25647 | 0.17395 | 0.15671 | Replication on steam matching LIGER paper evaluation (Best Val NDCG@10=0.18701) |
| 2026-06-07 | Full TIGER (VAE) (blocks=4, embed=384) on STEAM | Local (GeForce RTX 4080) | 0.15810 | 0.13978 | 0.18636 | 0.14886 | 0.22213 | 0.15789 | 0.13996 | Replication on steam matching TIGER paper evaluation (Best Val NDCG@10=0.17238) |
| 2026-06-07 | Full TIGER (K-Means) (blocks=4, embed=384) on STEAM | Local (GeForce RTX 4080) | 0.14024 | 0.12427 | 0.16521 | 0.13229 | 0.19851 | 0.14069 | 0.12461 | Replication on steam matching TIGER paper evaluation (Best Val NDCG@10=0.15363) |
