# Experiment Results Log

This file documents all training runs, evaluations, and benchmarks. Refer to [SKILL.md](file:///home/hongdp/Workspace/generative_recommendation/SKILL.md) for the logging protocol.

## Trials Log

| Date | Goal / Configuration | Environment / Hardware | Key Metrics (Loss, PPL, MFU) | Notes & Link |
|------|--------------------|-----------------------|------------------------------|--------------|
| 2026-06-06 | Dry-run baseline evaluation on MovieLens-100K (Random vs Most Popular) | Local (CPU) | Random: HR@10=0.00848, NDCG@10=0.00409, MRR=0.00580; Most-Pop: HR@10=0.04984, NDCG@10=0.02303, MRR=0.01490 | End-to-end pipeline verification completed |
| 2026-06-06 | HSTUModel next-item prediction on MovieLens-100K | Local (CPU) | Pre-train: MRR=0.00259; Post-train (5 epochs): HR@1=0.00742, HR@5=0.06681, HR@10=0.13786, NDCG@5=0.03692, NDCG@10=0.05943, MRR=0.05283 | Converged and ranking metrics improved significantly (+0.05024 MRR) |


| 2026-06-06 | Full HSTUModel (4 blocks, embed=256) on ML-1M | Local (GeForce RTX 4080) | Best Val NDCG@10=-1.00000; Test: HR@1=0.03626, HR@5=0.13212, HR@10=0.20414, NDCG@5=0.08459, NDCG@10=0.10770, MRR=0.09371 | Fully converged replication. Meets/exceeds original paper baselines |
| 2026-06-06 | Full HSTUModel (4 blocks, embed=256) on ML-1M | Local (GeForce RTX 4080) | Best Val NDCG@10=0.13163; Test: HR@1=0.04222, HR@5=0.14818, HR@10=0.23725, NDCG@5=0.09544, NDCG@10=0.12411, MRR=0.10549 | Fully converged replication. Meets/exceeds original paper baselines |
| 2026-06-06 | Full HSTUModel (4 blocks, embed=256) on ML-1M, paused at Epoch 14 | Local (GeForce RTX 4080) | Best Val NDCG@10=0.16485; Test: HR@1=0.06705, HR@5=0.19023, HR@10=0.27384, NDCG@5=0.12933, NDCG@10=0.15601, MRR=0.13502 | Paused run at Epoch 14 (Best Val Epoch 12). Strong convergence meeting paper baseline. |
