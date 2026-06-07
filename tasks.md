# Project Task List

This document tracks subsequent tasks and their progress.

## Execution Guidelines

- **Order of Execution**: Complete tasks in the order they are listed.
- **Resource Constraints**: When resources(GPU Memory, storage etc) are insufficient, tasks can be rejected (and clearly marked) or skipped temporarily.

## Backlog

- [ ] **Task 0**: Replicate full-scale HSTU results on MovieLens-1M (Paused at Epoch 14, Best Val NDCG@10 = 0.16485)
- [x] **Task 1**: Replicate full-scale TIGER results on MovieLens-1M 
    - [x] **Task 1.1**: Implement and train RQVAE for Semantic ID used in TIGER.
    - [x] **Task 1.2**: Implement and Train TIGER model
    - [x] **Task 1.3**: Evaluate TIGER model
- [] **Task 2**: Implement,train and eval Transformer model on MovieLens-1M that matches the number of parameters of the HSTU model we are using. 
- [x] **Task 3**: Implement and evaluate RQ-KMeans semantic ID generator and run comparative TIGER experiments.
    - [x] **Task 3.1**: Implement RQ-KMeans semantic ID generator.
    - [x] **Task 3.2**: Train and evaluate TIGER model with K-Means Semantic IDs.
- [x] **Task 4**: Rename `examples/train_full_movielens.py` to `examples/train_hstu.py` and update all imports and references.
- [/] **Task 5**: Train and evaluate HSTU, TIGER (VAE), and TIGER (K-Means) to full convergence on the Steam dataset.
    - [x] **Task 5.1**: Add `--patience` command-line argument to training runners.
    - [x] **Task 5.2**: Implement path validity metrics (Valid@1 and Valid@Beam) for TIGER TensorBoard logging.
    - [/] **Task 5.3**: Execute the Steam convergence sweep.

## Skipped / Rejected / Insufficient Resources

*(No tasks currently skipped or rejected)*

