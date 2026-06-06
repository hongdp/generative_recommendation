# Project Task List

This document tracks subsequent tasks and their progress.

## Execution Guidelines

- **Order of Execution**: Complete tasks in the order they are listed.
- **Resource Constraints**: When resources(GPU Memory, storage etc) are insufficient, tasks can be rejected (and clearly marked) or skipped temporarily.

## Backlog

- [ ] **Task 0**: Replicate full-scale HSTU results on MovieLens-1M (Paused at Epoch 14, Best Val NDCG@10 = 0.16485)
- [ ] **Task 1**: Replicate full-scale TIGER results on MovieLens-1M 
    - [] **Task 1.1**: Implement and train RQVAE for Semantic ID used in TIGER.
    - [] **Task 1.2**: Implement and Train TIGER model
    - [] **Task 1.3**: Evaluate TIGER model
- [] **Task 2**: Implement,train and eval Transformer model on MovieLens-1M that matches the number of parameters of the HSTU model we are using. 
- [ ] **Task 3**: Implement and evaluate RQ-KMeans semantic ID generator and run comparative TIGER experiments.
    - [x] **Task 3.1**: Implement RQ-KMeans semantic ID generator.
    - [ ] **Task 3.2**: Train and evaluate TIGER model with K-Means Semantic IDs.

## Skipped / Rejected / Insufficient Resources

*(No tasks currently skipped or rejected)*
