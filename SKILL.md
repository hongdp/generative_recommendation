---
name: basic-development-principles
description: Core principles for developing the generative recommendation codebase, emphasizing Test-Driven Development (TDD), rigorous documentation, and continuous learning.
---

# Basic Development Principles

This skill defines the core principles to follow when developing this repository. You must adhere to these practices to ensure high code quality, maintainability, and a clear history of decisions.

## 1. Test-Driven Development (TDD)
- **Write Tests First**: Before implementing new features or fixing bugs, write the test cases that define the expected behavior.
- **Run Tests Frequently**: Ensure that all tests pass locally before proposing changes.
- **Coverage**: Aim for high test coverage, especially for core model architecture components and data processing pipelines.
- **Testing Framework**: Use `pytest` for all unit and integration tests.

## 2. Documenting Changes
- **Docstrings**: Every new module, class, and function must have a clear docstring explaining its purpose, arguments, and return values.
- **Code Comments**: Use comments to explain *why* a particular approach was taken, especially for complex mathematical operations or JAX transformations.
- **Commit Messages**: Write clear, descriptive commit messages. Explain the problem being solved and how the commit addresses it.
- **Experiment Results Log**: Document every cloud training, local test, or evaluation run in `experiment_results.md` at the project root to preserve reproducibility and track performance metrics.

## 3. Continuous Learning and Skill Updates
- **Update Skills**: When you discover a new pattern, a recurring issue, or a best practice specific to JAX or this project's architecture, update this `SKILL.md` or create a new, specialized skill document.
- **Document Gotchas**: If you encounter a tricky bug (e.g., related to JAX's `jit` compilation, `vmap` dimension handling, or memory leaks), document the solution in the relevant skill or project documentation so others (and yourself) can learn from it.

## 4. JAX Specifics
- **Pure Functions**: Ensure functions intended for `jax.jit` or `jax.vmap` are pure (no side effects).
- **Static vs. Traced Arguments**: Be mindful of which arguments are static and which are traced during JAX transformations.
- **Random Number Generation**: Explicitly pass and split JAX PRNG keys; avoid implicit stateful randomness.

## 5. Environment Management
- **Conda Environments**: Always use `conda` to manage the project environment and its dependencies to ensure strict control over Python versions and native dependencies like CUDA and JAX.
- **Environment Exports**: Keep the `environment.yml` updated if using Conda-specific setups or `pyproject.toml` aligned with pip inside Conda.
- **Isolation**: Never install project dependencies in your global base environment. Always work within the activated project-specific Conda environment.

## 6. Memory-Constrained Training Strategies
When training large models (e.g., 1B parameters) on consumer hardware with limited VRAM (e.g., 16 GB RTX 4080), strict memory management is required:
- **Mixed Precision**: Always use `bfloat16` or `float16` for model weights and activations to halve memory requirements compared to `float32`.
- **Gradient Checkpointing**: Trade compute for memory by recomputing forward pass activations during the backward pass instead of storing them all in VRAM.
- **Gradient Accumulation**: Keep the physical batch size very small to avoid Out Of Memory (OOM) errors, and accumulate gradients over multiple steps to achieve the desired global batch size.
- **Hardware Checks**: If you encounter a `Driver/library version mismatch` error when checking GPU status via `nvidia-smi` or when JAX fails to initialize, reboot the machine to ensure the Nvidia kernel module matches the installed driver version.

## Workflow Example
1. Identify the task (e.g., "Implement a new attention mechanism").
2. Write tests for the attention mechanism in `tests/test_attention.py`.
3. Implement the attention mechanism in the source code.
4. Run tests; iterate until they pass.
5. Review and refine docstrings and comments.
6. If a new useful JAX pattern was learned, update `SKILL.md`.

## 7. Architectural Learnings & Experience Log

### [2026-06-04] Orbax Checkpointing on GCS FUSE
- **Orbax "Too many open files" Error**: When saving checkpoints to a Cloud Storage Bucket via GCS FUSE (`/gcs/...`), the default `ocp.CheckpointManager` spawns asynchronous threads that open thousands of tensor files concurrently. This hits the Linux `ulimit` for open file descriptors (`OSError: [Errno 24]`). **Always** disable async checkpointing using `ocp.CheckpointManagerOptions(enable_async_checkpointing=False)` when saving directly to GCS FUSE mounts. Additionally, for large models (e.g. 1B+ parameters, which split into 1000+ array files), you must dynamically increase the file descriptor limits (`RLIMIT_NOFILE`) inside python at the very start of the script:
  ```python
  import resource
  soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
  ```
  This prevents failures when writing numerous metadata/array files concurrently over GCS FUSE.

### [2026-06-03] Dual-Accelerator Sharding & Robust Data Pipelines
- **JAX JIT Decorator Syntax**: Never use `@jax.jit(donate_argnames=[...])` directly as it throws a `TypeError: missing argument 'fun'` on many JAX versions. **Always** use `@functools.partial(jax.jit, donate_argnames=[...])` to ensure robust syntax compatibility.
- **Universal TPU/GPU Sharding (FSDP)**: Use `jax.sharding.Mesh` and `NamedSharding` for multi-device/multi-host parallelism. This approach is completely architecture-agnostic. If deployed on a single A100 GPU, the mesh transparently degrades to a 1-device mesh without requiring any code changes.
- **TPU Distributed Initialization**: Multi-host TPU pods require calling `jax.distributed.initialize()`. Safely guard this by checking `if "TPU_NAME" in os.environ or jax.process_count() > 1` to prevent crashes on local single-GPU runs.
- **Robust Dataset Downloading**: For massive scale text datasets (e.g., HuggingFace 100BT splits), bypass API rate limits and connection drops by using pure `wget` inside multiprocess pools, and stream `.parquet` files iteratively using `pyarrow.parquet` to keep memory footprint under 50MB per worker.

### [2026-06-06] GCS FUSE Multiprocessing Deadlocks
- **Multiprocessing FUSE Deadlocks**: When using multiple asynchronous workers for data loading (such as `grain.DataLoader` with `worker_count > 0` or PyTorch `DataLoader` with `num_workers > 0`), do not open memory-mapped files (like `np.memmap`) on GCS FUSE in the parent process `__init__`. Spawning/forking child processes that inherit open FUSE file descriptors will trigger deadlocks inside the FUSE daemon during data access. **Always** use **Lazy Loading** to initialize and open the memmap file on demand inside the child process execution context (e.g., inside `__getitem__` on the first call).

### [2026-06-06] Generative Recommender (TIGER) & Model Evaluation Protocols
- **Discriminative vs. Generative Training Loops**: Discriminative/Index-based recommendation models (e.g., HSTU) compute cross-entropy loss over the total catalog items (`num_items + 1`) only on the final sequence position. Generative recommendation models (e.g., TIGER) tokenize sequences into flattened, shifted Semantic IDs (length `3 * L + 1` for $C=3$) and train using teacher-forcing cross-entropy loss over a vocabulary of size `3 * K + 2` at all positions. Keep their runner scripts separate to avoid complex branch logic.
- **TensorBoard Logging Horizontal Axis**: For TensorBoard logging during training, default PyTorch `SummaryWriter` plots horizontal curves against "Step". To enable meaningful step-level tracking, log metrics against `global_step = (epoch - 1) * num_batches + batch_idx` for detailed batch-level gradient steps, rather than just epoch indices.
- **Evaluation Frequency**: To capture tight convergence windows and prevent over-fitting, run validation evaluation on every epoch (`if True:` check) rather than skipping epochs.
- **Semantic Quantization Baselines**: For generating item Semantic IDs, sequential RQ-KMeans on item text embeddings (run sequentially over residual levels) serves as a fast, highly active codebook alternative (perfect codeword utilization) compared to standard neural VAE models. Ensure that padding tokens at index 0 are excluded from clustering and map directly to `[0, 0, 0]`.

## 8. Maintaining the Experiment Log
To ensure systematic progress, adhere to the following protocol when executing training or evaluation trials:
- **Immediate Logging**: Add an entry to `experiment_results.md` immediately after submitting a custom job, recording the date, environment, configuration, and the direct Vertex AI/Cloud console link.
- **Goal Definition**: Clearly state the objective of the trial (e.g., "Observe MFU on A100", "WikiText PPL benchmark validation").
- **Observation Metrics**: Upon completion, retrieve and fill in the key metrics:
  - **Train/Eval Loss**: Cross-entropy losses indicating convergence.
  - **Perplexity (PPL)**: The standard NLP test set PPL (e.g., on WikiText-103).
  - **Throughput & MFU**: Compute tokens/sec/device and MFU percentage to monitor hardware efficiency.
- **Problem & Fix Recording**: Document any runtime errors (e.g., GCS FUSE descriptor leaks, JAX compatibility crashes) alongside their specific remediation steps.
