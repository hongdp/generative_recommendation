# Generative Recommendation System

A JAX/Flax-based generative recommendation system utilizing LLM/Transformer architectures.

## Getting Started

1. Set up the Conda environment (see details in [SKILL.md](file:///home/hongdp/Workspace/generative_recommendation/SKILL.md)).
2. Install the package in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests to verify setup:
   ```bash
   pytest
   ```

## Project Structure

- `src/`: Core library code containing `models`, `datasets`, and `evaluation` components.
- `tests/`: Unit and integration tests.
- `SKILL.md`: Documented development principles, environment guidelines, and learnings.
- `experiment_results.md`: Logs for training runs and evaluation results.
