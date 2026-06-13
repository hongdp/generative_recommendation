#!/bin/bash
set -e

echo "Starting TPU Experiments..."

# 1. TIGER Random
echo "=============================="
echo "Experiment 1: TIGER Random IDs"
echo "=============================="
PYTHONPATH=src python examples/train_tiger_seq2seq.py \
    --dataset steam \
    --semantic_ids_path ./data/semantic_ids_random_steam.json \
    --checkpoint_dir ./data/tpu_tiger_random_checkpoints \
    --tb_log_dir ./data/tensorboard/tpu_tiger_random \
    --batch_size 2048 \
    --epochs 30

# 2. TIGER RQ-VAE
echo "=============================="
echo "Experiment 2: TIGER RQ-VAE IDs"
echo "=============================="
PYTHONPATH=src python examples/train_tiger_seq2seq.py \
    --dataset steam \
    --semantic_ids_path ./data/semantic_ids_rqvae_steam.json \
    --checkpoint_dir ./data/tpu_tiger_rqvae_checkpoints \
    --tb_log_dir ./data/tensorboard/tpu_tiger_rqvae \
    --batch_size 2048 \
    --epochs 30

# 3. Encoder+CE (dims 384)
echo "=============================="
echo "Experiment 3: TIGER Encoder+CE (384 dims)"
echo "=============================="
PYTHONPATH=src python examples/train_tiger_encoder_ce.py \
    --dataset steam \
    --semantic_ids_path ./data/semantic_ids_rqvae_steam.json \
    --embedding_dim 384 \
    --checkpoint_dir ./data/tpu_tiger_ce_384_checkpoints \
    --tb_log_dir ./data/tensorboard/tpu_tiger_ce_384 \
    --batch_size 2048 \
    --epochs 30

# 4. Encoder+CE (dims 1536)
echo "=============================="
echo "Experiment 4: TIGER Encoder+CE (1536 dims)"
echo "=============================="
PYTHONPATH=src python examples/train_tiger_encoder_ce.py \
    --dataset steam \
    --semantic_ids_path ./data/semantic_ids_rqvae_steam.json \
    --embedding_dim 1536 \
    --checkpoint_dir ./data/tpu_tiger_ce_1536_checkpoints \
    --tb_log_dir ./data/tensorboard/tpu_tiger_ce_1536 \
    --batch_size 256 \
    --epochs 30

echo "All TPU Experiments Completed Successfully!"
