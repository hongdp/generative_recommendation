#!/bin/bash
# Run all three TIGER ablation models to full convergence (30 epochs, patience=5)
set -e

export PYTHONPATH=src

echo "=============================================="
echo "=== Model 1/3: TIGER Seq2Seq (RQVAE IDs)  ==="
echo "=============================================="
python examples/train_tiger_seq2seq.py \
    --dataset steam \
    --semantic_ids_path ./data/semantic_ids_steam.json \
    --epochs 30 \
    --patience 5 \
    --batch_size 256 \
    --checkpoint_dir ./data/tiger_seq2seq_rqvae_steam_checkpoints \
    --tb_log_dir ./data/tensorboard/tiger_seq2seq_rqvae_steam_full

echo ""
echo "=============================================="
echo "=== Model 2/3: TIGER Seq2Seq (Random IDs)  ==="
echo "=============================================="
python examples/train_tiger_seq2seq.py \
    --dataset steam \
    --semantic_ids_path ./data/semantic_ids_random_steam.json \
    --epochs 30 \
    --patience 5 \
    --batch_size 256 \
    --checkpoint_dir ./data/tiger_seq2seq_random_steam_checkpoints \
    --tb_log_dir ./data/tensorboard/tiger_seq2seq_random_steam_full

echo ""
echo "=============================================="
echo "=== Model 3/3: TIGER Encoder + CE          ==="
echo "=============================================="
python examples/train_tiger_encoder_ce.py \
    --dataset steam \
    --semantic_ids_path ./data/semantic_ids_steam.json \
    --epochs 30 \
    --patience 5 \
    --batch_size 256 \
    --tb_log_dir ./data/tensorboard/tiger_encoder_ce_steam_full

echo ""
echo "=============================================="
echo "=== ALL 3 MODELS COMPLETED ==="
echo "=============================================="
