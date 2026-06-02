#!/bin/bash
set -e

# ============================================================
#  DarMo-STAS Training Script
#  - Backbone: pretrained ResNet-18
#  - LR schedule: cosine annealing
#
#  Launches two experiments in parallel:
#    GPU 0: 0x  (no generated data mixing)
#    GPU 1: 3x  (mix 300% generated data)
#
#  Usage:
#    chmod +x train.sh && ./train.sh
# ============================================================

# --- Paths (modify these to match your data location) ---
DATASET_EXCEL="<YOUR_DATASET_EXCEL_PATH>"   # e.g., ../data/dataset_split.xlsx
GEN_XLS="<YOUR_GEN_DATA_EXCEL_PATH>"        # e.g., ../data/generated_image.xls
DATA_ROOT="<YOUR_DATA_ROOT>"                 # e.g., ../dataset

# --- Hyperparameters ---
EPOCHS=100
BS=32
LR=0.0001
SEED=1265
GEN_SEED=1234

# --- Common arguments ---
COMMON="--dataset_excel $DATASET_EXCEL --data_root $DATA_ROOT \
    --epochs $EPOCHS --batch-size $BS --lr $LR --seed $SEED \
    --pretrained --backbone resnet18 --freeze_backbone 0 \
    --para_recon 1.0 --para_cls 0.5 --para_gcn 0.3 --para_cli 0.05 \
    --beta_max 0.05 --para_kld 1.0 --para_kld_dom 1.0 \
    --kl_warmup_epochs 20 --task_warmup_epochs 10 \
    --component_warmup_epochs 20 --latent_noise_warmup_epochs 20 \
    --decouple_weight 0.01 \
    --component_consistency_weight 0.2 --component_overlap_weight 0.1 \
    --residual_sparsity_weight 0.02 \
    --cls_label_smoothing 0.0 \
    --lr_schedule cosine"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ========== Experiment 1: 0x (no generated data mixing) ==========
echo "[GPU 0] Starting 0x (no gen data mixing) ..."
CUDA_VISIBLE_DEVICES=0 nohup python run.py \
    --saved exp_0x $COMMON \
    > "$LOG_DIR/train_0x.log" 2>&1 &
PID_0X=$!
echo "  PID $PID_0X -> 0x on GPU 0"

# ========== Experiment 2: 3x (300% generated data) ==========
echo "[GPU 1] Starting 3x (gen_data_ratio=3.0) ..."
CUDA_VISIBLE_DEVICES=1 nohup python run.py \
    --saved exp_3x $COMMON \
    --gen_data_ratio 3.0 --gen_data_xls "$GEN_XLS" --gen_data_seed $GEN_SEED \
    > "$LOG_DIR/train_3x.log" 2>&1 &
PID_3X=$!
echo "  PID $PID_3X -> 3x on GPU 1"

echo ""
echo "Both experiments launched. Logs in $LOG_DIR/"
echo "  tail -f $LOG_DIR/train_0x.log"
echo "  tail -f $LOG_DIR/train_3x.log"

wait $PID_0X
STATUS_0X=$?
wait $PID_3X
STATUS_3X=$?

echo ""
[ $STATUS_0X -eq 0 ] && echo "[0x] Done." || echo "[0x] FAILED (exit $STATUS_0X)"
[ $STATUS_3X -eq 0 ] && echo "[3x] Done." || echo "[3x] FAILED (exit $STATUS_3X)"
echo "All training jobs finished."
