#!/bin/bash
set -e

# ============================================================
#  DarMo-STAS Evaluation Script
#  Compute metrics (AUC, ACC, sensitivity, specificity)
#  on train / test_inter / test_exter / test_exter2 splits.
#
#  Usage:
#    chmod +x test.sh && ./test.sh
# ============================================================

# --- Paths (modify these to match your data location) ---
DATASET_EXCEL="<YOUR_DATASET_EXCEL_PATH>"   # e.g., ../data/dataset_split.xlsx
DATA_ROOT="<YOUR_DATA_ROOT>"                 # e.g., ../dataset

BS=32
SEED=1265

COMMON="--batch-size $BS --seed $SEED \
    --dataset_excel $DATASET_EXCEL --data_root $DATA_ROOT \
    --pretrained --backbone resnet18 --freeze_backbone 0"

# ========== Evaluate 0x model (best_inter_auc) ==========
echo "=== Evaluating 0x (best_inter_auc) ==="
python run_eval.py \
    --saved eval_0x \
    --resume exp_0x_res/checkpoints/best_inter_auc.pth.tar \
    $COMMON

# ========== Evaluate 3x model (best_inter_auc) ==========
echo ""
echo "=== Evaluating 3x (best_inter_auc) ==="
python run_eval.py \
    --saved eval_3x \
    --resume exp_3x_res/checkpoints/best_inter_auc.pth.tar \
    $COMMON

echo ""
echo "All evaluations finished. Metrics saved in eval_*_res/eval/*/metrics.json"
