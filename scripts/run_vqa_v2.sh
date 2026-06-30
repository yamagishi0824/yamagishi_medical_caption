#!/bin/bash
# Run VQA inference for all models, then compute accuracy summary.
#
# Usage (from project root):
#   bash scripts/run_vqa.sh
#   bash scripts/run_vqa.sh --num-workers 16
#
# Optional env overrides:
#   QA_CSV=output/v1/qa_all.csv bash scripts/run_vqa.sh

set -euo pipefail

# ========= 設定 =========
QA_CSV="${QA_CSV:-output/v2/qa_all.csv}"
ZIP="${ZIP:-data/cholec80/cholecseg.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-output/vqa_results_v2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_WORKERS=16

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "Usage: OPENAI_API_KEY=sk-... bash scripts/run_vqa_v2.sh"
    exit 1
fi

GPT5_EFFORTS=(low)

echo "========================================="
echo "  VQA Inference"
echo "  QA CSV    : $QA_CSV"
echo "  ZIP       : $ZIP"
echo "  Output    : $OUTPUT_DIR"
echo "  Workers   : $NUM_WORKERS"
echo "========================================="

# ========= gpt-5: effort 別に4回 =========
for EFFORT in "${GPT5_EFFORTS[@]}"; do
    echo ""
    echo "--- gpt-5 / effort: $EFFORT ---"
    uv run python src/vqa_inference.py \
        --model              "gpt-5"    \
        --reasoning-effort   "$EFFORT"  \
        --api-key     "$OPENAI_API_KEY" \
        --qa-csv      "$QA_CSV"         \
        --zip         "$ZIP"            \
        --output-dir  "$OUTPUT_DIR"     \
        --num-workers "$NUM_WORKERS"
done

for EFFORT in "${GPT5_EFFORTS[@]}"; do
    echo ""
    echo "--- gpt-5 / effort: $EFFORT ---"
    uv run python src/vqa_inference.py \
        --model              "gpt-5.2"    \
        --reasoning-effort   "$EFFORT"  \
        --api-key     "$OPENAI_API_KEY" \
        --qa-csv      "$QA_CSV"         \
        --zip         "$ZIP"            \
        --output-dir  "$OUTPUT_DIR"     \
        --num-workers "$NUM_WORKERS"
done

# ========= gpt-5-mini / gpt-5-nano: reasoning なし =========
for MODEL in gpt-5-mini gpt-5-nano gpt-4.1 gpt-4o; do
    echo ""
    echo "--- Model: $MODEL ---"
    uv run python src/vqa_inference.py \
        --model       "$MODEL"       \
        --api-key     "$OPENAI_API_KEY" \
        --qa-csv      "$QA_CSV"      \
        --zip         "$ZIP"         \
        --output-dir  "$OUTPUT_DIR"  \
        --num-workers "$NUM_WORKERS"
done

# ========= 精度計算 =========
echo ""
echo "--- Evaluating ---"
uv run python src/eval_vqa.py \
    --pred-dir "$OUTPUT_DIR" \
    --output   "$OUTPUT_DIR/accuracy_summary.csv"

echo ""
echo "Done. Results in $OUTPUT_DIR/"
