#!/bin/bash
# Run VQA inference for all models, then compute accuracy summary.
#
# Usage (from project root):
#   bash scripts/run_vqa_v3.sh
#
# Optional env overrides:
#   QA_CSV=output/v3/qa_all.csv bash scripts/run_vqa_v3.sh

set -euo pipefail

# =========================================================
# モデル設定
# =========================================================

# reasoning effort 付きで実行するモデル ("モデル名:effort" 形式)
REASONING_MODELS=(
    "gpt-5:low"
    "gpt-5.2:low"
)

# reasoning なしで実行するモデル
PLAIN_MODELS=(
    "gpt-5-mini"
    "gpt-5-nano"
    "gpt-4.1"
    "gpt-4o"
)

# =========================================================
# 実行設定
# =========================================================
QA_CSV="${QA_CSV:-output/v3/qa_all.csv}"
ZIP="${ZIP:-data/cholec80/cholecseg.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-output/vqa_results_v3}"
NUM_WORKERS=16

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "Usage: OPENAI_API_KEY=sk-... bash scripts/run_vqa_v3.sh"
    exit 1
fi

echo "========================================="
echo "  VQA Inference"
echo "  QA CSV    : $QA_CSV"
echo "  ZIP       : $ZIP"
echo "  Output    : $OUTPUT_DIR"
echo "  Workers   : $NUM_WORKERS"
echo "========================================="

# =========================================================
# 推論
# =========================================================

# reasoning effort 付きモデル (gpt-5 系)
for ENTRY in "${REASONING_MODELS[@]}"; do
    MODEL="${ENTRY%%:*}"
    EFFORT="${ENTRY##*:}"
    echo ""
    echo "--- $MODEL / effort: $EFFORT ---"
    uv run python src/vqa_inference.py \
        --model              "$MODEL"   \
        --reasoning-effort   "$EFFORT"  \
        --api-key     "$OPENAI_API_KEY" \
        --qa-csv      "$QA_CSV"         \
        --zip         "$ZIP"            \
        --output-dir  "$OUTPUT_DIR"     \
        --num-workers "$NUM_WORKERS"
done

# plain モデル
for MODEL in "${PLAIN_MODELS[@]}"; do
    echo ""
    echo "--- $MODEL ---"
    uv run python src/vqa_inference.py \
        --model       "$MODEL"       \
        --api-key     "$OPENAI_API_KEY" \
        --qa-csv      "$QA_CSV"      \
        --zip         "$ZIP"         \
        --output-dir  "$OUTPUT_DIR"  \
        --num-workers "$NUM_WORKERS"
done

# =========================================================
# 精度計算
# =========================================================
echo ""
echo "--- Evaluating ---"
uv run python src/eval_vqa.py \
    --pred-dir "$OUTPUT_DIR" \
    --output   "$OUTPUT_DIR/accuracy_summary.csv"

echo ""
echo "Done. Results in $OUTPUT_DIR/"
