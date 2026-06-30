"""
Compute accuracy from VQA prediction CSVs and write a summary.

Reads all *_pred.csv files in a directory, computes overall and per-phase
accuracy for each model, and outputs a single long-format summary CSV.

Usage:
    python src/eval_vqa.py
    python src/eval_vqa.py --pred-dir output/vqa_results --output output/vqa_results/accuracy_summary.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def accuracy_rows(df: pd.DataFrame, model: str) -> list[dict]:
    """Overall + per-phase accuracy rows for one model."""
    rows = []

    def _row(split: str, grp: pd.DataFrame) -> dict:
        n = len(grp)
        n_correct = int(grp["is_correct"].sum())
        return {
            "model": model,
            "split": split,
            "n_questions": n,
            "n_correct": n_correct,
            "accuracy": round(n_correct / n, 4) if n > 0 else 0.0,
        }

    rows.append(_row("Overall", df))

    if "phase" in df.columns:
        for phase, grp in df.groupby("phase"):
            rows.append(_row(phase, grp))

    return rows


def main(args: argparse.Namespace) -> None:
    project_root = Path(__file__).parent.parent
    pred_dir = project_root / args.pred_dir
    pred_files = sorted(pred_dir.glob("*_pred.csv"))

    if not pred_files:
        print(f"No *_pred.csv found in {pred_dir}")
        return

    print(f"Found {len(pred_files)} prediction file(s):")
    for p in pred_files:
        print(f"  {p.name}")

    all_rows = []
    for path in pred_files:
        df = pd.read_csv(path)
        model = df["model"].iloc[0] if "model" in df.columns else path.stem
        all_rows.extend(accuracy_rows(df, model))

    summary = (
        pd.DataFrame(all_rows)
        .sort_values(["model", "split"])
        .reset_index(drop=True)
    )

    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    # 表示テキストを組み立て
    lines = []

    overall = summary[summary["split"] == "Overall"].copy()
    lines.append("=== Overall Accuracy ===")
    lines.append(
        overall[["model", "n_questions", "n_correct", "accuracy"]]
        .to_string(index=False)
    )

    by_phase = summary[summary["split"] != "Overall"].copy()
    if not by_phase.empty:
        pivot = by_phase.pivot_table(
            index="split", columns="model", values="accuracy"
        ).round(4)
        lines.append("\n=== Accuracy by Phase ===")
        lines.append(pivot.to_string())

    report = "\n".join(lines)

    # コンソール出力
    print()
    print(report)

    # テキストファイルに保存
    txt_path = out_path.with_suffix(".txt")
    txt_path.write_text(report + "\n")

    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {txt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute accuracy from VQA prediction CSVs."
    )
    parser.add_argument(
        "--pred-dir", default="output/vqa_results",
        help="Directory containing *_pred.csv files (default: output/vqa_results)"
    )
    parser.add_argument(
        "--output", default="output/vqa_results/accuracy_summary.csv",
        help="Output path for accuracy summary CSV"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
