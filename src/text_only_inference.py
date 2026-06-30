"""
Text-only QA inference (no image) using OpenAI Responses API.

画像を入力せず、QA テキストのみでモデルに回答させることで、
視覚情報なしで解けてしまう問題がないかを検証するためのスクリプト。

Input : QA CSV
Output: prediction CSV per model (output file には "text_only" が付く)

Usage:
    python src/text_only_inference.py --model gpt-5-mini
    python src/text_only_inference.py --model gpt-5 --reasoning-effort high
    python src/text_only_inference.py --model gpt-5 --num-workers 16
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# ========= プロンプト & API =========
def build_prompt(row: pd.Series) -> str:
    options = "\n".join(f"  {i}: {row[f'option_{i}']}" for i in range(6))
    return (
        "Answer the following multiple-choice question.\n"
        "No image is provided. Choose the best answer based on the question and options alone.\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{options}\n\n"
        "Reply with ONLY the option number (0-5). No explanation, no punctuation."
    )


def parse_answer(text: str) -> int:
    m = re.search(r"[0-5]", text.strip())
    return int(m.group()) if m else -1


def run_one(
    client: OpenAI,
    model: str,
    row: pd.Series,
    reasoning_effort: str | None = None,
) -> tuple[int, int]:
    """1問推論（テキストのみ）。(predicted_index, is_correct) を返す。"""
    kwargs = {}
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_prompt(row)},
                ],
            }
        ],
        **kwargs,
    )
    predicted = parse_answer(response.output_text)
    is_correct = int(predicted == int(row["correct_option_index"]))
    return predicted, is_correct


def resolve_api_key(cli_api_key: str | None) -> str:
    api_key = cli_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Pass --api-key or set OPENAI_API_KEY."
        )
    return api_key.strip()


# ========= メイン処理 =========
def main(args: argparse.Namespace) -> None:
    project_root = Path(__file__).parent.parent

    client = OpenAI(api_key=resolve_api_key(args.api_key))

    df = pd.read_csv(project_root / args.qa_csv)

    # モデル識別子に "text_only" を付けて VQA 結果と区別する
    model_id = (
        f"{args.model}_{args.reasoning_effort}"
        if args.reasoning_effort
        else args.model
    )
    model_id_labeled = f"{model_id}_text_only"

    print(f"Model      : {args.model}")
    print(f"Effort     : {args.reasoning_effort or '(none)'}")
    print(f"Questions  : {len(df)}")
    print(f"Workers    : {args.num_workers}")
    print(f"Mode       : text-only (no image)")

    # 並列推論
    results: dict[int, tuple[int, int]] = {}
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                run_one, client, args.model, row, args.reasoning_effort,
            ): idx
            for idx, row in df.iterrows()
        }
        with tqdm(total=len(futures), desc=model_id_labeled, unit="q") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    tqdm.write(f"ERROR at index {idx}: {e}")
                    results[idx] = (-1, 0)
                pbar.update(1)

    # 元の行順に並べて保存
    sorted_idx = sorted(results.keys())
    df_out = df.loc[sorted_idx].copy()
    df_out["predicted_index"] = [results[i][0] for i in sorted_idx]
    df_out["is_correct"] = [results[i][1] for i in sorted_idx]
    df_out["model"] = model_id_labeled

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_slug = model_id_labeled.replace("/", "-")
    qa_stem = Path(args.qa_csv).stem
    out_path = output_dir / f"{qa_stem}_{model_slug}_pred.csv"
    df_out.to_csv(out_path, index=False)

    acc = df_out["is_correct"].mean()
    print(f"Accuracy   : {acc:.3f}  ({df_out['is_correct'].sum()}/{len(df_out)})")
    print(f"Saved      -> {out_path}")


# ========= CLI =========
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-only QA inference (no image) using OpenAI Responses API."
    )
    parser.add_argument(
        "--model", default="gpt-5-mini",
        help="OpenAI model (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="Reasoning effort for gpt-5 (minimal/low/medium/high). Omit for other models."
    )
    parser.add_argument(
        "--qa-csv", default="output/v3/qa_all.csv",
        help="QA CSV path relative to project root (default: output/v2/qa_all.csv)"
    )
    parser.add_argument(
        "--output-dir", default="output/vqa_results",
        help="Output directory for prediction CSVs (default: output/vqa_results)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="Parallel API workers (default: 8)"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY env var is used."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
