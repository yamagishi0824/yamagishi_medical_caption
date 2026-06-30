"""
VQA inference on surgical video frames using OpenAI Responses API.
Input : QA CSV + cholecseg.zip (original frames, not masks)
Output: prediction CSV per model

Usage:
    python src/vqa_inference.py --model gpt-5-mini
    python src/vqa_inference.py --model gpt-5 --reasoning-effort high
    python src/vqa_inference.py --model gpt-5 --reasoning-effort minimal --num-workers 16
"""

import argparse
import base64
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# ========= パス変換 =========
def mask_path_to_frame_key(mask_path: str) -> str:
    """CSV の mask_path から zip 内の元画像パス（マスクなし）を導出。

    例:
      ../data/cholec80/seg/video26/video26_01935/frame_1975_endo_color_mask.png
      -> video26/video26_01935/frame_1975_endo.png
    """
    key = mask_path.split("seg/", 1)[1]
    return key.replace("_color_mask", "")


def load_base64_from_zip(zf: zipfile.ZipFile, key: str) -> str:
    with zf.open(key) as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ========= プロンプト & API =========
def build_prompt(row: pd.Series) -> str:
    options = "\n".join(f"  {i}: {row[f'option_{i}']}" for i in range(6))
    return (
        "Look at the image carefully and answer the following multiple-choice question.\n\n"
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
    b64: str,
    reasoning_effort: str | None = None,
) -> tuple[int, int]:
    """1問推論。(predicted_index, is_correct) を返す。"""
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
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}",
                    },
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
    zip_path = project_root / args.zip

    # reasoning effort 付きの場合は識別子に含める (例: gpt-5_high)
    model_id = (
        f"{args.model}_{args.reasoning_effort}"
        if args.reasoning_effort
        else args.model
    )

    print(f"Model      : {args.model}")
    print(f"Effort     : {args.reasoning_effort or '(none)'}")
    print(f"Questions  : {len(df)}")
    print(f"Workers    : {args.num_workers}")

    # 同一フレームの重複読み込みを避けるため zip から事前キャッシュ
    print("Loading frames from zip ...")
    with zipfile.ZipFile(zip_path) as zf:
        b64_cache: dict[str, str] = {}
        for mask_path in df["mask_path"].unique():
            key = mask_path_to_frame_key(mask_path)
            b64_cache[mask_path] = load_base64_from_zip(zf, key)

    # 並列推論
    results: dict[int, tuple[int, int]] = {}
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                run_one, client, args.model, row, b64_cache[row["mask_path"]],
                args.reasoning_effort,
            ): idx
            for idx, row in df.iterrows()
        }
        with tqdm(total=len(futures), desc=model_id, unit="q") as pbar:
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
    df_out["model"] = model_id

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_slug = model_id.replace("/", "-")
    qa_stem = Path(args.qa_csv).stem
    out_path = output_dir / f"{qa_stem}_{model_slug}_pred.csv"
    df_out.to_csv(out_path, index=False)

    acc = df_out["is_correct"].mean()
    print(f"Accuracy   : {acc:.3f}  ({df_out['is_correct'].sum()}/{len(df_out)})")
    print(f"Saved      -> {out_path}")


# ========= CLI =========
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQA inference on surgical frames using OpenAI Responses API."
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
        "--qa-csv", default="output/v1/qa_all.csv",
        help="QA CSV path relative to project root (default: output/v1/qa_all.csv)"
    )
    parser.add_argument(
        "--zip", default="data/cholec80/cholecseg.zip",
        help="Path to cholecseg.zip relative to project root"
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
