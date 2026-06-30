"""
Generate captions and multiple-choice QA pairs for Cholec80 frames using an OpenAI model.

Usage:
    python src/make_cap_qa.py
    python src/make_cap_qa.py --n-samples 30
    python src/make_cap_qa.py --caption-output cholec_cap_v2.csv --qa-output cholec_qa_v2.csv
    python src/make_cap_qa.py --caption-model gpt-5-mini --qa-model gpt-5-mini --n-samples 30
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ========= 列定義 =========
# マスクピクセルカウントに基づく解剖構造（ツール系を除く）
ANATOMY_COLS = [
    "hepatic_vein",
    "liver_ligament",
    "fat",
    "abdominal_wall",
    "gi_tract",
    "blood",
    "connective_tissue",
    "liver",
    "gallbladder",
    "cystic_duct_yellow",
    "cystic_duct_white",
]

# マスクピクセルカウントに基づくツール（バイナリannotationは使わない）
MASK_TOOL_COLS = [
    "grasper",
    "l_hook_electrocautery",
]


# ========= プロンプト生成 =========
def build_caption_prompt(row: pd.Series) -> str:
    video = int(row["video"])
    frame = int(row["frame"])
    phase = row.get("phase", "Unknown")

    present_anatomy = [col for col in ANATOMY_COLS if col in row and row[col] > 0]
    present_tools   = [col for col in MASK_TOOL_COLS if col in row and row[col] > 0]

    def list2text(lst):
        return ", ".join(lst) if lst else "none"

    prompt = f"""You are an expert in describing laparoscopic cholecystectomy scenes.

Below is metadata for a single video frame from the Cholec80 dataset.
Use only this metadata to infer what is likely visible in the frame and generate a structured English description.

[Metadata]
- Dataset: Cholec80
- Video ID: video{video:02d}
- Frame index: {frame}
- Surgical phase label: {phase}
- Anatomical structures visible in this frame (from segmentation mask pixel counts):
  {list2text(present_anatomy)}
- Surgical instruments visible in this frame (from segmentation mask pixel counts):
  {list2text(present_tools)}

[Important notes]
- You do NOT see the image itself; you only see the metadata above.
- The segmentation class names are technical feature names from the dataset.
- You should convert them into natural anatomical or surgical terms when appropriate.
  For example, "gallbladder", "liver", "hepatic vein", "cystic duct", "abdominal wall",
  "gastrointestinal tract", "blood", etc.
- For instruments, convert technical mask names to natural surgical terms:
  "grasper" → "grasper", "l_hook_electrocautery" → "L-hook electrocautery (hook cautery)"
- Both anatomical structures and instruments are derived from segmentation mask pixel counts,
  so they reliably reflect what is physically visible in the frame.
- Use the surgical phase label to infer what is likely happening (e.g., preparation, dissection, clipping, etc.),
  but do not invent impossible details.

[Output format]
Return ONLY a valid JSON object, with no extra text, no explanation and no markdown.
Use exactly the following keys:

{{
  "caption": string,                 // 1-2 natural English sentences summarizing what is visible in this frame
  "phase_description": string,       // a short phrase describing the surgical phase context in plain English
  "anatomy_present": [string],       // list of anatomical structures or regions likely visible
  "tools_present": [string],         // list of surgical tools/instruments likely visible
  "focus_of_frame": string           // a concise description of what the surgeon is mainly doing or focusing on
}}

Requirements:
- The JSON must be syntactically valid.
- The values should be consistent with the metadata.
- If no tools are present, use an empty list [] for "tools_present".
- If you are unsure about some details, make a reasonable, conservative guess based on the metadata.

Now output ONLY the JSON object for this frame."""
    return prompt


def build_qa_prompt(caption_json: dict) -> str:
    caption = caption_json.get("caption", "")
    phase_desc = caption_json.get("phase_description", "")
    anatomy_present = caption_json.get("anatomy_present", [])
    tools_present = caption_json.get("tools_present", [])
    focus = caption_json.get("focus_of_frame", "")

    anatomy_text = ", ".join(anatomy_present) if anatomy_present else "none"
    tools_text = ", ".join(tools_present) if tools_present else "none"

    prompt = f"""You are creating multiple-choice question-answer (MCQA) items for a surgical vision-language dataset.

You are given a structured English description of a laparoscopic cholecystectomy frame:

[Structured description]
- Caption: {caption}
- Phase description: {phase_desc}
- Anatomy present: {anatomy_text}
- Tools present: {tools_text}
- Focus of frame: {focus}

Using ONLY this information, create 3 multiple-choice QA pairs (MCQ), each with:
- One question that can be answered from the description above
- Exactly 6 answer options (A-F), with only ONE correct answer
- Reasonable but clearly incorrect distractors that do not contradict the description

The questions should:
- Be about the surgical scene (phase, anatomy, tools, focus of action, etc.)
- Be answerable using the description alone
- Avoid trivial "copy one word" style; aim for clinically meaningful comprehension

[Output format]
Return ONLY a valid JSON object with the following structure, no extra text and no markdown:

{{
  "qas": [
    {{
      "id": 1,
      "question": "string",
      "options": ["option A", "option B", "option C", "option D", "option E", "option F"],
      "correct_option_index": int,   // 0-based index (0..5)
      "rationale": "short explanation of why the correct option is correct"
    }},
    {{
      "id": 2,
      "question": "...",
      "options": [... 6 strings ...],
      "correct_option_index": int,
      "rationale": "..."
    }},
    {{
      "id": 3,
      "question": "...",
      "options": [... 6 strings ...],
      "correct_option_index": int,
      "rationale": "..."
    }}
  ]
}}

Requirements:
- Exactly 3 questions.
- Each question must have exactly 6 options.
- Use 0-based indices for "correct_option_index" (0 to 5).
- All content must be consistent with the description above.

Now output ONLY this JSON object."""
    return prompt


# ========= API 呼び出し =========
def call_caption_llm(client: OpenAI, model: str, row: pd.Series) -> dict:
    prompt = build_caption_prompt(row)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful medical AI assistant."},
            {"role": "user", "content": prompt},
        ],
        #temperature=0.5,
    )
    return json.loads(resp.choices[0].message.content.strip())


def call_qa_llm(client: OpenAI, model: str, caption_json: dict) -> dict:
    prompt = build_qa_prompt(caption_json)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful medical AI assistant."},
            {"role": "user", "content": prompt},
        ],
        #temperature=0.7,
    )
    return json.loads(resp.choices[0].message.content.strip())


# ========= 行整形 =========
def format_caption_row(row: pd.Series, cap_json: dict) -> dict:
    return {
        "mask_path": row["mask_path"],
        "video": int(row["video"]),
        "frame": int(row["frame"]),
        "phase": row.get("phase", "Unknown"),
        "caption": cap_json.get("caption", ""),
        "phase_description": cap_json.get("phase_description", ""),
        "anatomy_present": ";".join(cap_json.get("anatomy_present", [])),
        "tools_present": ";".join(cap_json.get("tools_present", [])),
        "focus_of_frame": cap_json.get("focus_of_frame", ""),
    }


def format_qa_rows(row: pd.Series, qa_json: dict) -> list[dict]:
    rows = []
    for qa in qa_json.get("qas", []):
        options = qa.get("options", [""] * 6)
        rows.append({
            "mask_path": row["mask_path"],
            "video": int(row["video"]),
            "frame": int(row["frame"]),
            "phase": row.get("phase", "Unknown"),
            "qa_id": qa.get("id"),
            "question": qa.get("question", ""),
            "option_0": options[0] if len(options) > 0 else "",
            "option_1": options[1] if len(options) > 1 else "",
            "option_2": options[2] if len(options) > 2 else "",
            "option_3": options[3] if len(options) > 3 else "",
            "option_4": options[4] if len(options) > 4 else "",
            "option_5": options[5] if len(options) > 5 else "",
            "correct_option_index": qa.get("correct_option_index", -1),
            "rationale": qa.get("rationale", ""),
        })
    return rows


# ========= 1フレーム処理 (並列実行単位) =========
def process_row(
    client: OpenAI, caption_model: str, qa_model: str, idx: int, row: pd.Series
) -> tuple[int, dict, list[dict]]:
    cap_json = call_caption_llm(client, caption_model, row)
    cap_row = format_caption_row(row, cap_json)
    qa_json = call_qa_llm(client, qa_model, cap_json)
    qa_row_list = format_qa_rows(row, qa_json)
    return idx, cap_row, qa_row_list


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

    df_meta = pd.read_csv(project_root / args.meta_csv)
    df_meta_n = df_meta.head(args.n_samples).copy()
    print(f"Processing {len(df_meta_n)} frames with {args.num_workers} workers ...")

    caption_map: dict[int, dict] = {}
    qa_map: dict[int, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_row, client, args.caption_model, args.qa_model, idx, row): idx
            for idx, row in df_meta_n.iterrows()
        }
        with tqdm(total=len(futures), desc="Generating", unit="frame") as pbar:
            for future in as_completed(futures):
                orig_idx = futures[future]
                try:
                    idx, cap_row, qa_row_list = future.result()
                    caption_map[idx] = cap_row
                    qa_map[idx] = qa_row_list
                    pbar.set_postfix({"video": f"{cap_row['video']:02d}", "frame": cap_row["frame"]})
                except Exception as e:
                    tqdm.write(f"ERROR at index {orig_idx}: {e}")
                pbar.update(1)

    # 元の行順に並べ直す
    sorted_indices = sorted(caption_map.keys())
    caption_rows = [caption_map[i] for i in sorted_indices]
    qa_rows = [r for i in sorted_indices for r in qa_map.get(i, [])]

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    caption_path = output_dir / args.caption_output
    qa_path = output_dir / args.qa_output

    pd.DataFrame(caption_rows).to_csv(caption_path, index=False)
    pd.DataFrame(qa_rows).to_csv(qa_path, index=False)

    print(f"\nSaved {len(caption_rows)} caption rows  -> {caption_path}")
    print(f"Saved {len(qa_rows)} QA rows         -> {qa_path}")


# ========= CLI =========
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate captions and MCQ-QA pairs for Cholec80 frames using LLM."
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Number of frames to process (default: 10)"
    )
    parser.add_argument(
        "--caption-model", default="gpt-5-mini",
        help="OpenAI model for caption generation (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--qa-model", default="gpt-5-mini",
        help="OpenAI model for QA generation (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--meta-csv", default="data/cholec80_sampled_dataset.csv",
        help="Path to metadata CSV, relative to project root"
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory, relative to project root (default: output)"
    )
    parser.add_argument(
        "--caption-output", default="cholec_caption.csv",
        help="Filename for caption output CSV (default: cholec_caption.csv)"
    )
    parser.add_argument(
        "--qa-output", default="cholec_qa.csv",
        help="Filename for QA output CSV (default: cholec_qa.csv)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=16,
        help="Number of parallel API call workers (default: 4)"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY env var is used."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
