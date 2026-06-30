# surgery-vid

外科手術動画フレームから、LLMを用いてキャプションと多肢選択QAデータを生成し、VQAベンチマークとして評価するプロジェクト。

## セットアップ

```bash
uv sync
```

OpenAI APIキーを環境変数で指定:

```bash
export OPENAI_API_KEY="sk-..."
```

## データ構成

```text
data/
├── cholec80/
│   ├── cholecseg.zip                  # 腹腔鏡下胆嚢摘出術フレーム + セグメンテーションマスク
│   ├── phase_annotations/             # video01-phase.txt ... video80-phase.txt
│   ├── tool_annotations/              # video01-tool.txt  ... video80-tool.txt
│   └── seg/                           # セグメンテーションマスク画像（videoXX配下）
├── cholec80_sampled_dataset.csv       # サンプリング済みメタデータ (notebooks/mask_csv.ipynb で生成)
└── egosurgery/
    ├── images.zip
    ├── annotations.zip
    └── gaze.zip
```

## データ配置（`../surgery-vid/data` を参照）

このリポジトリは `data/` をGit管理しない前提です。  
ローカルにある参照データ `../surgery-vid/data` を使う場合は、必要なファイルを本リポジトリの `data/` に配置してください。

参照元で確認できる主な構成:

```text
../surgery-vid/data/
├── cholec80/
│   ├── cholecseg.zip
│   ├── phase_annotations/
│   ├── tool_annotations/
│   └── seg/
├── cholec80_sampled_dataset.csv
├── egosurgery/
│   ├── images.zip
│   ├── annotations.zip
│   └── gaze.zip
└── pit/
```

最小実行に必要なもの（本プロジェクト）:

- `data/cholec80/cholecseg.zip`（`src/vqa_inference.py` が参照）
- `data/cholec80_sampled_dataset.csv`（`src/make_cap_qa.py` が参照）

補助的に使うもの（主に notebook 生成工程）:

- `data/cholec80/phase_annotations/*.txt`
- `data/cholec80/tool_annotations/*.txt`
- `data/cholec80/seg/**`

## データセット概要

- `cholec80`: 腹腔鏡下胆嚢摘出術（Cholecystectomy）動画のデータ。  
  本リポジトリのキャプション生成・VQA評価の中心データです。
- `egosurgery`: 手術の一人称視点（egocentric）系データ。`images.zip` / `annotations.zip` / `gaze.zip` を含みます。
- `pit`: 内視鏡下下垂体手術（endoscopic pituitary surgery, eTSA）の動画データ（PitVis-2023）。参照元 `../surgery-vid/data` に存在しますが、本リポジトリの主要スクリプトでは未使用です。

## データセットの取得方法

各データセットは再配布が制限されている（多くが研究目的限定・CC BY-NC-SA 4.0）ため、本リポジトリには含めず、各自で公式配布元から取得してください。取得後は「データ配置」の節に従って `data/` 配下に配置します。

### cholec80（腹腔鏡下胆嚢摘出術 動画 + アノテーション）

- 配布元: University of Strasbourg の CAMMA 研究グループ <https://camma.unistra.fr/datasets/>
- 取得方法: データセットページのリクエストフォームに記入して申請すると、ダウンロード手順が案内されます（即時DLではなく承認制）。
- ライセンス: CC BY-NC-SA 4.0（非営利・研究目的）
- 内容: 80症例の手術動画（25fps）、phase アノテーション（25fps）、tool presence アノテーション（1fps）
- 引用: Twinanda et al., "EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos", IEEE TMI, 2016
- 本リポジトリで使うファイル: `phase_annotations/*.txt`, `tool_annotations/*.txt`

### CholecSeg8k（cholecseg.zip / セグメンテーションマスク）

Cholec80 から抽出した 17 動画・8,080 フレームにピクセル単位のセグメンテーション（13クラス）を付与した派生データセットです。`data/cholec80/cholecseg.zip` および `seg/` の元データになります。

- 配布元（公式・Kaggle）: <https://www.kaggle.com/datasets/newslab/cholecseg8k>
- ミラー（Hugging Face、参考）: <https://huggingface.co/datasets/minwoosun/CholecSeg8k>
- 取得方法（Kaggle CLI 例）:

  ```bash
  # 事前に Kaggle アカウントと API トークン(~/.kaggle/kaggle.json)が必要
  pip install kaggle
  kaggle datasets download -d newslab/cholecseg8k -p data/cholec80/
  # 取得した zip を本リポジトリの想定パスにあわせて配置/リネーム
  # -> data/cholec80/cholecseg.zip
  ```

- ライセンス: CC BY-NC-SA 4.0（非営利・研究目的）
- 引用: Hong et al., "CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80", 2020 (arXiv:2012.12453)

### egosurgery（一人称視点 開放手術 動画）

- 配布元: GitHub `Fujiry0/EgoSurgery` <https://github.com/Fujiry0/EgoSurgery>
- 取得方法: リポジトリ記載の Google フォームに記入して申請すると、ダウンロードリンクが送付されます（承認制）。
- ライセンス: CC BY-NC-SA 4.0（学術研究目的限定、商用不可）
- 内容: EgoSurgery-Phase（phase アノテーション）、EgoSurgery-Tool（手術器具・手のバウンディングボックス）等。本リポジトリの `images.zip` / `annotations.zip` / `gaze.zip` に対応。
- 引用: Fujii et al., "EgoSurgery-Phase", MICCAI 2024 / "EgoSurgery-Tool", arXiv:2406.03095

### pit（内視鏡下下垂体手術 動画 / PitVis-2023）

内視鏡下下垂体手術（endoscopic TransSphenoidal Approach, eTSA）の動画データセットです。MICCAI 2023 EndoVis のサブチャレンジ PitVis-2023 として公開されています。本リポジトリの主要スクリプトでは未使用です。

- 配布元（UCL RDR / Figshare）: <https://rdr.ucl.ac.uk/articles/dataset/PitVis_Challenge_Endoscopic_Pituitary_Surgery_videos/26531686>（DOI: 10.5522/04/26531686）
- 配布元（Hugging Face、ミラー）: <https://huggingface.co/datasets/UCL-WEISS/PitVis-2023>
- 取得方法: 上記ポータルから直接ダウンロード可能。25動画（`video_{n}.mp4`）と step/instrument アノテーション（`annotations_{n}.csv`）、ラベル対応表（`map_steps.csv` / `map_instrument.csv`）、動画メタデータ等を含みます。
- 補助スクリプト・ベースライン: <https://github.com/dreets/pitvis>
- 引用: Das et al., "PitVis-2023 Challenge: Workflow Recognition in videos of Endoscopic Pituitary Surgery", 2024 (arXiv:2409.01184)

### まとめ

| データセット | 配布元 | 取得方法 | ライセンス |
|---|---|---|---|
| cholec80 | CAMMA (University of Strasbourg) | リクエストフォーム申請（承認制） | CC BY-NC-SA 4.0 |
| CholecSeg8k | Kaggle (newslab/cholecseg8k) | 直接DL（要Kaggleアカウント） | CC BY-NC-SA 4.0 |
| egosurgery | GitHub (Fujiry0/EgoSurgery) | Googleフォーム申請（承認制） | CC BY-NC-SA 4.0 |
| pit（PitVis-2023） | UCL RDR / Hugging Face | 直接DL | （配布元の規約を確認） |

## パイプライン概要

```text
notebooks/mask_csv.ipynb
    ↓ data/cholec80_sampled_dataset.csv
src/make_cap_qa.py
    ↓ output/{version}/cap_all.csv
    ↓ output/{version}/qa_all.csv
src/vqa_inference.py or src/text_only_inference.py
    ↓ output/vqa_results*/{qa_stem}_{model}_pred.csv
src/eval_vqa.py
    ↓ output/vqa_results*/accuracy_summary.csv (+ accuracy_summary.txt)
```

## スクリプト

### `src/make_cap_qa.py`

フレームメタデータからキャプションとMCQ-QAを生成します。

```bash
python src/make_cap_qa.py
python src/make_cap_qa.py --n-samples 30 --output-dir output/v3 --caption-output cap_all.csv --qa-output qa_all.csv
```

主なオプション:

- `--n-samples` (default: `10`)
- `--num-workers` (default: `16`)
- `--caption-model` (default: `gpt-5-mini`)
- `--qa-model` (default: `gpt-5-mini`)
- `--meta-csv` (default: `data/cholec80_sampled_dataset.csv`)
- `--api-key` (省略時は `OPENAI_API_KEY` を使用)

### `src/vqa_inference.py`

QA CSV + `cholecseg.zip` の元画像でVQA推論を実行します。

```bash
python src/vqa_inference.py --model gpt-5-mini
python src/vqa_inference.py --model gpt-5 --reasoning-effort high
```

主なオプション:

- `--model` (default: `gpt-5-mini`)
- `--reasoning-effort` (`minimal` / `low` / `medium` / `high`)
- `--qa-csv` (default: `output/v1/qa_all.csv`)
- `--zip` (default: `data/cholec80/cholecseg.zip`)
- `--output-dir` (default: `output/vqa_results`)
- `--num-workers` (default: `8`)

### `src/text_only_inference.py`

画像なしのテキストのみでQA推論を実行し、リーク/易問化の有無を確認します。

```bash
python src/text_only_inference.py --model gpt-4.1 --qa-csv output/v3/qa_all.csv --output-dir output/vqa_results_v3
```

### `src/eval_vqa.py`

`*_pred.csv` を集計して `accuracy_summary.csv` と `accuracy_summary.txt` を出力します。

```bash
python src/eval_vqa.py --pred-dir output/vqa_results_v3 --output output/vqa_results_v3/accuracy_summary.csv
```

### `scripts/run_vqa_v3.sh`（現行の一括実行）

```bash
bash scripts/run_vqa_v3.sh
NUM_WORKERS=16 QA_CSV=output/v3/qa_all.csv bash scripts/run_vqa_v3.sh
OPENAI_API_KEY=sk-... bash scripts/run_vqa_v3.sh
```

補足:

- `scripts/run_vqa.sh` は v1 向け
- `scripts/run_vqa_v2.sh` は v2 向け
- 最新実験系は `run_vqa_v3.sh`

## 公開時の注意

- `data/` と `output/` は大容量のため、このリポジトリでは通常コミットしません（`.gitignore` で除外）。
- APIキーはファイルに保存せず、`OPENAI_API_KEY` 環境変数または `--api-key` 引数で渡してください。
- 公開用には、データの取得方法を README に記載し、必要なら別途配布先（Kaggle/Drive等）を使ってください。