#!/usr/bin/env python3
"""
Download TweetEval benchmark datasets for binary_factors_train.py.

Source: HuggingFace cardiffnlp/tweet_eval (no API needed)
Paper: Barbieri et al. 2020

Seven benchmark tasks are available through 11 Hugging Face configurations
(`--task` flag):
    sentiment   : negative / neutral / positive  (3 classes, ~50k tweets)
    emotion     : anger / joy / optimism / sadness  (4 classes, ~7k)
    irony       : non_irony / irony  (2 classes, ~3k)
    hate        : non-hate / hate  (2 classes, ~9k)
    offensive   : non-offensive / offensive  (2 classes, ~14k)
    stance_*    : none / against / favor  (3 classes, multiple topics)
    emoji       : 20 emoji classes  (20 classes, ~50k)

Output files in --output_dir/<task>/:
    <task>_all_ready.csv      id, text (all source partitions; SAE input)
    <task>_all.csv            labels and source_partition provenance
    <task>_stats.txt

Usage:
    pip install datasets

    # Single task:
    python download_tweeteval.py --output_dir data/tweeteval --task sentiment

    # All tasks at once:
    python download_tweeteval.py --output_dir data/tweeteval --task all

    # Subsample:
    python download_tweeteval.py --output_dir data/tweeteval --task emotion \\
        --max_per_class 1000
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

TASKS = ["sentiment", "emotion", "irony", "hate", "offensive",
         "stance_abortion", "stance_atheism", "stance_climate",
         "stance_feminist", "stance_hillary", "emoji"]

LABEL_NAMES = {
    "sentiment":        {0: "negative", 1: "neutral", 2: "positive"},
    "emotion":          {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"},
    "irony":            {0: "non_irony", 1: "irony"},
    "hate":             {0: "non-hate", 1: "hate"},
    "offensive":        {0: "non-offensive", 1: "offensive"},
    "stance_abortion":  {0: "none", 1: "against", 2: "favor"},
    "stance_atheism":   {0: "none", 1: "against", 2: "favor"},
    "stance_climate":   {0: "none", 1: "against", 2: "favor"},
    "stance_feminist":  {0: "none", 1: "against", 2: "favor"},
    "stance_hillary":   {0: "none", 1: "against", 2: "favor"},
    "emoji":            {
        0: "❤", 1: "😍", 2: "😂", 3: "💕", 4: "🔥",
        5: "😊", 6: "😎", 7: "✨", 8: "💙", 9: "😘",
        10: "📷", 11: "🇺🇸", 12: "☀", 13: "💜", 14: "😉",
        15: "💯", 16: "😁", 17: "🎄", 18: "📸", 19: "😜",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--task",          default="sentiment",
                   help=f"Task name or 'all'. Options: {', '.join(TASKS)}")
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--min_text_len",  type=int, default=5)
    return p.parse_args()


def process_task(task: str, out_root: Path,
                 max_per_class: int | None, min_text_len: int):
    from datasets import load_dataset
    print(f"\n── Task: {task} ──────────────────────────────────────")
    out = out_root / task
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("cardiffnlp/tweet_eval", task)
    # Prefer the dataset's own ClassLabel metadata so category names and numeric
    # ordering stay authoritative if the Hub dataset is revised. The local table
    # remains a fallback for older datasets versions lacking ClassLabel names.
    label_feature = ds["train"].features.get("label")
    feature_names = getattr(label_feature, "names", None)
    label_map = ({i: name for i, name in enumerate(feature_names)}
                 if feature_names else LABEL_NAMES.get(task, {}))

    splits_written = []
    ready_frames = []
    full_frames = []
    id_offset = 0
    for split_name in ["train", "test"]:
        if split_name not in ds:
            continue
        df = ds[split_name].to_pandas()
        df.columns = [c.lower() for c in df.columns]
        # Rename text column if needed
        for tc in ["text", "tweet", "content"]:
            if tc in df.columns:
                df = df.rename(columns={tc: "text"})
                break
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() >= min_text_len].reset_index(drop=True)
        df["label_name"] = df["label"].map(label_map).fillna(df["label"].astype(str))

        if max_per_class is not None:
            df = (df.groupby("label", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), max_per_class),
                                              random_state=42))
                    .reset_index(drop=True))

        df.insert(0, "id", range(id_offset, id_offset + len(df)))
        id_offset += len(df)

        ready = df[["id", "text"]]
        full  = df[["id", "text", "label", "label_name"]]
        ready_frames.append(ready)
        full_frames.append(
            full.assign(source_partition=f"official_{split_name}")
        )

        dist = full["label_name"].value_counts().to_dict()
        print(f"  {split_name}: {len(df):,} tweets   {dist}")
        splits_written.append((split_name, len(df)))

    all_ready = pd.concat(ready_frames, ignore_index=True)
    all_full = pd.concat(full_frames, ignore_index=True)
    all_ready.to_csv(out / f"{task}_all_ready.csv", index=False)
    all_full.to_csv(out / f"{task}_all.csv", index=False)
    for split_name in ("train", "test"):
        (out / f"{task}_{split_name}_ready.csv").unlink(missing_ok=True)
        (out / f"{task}_{split_name}.csv").unlink(missing_ok=True)

    # Stats
    lines = [f"Task: {task}"]
    for sn, n in splits_written:
        lines.append(f"  {sn}: {n:,}")
    (out / f"{task}_stats.txt").write_text("\n".join(lines) + "\n")
    print(f"  Saved to: {out}")
    return splits_written


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = TASKS if args.task == "all" else [args.task]

    failed = []
    for task in tasks:
        try:
            process_task(task, out_root, args.max_per_class, args.min_text_len)
        except Exception as e:
            print(f"  WARNING: {task} failed: {e}")
            failed.append(task)

    if failed:
        raise SystemExit(
            "TweetEval preparation failed for: " + ", ".join(failed)
        )

    print(f"""
Example — train on sentiment:
    python binary_factors_train.py \\
        --input {out_root}/sentiment/sentiment_all_ready.csv \\
        --output_dir runs/tweeteval_sentiment \\
        --model_name Qwen/Qwen3-Embedding-0.6B \\
        --pooling_mode last_token \\
        --objective sparse_ae \\
        --k_factors 16 --target_active 6 --epochs 3

    python binary_factors_train.py --mode encode \\
        --input {out_root}/sentiment/sentiment_all_ready.csv \\
        --output_dir runs/tweeteval_sentiment

    bash run_analysis.sh \\
        --encoded  runs/tweeteval_sentiment/encoded_sentiment_all_ready_best.csv \\
        --labels   {out_root}/sentiment/sentiment_all.csv \\
        --outdir   runs/tweeteval_sentiment/analysis \\
        --label_col label_name
""")


if __name__ == "__main__":
    main()
