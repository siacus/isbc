#!/usr/bin/env python3
"""
Download GoEmotions and prepare it for binary_factors_train.py.

Source: HuggingFace google-research-datasets/go_emotions
Paper: Demszky et al. 2020

~58k Reddit comments annotated with 27 fine-grained emotions + neutral.
Each comment can have multiple emotion labels (multi-label).

This script handles the multi-label nature in two ways:
    --mode single  : keep only comments with exactly 1 emotion label (clean)
    --mode primary : keep all comments, using the most-frequent emotion
                     as the primary label

Label columns produced:
    label_primary   : the single dominant emotion (or the one label if single mode)
    label_grouped   : collapsed into 4 sentiment groups:
                      positive / negative / ambiguous / neutral
    labels_all      : semicolon-separated list of all emotions (for reference)

Output files in --output_dir:
    goemotions_ready.csv    id, text
    goemotions_labels.csv   id, label_primary, label_grouped, labels_all
    goemotions_stats.txt

Usage:
    pip install datasets pandas

    # Clean single-label subset (45,432 comments with the default length filter):
    python download_goemotions.py --output_dir data/goemotions --mode single

    # All comments, primary label:
    python download_goemotions.py --output_dir data/goemotions --mode primary

    # Subsample:
    python download_goemotions.py --output_dir data/goemotions \\
        --mode single --max_per_class 1000
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# 27 emotions + neutral
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral",
]

# Grouped sentiment mapping (from the original paper)
SENTIMENT_GROUPS = {
    "positive":   {"admiration", "amusement", "approval", "caring",
                   "desire", "excitement", "gratitude", "joy", "love",
                   "optimism", "pride", "relief"},
    "negative":   {"anger", "annoyance", "disappointment", "disapproval",
                   "disgust", "embarrassment", "fear", "grief", "nervousness",
                   "remorse", "sadness"},
    "ambiguous":  {"confusion", "curiosity", "realization", "surprise"},
    "neutral":    {"neutral"},
}
EMOTION_TO_GROUP = {e: g for g, es in SENTIMENT_GROUPS.items() for e in es}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--mode",          default="single",
                   choices=["single", "primary"],
                   help="single: only keep unambiguous 1-label comments. "
                        "primary: keep all, use most common label as primary.")
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--min_text_len",  type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    print("Loading google-research-datasets/go_emotions ...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")

    frames = []
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        df = ds[split_name].to_pandas()
        df["_split"] = split_name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Raw rows: {len(combined):,}")

    # 'labels' column is a list of emotion indices
    combined["labels_all"] = combined["labels"].apply(
        lambda ls: ";".join(EMOTIONS[i] for i in ls) if len(ls) > 0 else "neutral")

    if args.mode == "single":
        combined = combined[combined["labels"].apply(len) == 1].reset_index(drop=True)
        combined["label_primary"] = combined["labels"].apply(
            lambda ls: EMOTIONS[ls[0]])
        print(f"Single-label rows: {len(combined):,}")
    else:
        # primary = first label (already sorted by frequency in dataset)
        combined = combined[combined["labels"].apply(len) > 0].reset_index(drop=True)
        combined["label_primary"] = combined["labels"].apply(
            lambda ls: EMOTIONS[ls[0]])

    combined["label_grouped"] = combined["label_primary"].map(EMOTION_TO_GROUP)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() >= args.min_text_len]

    if args.max_per_class is not None:
        combined = (combined.groupby("label_primary", group_keys=False)
                             .apply(lambda g: g.sample(
                                 min(len(g), args.max_per_class),
                                 random_state=42))
                             .reset_index(drop=True))

    combined = combined.drop(columns=["id"], errors="ignore").reset_index(drop=True)
    combined.insert(0, "id", range(len(combined)))

    ready_path  = out / "goemotions_ready.csv"
    labels_path = out / "goemotions_labels.csv"

    combined[["id", "text"]].to_csv(ready_path, index=False)
    combined[["id", "label_primary", "label_grouped",
              "labels_all"]].to_csv(labels_path, index=False)

    lines = [f"Mode: {args.mode}", f"Total: {len(combined):,}", ""]
    lines += ["label_primary (27 emotions)", "-" * 45]
    for val, cnt in combined["label_primary"].value_counts().items():
        lines.append(f"  {val:<20} {cnt:>6,}  ({cnt/len(combined)*100:.1f}%)")
    lines += ["", "label_grouped (4 sentiments)", "-" * 45]
    for val, cnt in combined["label_grouped"].value_counts().items():
        lines.append(f"  {val:<15} {cnt:>6,}  ({cnt/len(combined)*100:.1f}%)")
    stats_text = "\n".join(lines)
    print("\n" + stats_text)
    (out / "goemotions_stats.txt").write_text(stats_text + "\n")

    print(f"\nSaved: {ready_path}")
    print(f"       {labels_path}")
    print(f"""
Next — train with the standard K=16 setup:
    python binary_factors_train.py \\
        --input {ready_path} \\
        --output_dir runs/goemotions16 \\
        --model_name Qwen/Qwen3-Embedding-0.6B \\
        --pooling_mode last_token \\
        --objective sparse_ae \\
        --k_factors 16 --target_active 6 --epochs 5

    bash run_analysis.sh \\
        --encoded  runs/goemotions16/encoded_goemotions_ready_best.csv \\
        --labels   {labels_path} \\
        --outdir   runs/goemotions16/analysis \\
        --label_col label_primary    # 28 classes
    # or: --label_col label_grouped  # 4 sentiment groups
""")


if __name__ == "__main__":
    main()
