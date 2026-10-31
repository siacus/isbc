#!/usr/bin/env python3
"""
Download FakeNewsNet (text-only CSV version) for qwen_binary_factors_train.py.

Source: https://github.com/KaiDMML/FakeNewsNet
This script downloads the lightweight news-text CSVs directly from the repo
(no Twitter API keys required). The full social-graph version is NOT fetched.

Two datasets are available:
    politifact  : ~1,000 articles, labels: real / fake
    gossipcop   : ~22,000 articles, labels: real / fake

Each article has:
    title, text (body), label (fake/real), source (politifact/gossipcop)

Combined label columns in output:
    label            : fake / real
    source           : politifact / gossipcop
    label_x_source   : e.g. "fake_politifact", "real_gossipcop"

Output files in --output_dir:
    fakenews_ready.csv        id, text
    fakenews_labels.csv       id, label, source, label_x_source
    fakenews_stats.txt        dataset statistics

Usage:
    pip install requests pandas

    python download_fakenewsnet.py --output_dir data/fakenews

    # One source only:
    python download_fakenewsnet.py --output_dir data/fakenews --source politifact

    # Subsample per class (useful for balancing):
    python download_fakenewsnet.py --output_dir data/fakenews --max_per_class 2000

Then train:
    python qwen_binary_factors_train.py \\
        --input data/fakenews/fakenews_ready.csv \\
        --output_dir runs/fakenews_sae \\
        --model_name Qwen/Qwen3-Embedding-0.6B \\
        --pooling_mode last_token \\
        --objective sparse_ae \\
        --k_factors 16 --target_active 6 --epochs 10
"""

from __future__ import annotations
import argparse
import io
from pathlib import Path
import pandas as pd
import requests

# Raw CSV URLs from the FakeNewsNet GitHub repo
URLS = {
    "politifact": {
        "fake": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv",
        "real": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv",
    },
    "gossipcop": {
        "fake": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_fake.csv",
        "real": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_real.csv",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--source",        default="both",
                   choices=["politifact", "gossipcop", "both"])
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Max articles per (label, source) combination")
    p.add_argument("--min_text_len",  type=int, default=30)
    return p.parse_args()


def fetch_csv(url: str) -> pd.DataFrame:
    print(f"  Fetching {url.split('/')[-1]} ...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def load_source(source_name: str) -> pd.DataFrame:
    parts = []
    for label, url in URLS[source_name].items():
        df = fetch_csv(url)
        df["label"]  = label
        df["source"] = source_name
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_text(row: pd.Series) -> str:
    """Concatenate title and news_url/text fields into a single string."""
    parts = []
    for col in ["title", "news_url", "tweet_ids"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            # Only use title and news_url — tweet_ids are not text
            if col in ("title",):
                parts.append(str(row[col]).strip())
    # Some versions have a 'text' or 'content' column
    for col in ["text", "content", "body"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]).strip())
            break
    return " ".join(parts)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources = ["politifact", "gossipcop"] if args.source == "both" \
               else [args.source]

    frames = []
    for src in sources:
        print(f"\nLoading {src} ...")
        df = load_source(src)
        print(f"  Raw rows: {len(df):,}  columns: {list(df.columns)}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Build text column
    combined["text"] = combined.apply(build_text, axis=1)
    combined["text"] = combined["text"].str.strip()
    combined = combined[combined["text"].str.len() >= args.min_text_len]

    # Clean labels
    combined["label"]  = combined["label"].astype(str).str.strip().str.lower()
    combined["source"] = combined["source"].astype(str).str.strip()
    combined["label_x_source"] = combined["label"] + "_" + combined["source"]

    # Subsample
    if args.max_per_class is not None:
        combined = (combined.groupby("label_x_source", group_keys=False)
                             .apply(lambda g: g.sample(
                                 min(len(g), args.max_per_class),
                                 random_state=42))
                             .reset_index(drop=True))

    combined = combined.drop_duplicates().reset_index(drop=True)
    # Source files already carry dataset-local IDs. Replace them with one
    # collision-free ID sequence for the combined Politifact/GossipCop table.
    combined = combined.drop(columns=["id"], errors="ignore")
    combined.insert(0, "id", range(len(combined)))

    # Write training input
    ready_path = out / "fakenews_ready.csv"
    combined[["id", "text"]].to_csv(ready_path, index=False)

    # Write labels
    labels_path = out / "fakenews_labels.csv"
    combined[["id", "label", "source", "label_x_source"]].to_csv(
        labels_path, index=False)

    # Stats
    lines = [f"Total articles: {len(combined):,}", ""]
    lines += ["label distribution", "-" * 40]
    for val, cnt in combined["label"].value_counts().items():
        lines.append(f"  {val:<12} {cnt:>6,}  ({cnt/len(combined)*100:.1f}%)")
    lines += ["", "source distribution", "-" * 40]
    for val, cnt in combined["source"].value_counts().items():
        lines.append(f"  {val:<15} {cnt:>6,}  ({cnt/len(combined)*100:.1f}%)")
    lines += ["", "label x source", "-" * 40]
    for val, cnt in combined["label_x_source"].value_counts().items():
        lines.append(f"  {val:<25} {cnt:>6,}  ({cnt/len(combined)*100:.1f}%)")
    stats_text = "\n".join(lines)
    print("\n" + stats_text)
    (out / "fakenews_stats.txt").write_text(stats_text + "\n")

    print(f"\nSaved training input : {ready_path}")
    print(f"Saved labels        : {labels_path}")
    print(f"""
Next steps:
    python qwen_binary_factors_train.py \\
        --input {ready_path} \\
        --output_dir runs/fakenews_sae \\
        --model_name Qwen/Qwen3-Embedding-0.6B \\
        --pooling_mode last_token \\
        --objective sparse_ae \\
        --k_factors 16 --target_active 6 --epochs 10

    bash run_analysis.sh \\
        --encoded  runs/fakenews_sae/encoded_fakenews_ready_best.csv \\
        --labels   {labels_path} \\
        --outdir   runs/fakenews_sae/analysis \\
        --label_col label            # fake / real
    # or: --label_col label_x_source  # fake_politifact etc.
""")


if __name__ == "__main__":
    main()
