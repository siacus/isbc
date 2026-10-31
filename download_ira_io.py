#!/usr/bin/env python3
"""
Download IRA (Russian troll) tweets from FiveThirtyEight's GitHub repo
and prepare them for qwen_binary_factors_train.py.

Source: https://github.com/fivethirtyeight/russian-troll-tweets
        ~3M tweets from 2,848 IRA-linked handles, split across 13 CSVs.
        No API keys, no ZIP, no Archive.org — plain CSVs on GitHub raw.

Original data: Linvill & Warren (Clemson University), via FiveThirtyEight.
Period: February 2012 – May 2018 (bulk: 2015–2017).

Schema (relevant columns):
    content          : tweet text
    language         : tweet language (English = 'English')
    account_category : general theme coded by Linvill & Warren
                       (RightTroll, LeftTroll, NewsFeed, HashtagGamer,
                        Fearmonger, Commercial, Unknown, NonEnglish)
    account_type     : specific sub-theme
    publish_date     : timestamp

Label columns produced in ira_labels.csv:
    account_category          : 8 classes (main label — most interpretable)
    account_type              : finer sub-theme
    language                  : tweet language
    category_x_language       : cross product

Output files in --output_dir:
    ira_ready.csv        id, text
    ira_labels.csv       id, account_category, account_type, language,
                         category_x_language
    ira_stats.txt        dataset statistics

Usage:
    pip install requests pandas

    # English only, 100k tweets (fast):
    python download_ira_io.py --output_dir data/ira --lang_filter English \\
        --max_rows 100000

    # All languages, full ~3M dataset:
    python download_ira_io.py --output_dir data/ira

    # Specific files only (e.g. first 3 of 13):
    python download_ira_io.py --output_dir data/ira --files 1 2 3

Then train:
    python qwen_binary_factors_train.py \\
        --input data/ira/ira_ready.csv \\
        --output_dir runs/ira_sae \\
        --model_name Qwen/Qwen3-Embedding-0.6B \\
        --pooling_mode last_token \\
        --objective sparse_ae \\
        --k_factors 16 --target_active 6 --epochs 10

Analysis:
    bash run_analysis.sh \\
        --encoded  runs/ira_sae/encoded_ira_ready_best.csv \\
        --labels   data/ira/ira_labels.csv \\
        --outdir   runs/ira_sae/analysis \\
        --label_col account_category
"""

from __future__ import annotations
import argparse
import io
from pathlib import Path
import requests
import pandas as pd

BASE_URL = ("https://raw.githubusercontent.com/fivethirtyeight/"
            "russian-troll-tweets/master/IRAhandle_tweets_{n}.csv")
N_FILES = 13


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--lang_filter",   default=None,
                   help="Keep only tweets in this language. "
                        "Use 'English' for English-only. "
                        "Default: keep all languages.")
    p.add_argument("--max_rows",      type=int, default=None,
                   help="Max rows to keep after filtering (random sample).")
    p.add_argument("--min_text_len",  type=int, default=10)
    p.add_argument("--files",         type=int, nargs="+", default=None,
                   help="Which file numbers to download (1–13). "
                        "Default: all 13.")
    p.add_argument("--no_retweets",   action="store_true", default=False,
                   help="Drop retweets (retweet column == 1).")
    return p.parse_args()


def fetch_file(n: int) -> pd.DataFrame:
    url = BASE_URL.format(n=n)
    print(f"  Fetching file {n:>2}/13 ... ", end="", flush=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    print(f"{len(df):>8,} rows")
    return df


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    file_nums = args.files or list(range(1, N_FILES + 1))
    print(f"Downloading {len(file_nums)} of {N_FILES} files from GitHub ...")

    frames = []
    for n in file_nums:
        try:
            frames.append(fetch_file(n))
        except Exception as e:
            print(f"  WARNING: file {n} failed: {e}")

    if not frames:
        raise RuntimeError("No files downloaded successfully.")

    df = pd.concat(frames, ignore_index=True)
    print(f"\nRaw rows: {len(df):,}   columns: {list(df.columns)}")

    # ── Text ──────────────────────────────────────────────────────────────────
    df["text"] = df["content"].astype(str).str.strip()
    df = df[df["text"].str.len() >= args.min_text_len]

    # ── Optional: drop retweets ───────────────────────────────────────────────
    if args.no_retweets and "retweet" in df.columns:
        before = len(df)
        df = df[df["retweet"].astype(str) != "1"]
        print(f"Dropped {before - len(df):,} retweets")

    # ── Language filter ───────────────────────────────────────────────────────
    if args.lang_filter and "language" in df.columns:
        before = len(df)
        df = df[df["language"].astype(str).str.strip() == args.lang_filter]
        print(f"After lang_filter='{args.lang_filter}': {len(df):,} rows "
              f"(dropped {before - len(df):,})")

    # ── Subsample ─────────────────────────────────────────────────────────────
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=42)
        print(f"Subsampled to {args.max_rows:,} rows")

    df = df.reset_index(drop=True)
    df.insert(0, "id", df.index)

    # ── Label columns ─────────────────────────────────────────────────────────
    for col in ["account_category", "account_type", "language"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = "unknown"

    df["category_x_language"] = df["account_category"] + "_" + df["language"]

    # ── Save ──────────────────────────────────────────────────────────────────
    ready_path  = out / "ira_ready.csv"
    labels_path = out / "ira_labels.csv"

    df[["id", "text"]].to_csv(ready_path, index=False)
    df[["id", "account_category", "account_type",
        "language", "category_x_language"]].to_csv(labels_path, index=False)

    # ── Stats ─────────────────────────────────────────────────────────────────
    lines = [f"Total tweets: {len(df):,}",
             f"Source files: {file_nums}",
             f"Lang filter:  {args.lang_filter or 'all'}",
             ""]
    lines += ["account_category (main label)", "-" * 45]
    for val, cnt in df["account_category"].value_counts().items():
        lines.append(f"  {val:<25} {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")
    lines += ["", "language (top 10)", "-" * 45]
    for val, cnt in df["language"].value_counts().head(10).items():
        lines.append(f"  {val:<20} {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")
    stats_text = "\n".join(lines)
    print("\n" + stats_text)
    (out / "ira_stats.txt").write_text(stats_text + "\n")

    print(f"\nSaved: {ready_path}")
    print(f"       {labels_path}")
    print(f"""
Next:
    python qwen_binary_factors_train.py \\
        --input {ready_path} \\
        --output_dir runs/ira_sae \\
        --model_name Qwen/Qwen3-Embedding-0.6B \\
        --pooling_mode last_token \\
        --objective sparse_ae \\
        --k_factors 16 --target_active 6 --epochs 10

    bash run_analysis.sh \\
        --encoded  runs/ira_sae/encoded_ira_ready_best.csv \\
        --labels   {labels_path} \\
        --outdir   runs/ira_sae/analysis \\
        --label_col account_category
""")


if __name__ == "__main__":
    main()
