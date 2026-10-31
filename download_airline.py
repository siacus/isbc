#!/usr/bin/env python3
"""
Download and prepare the Twitter US Airline Sentiment dataset.

Source: osanseviero/twitter-airline-sentiment (HuggingFace)
        or Tweets.csv downloaded directly from Kaggle

Output files
------------
  <output_dir>/airline_ready.csv    — id, text            (input to training pipeline)
  <output_dir>/airline_labels.csv   — id + all label columns (for analysis scripts)
  <output_dir>/airline_stats.txt    — dataset statistics

Label columns in airline_labels.csv
------------------------------------
  airline_sentiment       3 classes: negative / neutral / positive

  sentiment_fine          ~12 classes: positive, neutral, and then each negative
                          reason as its own class ("neg_Late Flight", "neg_Rude
                          Service", etc.).  Unknown/missing negative reasons become
                          "neg_unknown".  Use this when you want the model to
                          discover which complaint types have distinct factor profiles.

  airline                 6 classes: United, US Airways, American, Southwest,
                          Delta, Virgin America

  sentiment_x_airline     18 classes: e.g. "negative_United" (cross-product)

Usage
-----
  # from HuggingFace (no login needed):
  python download_airline.py --output_dir data/airline

  # from a Kaggle CSV you already downloaded:
  python download_airline.py --output_dir data/airline --csv_path /path/to/Tweets.csv

Then in analysis, pick whichever label column you want:
  --label_col airline_sentiment    # coarse  (3 classes)
  --label_col sentiment_fine       # fine    (~12 classes)
  --label_col airline              # by carrier
  --label_col sentiment_x_airline  # cross-product
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="data/airline")
    p.add_argument("--csv_path", default=None,
                   help="Path to Tweets.csv from Kaggle. "
                        "If omitted, downloads from HuggingFace.")
    p.add_argument("--min_text_len", type=int, default=5)
    return p.parse_args()


def load_raw(csv_path) -> pd.DataFrame:
    if csv_path:
        print(f"Loading from local CSV: {csv_path}")
        return pd.read_csv(csv_path, encoding="latin-1")
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Run: pip install datasets")
        print("Downloading from HuggingFace: osanseviero/twitter-airline-sentiment ...")
        ds = load_dataset("osanseviero/twitter-airline-sentiment", split="train")
        return ds.to_pandas()


def make_sentiment_fine(sentiment: pd.Series, reason: pd.Series) -> pd.Series:
    """
    positive / neutral  →  kept as-is
    negative            →  "neg_<reason>", or "neg_unknown" if reason is missing
    """
    UNKNOWN = {"", "nan", "n/a", "can't tell"}

    def map_one(sent, rsn):
        if sent != "negative":
            return sent  # "positive" or "neutral" unchanged
        rsn_clean = str(rsn).strip()
        if rsn_clean.lower() in UNKNOWN:
            return "neg_unknown"
        return f"neg_{rsn_clean}"

    return pd.Series(
        [map_one(s, r) for s, r in zip(sentiment, reason)],
        index=sentiment.index,
    )


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_raw(args.csv_path)
    print(f"Raw rows: {len(df):,}   Columns: {list(df.columns)}")

    # ── Normalise column names ────────────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for id_col in ["tweet_id", "id", "tweetid"]:
        if id_col in df.columns:
            df = df.rename(columns={id_col: "id"})
            break
    else:
        df["id"] = range(len(df))

    for text_col in ["text", "tweet_text", "content"]:
        if text_col in df.columns:
            df = df.rename(columns={text_col: "text"})
            break

    # ── Clean text ────────────────────────────────────────────────────────────
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= args.min_text_len]
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    print(f"After dedup/clean: {len(df):,} tweets")

    # ── Normalise sentiment column ────────────────────────────────────────────
    if "airline_sentiment" not in df.columns:
        for c in df.columns:
            if "sentiment" in c:
                df = df.rename(columns={c: "airline_sentiment"})
                break
    df["airline_sentiment"] = df["airline_sentiment"].astype(str).str.strip().str.lower()

    # ── Normalise negative reason column ─────────────────────────────────────
    if "negativereason" not in df.columns:
        for c in df.columns:
            if "reason" in c:
                df = df.rename(columns={c: "negativereason"})
                break
        else:
            df["negativereason"] = ""
    df["negativereason"] = df["negativereason"].fillna("").astype(str).str.strip()
    # non-negative tweets: blank out reason (they'll map to the sentiment as-is)
    df.loc[df["airline_sentiment"] != "negative", "negativereason"] = ""

    # ── Normalise airline column ──────────────────────────────────────────────
    if "airline" not in df.columns:
        df["airline"] = "unknown"
    df["airline"] = df["airline"].astype(str).str.strip()

    # ── Build label columns ───────────────────────────────────────────────────
    # coarse: original 3-class
    # (airline_sentiment already normalised above)

    # fine: positive / neutral / neg_<reason>
    df["sentiment_fine"] = make_sentiment_fine(
        df["airline_sentiment"], df["negativereason"]
    )

    # cross-product
    df["sentiment_x_airline"] = df["airline_sentiment"] + "_" + df["airline"]

    # ── Write training input (id + text only) ─────────────────────────────────
    ready_path = out / "airline_ready.csv"
    df[["id", "text"]].to_csv(ready_path, index=False)
    print(f"Saved training input : {ready_path}")

    # ── Write labels CSV ──────────────────────────────────────────────────────
    labels_path = out / "airline_labels.csv"
    df[["id", "airline_sentiment", "sentiment_fine",
        "airline", "sentiment_x_airline"]].to_csv(labels_path, index=False)
    print(f"Saved labels        : {labels_path}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    lines = [f"Total tweets: {len(df):,}", ""]

    lines += ["airline_sentiment (3 classes)", "-" * 45]
    for val, cnt in df["airline_sentiment"].value_counts().items():
        lines.append(f"  {val:<20} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

    lines += ["", "sentiment_fine (~12 classes)", "-" * 45]
    for val, cnt in df["sentiment_fine"].value_counts().items():
        lines.append(f"  {val:<35} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

    lines += ["", "airline (6 classes)", "-" * 45]
    for val, cnt in df["airline"].value_counts().items():
        lines.append(f"  {val:<25} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

    stats_text = "\n".join(lines)
    print("\n" + stats_text)
    (out / "airline_stats.txt").write_text(stats_text + "\n")
    print(f"\nSaved stats: {out / 'airline_stats.txt'}")

    print(f"""
Next steps
──────────
1. Train:
   python binary_factors_train.py \\
     --input {ready_path} \\
     --output_dir runs/airline_sae \\
     --model_name Qwen/Qwen3-Embedding-0.6B \\
     --pooling_mode last_token \\
     --objective sparse_ae \\
     --k_factors 16 --target_active 6 --epochs 10

2. Encode:
   python binary_factors_train.py \\
     --mode encode \\
     --input {ready_path} \\
     --output_dir runs/airline_sae

3. Analyze — pick label dimension:

   # coarse sentiment (3 classes):
   bash run_analysis.sh \\
     --encoded   runs/airline_sae/encoded_airline_ready_best.csv \\
     --labels    {labels_path} \\
     --outdir    runs/airline_sae/analysis_sentiment \\
     --label_col airline_sentiment

   # fine sentiment (~12 classes — most informative):
   bash run_analysis.sh \\
     --encoded   runs/airline_sae/encoded_airline_ready_best.csv \\
     --labels    {labels_path} \\
     --outdir    runs/airline_sae/analysis_fine \\
     --label_col sentiment_fine

   # by airline:
   bash run_analysis.sh \\
     --encoded   runs/airline_sae/encoded_airline_ready_best.csv \\
     --labels    {labels_path} \\
     --outdir    runs/airline_sae/analysis_airline \\
     --label_col airline
""")


if __name__ == "__main__":
    main()
