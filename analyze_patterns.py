#!/usr/bin/env python3
"""
Analyze unique binary factor patterns and their distribution across label categories.

For each unique binary vector (combination of active factors), shows:
  - how many documents have that exact pattern
  - the count and percentage of each category
  - which category dominates
  - sorted by dominant category % descending

For patterns above --min_dominant_pct, prints all documents in that pattern,
with labels ordered by their abundance within the pattern (most common label first).

Usage:
    # AG News (integer label column):
    python analyze_patterns.py \
        --encoded   runs/agnews_sae/encoded_agnews_all_ready_best.csv \
        --labels    data/agnews/agnews_all.csv \
        --label_col label \
        --output    runs/agnews_sae/pattern_analysis.csv

    # Airline sentiment:
    python analyze_patterns.py \
        --encoded   runs/airline_sae/encoded_airline_ready_best.csv \
        --labels    data/airline/airline_labels.csv \
        --label_col airline_sentiment \
        --output    runs/airline_sae/pattern_analysis_sentiment.csv

    # Airline negative reason:
    python analyze_patterns.py \
        --encoded   runs/airline_sae/encoded_airline_ready_best.csv \
        --labels    data/airline/airline_labels.csv \
        --label_col negativereason \
        --output    runs/airline_sae/pattern_analysis_reason.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoded",    required=True, help="CSV output from --mode encode")
    p.add_argument("--labels",     required=True, help="CSV with columns: id, <label_col>")
    p.add_argument("--label_col",  type=str, default="label_name",
                   help="Column in the labels CSV to use as category. "
                        "e.g. label_name, airline_sentiment, negativereason, airline. "
                        "For AG News pass 'label' (integer column, auto-mapped) or "
                        "'label_name' (string column).")
    p.add_argument("--output",     required=True, help="Where to write the pattern summary CSV")
    p.add_argument("--min_count",        type=int,   default=1,
                   help="Minimum number of documents for a pattern to appear in the summary table")
    p.add_argument("--min_dominant_pct", type=float, default=80.0,
                   help="Only print document listings for patterns where the dominant category "
                        "exceeds this %% (default: 80)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load encoded factors ──────────────────────────────────────────────────
    enc = pd.read_csv(args.encoded)
    bit_cols = sorted([c for c in enc.columns if c.startswith("bit_")])
    if not bit_cols:
        raise ValueError("No bit_* columns found in encoded CSV. Did you run --mode encode?")
    print(f"Loaded {len(enc):,} documents, {len(bit_cols)} factors.")

    # ── Load labels ───────────────────────────────────────────────────────────
    lab = pd.read_csv(args.labels)
    label_col = args.label_col

    # Special case: AG News integer 'label' column — map to strings
    AG_NEWS_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci_Tech"}
    if label_col == "label" and "label" in lab.columns and pd.api.types.is_integer_dtype(lab["label"]):
        lab["label_name"] = lab["label"].map(AG_NEWS_MAP)
        label_col = "label_name"

    if label_col not in lab.columns:
        available = [c for c in lab.columns if c != "id"]
        raise ValueError(f"--label_col '{label_col}' not found. Available columns: {available}")

    lab["label_name"] = lab[label_col].astype(str).str.strip()

    df = enc.merge(lab[["id", "label_name"]], on="id", how="inner")
    n_lost = len(enc) - len(df)
    if n_lost:
        print(f"Warning: {n_lost} documents had no matching label and were dropped.")

    categories = sorted(df["label_name"].unique())
    print(f"Categories: {categories}")

    # ── Build pattern strings ─────────────────────────────────────────────────
    # Represent each binary vector as a compact string, e.g. "0010100001..."
    bit_matrix = df[bit_cols].astype(int)
    df["pattern"] = bit_matrix.apply(lambda row: "".join(row.astype(str)), axis=1)

    n_patterns = df["pattern"].nunique()
    print(f"Unique binary patterns: {n_patterns:,}  (out of {2**len(bit_cols):,} possible)")

    # ── Aggregate: for each pattern, count docs per category ─────────────────
    grouped = (
        df.groupby(["pattern", "label_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=categories, fill_value=0)
    )
    grouped["total"] = grouped.sum(axis=1)
    grouped = grouped[grouped["total"] >= args.min_count].copy()

    # Percentage columns
    for cat in categories:
        grouped[f"pct_{cat}"] = (grouped[cat] / grouped["total"] * 100).round(1)

    # Dominant category and its percentage
    grouped["dominant"]     = grouped[categories].idxmax(axis=1)
    grouped["dominant_pct"] = grouped[categories].max(axis=1) / grouped["total"] * 100
    grouped["dominant_pct"] = grouped["dominant_pct"].round(1)

    # Entropy (lower = more category-pure)
    def entropy(row):
        p = np.array([row[c] for c in categories], dtype=float)
        p = p / p.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    grouped["entropy"] = grouped.apply(entropy, axis=1).round(3)

    # Also store the active factor indices for readability
    def active_factors(pattern):
        return ",".join(str(i) for i, b in enumerate(pattern) if b == "1") or "(none)"

    grouped = grouped.reset_index()
    grouped["active_factors"] = grouped["pattern"].apply(active_factors)
    grouped["n_active"]       = grouped["pattern"].apply(lambda p: p.count("1"))

    # ── Sort by dominant_pct descending (then total descending as tiebreak) ───
    grouped = grouped.sort_values(
        ["dominant_pct", "total"], ascending=[False, False]
    ).reset_index(drop=True)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Pattern analysis  (min_count={args.min_count})")
    print(f"Sorted by: dominant category % descending")
    print(f"{'='*80}")
    header = (f"{'#':>4}  {'total':>6}  {'n_act':>5}  {'dominant':<10}  {'dom%':>5}  "
              f"{'entropy':>7}  " +
              "  ".join(f"{c[:4]:>6}" for c in categories) +
              "  active factors")
    print(header)
    print("-" * len(header))
    for rank, row in grouped.iterrows():
        cat_counts = "  ".join(f"{int(row[c]):>6}" for c in categories)
        print(f"{rank+1:>4}  {row['total']:>6,}  {row['n_active']:>5}  "
              f"{row['dominant']:<10}  {row['dominant_pct']:>4.0f}%  "
              f"{row['entropy']:>7.3f}  {cat_counts}  {row['active_factors']}")

    # Overall stats
    print(f"\nTotal patterns with >= {args.min_count} docs: {len(grouped):,}")
    print(f"Pure patterns (entropy=0):  {(grouped['entropy']==0).sum():,}")
    print(f"Median entropy:             {grouped['entropy'].median():.3f}")
    print(f"Mean n_active per pattern:  {grouped['n_active'].mean():.2f}")

    # ── Per-pattern document listing (above dominance threshold) ─────────────
    threshold_patterns = grouped[grouped["dominant_pct"] >= args.min_dominant_pct]
    print(f"\n{'='*80}")
    print(f"Document listing for patterns with dominant_pct >= {args.min_dominant_pct}%")
    print(f"({len(threshold_patterns)} patterns qualify)")
    print(f"{'='*80}")

    for rank, prow in threshold_patterns.iterrows():
        pat = prow["pattern"]
        pat_docs = df[df["pattern"] == pat].copy()

        # Build label order by descending abundance for this pattern
        label_order = (
            pat_docs["label_name"]
            .value_counts()
            .index
            .tolist()
        )
        pat_docs["_label_rank"] = pat_docs["label_name"].map(
            {lbl: i for i, lbl in enumerate(label_order)}
        )
        pat_docs = pat_docs.sort_values("_label_rank")

        print(f"\nPattern #{rank+1}  [{prow['active_factors']}]  "
              f"n={int(prow['total'])}  dominant={prow['dominant']} ({prow['dominant_pct']:.0f}%)  "
              f"entropy={prow['entropy']:.3f}")
        print(f"  Label order (by abundance): {' > '.join(label_order)}")
        print(f"  {'label':<10}  {'id':<12}  text")
        print(f"  {'-'*9}  {'-'*11}  {'-'*55}")
        for _, doc in pat_docs.iterrows():
            text_preview = str(doc["text"])[:80].replace("\n", " ")
            print(f"  {doc['label_name']:<10}  {str(doc['id']):<12}  {text_preview}")

    # ── Per-category coverage by pure patterns ────────────────────────────────
    print(f"\n{'='*80}")
    print("Coverage: fraction of each category in patterns with entropy=0")
    print("-"*80)
    pure_patterns = set(grouped[grouped["entropy"] == 0]["pattern"])
    df["in_pure_pattern"] = df["pattern"].isin(pure_patterns)
    for cat in categories:
        cat_df = df[df["label_name"] == cat]
        pct = cat_df["in_pure_pattern"].mean() * 100
        print(f"  {cat:<10}: {pct:5.1f}% of {len(cat_df):,} docs in a pure pattern")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    col_order = (["pattern", "active_factors", "n_active", "total"]
                 + categories
                 + [f"pct_{c}" for c in categories]
                 + ["dominant", "dominant_pct", "entropy"])
    grouped[col_order].to_csv(args.output, index=False)
    print(f"\nSaved full pattern table to: {args.output}")


if __name__ == "__main__":
    main()
