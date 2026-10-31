#!/usr/bin/env python3
"""
Step 7: Code uniqueness and entropy analysis.

Characterises how much of the binary codebook the model is actually using:

  - Pattern count       : unique binary vectors out of 2^K possible
  - Pattern entropy     : H(pattern distribution) in bits — max is log2(n_unique)
  - Coverage            : n_unique / n_documents  (how many docs share a code?)
  - Concentration       : what fraction of docs are covered by the top-N patterns?
  - Per-category entropy: does each label have a diverse or concentrated code profile?
  - Singleton patterns  : patterns used by exactly 1 document (over-specificity signal)

High entropy = model uses a rich, diverse codebook.
Low entropy  = most documents collapse into a few patterns (codebook collapse).

Usage:
    python analyze_7_code_entropy.py \\
        --encoded   runs/airline_sae/encoded_airline_ready_best.csv \\
        --labels    data/airline/airline_labels.csv \\
        --label_col airline_sentiment \\
        --output    runs/airline_sae/analysis/code_entropy.csv \\
        [--top_patterns 20]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoded",      required=True)
    p.add_argument("--labels",       required=True)
    p.add_argument("--label_col",    type=str, default="label_name")
    p.add_argument("--output",       required=True)
    p.add_argument("--top_patterns", type=int, default=20,
                   help="How many most-frequent patterns to display")
    return p.parse_args()


def load_data(args):
    enc = pd.read_csv(args.encoded)
    bit_cols = sorted([c for c in enc.columns if c.startswith("bit_")])
    lab = pd.read_csv(args.labels)
    if args.label_col not in lab.columns:
        available = [c for c in lab.columns if c != "id"]
        raise ValueError(f"--label_col '{args.label_col}' not found. Available: {available}")
    lab["label_name"] = lab[args.label_col].astype(str).str.strip()
    df = enc.merge(lab[["id", "label_name"]], on="id", how="inner")
    return df, bit_cols


def entropy_from_counts(counts: np.ndarray) -> float:
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def main():
    args = parse_args()
    df, bit_cols = load_data(args)
    categories = sorted(df["label_name"].unique())
    n_docs = len(df)
    k = len(bit_cols)

    # ── Build pattern strings ─────────────────────────────────────────────────
    df["pattern"] = df[bit_cols].astype(int).apply(
        lambda row: "".join(row.astype(str)), axis=1
    )

    # ── Global stats ──────────────────────────────────────────────────────────
    pattern_counts = df["pattern"].value_counts()
    n_unique   = len(pattern_counts)
    n_possible = 2 ** k
    h_patterns = entropy_from_counts(pattern_counts.values)
    h_max      = math.log2(n_unique) if n_unique > 1 else 0.0
    n_active_per_doc = df[bit_cols].sum(axis=1)
    efficiency = h_patterns / h_max if h_max > 0 else 0.0
    n_singletons = int((pattern_counts == 1).sum())

    print(f"{'='*70}")
    print(f"CODE ENTROPY & UNIQUENESS ANALYSIS")
    print(f"{'='*70}")
    print(f"Documents        : {n_docs:,}")
    print(f"Factors (K)      : {k}")
    print(f"Possible patterns: {n_possible:,}  (2^{k})")
    print(f"Observed patterns: {n_unique:,}  ({n_unique/n_possible*100:.4f}% of space)")
    print(f"")
    print(f"Pattern entropy  : {h_patterns:.4f} bits")
    print(f"Max entropy      : {h_max:.4f} bits  (uniform over {n_unique} patterns)")
    print(f"Efficiency       : {efficiency*100:.1f}%  (how uniformly patterns are used)")
    print(f"")
    print(f"Singleton patterns (used by exactly 1 doc): "
          f"{n_singletons:,}  "
          f"({n_singletons/n_unique*100:.1f}% of unique patterns)")
    print(f"Mean n_active per doc: {n_active_per_doc.mean():.2f}  "
          f"(std={n_active_per_doc.std():.2f})")

    # ── Concentration: top-N coverage ────────────────────────────────────────
    print(f"\n── Top-{args.top_patterns} pattern concentration ──────────────────────────────────")
    top_n_count = pattern_counts.head(args.top_patterns).sum()
    print(f"  Top {args.top_patterns} patterns cover "
          f"{top_n_count:,} / {n_docs:,} docs "
          f"({top_n_count/n_docs*100:.1f}%)")
    patterns_for_coverage = {}
    for thresh in [0.50, 0.75, 0.90]:
        cumsum = pattern_counts.cumsum()
        n_needed = int((cumsum <= thresh * n_docs).sum()) + 1
        patterns_for_coverage[f"patterns_covering_{int(thresh * 100)}pct_docs"] = n_needed
        print(f"  {thresh*100:.0f}% of docs covered by top {n_needed} patterns")

    # ── Per-category pattern entropy ──────────────────────────────────────────
    print(f"\n── Per-category pattern entropy ─────────────────────────────────────────────")
    print(f"  {'category':<30} {'n_docs':>7} {'n_unique':>9} {'H(pattern)':>11} "
          f"{'efficiency':>11}")
    print(f"  {'-'*72}")
    cat_rows = []
    for cat in categories:
        sub = df[df["label_name"] == cat]
        cat_counts = sub["pattern"].value_counts()
        n_cat = len(sub)
        n_cat_unique = len(cat_counts)
        h_cat = entropy_from_counts(cat_counts.values)
        h_cat_max = math.log2(n_cat_unique) if n_cat_unique > 1 else 0.0
        eff = h_cat / h_cat_max if h_cat_max > 0 else 0.0
        print(f"  {cat:<30} {n_cat:>7,} {n_cat_unique:>9,} {h_cat:>11.4f} {eff*100:>10.1f}%")
        cat_rows.append({
            "category": cat,
            "n_docs": n_cat,
            "n_unique_patterns": n_cat_unique,
            "H_pattern_bits": round(h_cat, 4),
            "H_max_bits": round(h_cat_max, 4),
            "efficiency_pct": round(eff * 100, 2),
        })

    # ── Top patterns table ────────────────────────────────────────────────────
    print(f"\n── Top {args.top_patterns} most frequent patterns ──────────────────────────────────")
    print(f"  {'rank':>4}  {'count':>6}  {'pct':>5}  {'pattern':<{k+2}}  "
          f"{'n_active':>8}  dominant_category")
    print(f"  {'-'*75}")

    pattern_rows = []
    for rank, (pat, cnt) in enumerate(pattern_counts.head(args.top_patterns).items(), 1):
        n_active_pat = sum(int(b) for b in pat)
        sub = df[df["pattern"] == pat]
        dom_cat = sub["label_name"].value_counts().index[0]
        dom_pct = sub["label_name"].value_counts().iloc[0] / len(sub) * 100
        print(f"  {rank:>4}  {cnt:>6,}  {cnt/n_docs*100:>4.1f}%  {pat:<{k+2}}  "
              f"{n_active_pat:>8}  {dom_cat} ({dom_pct:.0f}%)")
        pattern_rows.append({
            "rank": rank,
            "pattern": pat,
            "count": cnt,
            "pct": round(cnt / n_docs * 100, 2),
            "n_active": n_active_pat,
            "dominant_category": dom_cat,
            "dominant_pct": round(dom_pct, 1),
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Save global stats + top patterns
    pd.DataFrame(pattern_rows).to_csv(out, index=False)

    global_path = out.parent / (out.stem + "_global.csv")
    global_row = {
        "n_documents": n_docs,
        "k_factors": k,
        "n_possible_patterns": n_possible,
        "n_observed_patterns": n_unique,
        "observed_space_pct": n_unique / n_possible * 100,
        "pattern_entropy_bits": h_patterns,
        "max_observed_entropy_bits": h_max,
        "efficiency_pct": efficiency * 100,
        "n_singleton_patterns": n_singletons,
        "singleton_patterns_pct": n_singletons / n_unique * 100,
        "mean_active_factors_per_document": float(n_active_per_doc.mean()),
        "std_active_factors_per_document": float(n_active_per_doc.std()),
        "top_patterns_requested": args.top_patterns,
        "top_patterns_document_coverage_pct": top_n_count / n_docs * 100,
        **patterns_for_coverage,
    }
    pd.DataFrame([global_row]).to_csv(global_path, index=False)

    # Save per-category entropy
    cat_path = out.parent / (out.stem + "_per_category.csv")
    pd.DataFrame(cat_rows).to_csv(cat_path, index=False)

    print(f"\nSaved pattern table to  : {out}")
    print(f"Saved global code stats : {global_path}")
    print(f"Saved per-category stats: {cat_path}")


if __name__ == "__main__":
    main()
