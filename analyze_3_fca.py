#!/usr/bin/env python3
"""
Step 3: Formal Concept Analysis (FCA) over binary factors.

A formal concept is a pair (documents, factors) where:
  - every document in the set has ALL the factors in the set active
  - no additional document or factor can be added while preserving that property

This gives the mathematically exact union/intersection structure:
  "all documents that share at least factors {4,5} = 1847 docs, 91% Sports"
  "restrict to {4,5,7} = 583 docs, 98% Sports"
  ...forming a lattice from general (few shared bits) to specific (many shared bits)

Requires:  pip install concepts

Usage:
    python analyze_3_fca.py \
        --encoded   runs/agnews_sae/encoded_agnews_all_ready_best.csv \
        --labels    data/agnews/agnews_all.csv \
        --output    runs/agnews_sae/fca_concepts.csv \
        [--min_support 10]    # minimum documents for a concept to be reported
        [--max_concepts 200]  # cap output to avoid combinatorial explosion
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoded",      required=True)
    p.add_argument("--labels",       required=True)
    p.add_argument("--label_col", type=str, default="label_name",
                   help="Column in the labels CSV to use as category. "
                        "e.g. label_name, airline_sentiment, negativereason, airline")
    p.add_argument("--output",       required=True)
    p.add_argument("--min_support",  type=int,   default=10)
    p.add_argument("--max_concepts", type=int,   default=200)
    return p.parse_args()


def load_data(args):
    enc = pd.read_csv(args.encoded)
    bit_cols = sorted([c for c in enc.columns if c.startswith("bit_")])
    lab = pd.read_csv(args.labels)
    label_col = args.label_col
    if label_col not in lab.columns:
        available = [c for c in lab.columns if c != "id"]
        raise ValueError(f"--label_col '{label_col}' not found. Available columns: {available}")
    lab["label_name"] = lab[label_col].astype(str).str.strip()
    df = enc.merge(lab[["id", "label_name"]], on="id", how="inner")
    return df, bit_cols


def compute_concepts_manual(bit_matrix: np.ndarray, min_support: int):
    """
    Enumerate formal concepts via the Next Closure algorithm (lightweight implementation).
    bit_matrix: (n_docs, k_factors) binary numpy array
    Returns list of (extent_indices, intent_set) sorted by |extent| descending.

    For large K this can be slow — we cap with max_concepts in the caller.
    """
    n, k = bit_matrix.shape

    def closure(doc_set):
        """Given a set of doc indices, return the maximal shared factor set."""
        if not doc_set:
            return frozenset(range(k))
        shared = np.ones(k, dtype=bool)
        for d in doc_set:
            shared &= bit_matrix[d].astype(bool)
        return frozenset(np.where(shared)[0])

    def extent(factor_set):
        """Given a set of factor indices, return all docs that have ALL of them active."""
        if not factor_set:
            return list(range(n))
        mask = np.ones(n, dtype=bool)
        for f in factor_set:
            mask &= bit_matrix[:, f].astype(bool)
        return list(np.where(mask)[0])

    # Start from every singleton factor set and build upward
    seen_intents = set()
    concepts = []

    # Also include the empty intent (all docs)
    all_docs = list(range(n))
    all_intent = closure(all_docs)
    if len(all_docs) >= min_support:
        seen_intents.add(all_intent)
        concepts.append((all_docs, all_intent))

    # For each subset of active factors (via all unique row patterns), derive concept
    # Use unique binary rows as seeds to avoid exponential enumeration
    unique_rows = np.unique(bit_matrix, axis=0)
    for row in unique_rows:
        active = frozenset(np.where(row)[0])
        if not active:
            continue
        # Grow to concept: find all docs that have ALL these factors
        ext = extent(active)
        if len(ext) < min_support:
            continue
        # Compute closure (maximal shared factors among these docs)
        intent = closure(ext)
        if intent in seen_intents:
            continue
        seen_intents.add(intent)
        # Re-derive extent from intent (may be larger than original ext)
        full_ext = extent(intent)
        if len(full_ext) >= min_support:
            concepts.append((full_ext, intent))

    # Also seed from each single factor
    for f in range(k):
        single = frozenset([f])
        ext = extent(single)
        if len(ext) < min_support:
            continue
        intent = closure(ext)
        if intent in seen_intents:
            continue
        seen_intents.add(intent)
        full_ext = extent(intent)
        if len(full_ext) >= min_support:
            concepts.append((full_ext, intent))

    concepts.sort(key=lambda x: -len(x[0]))
    return concepts


def main():
    args = parse_args()
    df, bit_cols = load_data(args)
    categories = sorted(df["label_name"].unique())
    k = len(bit_cols)

    bit_matrix = df[bit_cols].values.astype(np.int8)

    print(f"Documents: {len(df):,}  |  Factors: {k}")
    print(f"Computing formal concepts (min_support={args.min_support})...")

    concepts = compute_concepts_manual(bit_matrix, args.min_support)
    print(f"Found {len(concepts):,} concepts before cap.")
    concepts = concepts[:args.max_concepts]

    # ── Build results table ───────────────────────────────────────────────────
    rows = []
    for ext, intent in concepts:
        doc_labels = df.iloc[ext]["label_name"]
        counts     = doc_labels.value_counts()
        total      = len(ext)
        dominant   = counts.index[0]
        dom_pct    = counts.iloc[0] / total * 100

        p = np.array([counts.get(c, 0) for c in categories], dtype=float) / total
        p_nz = p[p > 0]
        entropy = float(-np.sum(p_nz * np.log2(p_nz))) if len(p_nz) > 1 else 0.0

        intent_str   = ",".join(str(f) for f in sorted(intent))
        intent_names = ",".join(bit_cols[f] for f in sorted(intent))

        row = {
            "n_factors_shared": len(intent),
            "shared_factor_indices": intent_str,
            "shared_factor_names":   intent_names,
            "n_docs":       total,
            "dominant":     dominant,
            "dominant_pct": round(dom_pct, 1),
            "entropy":      round(entropy, 3),
        }
        for c in categories:
            row[c]          = int(counts.get(c, 0))
            row[f"pct_{c}"] = round(counts.get(c, 0) / total * 100, 1)
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["dominant_pct", "n_docs"], ascending=[False, False]
    ).reset_index(drop=True)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"FORMAL CONCEPTS  (min_support={args.min_support}, top {len(result)} shown)")
    print(f"Sorted by dominant category % descending")
    print(f"{'='*90}")
    header = (f"{'#':>4}  {'n_docs':>7}  {'n_bits':>6}  {'dominant':<10}  {'dom%':>5}  "
              f"{'entropy':>7}  shared factors")
    print(header)
    print("-" * len(header))
    for i, row in result.iterrows():
        print(f"{i+1:>4}  {row['n_docs']:>7,}  {row['n_factors_shared']:>6}  "
              f"{row['dominant']:<10}  {row['dominant_pct']:>4.0f}%  "
              f"{row['entropy']:>7.3f}  {row['shared_factor_indices']}")

    print(f"\nInterpretation:")
    print(f"  Each row = a maximal group of documents that ALL share the listed factors.")
    print(f"  Moving DOWN the table = more factors shared = smaller, purer groups.")
    print(f"  The top row is the most general concept (fewest shared factors, most docs).")

    # ── Category summary: which factors define each category? ─────────────────
    print(f"\n── Per-category top concepts ────────────────────────────────────────────────")
    for cat in categories:
        cat_concepts = result[result["dominant"] == cat].head(5)
        print(f"\n  {cat}:")
        for _, row in cat_concepts.iterrows():
            print(f"    n={row['n_docs']:>5,}  {row['dominant_pct']:>4.0f}%  "
                  f"shared bits: [{row['shared_factor_indices']}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    col_order = (["n_factors_shared", "shared_factor_indices", "shared_factor_names",
                  "n_docs", "dominant", "dominant_pct", "entropy"]
                 + categories + [f"pct_{c}" for c in categories])
    result[col_order].to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
