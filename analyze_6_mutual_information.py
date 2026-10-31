#!/usr/bin/env python3
"""
Step 6: Mutual information between each factor and the label.

For each factor k, computes I(bit_k ; label) — how much knowing whether
factor k is active reduces uncertainty about the document's category.

Also computes:
  - Normalised MI (NMI): I / H(label), bounded [0,1]
  - Label-to-factor uncertainty coefficient: I / H(bit), bounded [0,1]. This
    is the primary quantity for comparing which label column best explains a
    factor's activation in the cross-label summary.
  - Per-category specificity: for each label value, the pointwise MI
    PMI(bit=1, cat) = log P(bit=1, cat) / (P(bit=1) * P(cat))
  - Factor ranking by MI (most informative first)

Useful for:
  - Identifying which factors carry semantic signal vs. noise
  - Comparing MI across label columns (e.g. airline_sentiment vs sentiment_fine)
    to see which label dimension the factors align with most
  - Spotting dead factors (MI ~ 0) that can be pruned

Usage:
    python analyze_6_mutual_information.py \\
        --encoded   runs/airline_sae/encoded_airline_ready_best.csv \\
        --labels    data/airline/airline_labels.csv \\
        --label_col airline_sentiment \\
        --output    runs/airline_sae/analysis/mutual_information.csv

    # Compare two label columns:
    python analyze_6_mutual_information.py ... --label_col airline_sentiment
    python analyze_6_mutual_information.py ... --label_col sentiment_fine
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoded",   required=True, help="CSV from --mode encode")
    p.add_argument("--labels",    required=True, help="CSV with id + label columns")
    p.add_argument("--label_col", type=str, default="label_name",
                   help="Column to use as category label")
    p.add_argument("--output",    required=True, help="Output CSV path")
    p.add_argument("--top_n",     type=int, default=None,
                   help="Print only top N factors by MI (default: all)")
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
    n_lost = len(enc) - len(df)
    if n_lost:
        print(f"Warning: {n_lost} documents had no matching label and were dropped.")
    return df, bit_cols


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits from a probability array."""
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def mutual_information(bit: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Compute I(bit ; label) in bits.
    Returns (mi, nmi_label, label_explains_factor, pmi_per_category_dict,
    h_label, h_factor)
    """
    categories = sorted(set(labels))
    n = len(bit)

    # Marginal P(bit=1) and P(bit=0)
    p_bit = np.array([np.mean(bit == v) for v in [0, 1]])
    # Marginal P(cat)
    p_cat = np.array([np.mean(labels == c) for c in categories])
    h_label = entropy(p_cat)
    h_factor = entropy(p_bit)

    mi = 0.0
    pmi_dict = {}
    for bi, pbi in enumerate(p_bit):
        if pbi == 0:
            continue
        for ci, cat in enumerate(categories):
            pci = p_cat[ci]
            if pci == 0:
                continue
            p_joint = np.mean((bit == bi) & (labels == cat))
            if p_joint == 0:
                continue
            mi += p_joint * math.log2(p_joint / (pbi * pci))
            if bi == 1:  # pointwise MI for bit=1 only
                pmi_dict[cat] = math.log2(p_joint / (pbi * pci)) if (pbi * pci) > 0 else 0.0

    nmi = mi / h_label if h_label > 0 else 0.0
    label_explains_factor = mi / h_factor if h_factor > 0 else 0.0
    return mi, nmi, label_explains_factor, pmi_dict, h_label, h_factor


def main():
    args = parse_args()
    df, bit_cols = load_data(args)
    categories = sorted(df["label_name"].unique())
    n_total = len(df)
    labels_arr = df["label_name"].values

    print(f"Documents : {n_total:,}")
    print(f"Factors   : {len(bit_cols)}")
    print(f"Categories: {len(categories)}  {categories}")

    # Label entropy (upper bound on MI)
    p_cat = np.array([np.mean(labels_arr == c) for c in categories])
    h_label = entropy(p_cat)
    print(f"H(label)  : {h_label:.4f} bits  (max possible MI)")

    rows = []
    for col in bit_cols:
        bit_arr = df[col].astype(int).values
        mi, nmi, label_explains_factor, pmi_dict, _, h_factor = mutual_information(
            bit_arr, labels_arr
        )
        marginal = float(bit_arr.mean())

        row = {
            "bit":      col,
            "bit_idx":  int(col.split("_")[1]),
            "marginal": round(marginal, 4),
            "MI_bits":  round(mi, 6),
            "NMI":      round(nmi, 6),
            "H_factor_bits": round(h_factor, 6),
            "label_explains_factor": round(label_explains_factor, 6),
            "H_label_bits": round(h_label, 6),
        }
        for cat in categories:
            row[f"PMI_{cat}"] = round(pmi_dict.get(cat, 0.0), 4)
        rows.append(row)

    result = (pd.DataFrame(rows)
              .sort_values("MI_bits", ascending=False)
              .reset_index(drop=True))

    # ── Print ─────────────────────────────────────────────────────────────────
    top_n = args.top_n or len(result)
    print(f"\n{'='*80}")
    print(f"MUTUAL INFORMATION: factor ↔ {args.label_col}")
    print(f"H(label) = {h_label:.4f} bits  |  NMI = MI / H(label)  |  "
          f"label_explains_factor = MI / H(factor)")
    print(f"{'='*80}")
    print(f"\n{'bit':<10} {'marg':>6} {'MI':>8} {'NMI':>7} {'L→F':>7}   "
          "pointwise MI per category (bit=1)")
    print(f"{'':10} {'':6} {'(bits)':>8} {'':>7}   " +
          "  ".join(f"{c[:10]:>10}" for c in categories))
    print("-" * (38 + 12 * len(categories)))

    for _, row in result.head(top_n).iterrows():
        pmi_vals = "  ".join(f"{row[f'PMI_{c}']:>10.3f}" for c in categories)
        print(f"{row['bit']:<10} {row['marginal']:>6.3f} {row['MI_bits']:>8.5f} "
              f"{row['NMI']:>7.4f} {row['label_explains_factor']:>7.4f}   {pmi_vals}")

    # Summary stats
    dead = (result["MI_bits"] < 1e-4).sum()
    print(f"\nDead factors (MI < 0.0001 bits): {dead} / {len(result)}")
    print(f"Top factor MI: {result['MI_bits'].iloc[0]:.5f} bits "
          f"({result['NMI'].iloc[0]*100:.1f}% of label entropy)")
    print(f"Mean MI across factors: {result['MI_bits'].mean():.5f} bits")

    result.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
