#!/usr/bin/env python3
"""
Step 1: Marginal factor-category association table.

For each factor independently, computes:
  - P(bit=1 | category)  — how often is this factor active within each category
  - P(category | bit=1)  — given this factor fires, what category is it?
  - Cramér's V           — overall association strength (0=none, 1=perfect)
  - Chi-squared p-value  — statistical significance of the association
  - Role classification  — 'discriminative' / 'universal' / 'silent' / 'modifier'

Role rules:
  silent        → P(bit=1) < 0.02  (barely ever fires)
  universal     → P(bit=1) > 0.50  (fires for most documents regardless of category)
  discriminative→ Cramér's V > 0.3  (strongly tied to one or more categories)
  modifier      → everything else   (fires moderately, weakly associated)

Usage:
    python analyze_1_marginals.py \
        --encoded   runs/agnews_sae/encoded_agnews_all_ready_best.csv \
        --labels    data/agnews/agnews_all.csv \
        --output    runs/agnews_sae/marginal_associations.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROLE_SILENT        = "silent"
ROLE_UNIVERSAL     = "universal"
ROLE_DISCRIMINATIVE = "discriminative"
ROLE_MODIFIER      = "modifier"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoded",  required=True)
    p.add_argument("--labels",    required=True)
    p.add_argument("--label_col", type=str, default="label_name",
                   help="Column in the labels CSV to use as category. "  
                        "e.g. label_name, airline_sentiment, negativereason, airline")
    p.add_argument("--output",   required=True)
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


def cramers_v(contingency: np.ndarray) -> float:
    n = contingency.sum()
    if n == 0:
        return 0.0
    # Skip if any row or column is entirely zero — chi2 would fail
    if contingency.min() == 0:
        row_sums = contingency.sum(axis=1)
        col_sums = contingency.sum(axis=0)
        if (row_sums == 0).any() or (col_sums == 0).any():
            return 0.0
    try:
        chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    except ValueError:
        return 0.0
    r, c = contingency.shape
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - (c - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def classify_role(marginal: float, v: float) -> str:
    if marginal < 0.02:
        return ROLE_SILENT
    if marginal > 0.50:
        return ROLE_UNIVERSAL
    if v > 0.30:
        return ROLE_DISCRIMINATIVE
    return ROLE_MODIFIER


def main():
    args = parse_args()
    df, bit_cols = load_data(args)
    categories = sorted(df["label_name"].unique())
    n_total = len(df)

    rows = []
    for col in bit_cols:
        bit_idx = int(col.split("_")[1])
        active  = df[col].astype(int)
        marginal = active.mean()

        # Contingency table: rows = bit (0/1), cols = categories
        ct = pd.crosstab(active, df["label_name"]).reindex(
            index=[0, 1], columns=categories, fill_value=0
        ).values  # shape (2, n_categories)

        # Chi-squared and Cramér's V
        try:
            chi2, pval, _, _ = chi2_contingency(ct, correction=False)
        except ValueError:
            chi2, pval = 0.0, 1.0
        v = cramers_v(ct)

        # P(category | bit=1) — given factor fires, what do we see?
        active_rows = ct[1]   # bit=1 row
        n_active = active_rows.sum()
        p_cat_given_active = active_rows / max(1, n_active)
        dominant_cat = categories[p_cat_given_active.argmax()]
        dominant_pct = p_cat_given_active.max() * 100

        # P(bit=1 | category) — per-category activation rate
        p_active_given_cat = {}
        for cat in categories:
            cat_mask = df["label_name"] == cat
            p_active_given_cat[cat] = df.loc[cat_mask, col].mean()

        role = classify_role(marginal, v)

        row = {
            "bit":           col,
            "bit_idx":       bit_idx,
            "marginal":      round(float(marginal), 4),
            "n_active":      int(n_active),
            "cramers_v":     round(v, 4),
            "chi2":          round(float(chi2), 2),
            "pval":          float(pval),
            "dominant_cat":  dominant_cat,
            "dominant_pct":  round(dominant_pct, 1),
            "role":          role,
        }
        for cat in categories:
            row[f"P(bit=1|{cat})"]   = round(p_active_given_cat[cat], 4)
            row[f"P({cat}|bit=1)"]   = round(float(p_cat_given_active[categories.index(cat)]), 4)

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"{'='*90}")
    print(f"MARGINAL FACTOR-CATEGORY ASSOCIATIONS  (n={n_total:,} documents)")
    print(f"Sorted by Cramér's V (association strength) descending")
    print(f"{'='*90}")

    print(f"\n── Per-factor summary ────────────────────────────────────────────────────────")
    print(f"{'bit':<10} {'marg':>6} {'V':>6} {'p':>8} {'role':<15} "
          f"{'dominant_cat':<12} {'dom%':>5}")
    print("-" * 75)
    for _, row in result.iterrows():
        sig = "***" if row["pval"] < 0.001 else ("**" if row["pval"] < 0.01
              else ("*" if row["pval"] < 0.05 else "   "))
        print(f"{row['bit']:<10} {row['marginal']:>6.3f} {row['cramers_v']:>6.3f} "
              f"{row['pval']:>8.2e}{sig} {row['role']:<15} "
              f"{row['dominant_cat']:<12} {row['dominant_pct']:>4.0f}%")

    # ── Role breakdown ────────────────────────────────────────────────────────
    print(f"\n── Role breakdown ────────────────────────────────────────────────────────────")
    for role in [ROLE_DISCRIMINATIVE, ROLE_MODIFIER, ROLE_UNIVERSAL, ROLE_SILENT]:
        sub = result[result["role"] == role]
        bits = ", ".join(sub["bit"].tolist()) or "(none)"
        print(f"  {role:<15} ({len(sub):>2} factors): {bits}")

    # ── P(bit=1 | category) heatmap-style table ───────────────────────────────
    print(f"\n── P(bit=1 | category)  — activation rate per category ──────────────────────")
    header = f"{'bit':<10}" + "".join(f"  {c[:8]:>8}" for c in categories) + "   V      role"
    print(header)
    print("-" * len(header))
    for _, row in result.iterrows():
        vals = "".join(f"  {row[f'P(bit=1|{c})']:>8.3f}" for c in categories)
        print(f"{row['bit']:<10}{vals}   {row['cramers_v']:.3f}  {row['role']}")

    # ── P(category | bit=1) — given factor fires ─────────────────────────────
    print(f"\n── P(category | bit=1)  — what category does this factor predict? ────────────")
    print(header)
    print("-" * len(header))
    for _, row in result.iterrows():
        vals = "".join(f"  {row[f'P({c}|bit=1)']:>8.3f}" for c in categories)
        print(f"{row['bit']:<10}{vals}   {row['cramers_v']:.3f}  {row['role']}")

    # ── Save ──────────────────────────────────────────────────────────────────
    result.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
