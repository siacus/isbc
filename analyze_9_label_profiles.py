#!/usr/bin/env python3
"""
Step 9: Label-conditional factor profiles.

For each label value, computes the mean probability vector across all documents
in that class — giving an (n_classes × K) matrix where each row is the
"factor fingerprint" of that class.

From this matrix, computes:
  - Per-class factor fingerprint (mean activation per factor)
  - Class similarity matrix: cosine similarity between class fingerprints
  - Class-exclusive factors: activated strongly by one class but not others
  - Class-shared factors: activated similarly across all classes
  - Factor discrimination score: std of per-class activation (high = discriminative)

Useful for:
  - Understanding what binary profile each label value has
  - Seeing which classes are most similar in factor space
  - Comparing how cleanly separated classes are (e.g. airline vs. sentiment)

Usage:
    python analyze_9_label_profiles.py \\
        --encoded   runs/airline_sae/encoded_airline_ready_best.csv \\
        --labels    data/airline/airline_labels.csv \\
        --label_col airline_sentiment \\
        --output    runs/airline_sae/analysis/label_profiles.csv \\
        [--exclusive_threshold 0.2]   # min gap to dominant class to call exclusive
        [--shared_threshold 0.05]     # max std across classes to call shared
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoded",              required=True)
    p.add_argument("--labels",               required=True)
    p.add_argument("--label_col",            type=str, default="label_name")
    p.add_argument("--output",               required=True)
    p.add_argument("--exclusive_threshold",  type=float, default=0.20,
                   help="A factor is class-exclusive if its activation in the "
                        "dominant class exceeds the second highest by this margin")
    p.add_argument("--shared_threshold",     type=float, default=0.05,
                   help="A factor is shared if std of per-class activation < this")
    p.add_argument("--use_probs",            action="store_true", default=False,
                   help="Use soft prob_* columns instead of hard bit_* columns")
    return p.parse_args()


def load_data(args):
    enc = pd.read_csv(args.encoded)
    if args.use_probs:
        feat_cols = sorted([c for c in enc.columns if c.startswith("prob_")])
        feat_type = "prob"
    else:
        feat_cols = sorted([c for c in enc.columns if c.startswith("bit_")])
        feat_type = "bit"
    if not feat_cols:
        raise ValueError(f"No {'prob' if args.use_probs else 'bit'}_* columns found.")
    lab = pd.read_csv(args.labels)
    if args.label_col not in lab.columns:
        available = [c for c in lab.columns if c != "id"]
        raise ValueError(f"--label_col '{args.label_col}' not found. Available: {available}")
    lab["label_name"] = lab[args.label_col].astype(str).str.strip()
    df = enc.merge(lab[["id", "label_name"]], on="id", how="inner")
    return df, feat_cols, feat_type


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    args = parse_args()
    df, feat_cols, feat_type = load_data(args)
    categories = sorted(df["label_name"].unique())
    n_docs = len(df)
    k = len(feat_cols)

    print(f"Documents : {n_docs:,}")
    print(f"Factors   : {k}  (using {feat_type}_* columns)")
    print(f"Categories: {len(categories)}  {categories}")

    # ── Build (n_classes × K) fingerprint matrix ──────────────────────────────
    fingerprints = {}
    class_sizes = {}
    for cat in categories:
        sub = df[df["label_name"] == cat][feat_cols].values.astype(float)
        fingerprints[cat] = sub.mean(axis=0)
        class_sizes[cat] = len(sub)

    fp_matrix = np.stack([fingerprints[c] for c in categories])  # (n_classes, K)

    # ── Class similarity matrix ───────────────────────────────────────────────
    sim_matrix = np.zeros((len(categories), len(categories)))
    for i, ca in enumerate(categories):
        for j, cb in enumerate(categories):
            sim_matrix[i, j] = cosine_sim(fingerprints[ca], fingerprints[cb])

    print(f"\n{'='*70}")
    print(f"LABEL-CONDITIONAL FACTOR PROFILES  —  {args.label_col}")
    print(f"{'='*70}")

    # ── Per-factor discrimination score ──────────────────────────────────────
    factor_std = fp_matrix.std(axis=0)          # std across classes per factor
    factor_mean = fp_matrix.mean(axis=0)        # mean across classes per factor

    # ── Print fingerprint table ───────────────────────────────────────────────
    print(f"\n── Factor fingerprints: mean activation P(factor=1 | class) ─────────────────")
    header = f"{'factor':<10}" + "".join(f"  {c[:10]:>10}" for c in categories) + \
             f"  {'std':>6}  {'role':<12}"
    print(header)
    print("-" * len(header))

    factor_roles = []
    rows = []
    for fi, col in enumerate(feat_cols):
        vals = fp_matrix[:, fi]
        std = float(factor_std[fi])
        sorted_vals = np.sort(vals)[::-1]
        gap = float(sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else float(sorted_vals[0])
        dominant_idx = int(np.argmax(vals))
        dominant_cat = categories[dominant_idx]

        if std < args.shared_threshold:
            role = "shared"
        elif gap >= args.exclusive_threshold:
            role = f"excl_{dominant_cat[:8]}"
        else:
            role = "discriminative"

        vals_str = "".join(f"  {v:>10.3f}" for v in vals)
        print(f"{col:<10}{vals_str}  {std:>6.3f}  {role:<12}")
        factor_roles.append(role)

        row = {"factor": col, "std_across_classes": round(std, 4), "role": role,
               "dominant_class": dominant_cat, "gap_to_second": round(gap, 4)}
        for ci, cat in enumerate(categories):
            row[f"mean_{cat}"] = round(float(fp_matrix[ci, fi]), 4)
        rows.append(row)

    # ── Class similarity matrix ───────────────────────────────────────────────
    print(f"\n── Class fingerprint cosine similarity ──────────────────────────────────────")
    header2 = f"{'':30}" + "".join(f"  {c[:10]:>10}" for c in categories)
    print(header2)
    for i, ca in enumerate(categories):
        vals_str = "".join(f"  {sim_matrix[i, j]:>10.3f}" for j in range(len(categories)))
        n = class_sizes[ca]
        print(f"  {ca[:28]:<28}{vals_str}  (n={n:,})")

    # ── Most similar / dissimilar class pairs ─────────────────────────────────
    pairs = []
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            pairs.append((categories[i], categories[j], sim_matrix[i, j]))
    pairs.sort(key=lambda x: -x[2])

    if len(pairs) > 0:
        print(f"\n  Most similar pair   : {pairs[0][0]} ↔ {pairs[0][1]}  (cos={pairs[0][2]:.4f})")
        print(f"  Most different pair : {pairs[-1][0]} ↔ {pairs[-1][1]}  (cos={pairs[-1][2]:.4f})")

    # ── Role breakdown ────────────────────────────────────────────────────────
    print(f"\n── Factor role breakdown ────────────────────────────────────────────────────")
    shared_factors = [feat_cols[i] for i, r in enumerate(factor_roles) if r == "shared"]
    excl_factors   = [feat_cols[i] for i, r in enumerate(factor_roles) if r.startswith("excl_")]
    disc_factors   = [feat_cols[i] for i, r in enumerate(factor_roles) if r == "discriminative"]
    print(f"  shared        ({len(shared_factors):2d}): {', '.join(shared_factors) or '(none)'}")
    print(f"  exclusive     ({len(excl_factors):2d}): {', '.join(excl_factors) or '(none)'}")
    print(f"  discriminative({len(disc_factors):2d}): {', '.join(disc_factors) or '(none)'}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(rows).sort_values("std_across_classes", ascending=False)
    result_df.to_csv(out, index=False)

    sim_path = out.parent / (out.stem + "_class_similarity.csv")
    pd.DataFrame(sim_matrix, index=categories, columns=categories).to_csv(sim_path)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(max(14, k * 0.6 + 6), max(5, len(categories) * 0.5 + 3)))

        # Left: fingerprint heatmap (categories × factors)
        ax = axes[0]
        im = ax.imshow(fp_matrix, vmin=0, vmax=fp_matrix.max(), cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(k))
        ax.set_xticklabels(feat_cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=8)
        plt.colorbar(im, ax=ax, label="mean activation")
        ax.set_title(f"Factor fingerprints by class\n({args.label_col})", fontsize=10)

        # Right: class similarity heatmap
        ax2 = axes[1]
        im2 = ax2.imshow(sim_matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
        ax2.set_yticks(range(len(categories)))
        ax2.set_yticklabels(categories, fontsize=8)
        for i in range(len(categories)):
            for j in range(len(categories)):
                ax2.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center",
                         fontsize=7, color="black" if sim_matrix[i,j] < 0.7 else "white")
        plt.colorbar(im2, ax=ax2, label="cosine similarity")
        ax2.set_title("Class fingerprint similarity", fontsize=10)

        plt.tight_layout()
        heatmap_path = out.parent / (out.stem + "_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved heatmap to: {heatmap_path}")
    except ImportError:
        print("\n(matplotlib not available — skipping heatmap)")

    print(f"Saved factor profiles to    : {out}")
    print(f"Saved class similarity to   : {sim_path}")


if __name__ == "__main__":
    main()
