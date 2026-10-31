#!/usr/bin/env python3
"""
Step 8: Factor co-occurrence structure via Jaccard similarity.

Computes the K×K Jaccard similarity matrix between all factor pairs:
  J(i, j) = |docs where both i=1 and j=1| / |docs where i=1 OR j=1|

Bounded [0, 1] and symmetric. More interpretable than raw PMI for clustering
because it accounts for marginal activation rates directly.

Outputs:
  - Jaccard matrix CSV
  - Ranked edge list (all pairs, sorted by J descending)
  - Console summary of top co-occurring pairs and mutual exclusions
  - Cluster assignments (agglomerative on 1-J distance)
  - Heatmap PNG (requires matplotlib + scipy)

Usage:
    python analyze_8_jaccard.py \\
        --encoded   runs/airline_sae/encoded_airline_ready_best.csv \\
        --labels    data/airline/airline_labels.csv \\
        --label_col airline_sentiment \\
        --output    runs/airline_sae/analysis/jaccard \\
        [--n_clusters 4] [--jaccard_threshold 0.1]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoded",           required=True)
    p.add_argument("--labels",            required=True)
    p.add_argument("--label_col",         type=str, default="label_name")
    p.add_argument("--output",            required=True,
                   help="Output prefix — files written as <output>_matrix.csv, "
                        "<output>_edges.csv, <output>_heatmap.png")
    p.add_argument("--n_clusters",        type=int, default=4)
    p.add_argument("--jaccard_threshold", type=float, default=0.05,
                   help="Minimum J to include in the edge list summary")
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


def jaccard_matrix(B: np.ndarray) -> np.ndarray:
    """
    B: (n_docs, k_factors) binary matrix.
    Returns K×K Jaccard matrix.
    """
    k = B.shape[1]
    J = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            both = np.sum((B[:, i] == 1) & (B[:, j] == 1))
            either = np.sum((B[:, i] == 1) | (B[:, j] == 1))
            jac = both / either if either > 0 else 0.0
            J[i, j] = J[j, i] = jac
    np.fill_diagonal(J, 1.0)
    return J


def main():
    args = parse_args()
    df, bit_cols = load_data(args)
    categories = sorted(df["label_name"].unique())
    n_docs = len(df)
    k = len(bit_cols)

    B = df[bit_cols].astype(int).values
    marginals = B.mean(axis=0)

    print(f"Computing {k}×{k} Jaccard matrix over {n_docs:,} documents ...")
    J = jaccard_matrix(B)

    # ── Save matrix ───────────────────────────────────────────────────────────
    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    matrix_df = pd.DataFrame(J, index=bit_cols, columns=bit_cols)
    matrix_path = Path(str(out_prefix) + "_matrix.csv")
    matrix_df.to_csv(matrix_path)

    # ── Build edge list (upper triangle, excluding diagonal) ──────────────────
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append({
                "factor_a":   bit_cols[i],
                "factor_b":   bit_cols[j],
                "jaccard":    round(float(J[i, j]), 6),
                "marginal_a": round(float(marginals[i]), 4),
                "marginal_b": round(float(marginals[j]), 4),
            })
    edges_df = (pd.DataFrame(edges)
                .sort_values("jaccard", ascending=False)
                .reset_index(drop=True))
    edges_path = Path(str(out_prefix) + "_edges.csv")
    edges_df.to_csv(edges_path, index=False)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"JACCARD CO-OCCURRENCE STRUCTURE  (K={k})")
    print(f"{'='*70}")

    above = edges_df[edges_df["jaccard"] >= args.jaccard_threshold]
    print(f"\nPairs with J >= {args.jaccard_threshold}: {len(above)} / {len(edges_df)}")

    print(f"\n── Top 15 co-occurring factor pairs ─────────────────────────────────────────")
    print(f"  {'factor_a':<10}  {'factor_b':<10}  {'J':>6}  "
          f"{'P(a)':>6}  {'P(b)':>6}")
    print(f"  {'-'*50}")
    for _, row in edges_df.head(15).iterrows():
        print(f"  {row['factor_a']:<10}  {row['factor_b']:<10}  "
              f"{row['jaccard']:>6.4f}  {row['marginal_a']:>6.3f}  {row['marginal_b']:>6.3f}")

    print(f"\n── Bottom 10 pairs (most mutually exclusive) ────────────────────────────────")
    print(f"  {'factor_a':<10}  {'factor_b':<10}  {'J':>6}  "
          f"{'P(a)':>6}  {'P(b)':>6}")
    print(f"  {'-'*50}")
    for _, row in edges_df.tail(10).iloc[::-1].iterrows():
        print(f"  {row['factor_a']:<10}  {row['factor_b']:<10}  "
              f"{row['jaccard']:>6.4f}  {row['marginal_a']:>6.3f}  {row['marginal_b']:>6.3f}")

    # ── Agglomerative clustering on 1-J distance ──────────────────────────────
    try:
        from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
        dist = 1.0 - J
        np.fill_diagonal(dist, 0.0)
        # Convert to condensed form
        from scipy.spatial.distance import squareform
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        cluster_ids = fcluster(Z, t=args.n_clusters, criterion="maxclust")
        print(f"\n── Agglomerative clusters (n={args.n_clusters}, 1-Jaccard distance) ──────────")
        for cid in sorted(set(cluster_ids)):
            members = [bit_cols[i] for i, c in enumerate(cluster_ids) if c == cid]
            margs = [f"{marginals[bit_cols.index(b)]:.3f}" for b in members]
            print(f"  Cluster {cid}: {', '.join(f'{b}(p={m})' for b, m in zip(members, margs))}")
        cluster_col = cluster_ids
    except ImportError:
        print("\n(scipy not available — skipping clustering)")
        cluster_col = [0] * k

    # ── Heatmap ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform

        dist = 1.0 - J
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        order = dendrogram(Z, no_plot=True)["leaves"]

        J_ordered = J[np.ix_(order, order)]
        labels_ordered = [bit_cols[i] for i in order]

        fig, ax = plt.subplots(figsize=(max(8, k * 0.5), max(7, k * 0.45)))
        im = ax.imshow(J_ordered, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(labels_ordered, rotation=90, fontsize=7)
        ax.set_yticklabels(labels_ordered, fontsize=7)
        plt.colorbar(im, ax=ax, label="Jaccard similarity")
        ax.set_title(f"Factor co-occurrence (Jaccard)  —  {args.label_col}", fontsize=11)
        plt.tight_layout()
        heatmap_path = Path(str(out_prefix) + "_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved heatmap to: {heatmap_path}")
    except ImportError:
        print("\n(matplotlib/scipy not available — skipping heatmap)")

    print(f"\nSaved Jaccard matrix to : {matrix_path}")
    print(f"Saved edge list to      : {edges_path}")


if __name__ == "__main__":
    main()
