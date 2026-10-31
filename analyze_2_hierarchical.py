#!/usr/bin/env python3
"""
Step 2: Hierarchical clustering of unique binary patterns by Hamming distance.

Treats each unique binary vector as a point, weighted by how many documents
have that pattern. Computes pairwise Hamming distances between patterns and
runs agglomerative clustering, revealing which patterns differ by only 1-2 bits
(i.e. the same semantic concept with slight variation).

Outputs:
  - dendrogram plot as PNG
  - flat cluster assignments (at a given cut height)
  - per-cluster category composition summary

Requires:  pip install scipy matplotlib

Usage:
    python analyze_2_hierarchical.py \
        --encoded    runs/agnews_sae/encoded_agnews_all_ready_best.csv \
        --labels     data/agnews/agnews_all.csv \
        --output     runs/agnews_sae/hierarchical \
        [--min_count     5]      # ignore rare patterns
        [--n_clusters    8]      # number of flat clusters to cut
        [--linkage       ward]   # ward / complete / average / single
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoded",    required=True)
    p.add_argument("--labels",     required=True)
    p.add_argument("--label_col", type=str, default="label_name",
                   help="Column in the labels CSV to use as category. "
                        "e.g. label_name, airline_sentiment, negativereason, airline")
    p.add_argument("--output",     required=True,
                   help="Output prefix — creates <o>_dendrogram.png, <o>_clusters.csv")
    p.add_argument("--min_count",  type=int,   default=5)
    p.add_argument("--n_clusters", type=int,   default=8)
    p.add_argument("--linkage",    type=str,   default="average",
                   choices=["ward", "complete", "average", "single"])
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


def main():
    args = parse_args()
    df, bit_cols = load_data(args)
    categories = sorted(df["label_name"].unique())
    k = len(bit_cols)
    out_prefix = Path(args.output)

    # ── Build pattern table ───────────────────────────────────────────────────
    bit_matrix = df[bit_cols].astype(int)
    df["pattern"] = bit_matrix.apply(lambda r: "".join(r.astype(str)), axis=1)

    grouped = (
        df.groupby(["pattern", "label_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=categories, fill_value=0)
    )
    grouped["total"] = grouped.sum(axis=1)
    grouped = grouped[grouped["total"] >= args.min_count].copy()
    grouped["dominant"]     = grouped[categories].idxmax(axis=1)
    grouped["dominant_pct"] = grouped[categories].max(axis=1) / grouped["total"] * 100

    n_patterns = len(grouped)
    print(f"Patterns with >= {args.min_count} docs: {n_patterns}")

    if n_patterns < 2:
        print("Not enough patterns to cluster. Lower --min_count.")
        return

    # ── Hamming distance matrix ───────────────────────────────────────────────
    pattern_strs = grouped.index.tolist()
    P = np.array([[int(b) for b in pat] for pat in pattern_strs], dtype=np.float32)
    # P shape: (n_patterns, k)

    # Hamming distance = number of differing bits / k  (normalised)
    # dist(i,j) = sum(|P[i] - P[j]|) / k
    from scipy.spatial.distance import pdist, squareform
    dist_vec  = pdist(P, metric="hamming")   # condensed
    dist_sq   = squareform(dist_vec)         # (n_patterns, n_patterns)

    # ── Hierarchical clustering ───────────────────────────────────────────────
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ward requires euclidean; use euclidean for ward, precomputed for others
    if args.linkage == "ward":
        Z = linkage(P, method="ward", metric="euclidean")
    else:
        Z = linkage(dist_vec, method=args.linkage)

    flat_labels = fcluster(Z, t=args.n_clusters, criterion="maxclust")
    grouped["cluster"] = flat_labels

    # ── Dendrogram ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, n_patterns * 0.4), 7))
    # Label each leaf with dominant category + n_docs
    leaf_labels = [
        f"{grouped.iloc[i]['dominant'][:3]}({grouped.iloc[i]['total']})"
        for i in range(n_patterns)
    ]
    dendrogram(Z, ax=ax, labels=leaf_labels, leaf_rotation=90, leaf_font_size=7,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(f"Hierarchical clustering of binary patterns  "
                 f"(linkage={args.linkage}, {n_patterns} patterns, min_count={args.min_count})")
    ax.set_xlabel("Pattern (dominant_category(n_docs))")
    ax.set_ylabel("Hamming distance")
    plt.tight_layout()
    dendro_path = str(out_prefix) + "_dendrogram.png"
    plt.savefig(dendro_path, dpi=150)
    plt.close()
    print(f"Saved dendrogram to: {dendro_path}")

    # ── Per-cluster summary ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"CLUSTER SUMMARY  ({args.n_clusters} clusters, linkage={args.linkage})")
    print(f"{'='*80}")

    cluster_rows = []
    for cid in sorted(grouped["cluster"].unique()):
        members = grouped[grouped["cluster"] == cid]
        # Aggregate all documents in this cluster
        total   = members["total"].sum()
        cat_counts = {c: int(members[c].sum()) for c in categories}
        dominant   = max(cat_counts, key=cat_counts.get)
        dom_pct    = cat_counts[dominant] / total * 100

        p_arr = np.array([cat_counts[c] for c in categories], dtype=float) / total
        p_nz  = p_arr[p_arr > 0]
        entropy = float(-np.sum(p_nz * np.log2(p_nz))) if len(p_nz) > 1 else 0.0

        # Core bits: bits that are 1 in ALL patterns of this cluster
        pattern_matrix = np.array([[int(b) for b in pat] for pat in members.index])
        core_bits    = np.where(pattern_matrix.min(axis=0) == 1)[0]
        variable_bits = np.where(
            (pattern_matrix.max(axis=0) - pattern_matrix.min(axis=0)) == 1
        )[0]

        core_str     = ",".join(str(i) for i in core_bits)     or "(none)"
        variable_str = ",".join(str(i) for i in variable_bits) or "(none)"

        cluster_rows.append({
            "cluster":       cid,
            "n_patterns":    len(members),
            "n_docs":        int(total),
            "dominant":      dominant,
            "dominant_pct":  round(dom_pct, 1),
            "entropy":       round(entropy, 3),
            "core_bits":     core_str,
            "variable_bits": variable_str,
            **{c: cat_counts[c] for c in categories},
        })

        print(f"\nCluster {cid}  ({len(members)} patterns, {total:,} docs)")
        print(f"  Dominant:      {dominant} ({dom_pct:.0f}%)")
        print(f"  Entropy:       {entropy:.3f}")
        print(f"  Core bits:     [{core_str}]   (present in ALL patterns of this cluster)")
        print(f"  Variable bits: [{variable_str}]   (differ across patterns)")
        cat_str = "  ".join(f"{c}={cat_counts[c]}" for c in categories)
        print(f"  Counts:        {cat_str}")
        print(f"  Patterns:")
        for pat, prow in members.iterrows():
            active = ",".join(str(i) for i, b in enumerate(pat) if b == "1") or "(none)"
            print(f"    [{active}]  n={int(prow['total'])}  "
                  f"{prow['dominant']} ({prow['dominant_pct']:.0f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    cluster_df = pd.DataFrame(cluster_rows)
    clusters_path = str(out_prefix) + "_clusters.csv"
    cluster_df.to_csv(clusters_path, index=False)
    print(f"\nSaved cluster summary to: {clusters_path}")

    # Pattern-level CSV with cluster assignments
    grouped_out = grouped.reset_index()
    grouped_out["active_factors"] = grouped_out["pattern"].apply(
        lambda p: ",".join(str(i) for i, b in enumerate(p) if b == "1") or "(none)"
    )
    patterns_path = str(out_prefix) + "_patterns_with_clusters.csv"
    grouped_out.to_csv(patterns_path, index=False)
    print(f"Saved pattern+cluster table to: {patterns_path}")


if __name__ == "__main__":
    main()
