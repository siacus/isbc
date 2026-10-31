#!/usr/bin/env python3
"""
Step 5: Bit interaction network via Pointwise Mutual Information (PMI).

For each pair of factors (i, j), compute:
  PMI(i,j) = log2[ P(i=1, j=1) / (P(i=1) * P(j=1)) ]

Positive PMI = factors co-occur more than chance (part of same "coalition").
Negative PMI = factors are mutually exclusive (different semantic regions).

Outputs:
  - PMI matrix as CSV
  - ranked edge list (top co-occurring and top mutually exclusive pairs)
  - console summary of factor coalitions (connected components at PMI threshold)

Optionally writes a GraphML file for Gephi/networkx visualisation.

Requires:  pip install networkx

Usage:
    python analyze_5_pmi_network.py \
        --encoded   runs/agnews_sae/encoded_agnews_all_ready_best.csv \
        --labels    data/agnews/agnews_all.csv \
        --output    runs/agnews_sae/pmi_network \
        [--pmi_threshold 0.5]   # min PMI to draw an edge
        [--top_n_edges 30]      # edges to print in ranked list
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoded",       required=True)
    p.add_argument("--labels",        required=True)
    p.add_argument("--label_col", type=str, default="label_name",
                   help="Column in the labels CSV to use as category. "
                        "e.g. label_name, airline_sentiment, negativereason, airline")
    p.add_argument("--output",        required=True,
                   help="Output prefix — will create <output>_pmi.csv and <output>.graphml")
    p.add_argument("--pmi_threshold", type=float, default=0.5,
                   help="Minimum PMI to include an edge in the network (default: 0.5)")
    p.add_argument("--top_n_edges",   type=int,   default=30)
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


def compute_pmi_matrix(B: np.ndarray, smoothing: float = 0.5):
    """
    B: (n_docs, k_factors) binary matrix
    smoothing: Laplace smoothing count (avoids log(0))
    Returns PMI matrix (k x k), marginal probabilities.
    """
    n, k = B.shape
    # Marginal probabilities P(bit_i = 1)
    marginals = (B.sum(axis=0) + smoothing) / (n + 2 * smoothing)  # (k,)

    # Joint probabilities P(bit_i=1, bit_j=1)
    # = B^T @ B / n  (with smoothing)
    joint = (B.T @ B + smoothing) / (n + 2 * smoothing)  # (k, k)

    # PMI(i,j) = log2(P(i,j) / (P(i)*P(j)))
    outer = np.outer(marginals, marginals)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(joint / outer)
    # Set diagonal to NaN (self-PMI is undefined/trivially 1)
    np.fill_diagonal(pmi, np.nan)
    return pmi, marginals


def dominant_category(bit_idx: int, df: pd.DataFrame, bit_cols: list) -> str:
    """Return the category most associated with a given factor being active."""
    col = bit_cols[bit_idx]
    active_docs = df[df[col] == 1]["label_name"]
    if len(active_docs) == 0:
        return "none"
    return active_docs.value_counts().index[0]


def main():
    args = parse_args()
    df, bit_cols = load_data(args)
    categories = sorted(df["label_name"].unique())
    k = len(bit_cols)

    B = df[bit_cols].values.astype(np.float32)
    pmi, marginals = compute_pmi_matrix(B)

    # ── Marginal stats ────────────────────────────────────────────────────────
    print(f"{'='*70}")
    print(f"FACTOR MARGINALS  (P(bit=1) across all documents)")
    print(f"{'='*70}")
    print(f"{'bit':<10} {'P(active)':>10}  {'dominant_category':<14}  bar")
    print("-" * 60)
    for i, col in enumerate(bit_cols):
        dom = dominant_category(i, df, bit_cols)
        bar = "█" * int(marginals[i] * 40)
        print(f"{col:<10} {marginals[i]:>10.3f}  {dom:<14}  {bar}")

    # ── Build edge list ───────────────────────────────────────────────────────
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            pmi_val = pmi[i, j]
            if np.isnan(pmi_val):
                continue
            dom_i = dominant_category(i, df, bit_cols)
            dom_j = dominant_category(j, df, bit_cols)
            edges.append({
                "bit_i":    i,
                "bit_j":    j,
                "name_i":   bit_cols[i],
                "name_j":   bit_cols[j],
                "pmi":      round(float(pmi_val), 4),
                "dominant_i": dom_i,
                "dominant_j": dom_j,
                "same_category": dom_i == dom_j,
            })

    edges_df = pd.DataFrame(edges).sort_values("pmi", ascending=False)

    # ── Print top positive edges (coalition) ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TOP {args.top_n_edges} POSITIVE PMI PAIRS  (factors that co-occur above chance)")
    print(f"{'='*70}")
    print(f"{'bit_i':<10} {'bit_j':<10} {'pmi':>7}  {'dom_i':<12} {'dom_j':<12}  same?")
    print("-" * 70)
    for _, row in edges_df.head(args.top_n_edges).iterrows():
        same = "✓" if row["same_category"] else "✗"
        print(f"{row['name_i']:<10} {row['name_j']:<10} {row['pmi']:>7.3f}  "
              f"{row['dominant_i']:<12} {row['dominant_j']:<12}  {same}")

    # ── Print top negative edges (mutual exclusion) ───────────────────────────
    print(f"\n{'='*70}")
    print(f"TOP {args.top_n_edges} NEGATIVE PMI PAIRS  (mutually exclusive factors)")
    print(f"{'='*70}")
    print(f"{'bit_i':<10} {'bit_j':<10} {'pmi':>7}  {'dom_i':<12} {'dom_j':<12}")
    print("-" * 70)
    for _, row in edges_df.tail(args.top_n_edges).iloc[::-1].iterrows():
        print(f"{row['name_i']:<10} {row['name_j']:<10} {row['pmi']:>7.3f}  "
              f"{row['dominant_i']:<12} {row['dominant_j']:<12}")

    # ── Connected components at PMI threshold ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FACTOR COALITIONS  (connected components at PMI >= {args.pmi_threshold})")
    print(f"{'='*70}")

    # Simple union-find
    parent = list(range(k))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        parent[find(x)] = find(y)

    strong_edges = edges_df[edges_df["pmi"] >= args.pmi_threshold]
    for _, row in strong_edges.iterrows():
        union(int(row["bit_i"]), int(row["bit_j"]))

    # Group by component
    components: dict = {}
    for i in range(k):
        root = find(i)
        components.setdefault(root, []).append(i)

    components_list = sorted(components.values(), key=lambda c: -len(c))
    for ci, comp in enumerate(components_list):
        bit_names = [bit_cols[i] for i in comp]
        doms = [dominant_category(i, df, bit_cols) for i in comp]
        dom_counts = pd.Series(doms).value_counts()
        print(f"\n  Coalition {ci+1}  ({len(comp)} factors):")
        print(f"    Factors:    {', '.join(bit_names)}")
        print(f"    Dominants:  {dict(dom_counts)}")

    # ── Save PMI matrix ───────────────────────────────────────────────────────
    out_prefix = Path(args.output)
    pmi_df = pd.DataFrame(pmi, index=bit_cols, columns=bit_cols)
    pmi_csv = str(out_prefix) + "_pmi_matrix.csv"
    pmi_df.to_csv(pmi_csv)
    print(f"\nSaved PMI matrix to: {pmi_csv}")

    edges_csv = str(out_prefix) + "_pmi_edges.csv"
    edges_df.to_csv(edges_csv, index=False)
    print(f"Saved edge list to:  {edges_csv}")

    # ── Save GraphML for Gephi/networkx ───────────────────────────────────────
    try:
        import networkx as nx
        G = nx.Graph()
        for i, col in enumerate(bit_cols):
            dom = dominant_category(i, df, bit_cols)
            G.add_node(col, dominant_category=dom,
                       marginal_prob=float(marginals[i]),
                       factor_index=i)
        for _, row in strong_edges.iterrows():
            G.add_edge(row["name_i"], row["name_j"],
                       pmi=float(row["pmi"]),
                       same_category=bool(row["same_category"]))
        graphml_path = str(out_prefix) + ".graphml"
        nx.write_graphml(G, graphml_path)
        print(f"Saved GraphML to:    {graphml_path}")
    except ImportError:
        print("networkx not installed — skipping GraphML export. pip install networkx")


if __name__ == "__main__":
    main()
