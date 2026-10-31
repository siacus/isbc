#!/usr/bin/env python3
"""
Visualize the PMI factor interaction network from a .graphml file.

Nodes  = binary factors, sized by marginal activation probability
         coloured by dominant category
Edges  = PMI >= threshold, thickness proportional to PMI strength

Produces two plots:
  1. Full network (all edges above threshold)
  2. Positive-only network (PMI > 0, co-occurrence coalitions)

Requires:  pip install networkx matplotlib

Usage:
    python visualize_pmi_network.py \
        --graphml   runs/agnews_sae/analysis/pmi.graphml \
        --output    runs/agnews_sae/analysis/pmi_network.png \
        [--layout   spring]        # spring / kamada_kawai / circular / spectral
        [--figsize  14 10]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


KNOWN_CATEGORY_COLORS = {
    # AG News
    "World":    "#4C72B0",
    "Sports":   "#DD8452",
    "Business": "#55A868",
    "Sci_Tech": "#C44E52",
    # Airline sentiment coarse
    "positive": "#55A868",
    "neutral":  "#8172B2",
    "negative": "#C44E52",
    # Airline fine-grained negatives get auto-assigned below
}

# Distinct palette for auto-assignment (up to ~20 categories)
_AUTO_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#AEC7E8",
    "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5", "#C49C94",
]


def build_color_map(categories: list) -> dict:
    """
    Return a color dict for the given category list.
    Known categories get their fixed color; unknown ones are auto-assigned
    from the palette in sorted order so the mapping is deterministic.
    """
    color_map = {}
    auto_idx = 0
    for cat in sorted(categories):
        if cat in KNOWN_CATEGORY_COLORS:
            color_map[cat] = KNOWN_CATEGORY_COLORS[cat]
        else:
            color_map[cat] = _AUTO_PALETTE[auto_idx % len(_AUTO_PALETTE)]
            auto_idx += 1
    color_map["none"] = "#8C8C8C"
    return color_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graphml", required=True)
    p.add_argument("--output",  required=True,
                   help="Output PNG path (a second file _coalitions.png is also written)")
    p.add_argument("--layout",  default="spring",
                   choices=["spring", "kamada_kawai", "circular", "spectral"])
    p.add_argument("--figsize", nargs=2, type=float, default=[14, 10])
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for spring layouts (default: 42)")
    return p.parse_args()


def get_layout(G, layout: str, seed: int = 42):
    if layout == "spring":
        return nx.spring_layout(G, seed=seed, k=2.5 / np.sqrt(len(G.nodes)))
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "spectral":
        return nx.spectral_layout(G)
    return nx.spring_layout(G, seed=seed)


def draw_network(G, pos, ax, title: str,
                 node_colors, node_sizes,
                 edge_colors, edge_widths, edge_alphas,
                 show_edge_labels: bool = False):

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.92, linewidths=1.5,
                           edgecolors="white")

    edges = list(G.edges())
    if edges:
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edgelist=edges,
                               edge_color=edge_colors,
                               width=edge_widths,
                               alpha=edge_alphas,
                               arrows=False)

    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_size=8, font_color="white",
                            font_weight="bold")

    if show_edge_labels and edges:
        edge_labels = {(u, v): f"{G[u][v]['pmi']:.2f}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     ax=ax, font_size=6, alpha=0.7)

    ax.set_title(title, fontsize=13, pad=12)
    ax.axis("off")


def main():
    args = parse_args()
    out_path = Path(args.output)

    G = nx.read_graphml(args.graphml)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_nodes() == 0:
        print("Graph is empty — nothing to plot.")
        return

    # ── Build color map from actual categories in graph ──────────────────────
    all_categories = list({
        G.nodes[n].get("dominant_category", "none")
        for n in G.nodes()
    })
    CATEGORY_COLORS = build_color_map(all_categories)

    # ── Node attributes ───────────────────────────────────────────────────────
    node_colors = []
    node_sizes  = []
    for node in G.nodes():
        attrs = G.nodes[node]
        dom   = attrs.get("dominant_category", "none")
        marg  = float(attrs.get("marginal_prob", 0.1))
        node_colors.append(CATEGORY_COLORS.get(dom, CATEGORY_COLORS["none"]))
        node_sizes.append(300 + marg * 2500)   # size proportional to marginal P(active)

    # ── Edge attributes ───────────────────────────────────────────────────────
    all_pmis = [G[u][v].get("pmi", 0) for u, v in G.edges()]
    pmi_max  = max(abs(p) for p in all_pmis) if all_pmis else 1.0

    edge_colors = []
    edge_widths = []
    edge_alphas = []
    for u, v in G.edges():
        pmi = G[u][v].get("pmi", 0)
        same = G[u][v].get("same_category", False)
        # Positive PMI: green (same category) or grey (cross-category)
        # Negative PMI: red dashed feel — we encode as red with low alpha
        if pmi >= 0:
            color = "#2ecc71" if same else "#95a5a6"
        else:
            color = "#e74c3c"
        edge_colors.append(color)
        edge_widths.append(0.5 + abs(pmi) / pmi_max * 5.0)
        edge_alphas.append(0.4 + abs(pmi) / pmi_max * 0.5)

    # ── Layout ────────────────────────────────────────────────────────────────
    pos = get_layout(G, args.layout, args.seed)

    # ── Plot 1: Full network ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=args.figsize)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    draw_network(G, pos, ax,
                 title=f"PMI Factor Interaction Network  ({G.number_of_nodes()} factors, "
                       f"{G.number_of_edges()} edges)",
                 node_colors=node_colors, node_sizes=node_sizes,
                 edge_colors=edge_colors, edge_widths=edge_widths, edge_alphas=edge_alphas,
                 show_edge_labels=G.number_of_edges() <= 40)

    # Legend: categories
    cat_patches = [mpatches.Patch(color=c, label=cat)
                   for cat, c in CATEGORY_COLORS.items() if cat != "none"]
    edge_patches = [
        mpatches.Patch(color="#2ecc71", label="positive PMI, same category"),
        mpatches.Patch(color="#95a5a6", label="positive PMI, cross-category"),
        mpatches.Patch(color="#e74c3c", label="negative PMI (mutual exclusion)"),
    ]
    legend = ax.legend(handles=cat_patches + edge_patches,
                       loc="lower left", fontsize=8,
                       facecolor="#2d2d4e", edgecolor="white",
                       labelcolor="white", framealpha=0.8)

    # Node size legend
    size_note = ax.text(0.01, 0.01,
                        "Node size ∝ P(factor active)",
                        transform=ax.transAxes, fontsize=7,
                        color="#aaaaaa", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved full network to: {out_path}")

    # ── Plot 2: Coalitions only (positive PMI edges, same category) ───────────
    G_pos = nx.Graph()
    G_pos.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if data.get("pmi", 0) > 0 and data.get("same_category", False):
            G_pos.add_edge(u, v, **data)

    pos2 = get_layout(G_pos, args.layout)

    # Recompute edge attributes for filtered graph
    pmis2   = [G_pos[u][v].get("pmi", 0) for u, v in G_pos.edges()]
    pmax2   = max(pmis2) if pmis2 else 1.0
    ecols2  = ["#2ecc71"] * len(pmis2)
    ewidths2 = [0.5 + p / pmax2 * 5.0 for p in pmis2]
    ealphas2 = [0.5 + p / pmax2 * 0.45 for p in pmis2]

    fig2, ax2 = plt.subplots(figsize=args.figsize)
    fig2.patch.set_facecolor("#1a1a2e")
    ax2.set_facecolor("#1a1a2e")

    draw_network(G_pos, pos2, ax2,
                 title=f"Factor Coalitions  (positive PMI, same-category edges only)",
                 node_colors=node_colors, node_sizes=node_sizes,
                 edge_colors=ecols2, edge_widths=ewidths2, edge_alphas=ealphas2,
                 show_edge_labels=True)

    ax2.legend(handles=cat_patches,
               loc="lower left", fontsize=8,
               facecolor="#2d2d4e", edgecolor="white",
               labelcolor="white", framealpha=0.8)

    plt.tight_layout()
    coal_path = out_path.parent / (out_path.stem + "_coalitions.png")
    plt.savefig(coal_path, dpi=150, bbox_inches="tight",
                facecolor=fig2.get_facecolor())
    plt.close()
    print(f"Saved coalitions network to: {coal_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\nNode summary:")
    for node in sorted(G.nodes()):
        attrs = G.nodes[node]
        dom  = attrs.get("dominant_category", "?")
        marg = float(attrs.get("marginal_prob", 0))
        deg  = G.degree(node)
        print(f"  {node:<10}  dominant={dom:<10}  P(active)={marg:.3f}  degree={deg}")


if __name__ == "__main__":
    main()
