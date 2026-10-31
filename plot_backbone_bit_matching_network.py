#!/usr/bin/env python3
"""Render a cross-backbone network from a factor-collection comparison.

The independently fitted bit numbers are arbitrary.  This script therefore uses
only the pairwise full Hungarian assignments already stored in
``factor_collection_comparison.json``.  It produces two genuine multipartite
networks over nodes ``(backbone, bit)``:

* **semantic network**: the assignment selected from class-profile direction;
* **activation network**: the independent assignment selected from document-level
  normalized mutual information (NMI).

Nodes are never merged: a connected component is an exploratory visual summary of
recurring pairwise partners, not proof of a globally identical latent factor.
No encoder, embedding, or classifier is recomputed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score


CMAP = LinearSegmentedColormap.from_list(
    "matching_nmi_blue_yellow_red", ["#0077ff", "#ffd60a", "#ff3b30"]
)
MODEL_COLORS = {
    "qwen": "#1f77b4",
    "minilm": "#ff7f0e",
    "nvidia": "#2ca02c",
    "harrier": "#d62728",
    "gemma": "#9467bd",
}
MODEL_DISPLAY_NAMES = {
    "qwen": "Qwen",
    "minilm": "MiniLM",
    "nvidia": "NVIDIA",
    "harrier": "Harrier",
    "gemma": "Gemma",
}
LAYOUT_SEED = 20261026
LEIDEN_SEEDS = tuple(range(1234, 1254))
COMMUNITY_RESOLUTION = 1.0
COMMUNITY_PASTELS = (
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462",
    "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cross-backbone bit matching networks from an existing comparison JSON."
    )
    parser.add_argument("--comparison_json", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to the directory containing --comparison_json.",
    )
    parser.add_argument(
        "--update_report",
        type=Path,
        default=None,
        help="Optional factor_collection_comparison.md to receive a linked section.",
    )
    return parser.parse_args()


def node_id(model: str, bit: str) -> str:
    return f"{model}::{bit}"


def bit_number(bit: str) -> str:
    return bit.removeprefix("bit_")


def display_bit(model: str, bit: str) -> str:
    """Readable zero-based node labels (e.g. Q0 = Qwen bit_00)."""
    prefix = {"qwen": "Q", "minilm": "M", "nvidia": "N", "harrier": "H", "gemma": "G"}.get(
        model, model[:1].upper()
    )
    return f"{prefix}{int(bit_number(bit))}"


def model_color(model: str, index: int) -> str:
    return MODEL_COLORS.get(model, plt.get_cmap("tab10")(index % 10))


def community_colour_mapping(labels: list[int]) -> dict[int, str]:
    """Pastel fills stay visually distinct from saturated backbone outlines."""
    if len(labels) > len(COMMUNITY_PASTELS):
        raise ValueError("Community palette needs more distinct pastel colours")
    return {label: COMMUNITY_PASTELS[index] for index, label in enumerate(labels)}


def edge_rows(payload: dict[str, Any], kind: str) -> pd.DataFrame:
    """Flatten all full pairwise Hungarian assignments into an edge table."""
    rows: list[dict[str, Any]] = []
    for pair_key, pair in payload["pairs"].items():
        run_a, run_b = pair["run_a"], pair["run_b"]
        if kind == "semantic":
            for item in pair["semantic"]["pairs"]:
                rows.append(
                    {
                        "matching": "semantic_hungarian",
                        "pair_key": pair_key,
                        "model_a": run_a,
                        "bit_a": item["bit_a"],
                        "model_b": run_b,
                        "bit_b": item["bit_b"],
                        "semantic_direction": item["similarity"],
                        "strength_aware_similarity": item["strength_aware_similarity"],
                        "activation_nmi": item["activation_nmi"],
                        "complemented": item["complemented"],
                    }
                )
        elif kind == "activation":
            for item in pair["activation"]["pairs"]:
                rows.append(
                    {
                        "matching": "activation_nmi_hungarian",
                        "pair_key": pair_key,
                        "model_a": run_a,
                        "bit_a": item["bit_a"],
                        "model_b": run_b,
                        "bit_b": item["bit_b"],
                        "semantic_direction": np.nan,
                        "strength_aware_similarity": np.nan,
                        "activation_nmi": item["normalized_mutual_information"],
                        "complemented": False,
                    }
                )
        else:  # pragma: no cover - internal invariant
            raise ValueError(kind)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No {kind} edges found")
    return frame.sort_values(["model_a", "model_b", "bit_a", "bit_b"]).reset_index(drop=True)


def node_rows(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in payload["run_order"]:
        per_bit = payload["runs"][model]["code_summary"]["per_bit"]
        for record in per_bit:
            rows.append(
                {
                    "node": node_id(model, record["bit"]),
                    "model": model,
                    "bit": record["bit"],
                    "activation_share": record["prevalence"],
                    "label_nmi": record["label_normalized_mutual_information"],
                    "constant": record["constant"],
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "bit"]).reset_index(drop=True)


def make_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(
            row.node,
            model=row.model,
            bit=row.bit,
            activation_share=float(row.activation_share),
        )
    for row in edges.itertuples(index=False):
        graph.add_edge(
            node_id(row.model_a, row.bit_a),
            node_id(row.model_b, row.bit_b),
            activation_nmi=float(row.activation_nmi),
            semantic_direction=(None if pd.isna(row.semantic_direction) else float(row.semantic_direction)),
            strength_aware_similarity=(
                None
                if pd.isna(row.strength_aware_similarity)
                else float(row.strength_aware_similarity)
            ),
            complemented=bool(row.complemented),
        )
    return graph


def components(graph: nx.Graph) -> dict[str, int]:
    """Stable component IDs: largest components first, then lexical tie-break."""
    ordered = sorted(
        (sorted(part) for part in nx.connected_components(graph)),
        key=lambda part: (-len(part), part),
    )
    return {node: index + 1 for index, part in enumerate(ordered) for node in part}


def canonical_membership(membership: list[int], nodes: list[str]) -> list[int]:
    """Canonicalize arbitrary Leiden labels by size, then lexical node membership."""
    groups: dict[int, list[str]] = defaultdict(list)
    for node, value in zip(nodes, membership):
        groups[int(value)].append(node)
    ordered = sorted(groups.values(), key=lambda values: (-len(values), sorted(values)))
    remap = {
        old: new
        for new, values in enumerate(ordered, start=1)
        for old, old_values in groups.items()
        if old_values is values
    }
    return [remap[int(value)] for value in membership]


def leiden_communities(graph: nx.Graph) -> tuple[dict[str, int], dict[str, Any]]:
    """Weighted Leiden partitions over selected matching edges, with seed stability."""
    nodes = sorted(graph.nodes)
    index = {node: position for position, node in enumerate(nodes)}
    edges = [(index[left], index[right]) for left, right in graph.edges]
    weights = [float(graph[left][right]["activation_nmi"]) for left, right in graph.edges]
    if not edges:
        membership = list(range(1, len(nodes) + 1))
        return dict(zip(nodes, membership)), {
            "objective": "weighted RBConfigurationVertexPartition modularity",
            "resolution": COMMUNITY_RESOLUTION,
            "seeds": list(LEIDEN_SEEDS),
            "selected_seed": None,
            "selected_modularity": 0.0,
            "n_communities": len(nodes),
            "mean_pairwise_ari": 1.0,
            "min_pairwise_ari": 1.0,
            "edgeless": True,
        }
    graph_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
    graph_ig.es["weight"] = weights
    runs: list[tuple[int, list[int], float]] = []
    for seed in LEIDEN_SEEDS:
        partition = leidenalg.find_partition(
            graph_ig,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=COMMUNITY_RESOLUTION,
            seed=seed,
        )
        membership = canonical_membership(list(map(int, partition.membership)), nodes)
        grouped: dict[int, set[str]] = defaultdict(set)
        for node, community in zip(nodes, membership):
            grouped[community].add(node)
        # Report the conventional normalized weighted modularity, not igraph's
        # unnormalised objective value.  This keeps Q on its familiar scale.
        modularity = float(
            nx.algorithms.community.modularity(
                graph,
                list(grouped.values()),
                weight="activation_nmi",
                resolution=COMMUNITY_RESOLUTION,
            )
        )
        runs.append((seed, membership, modularity))
    # Ties are deterministic: the lower seed wins.
    best_seed, best_membership, best_quality = max(runs, key=lambda item: (item[2], -item[0]))
    aris = [
        adjusted_rand_score(left[1], right[1])
        for offset, left in enumerate(runs)
        for right in runs[offset + 1 :]
    ]
    membership_map = dict(zip(nodes, best_membership))
    return membership_map, {
        "objective": "weighted RBConfigurationVertexPartition modularity",
        "resolution": COMMUNITY_RESOLUTION,
        "seeds": list(LEIDEN_SEEDS),
        "selected_seed": best_seed,
        "selected_modularity": best_quality,
        "n_communities": len(set(best_membership)),
        "mean_pairwise_ari": float(np.mean(aris)) if aris else 1.0,
        "min_pairwise_ari": float(np.min(aris)) if aris else 1.0,
        "edgeless": False,
    }


def _layout(graph: nx.Graph) -> dict[str, np.ndarray]:
    # Add a tiny floor so weak selected edges preserve the matching topology.
    weighted = graph.copy()
    for _, _, attrs in weighted.edges(data=True):
        attrs["layout_weight"] = max(float(attrs["activation_nmi"]), 0.002)
    return nx.spring_layout(weighted, seed=LAYOUT_SEED, weight="layout_weight", k=1.35, iterations=600)


def community_packed_layout(graph: nx.Graph, communities: dict[str, int]) -> dict[str, np.ndarray]:
    """A deterministic two-level layout that deliberately separates communities.

    A single force layout permits weak inter-community edges to pull clusters over
    one another.  Here each Leiden community receives a well-spaced ring position;
    a second, weighted local force layout arranges only its own nodes.  The result
    is a visual clustering device rather than an embedding-distance representation.
    """
    grouped: dict[int, list[str]] = defaultdict(list)
    for node, community in communities.items():
        grouped[community].append(node)
    ordered = sorted(grouped, key=lambda community: (-len(grouped[community]), community))
    n_communities = len(ordered)
    # This radius leaves a gap between adjacent clusters even for the largest
    # observed community.  Community 1 is intentionally not privileged at the
    # centre: every cluster is equally readable.
    centre_radius = max(5.2, 0.72 * n_communities)
    centres = {
        community: np.array(
            [
                centre_radius * np.cos(2 * np.pi * index / n_communities + np.pi / 2),
                centre_radius * np.sin(2 * np.pi * index / n_communities + np.pi / 2),
            ]
        )
        for index, community in enumerate(ordered)
    }
    result: dict[str, np.ndarray] = {}
    for index, community in enumerate(ordered):
        members = sorted(grouped[community])
        subgraph = graph.subgraph(members).copy()
        local_radius = 0.48 + 0.32 * np.sqrt(len(members))
        if len(members) == 1:
            local = {members[0]: np.zeros(2)}
        elif subgraph.number_of_edges() == 0:
            local = nx.circular_layout(subgraph, scale=local_radius)
        else:
            for _, _, attrs in subgraph.edges(data=True):
                attrs["layout_weight"] = max(float(attrs["activation_nmi"]), 0.002)
            local = nx.spring_layout(
                subgraph,
                seed=LAYOUT_SEED + index,
                weight="layout_weight",
                k=1.25,
                iterations=500,
                scale=local_radius,
            )
        local = resolve_position_collisions(local, minimum_distance=0.62)
        for node, coordinate in local.items():
            result[node] = centres[community] + np.asarray(coordinate, dtype=float)
    # The individual figures are intentionally landscape for a wide LaTex panel.
    # This affine stretch is visual only; the caption explicitly rules out reading
    # geometric distances as a statistic.
    result = {node: coordinate * np.array([1.55, 1.0]) for node, coordinate in result.items()}
    return resolve_position_collisions(result, minimum_distance=0.62)


def resolve_position_collisions(
    positions: dict[str, np.ndarray], minimum_distance: float
) -> dict[str, np.ndarray]:
    """Deterministically repel nodes until their drawn centres do not overlap.

    The local force layout has no hard collision constraint.  This small relaxation
    is applied after layout, using lexical node order and a deterministic fallback
    direction for coincident coordinates, so reruns give the same figure.
    """
    nodes = sorted(positions)
    resolved = {node: np.asarray(value, dtype=float).copy() for node, value in positions.items()}
    for _ in range(800):
        moved = False
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                delta = resolved[right] - resolved[left]
                distance = float(np.linalg.norm(delta))
                if distance >= minimum_distance:
                    continue
                if distance <= 1e-12:
                    # Stable, non-random direction from lexical positions.
                    angle = 2 * np.pi * (left_index + nodes.index(right) + 1) / max(3, len(nodes))
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    direction = delta / distance
                push = (minimum_distance - distance) / 2 + 1e-4
                resolved[left] -= direction * push
                resolved[right] += direction * push
                moved = True
        if not moved:
            break
    return resolved


def draw_network(
    ax: plt.Axes,
    graph: nx.Graph,
    nodes: pd.DataFrame,
    positions: dict[str, np.ndarray],
    title: str,
    models: list[str],
) -> None:
    norm = Normalize(vmin=0.0, vmax=max(0.05, max(attrs["activation_nmi"] for _, _, attrs in graph.edges(data=True))))
    for left, right, attrs in graph.edges(data=True):
        score = float(attrs["activation_nmi"])
        x = [positions[left][0], positions[right][0]]
        y = [positions[left][1], positions[right][1]]
        ax.plot(
            x,
            y,
            color=CMAP(norm(score)),
            linewidth=0.25 + 5.8 * norm(score),
            alpha=0.60,
            zorder=1,
        )
    node_lookup = nodes.set_index("node")
    for model_index, model in enumerate(models):
        current = [node for node in graph.nodes if node_lookup.loc[node, "model"] == model]
        ax.scatter(
            [positions[node][0] for node in current],
            [positions[node][1] for node in current],
            s=[90 + 500 * float(node_lookup.loc[node, "activation_share"]) for node in current],
            c=[model_color(model, model_index)],
            edgecolors="white",
            linewidths=0.85,
            alpha=0.98,
            zorder=2,
            label=model,
        )
        for node in current:
            ax.text(
                positions[node][0],
                positions[node][1],
                bit_number(str(node_lookup.loc[node, "bit"])),
                ha="center",
                va="center",
                fontsize=5.7,
                color="black",
                fontweight="bold",
                zorder=3,
            )
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_axis_off()


def draw_community_network(
    ax: plt.Axes,
    graph: nx.Graph,
    nodes: pd.DataFrame,
    positions: dict[str, np.ndarray],
    communities: dict[str, int],
    title: str,
    models: list[str],
    community_colour: dict[int, Any],
) -> None:
    """Draw a matching network with Leiden communities as node fills.

    Backbone remains recoverable from both the initial in the node label and its
    coloured outline.  This avoids using a fill colour for two incompatible roles.
    """
    norm = Normalize(vmin=0.0, vmax=max(0.05, max(attrs["activation_nmi"] for _, _, attrs in graph.edges(data=True))))
    for left, right, attrs in graph.edges(data=True):
        score = float(attrs["activation_nmi"])
        ax.plot(
            [positions[left][0], positions[right][0]],
            [positions[left][1], positions[right][1]],
            color=CMAP(norm(score)),
            linewidth=0.25 + 5.8 * norm(score),
            alpha=0.52,
            zorder=1,
        )
    node_lookup = nodes.set_index("node")
    for model_index, model in enumerate(models):
        current = [node for node in graph.nodes if node_lookup.loc[node, "model"] == model]
        node_sizes = np.asarray(
            [90 + 500 * float(node_lookup.loc[node, "activation_share"]) for node in current],
            dtype=float,
        )
        # Nested markers make two independent colour encodings legible in PDF as
        # well as PNG: outer backbone ring → white separator → community fill.
        # A white separator is robust against the coloured community palette and
        # prevents a coloured outline from visually blending into the fill.
        ax.scatter(
            [positions[node][0] for node in current],
            [positions[node][1] for node in current],
            s=node_sizes + 150,
            c=[model_color(model, model_index)] * len(current),
            edgecolors=[model_color(model, model_index)] * len(current),
            linewidths=0.0,
            alpha=1.0,
            zorder=2,
        )
        ax.scatter(
            [positions[node][0] for node in current],
            [positions[node][1] for node in current],
            s=node_sizes + 65,
            c="#ffffff",
            edgecolors="#ffffff",
            linewidths=0.0,
            alpha=1.0,
            zorder=2.1,
        )
        ax.scatter(
            [positions[node][0] for node in current],
            [positions[node][1] for node in current],
            s=node_sizes,
            c=[community_colour[communities[node]] for node in current],
            edgecolors="#222222",
            linewidths=0.32,
            alpha=0.98,
            zorder=2.2,
        )
        for node in current:
            ax.text(
                positions[node][0],
                positions[node][1],
                display_bit(model, str(node_lookup.loc[node, "bit"])),
                ha="center",
                va="center",
                fontsize=5.5,
                color="black",
                fontweight="bold",
                zorder=3,
            )
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_axis_off()


def save_community_figure(
    nodes: pd.DataFrame,
    semantic_graph: nx.Graph,
    activation_graph: nx.Graph,
    semantic_communities: dict[str, int],
    activation_communities: dict[str, int],
    semantic_metadata: dict[str, Any],
    activation_metadata: dict[str, Any],
    models: list[str],
    png_path: Path,
    pdf_path: Path,
) -> None:
    semantic_position = community_packed_layout(semantic_graph, semantic_communities)
    activation_position = community_packed_layout(activation_graph, activation_communities)
    fig, axes = plt.subplots(1, 2, figsize=(19, 10.4), constrained_layout=True)
    all_community_labels = sorted(
        set(semantic_communities.values()) | set(activation_communities.values())
    )
    community_colour = community_colour_mapping(all_community_labels)
    draw_community_network(
        axes[0], semantic_graph, nodes, semantic_position, semantic_communities,
        "Semantic Hungarian network: Leiden communities\n"
        f"weighted by same-pair activation NMI; Q={semantic_metadata['selected_modularity']:.3f}",
        models, community_colour,
    )
    draw_community_network(
        axes[1], activation_graph, nodes, activation_position, activation_communities,
        "Activation-optimal network: Leiden communities\n"
        f"weighted by activation NMI; Q={activation_metadata['selected_modularity']:.3f}",
        models, community_colour,
    )
    community_handles = [Line2D([], [], linestyle="", marker=None, label="Community:")] + [
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=community_colour[community],
               markeredgecolor="#333333", label=f"C{community}")
        for community in all_community_labels
    ]
    backbone_handles = [Line2D([], [], linestyle="", marker=None, label="Model:")] + [
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="white",
               markeredgecolor=model_color(model, index), markeredgewidth=2.0,
               label=f"{display_bit(model, 'bit_00')[0]} {MODEL_DISPLAY_NAMES.get(model, model)}")
        for index, model in enumerate(models)
    ]
    model_legend = fig.legend(
        handles=backbone_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.070),
        ncol=len(backbone_handles),
        frameon=False,
    )
    fig.add_artist(model_legend)
    fig.legend(handles=community_handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=len(community_handles), frameon=False)
    fig.suptitle("Cross-backbone bit matching: community overlays", fontsize=15, y=1.01)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_single_community_figure(
    nodes: pd.DataFrame,
    graph: nx.Graph,
    communities: dict[str, int],
    metadata: dict[str, Any],
    models: list[str],
    matching_title: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Render one community network per page for inspection and paper-quality PDF."""
    positions = community_packed_layout(graph, communities)
    labels = sorted(set(communities.values()))
    community_colour = community_colour_mapping(labels)
    fig, ax = plt.subplots(figsize=(15.5, 8.7), constrained_layout=False)
    draw_community_network(
        ax,
        graph,
        nodes,
        positions,
        communities,
        f"{matching_title}\n"
        f"Leiden weighted modularity Q={metadata['selected_modularity']:.3f}; "
        f"{metadata['n_communities']} communities",
        models,
        community_colour,
    )
    community_handles = [Line2D([], [], linestyle="", marker=None, label="Community:")] + [
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=community_colour[community],
               markeredgecolor="#333333", label=f"C{community}")
        for community in labels
    ]
    backbone_handles = [Line2D([], [], linestyle="", marker=None, label="Model:")] + [
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="white",
               markeredgecolor=model_color(model, index), markeredgewidth=2.0,
               label=f"{display_bit(model, 'bit_00')[0]} {MODEL_DISPLAY_NAMES.get(model, model)}")
        for index, model in enumerate(models)
    ]
    community_legend = fig.legend(
        handles=community_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=len(community_handles),
        frameon=False,
    )
    fig.add_artist(community_legend)
    fig.legend(
        handles=backbone_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.037),
        ncol=len(backbone_handles),
        frameon=False,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.105)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def activation_community_label_analysis(
    payload: dict[str, Any], nodes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Describe right-panel communities post hoc without changing the matching.

    A document activates a community under the primary *cross-backbone consensus*
    rule when at least two distinct backbones have at least one active member bit in
    that community.  It is intentionally permissive about which member bit fires,
    since a community can contain more than one bit from a backbone.  Community
    memberships overlap; this is a descriptive diagnostic, not a partition of the
    corpus or a new classifier.
    """
    models = list(payload["run_order"])
    codes: dict[str, pd.DataFrame] = {}
    reference_ids: pd.Index | None = None
    for model in models:
        encoded_path = Path(payload["runs"][model]["encoded"]["path"])
        frame = pd.read_csv(
            encoded_path,
            usecols=lambda column: column == "id" or column.startswith("bit_"),
            dtype={"id": str},
        ).set_index("id").sort_index()
        if reference_ids is None:
            reference_ids = frame.index
        elif not frame.index.equals(reference_ids):
            raise ValueError(f"Encoded IDs do not align for community label analysis: {model}")
        codes[model] = frame
    if reference_ids is None:  # pragma: no cover - payload invariant
        raise ValueError("No runs available for community label analysis")
    label_column = str(payload["label_column"])
    labels_path = Path(payload["labels"]["path"])
    labels = pd.read_csv(labels_path, usecols=["id", label_column], dtype={"id": str}).set_index("id")
    y = labels.reindex(reference_ids)[label_column]
    if y.isna().any():
        raise ValueError("Community label analysis has missing labels after ID alignment")
    y = y.astype(str)
    categories = [str(value) for value in payload.get("classes", sorted(y.unique()))]
    corpus_share = y.value_counts(normalize=True).reindex(categories, fill_value=0.0)

    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for community, group in nodes.groupby("activation_community", sort=True):
        votes = np.zeros(len(reference_ids), dtype=np.int8)
        for model in models:
            bits = group.loc[group["model"].eq(model), "bit"].tolist()
            if bits:
                votes += codes[model][bits].to_numpy(dtype=bool).any(axis=1).astype(np.int8)
        any_member = votes >= 1
        consensus = votes >= 2
        distribution = y[consensus].value_counts().reindex(categories, fill_value=0)
        shares = distribution / max(1, int(consensus.sum()))
        top_category = shares.idxmax()
        summary_rows.append(
            {
                "community": int(community),
                "n_member_bits": int(len(group)),
                "n_backbones_represented": int(group["model"].nunique()),
                "member_bits": ", ".join(group.sort_values(["model", "bit"])["display_bit"]),
                "any_member_documents": int(any_member.sum()),
                "any_member_share": float(any_member.mean()),
                "consensus_documents": int(consensus.sum()),
                "consensus_share": float(consensus.mean()),
                "top_category": top_category,
                "top_category_share": float(shares.loc[top_category]),
                "top_category_lift_vs_corpus": float(shares.loc[top_category] / corpus_share.loc[top_category])
                if corpus_share.loc[top_category] > 0
                else np.nan,
            }
        )
        for category in categories:
            long_rows.append(
                {
                    "community": int(community),
                    "minimum_active_backbones": 2,
                    "activation_definition": "at_least_2_backbones",
                    "category": category,
                    "documents": int(consensus.sum()),
                    "document_share": float(consensus.mean()),
                    "category_documents": int(distribution.loc[category]),
                    "category_share_within_community": float(shares.loc[category]),
                    "category_share_in_corpus": float(corpus_share.loc[category]),
                    "lift_vs_corpus": float(shares.loc[category] / corpus_share.loc[category])
                    if corpus_share.loc[category] > 0
                    else np.nan,
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values("community").reset_index(drop=True)
    long = pd.DataFrame(long_rows).sort_values(["community", "category"]).reset_index(drop=True)
    metadata = {
        "label_column": label_column,
        "categories": categories,
        "document_activation_definition": (
            "At least two distinct backbones have at least one active member bit in the community."
        ),
        "minimum_active_backbones": 2,
        "overlap": "Documents may activate more than one community.",
        "n_documents": int(len(reference_ids)),
        "corpus_category_shares": {category: float(corpus_share.loc[category]) for category in categories},
    }
    return summary, long, metadata


def label_description_table(summary: pd.DataFrame, long: pd.DataFrame) -> list[str]:
    """Compact Markdown table intended for a handover, not inferential reporting."""
    pivot = long.pivot(index="community", columns="category", values="category_share_within_community")
    lines = [
        "| Community | Member bits | Consensus documents | Business | Sci-Tech | Sports | World | Dominant label (lift) |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        shares = pivot.loc[row.community]
        def share(category: str) -> str:
            return f"{100 * float(shares.get(category, 0.0)):.1f}%"
        lines.append(
            f"| C{row.community} | {row.member_bits} | "
            f"{row.consensus_documents:,} ({100 * row.consensus_share:.1f}%) | "
            f"{share('Business')} | {share('Sci_Tech')} | {share('Sports')} | {share('World')} | "
            f"{row.top_category} ({row.top_category_lift_vs_corpus:.2f}×) |"
        )
    return lines


def write_handoff(
    output_dir: Path,
    payload: dict[str, Any],
    nodes: pd.DataFrame,
    semantic_metadata: dict[str, Any],
    activation_metadata: dict[str, Any],
    label_summary: pd.DataFrame,
    label_long: pd.DataFrame,
    label_metadata: dict[str, Any],
) -> None:
    """Write an agent-ready account of the bit-network analysis and exact artifacts."""
    table = label_description_table(label_summary, label_long)
    lines = [
        "# Handover: AG News cross-backbone bit-matching networks",
        "",
        "## Purpose and scope",
        "",
        f"This bundle compares **{len(payload['run_order'])} independently trained 16-bit sparse "
        f"codes**—Qwen, MiniLM, NVIDIA, Harrier, and Gemma—on the same "
        f"**{payload['n_aligned_rows']:,} AG News documents**. It reuses the stored hard codes "
        "and matching results; it does **not** "
        "train an SAE, regenerate embeddings, or fit a new predictive model.",
        "",
        "## What was constructed",
        "",
        "1. **Semantic Hungarian network (left / first figure).** For every backbone pair, all "
        "16×16 bit pairs are scored by absolute correlation of their four category-conditional "
        "activation profiles. A full Hungarian assignment selects 16 one-to-one partners. Edge "
        "colour and width subsequently show same-pair document-level activation NMI, not the "
        "semantic score used to choose the edge.",
        "2. **Activation-optimal Hungarian network (right / second figure).** Category labels are "
        "ignored when matching. Each pairwise assignment maximizes total document-level NMI over "
        "the two hard activation vectors.",
        "3. **Community overlay.** Weighted Leiden configuration-model modularity, resolution 1, "
        "is applied separately to each network with activation NMI as edge weight. Twenty fixed "
        "seeds are evaluated and the maximum-modularity partition is retained. Node fill is the "
        "Leiden community; outer coloured ring is the backbone; the white ring separates these "
        "two encodings. Labels use original zero-based indices (`Q0` = Qwen `bit_00`).",
        "",
        "The network layout is deliberately two-level and community-packed for legibility. It is "
        "not an embedding geometry: distances between communities have no quantitative meaning.",
        "",
        "## Figures and visual legend",
        "",
        "![Semantic Hungarian Leiden communities](backbone_bit_matching_semantic_communities.png)",
        "",
        "![Activation-optimal Leiden communities](backbone_bit_matching_activation_communities.png)",
        "",
        "In both figures, the outer ring and initial identify the model: `Q` Qwen, `M` MiniLM, "
        "`N` NVIDIA, `H` Harrier, and `G` Gemma. The node label then gives its run-local, zero-based "
        "bit number (for example, `Q0` is Qwen's `bit_00`). Node fill is the Leiden community. "
        "Edge colour and width encode document-level activation NMI. The two-level community-packed "
        "layout is only visual separation: distances between communities are not statistical distances.",
        "",
        "## Exact results",
        "",
        f"- Semantic network: {semantic_metadata['n_communities']} communities; weighted modularity "
        f"{semantic_metadata['selected_modularity']:.3f}; mean seed-pair ARI "
        f"{semantic_metadata['mean_pairwise_ari']:.3f} (minimum "
        f"{semantic_metadata['min_pairwise_ari']:.3f}).",
        f"- Activation-optimal network: {activation_metadata['n_communities']} communities; weighted "
        f"modularity {activation_metadata['selected_modularity']:.3f}; mean seed-pair ARI "
        f"{activation_metadata['mean_pairwise_ari']:.3f} (minimum "
        f"{activation_metadata['min_pairwise_ari']:.3f}).",
        "",
        "## Right-panel community label composition (post hoc)",
        "",
        "The table below was **not** used to make the right network or choose its communities. "
        "For a given community, a document enters the primary diagnostic if at least two distinct "
        "backbones have at least one active member bit. Rows are therefore overlapping descriptions, "
        "not a partition of documents. The AG News corpus is balanced: each category is 25%, so "
        "a lift above 1 means over-representation relative to the complete corpus.",
        "",
    ]
    lines.extend(table)
    lines.extend(
        [
            "",
            "### How to interpret the table",
            "",
            "The label shares describe the empirical composition of documents selected by a "
            "community’s cross-backbone activation pattern. They can motivate close reading and "
            "profile-level inspection, but they do not establish the semantic meaning of a community. "
            "In particular, high association with a category does not exclude a transverse theme "
            "within that category, and an apparently mixed row can still encode an interpretable "
            "dimension orthogonal to the AG News taxonomy.",
            "",
            "## Artifact map",
            "",
            "- `backbone_bit_matching_semantic_communities.png` / `.pdf`: semantic-Hungarian network.",
            "- `backbone_bit_matching_activation_communities.png` / `.pdf`: label-free, "
            "activation-optimal network.",
            "- `backbone_bit_matching_edges.csv`: all 160 selected pairwise edges for each matching "
            "objective (320 rows total).",
            "- `backbone_bit_matching_nodes.csv`: all 80 nodes, prevalence, display labels, and the "
            "two community memberships.",
            "- `backbone_bit_matching_activation_community_summary.csv`: coverage and dominant-label "
            "summary for the right-panel communities.",
            "- `backbone_bit_matching_activation_community_labels.csv`: full community × category "
            "composition and lift table.",
            "- `backbone_bit_matching_network.json`: source provenance, matching counts, and Leiden "
            "stability metadata.",
            "- `factor_collection_comparison.md`: main comparative report embedding the network figures.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "source .venv/bin/activate",
            "MPLCONFIGDIR=/tmp/latent-mpl python plot_backbone_bit_matching_network.py \\",
            "  --comparison_json runs/agnews_embedding_comparison/all_fixed_balance_label_name/factor_collection_comparison.json \\",
            "  --output_dir runs/agnews_embedding_comparison/all_fixed_balance_label_name \\",
            "  --update_report runs/agnews_embedding_comparison/all_fixed_balance_label_name/factor_collection_comparison.md",
            "```",
            "",
            "Do not reinterpret run-local bit numbers as pre-aligned. Do not write that the labels "
            "created the activation-optimal graph: they appear only in the post-hoc table above.",
        ]
    )
    (output_dir / "backbone_bit_matching_handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(
    nodes: pd.DataFrame,
    semantic_graph: nx.Graph,
    activation_graph: nx.Graph,
    models: list[str],
    png_path: Path,
    pdf_path: Path,
) -> None:
    # Separate layouts retain each objective's actual graph topology.  The fixed seed
    # makes each panel reproducible, but coordinates are not a cross-panel statistic.
    semantic_position = _layout(semantic_graph)
    activation_position = _layout(activation_graph)
    fig, axes = plt.subplots(1, 2, figsize=(19, 10.2), constrained_layout=True)
    draw_network(
        axes[0], semantic_graph, nodes, semantic_position,
        "Semantic Hungarian network\n(edge colour/width = same-pair activation NMI)", models,
    )
    draw_network(
        axes[1], activation_graph, nodes, activation_position,
        "Activation-optimal Hungarian network\n(edge colour/width = activation NMI)", models,
    )
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=model_color(model, index),
            markeredgecolor="white", markersize=8, label=model,
        )
        for index, model in enumerate(models)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(models),
        frameon=False,
        title="Node colour = backbone; node label = run-local bit number; node area = activation prevalence",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=CMAP),
        ax=axes,
        shrink=0.60,
        pad=0.015,
        label="Document-level normalized mutual information (NMI)",
    )
    fig.suptitle("Cross-backbone bit matching networks", fontsize=15, y=1.01)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def report_fragment(
    output_dir: Path,
    nodes: pd.DataFrame,
    semantic_edges: pd.DataFrame,
    activation_edges: pd.DataFrame,
    semantic_components: dict[str, int],
    activation_components: dict[str, int],
    semantic_community_metadata: dict[str, Any],
    activation_community_metadata: dict[str, Any],
) -> str:
    semantic_count = len(set(semantic_components.values()))
    activation_count = len(set(activation_components.values()))
    semantic_med = semantic_edges["activation_nmi"].median()
    activation_med = activation_edges["activation_nmi"].median()
    return "\n".join(
        [
            "## Cross-backbone bit-matching network",
            "",
            "![Cross-backbone bit matching networks](backbone_bit_matching_network.png)",
            "",
            f"This graph uses all **{len(nodes)} run-local bit nodes** and all pairwise full "
            f"Hungarian assignments ({len(semantic_edges)} edges per panel). The left panel "
            f"uses the semantic assignments; the right panel independently reassigns partners "
            f"to maximize document-level activation NMI. The underlying values are in "
            f"[the edge table](backbone_bit_matching_edges.csv) and [node table]"
            f"(backbone_bit_matching_nodes.csv).",
            "",
            "An edge means only that its endpoints were chosen as partners in the relevant "
            "two-backbone full matching. It is not a claim that a five-backbone cluster is "
            "globally identified or transitive. Colour and width encode activation NMI; the "
            "semantic panel is still *selected* by label-profile direction, whereas the "
            "activation panel is selected directly by NMI. The fixed-seed force layouts are "
            "visual aids, not distances or effect sizes.",
            "",
            f"The semantic network has **{semantic_count}** connected component(s) and median "
            f"same-pair NMI **{semantic_med:.3f}**; the activation-optimal network has "
            f"**{activation_count}** component(s) and median NMI **{activation_med:.3f}**. "
            "A recurrent cross-backbone neighbourhood is evidence consistent with a shared "
            "factor, but must be interpreted together with the numerical pairwise tables and "
            "cannot establish semantic equivalence on its own.",
            "",
            "### Community overlay",
            "",
            "#### Semantic Hungarian communities",
            "",
            "![Semantic Hungarian Leiden communities](backbone_bit_matching_semantic_communities.png)",
            "",
            "[PDF version](backbone_bit_matching_semantic_communities.pdf)",
            "",
            "#### Activation-optimal communities",
            "",
            "![Activation-optimal Leiden communities](backbone_bit_matching_activation_communities.png)",
            "",
            "[PDF version](backbone_bit_matching_activation_communities.pdf)",
            "",
            "The overlay uses weighted Leiden configuration-model modularity (resolution 1) "
            "on the selected network edges, weighting every edge by document-level activation "
            "NMI. Fill colours identify communities; node outlines and the first label letter "
            "identify the backbone. Labels use the original zero-based bit indices: `Q0` is "
            "Qwen `bit_00`, while `M0`, `N0`, `H0`, and `G0` analogously denote MiniLM, "
            "NVIDIA, Harrier, and Gemma. "
            "Community colours are panel-specific because each panel has a different matching "
            "objective; matching colours across panels do not assert correspondence.",
            "",
            f"Across 20 fixed Leiden seeds, the selected semantic partition has "
            f"**{semantic_community_metadata['n_communities']}** communities, weighted "
            f"modularity **{semantic_community_metadata['selected_modularity']:.3f}**, and "
            f"seed-to-seed ARI **{semantic_community_metadata['mean_pairwise_ari']:.3f}** "
            f"(minimum **{semantic_community_metadata['min_pairwise_ari']:.3f}**). The "
            f"activation-optimal partition has **{activation_community_metadata['n_communities']}** "
            f"communities, modularity **{activation_community_metadata['selected_modularity']:.3f}**, "
            f"and mean ARI **{activation_community_metadata['mean_pairwise_ari']:.3f}** "
            f"(minimum **{activation_community_metadata['min_pairwise_ari']:.3f}**). Seed "
            "stability concerns the optimizer on this fixed graph, not sampling robustness.",
            "",
            "A [standalone handover](backbone_bit_matching_handoff.md) documents the construction "
            "and includes a post-hoc right-panel community-by-label table. Those labels were not "
            "used for activation matching or community detection.",
        ]
    )


def rebuild_navigation(markdown: str) -> str:
    """Remove the generated contents/anchors and recreate them after insertion."""
    source = markdown.splitlines()
    cleaned: list[str] = []
    skipping_contents = False
    skip_anchor_padding = False
    for line in source:
        if line == "## Contents":
            skipping_contents = True
            continue
        if skipping_contents:
            if line.startswith("<a id=\"report-section-"):
                skipping_contents = False
            else:
                continue
        if line.startswith("<a id=\"report-section-"):
            skip_anchor_padding = True
            continue
        if skip_anchor_padding and not line:
            skip_anchor_padding = False
            continue
        skip_anchor_padding = False
        cleaned.append(line)

    sections: list[tuple[str, str]] = []
    positions: dict[int, str] = {}
    for index, line in enumerate(cleaned):
        if line.startswith("## ") and line != "## Contents":
            anchor = f"report-section-{len(sections) + 1}"
            sections.append((anchor, line[3:]))
            positions[index] = anchor
    contents = ["## Contents", ""]
    contents.extend(f"- [{title}](#{anchor})" for anchor, title in sections)
    contents.append("")
    rendered: list[str] = []
    inserted = False
    for index, line in enumerate(cleaned):
        if index == 1 and not inserted:
            rendered.extend([""] + contents)
            inserted = True
        if index in positions:
            rendered.extend([f'<a id="{positions[index]}"></a>', ""])
        rendered.append(line)
    return "\n".join(rendered).rstrip() + "\n"


def update_report(report_path: Path, fragment: str) -> None:
    text = report_path.read_text(encoding="utf-8")
    marker = "## Cross-backbone bit-matching network"
    if marker in text:
        before, rest = text.split(marker, 1)
        # The network section is always inserted immediately before the pairwise
        # heatmaps. Replacing precisely this range makes reruns idempotent.
        _, after = rest.split("## Pairwise comparison heatmaps", 1)
        text = before.rstrip() + "\n\n" + fragment + "\n\n## Pairwise comparison heatmaps" + after
    else:
        marker = "## Pairwise comparison heatmaps"
        if marker not in text:
            raise ValueError(f"Cannot find insertion point in {report_path}")
        text = text.replace(marker, fragment + "\n\n" + marker, 1)
    report_path.write_text(rebuild_navigation(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.comparison_json.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.comparison_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    models = list(payload["run_order"])
    nodes = node_rows(payload)
    semantic_edges = edge_rows(payload, "semantic")
    activation_edges = edge_rows(payload, "activation")
    combined = pd.concat([semantic_edges, activation_edges], ignore_index=True)
    semantic_graph = make_graph(nodes, semantic_edges)
    activation_graph = make_graph(nodes, activation_edges)
    semantic_components = components(semantic_graph)
    activation_components = components(activation_graph)
    semantic_communities, semantic_community_metadata = leiden_communities(semantic_graph)
    activation_communities, activation_community_metadata = leiden_communities(activation_graph)
    nodes["semantic_component"] = nodes["node"].map(semantic_components).astype(int)
    nodes["activation_component"] = nodes["node"].map(activation_components).astype(int)
    nodes["display_bit"] = [display_bit(model, bit) for model, bit in zip(nodes["model"], nodes["bit"])]
    nodes["semantic_community"] = nodes["node"].map(semantic_communities).astype(int)
    nodes["activation_community"] = nodes["node"].map(activation_communities).astype(int)
    label_summary, label_long, label_metadata = activation_community_label_analysis(payload, nodes)
    nodes.to_csv(output_dir / "backbone_bit_matching_nodes.csv", index=False)
    combined.to_csv(output_dir / "backbone_bit_matching_edges.csv", index=False)
    label_summary.to_csv(
        output_dir / "backbone_bit_matching_activation_community_summary.csv", index=False
    )
    label_long.to_csv(
        output_dir / "backbone_bit_matching_activation_community_labels.csv", index=False
    )
    summary = {
        "schema_version": 1,
        "source_comparison_json": str(args.comparison_json),
        "n_documents": int(payload["n_aligned_rows"]),
        "run_order": models,
        "n_nodes": int(len(nodes)),
        "semantic": {
            "n_edges": int(len(semantic_edges)),
            "n_components": int(len(set(semantic_components.values()))),
            "median_activation_nmi": float(semantic_edges["activation_nmi"].median()),
            "community_detection": semantic_community_metadata,
        },
        "activation": {
            "n_edges": int(len(activation_edges)),
            "n_components": int(len(set(activation_components.values()))),
            "median_activation_nmi": float(activation_edges["activation_nmi"].median()),
            "community_detection": activation_community_metadata,
        },
        "layout": {"algorithm": "networkx.spring_layout", "seed": LAYOUT_SEED},
        "activation_community_label_diagnostic": {
            **label_metadata,
            "summary_csv": "backbone_bit_matching_activation_community_summary.csv",
            "long_csv": "backbone_bit_matching_activation_community_labels.csv",
        },
    }
    (output_dir / "backbone_bit_matching_network.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_figure(
        nodes, semantic_graph, activation_graph, models,
        output_dir / "backbone_bit_matching_network.png",
        output_dir / "backbone_bit_matching_network.pdf",
    )
    save_community_figure(
        nodes,
        semantic_graph,
        activation_graph,
        semantic_communities,
        activation_communities,
        semantic_community_metadata,
        activation_community_metadata,
        models,
        output_dir / "backbone_bit_matching_communities.png",
        output_dir / "backbone_bit_matching_communities.pdf",
    )
    save_single_community_figure(
        nodes,
        semantic_graph,
        semantic_communities,
        semantic_community_metadata,
        models,
        "Semantic Hungarian bit-matching network",
        output_dir / "backbone_bit_matching_semantic_communities.png",
        output_dir / "backbone_bit_matching_semantic_communities.pdf",
    )
    save_single_community_figure(
        nodes,
        activation_graph,
        activation_communities,
        activation_community_metadata,
        models,
        "Activation-optimal bit-matching network",
        output_dir / "backbone_bit_matching_activation_communities.png",
        output_dir / "backbone_bit_matching_activation_communities.pdf",
    )
    fragment = report_fragment(
        output_dir,
        nodes,
        semantic_edges,
        activation_edges,
        semantic_components,
        activation_components,
        semantic_community_metadata,
        activation_community_metadata,
    )
    write_handoff(
        output_dir,
        payload,
        nodes,
        semantic_community_metadata,
        activation_community_metadata,
        label_summary,
        label_long,
        label_metadata,
    )
    (output_dir / "backbone_bit_matching_network.md").write_text(fragment + "\n", encoding="utf-8")
    if args.update_report:
        update_report(args.update_report, fragment)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
