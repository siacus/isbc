#!/usr/bin/env python3
"""Create a label-free, single-run report of a binary code's internal structure.

This is deliberately separate from ``compare_factor_run_collection.py``.  It is the
first descriptive step in a paper: the report sees only encoded documents and training
artifacts, never a target label.  Consequently it describes *selectivity* and
co-activation, not a bit's ability to discriminate a particular taxonomy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Any

import igraph as ig
import leidenalg
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.path import Path as MatplotlibPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.ticker import PercentFormatter
from adjustText import adjust_text
from sklearn.metrics import adjusted_rand_score


BIT_RE = re.compile(r"^bit_(\d+)$")
PMI_CMAP = LinearSegmentedColormap.from_list(
    "pmi_blue_gray_red", ["#2166ac", "#bdbdbd", "#b2182b"]
)
JOINT_SHARE_CMAP = LinearSegmentedColormap.from_list(
    "joint_share_blue_red",
    # Saturated, high-luminance endpoints remain legible on the white report
    # background while retaining an intuitive low-to-high blue → yellow → red
    # ordering.
    ["#0077ff", "#ffd60a", "#ff3b30"],
)
COMMUNITY_RESOLUTIONS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
COMMUNITY_SEEDS = tuple(range(1234, 1254))
COMMUNITY_LAYOUT_SEED = 20261023


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a label-free, single-backbone binary-code report."
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument(
        "--encoded_csv",
        type=Path,
        default=None,
        help="Defaults to the unique encoded_*_best.csv in run_dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to run_dir/bit_structure.",
    )
    parser.add_argument(
        "--posthoc_category_json",
        type=Path,
        default=None,
        help=(
            "Optional factor_collection_comparison.json used only to overlay the "
            "heatmap's selected external categories on the community graph."
        ),
    )
    parser.add_argument(
        "--posthoc_category_run",
        default=None,
        help="Run key in --posthoc_category_json, for example qwen (required with that JSON).",
    )
    parser.add_argument("--top_profiles", type=int, default=20)
    return parser.parse_args()


def bit_key(name: str) -> int:
    match = BIT_RE.match(name)
    if not match:
        raise ValueError(f"Not a hard-bit column: {name}")
    return int(match.group(1))


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    """Binary entropy in bits, defined as zero at probabilities 0 and 1."""
    values = np.asarray(probability, dtype=float)
    out = np.zeros_like(values, dtype=float)
    interior = (values > 0.0) & (values < 1.0)
    p = values[interior]
    out[interior] = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return out


def profile_strings(matrix: np.ndarray) -> np.ndarray:
    """Return fixed-width binary strings in the visible ``bit_00 ...`` order."""
    n_bits = matrix.shape[1]
    if n_bits > 62:
        return np.asarray(["".join(map(str, row)) for row in matrix], dtype=object)
    weights = 1 << np.arange(n_bits - 1, -1, -1, dtype=np.int64)
    values = matrix.astype(np.int64) @ weights
    width = f"0{n_bits}b"
    return np.asarray([format(int(value), width) for value in values], dtype=object)


def contents(markdown: str) -> str:
    """Insert stable HTML anchors and a compact, renderer-independent contents list."""
    source = markdown.splitlines()
    anchors: dict[int, str] = {}
    sections: list[tuple[str, str]] = []
    for index, line in enumerate(source):
        if line.startswith("## ") and line != "## Contents":
            anchor = f"report-section-{len(sections) + 1}"
            anchors[index] = anchor
            sections.append((anchor, line[3:]))
    toc = ["## Contents", ""]
    toc.extend(f"- [{title}](#{anchor})" for anchor, title in sections)
    toc.append("")
    rendered: list[str] = []
    for index, line in enumerate(source):
        if index == 1:
            rendered.extend([""] + toc)
        if index in anchors:
            rendered.extend([f'<a id="{anchors[index]}"></a>', ""])
        rendered.append(line)
    return "\n".join(rendered) + "\n"


def save_activation_plot(stats: pd.DataFrame, path: Path) -> None:
    ordered = stats.sort_values("activation_share", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = np.where(
        ordered["activation_share"].to_numpy() >= 0.95,
        "#b2182b",
        np.where(ordered["activation_share"].to_numpy() <= 0.05, "#2166ac", "#4daf4a"),
    )
    ax.bar(ordered["bit"], ordered["activation_share"], color=colors, width=0.78)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="50% activation")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Bit (ordered by observed activation)")
    ax.set_ylabel("Share of documents with bit = 1")
    ax.set_title("Hard-bit prevalence (label-free)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_heatmap(
    values: np.ndarray,
    bit_names: list[str],
    path: Path,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    colorbar_label: str,
) -> None:
    size = max(6.5, len(bit_names) * 0.46)
    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    image = ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(bit_names)), bit_names, rotation=90)
    ax.set_yticks(range(len(bit_names)), bit_names)
    ax.set_title(title)
    bar = fig.colorbar(image, ax=ax, shrink=0.82)
    bar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_profile_plot(profiles: pd.DataFrame, path: Path) -> None:
    shown = profiles.head(20)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(shown["rank"].astype(str), shown["share_pct"], color="#377eb8")
    ax.set_xlabel("Profile rank by observed frequency")
    ax.set_ylabel("Share of all documents (%)")
    ax.set_title("Twenty most frequent exact hard-code profiles")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_pmi_phi_scatter(
    pairs: pd.DataFrame, path: Path, *, color_by: str = "pmi"
) -> None:
    """Render a PMI--phi scatter, colouring by association or observed volume."""
    finite = pairs[
        np.isfinite(pairs["pmi_bits"])
        & pd.to_numeric(pairs["phi_correlation"], errors="coerce").notna()
    ].copy()
    if finite.empty:
        raise ValueError("No finite PMI--phi pairs are available for plotting")
    labelled = pd.concat(
        [finite.nlargest(20, "pmi_bits"), finite.nsmallest(10, "pmi_bits")],
        ignore_index=True,
    ).drop_duplicates(subset=["bit_a", "bit_b"])
    fig, ax = plt.subplots(figsize=(13.5, 9.5))
    if color_by == "pmi":
        bound = max(1.0, float(np.quantile(np.abs(finite["pmi_bits"]), 0.95)))
        positive = finite[finite["pmi_bits"] > 0]
        remainder = finite[finite["pmi_bits"] <= 0]
        normalization = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
        remainder_scatter = ax.scatter(
            remainder["pmi_bits"], remainder["phi_correlation"],
            c=remainder["pmi_bits"], cmap=PMI_CMAP, norm=normalization,
            s=28, alpha=0.78, zorder=2, label="PMI ≤ 0",
        )
        color_artist = ax.scatter(
            positive["pmi_bits"], positive["phi_correlation"],
            c=positive["pmi_bits"], cmap=PMI_CMAP, norm=normalization,
            s=34, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3,
            label="PMI > 0",
        )
        colorbar_label = "PMI (bits): colour intensity"
        title_suffix = ""
    elif color_by == "joint_share":
        share_min = float(finite["joint_activation_share"].min())
        share_max = float(finite["joint_activation_share"].max())
        color_artist = ax.scatter(
            finite["pmi_bits"], finite["phi_correlation"],
            c=finite["joint_activation_share"], cmap=JOINT_SHARE_CMAP,
            norm=Normalize(vmin=share_min, vmax=max(share_max, share_min + 1e-12)),
            s=31, alpha=0.85, edgecolor="white", linewidth=0.35, zorder=3,
        )
        colorbar_label = "Observed joint activation share"
        title_suffix = " (dot colour: joint share)"
    else:
        raise ValueError(f"Unsupported scatter colour: {color_by}")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Pointwise mutual information, PMI (bits)")
    ax.set_ylabel(r"$\phi$ correlation")
    ax.set_title("Pairwise hard-bit association: PMI versus phi" + title_suffix)
    colorbar = fig.colorbar(color_artist, ax=ax)
    colorbar.set_label(colorbar_label)
    if color_by == "joint_share":
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    texts = [
        ax.text(
            row.pmi_bits,
            row.phi_correlation,
            f"({bit_key(row.bit_a)},{bit_key(row.bit_b)})",
            color="black",
            fontsize=7.2,
            ha="center",
            va="center",
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.18},
        )
        for row in labelled.itertuples(index=False)
    ]
    np.random.seed(1234)
    adjust_text(
        texts,
        x=labelled["pmi_bits"].to_numpy(),
        y=labelled["phi_correlation"].to_numpy(),
        ax=ax,
        force_text=(0.9, 1.2),
        force_static=(0.45, 0.65),
        force_pull=(0.08, 0.14),
        expand=(1.25, 1.40),
        max_move=(40, 55),
        only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
        ensure_inside_axes=False,
        min_arrow_len=0,
        time_lim=8,
        arrowprops={"arrowstyle": "-", "color": "#202020", "alpha": 0.72, "lw": 0.72},
    )
    if color_by == "pmi":
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def source_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def canonical_membership(membership: list[int] | np.ndarray) -> list[int]:
    """Relabel communities in stable order of their smallest bit index."""
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(membership):
        groups.setdefault(int(label), []).append(index)
    mapping = {
        original: new
        for new, (original, _) in enumerate(
            sorted(groups.items(), key=lambda item: min(item[1]))
        )
    }
    return [mapping[int(label)] for label in membership]


def weighted_modularity(
    n_bits: int, edges: list[tuple[int, int]], weights: list[float], membership: list[int],
    resolution: float,
) -> float:
    """Return NetworkX's normalized weighted modularity, including isolates."""
    if not edges:
        return 0.0
    graph = nx.Graph()
    graph.add_nodes_from(range(n_bits))
    graph.add_weighted_edges_from(
        [(left, right, weight) for (left, right), weight in zip(edges, weights)]
    )
    groups: dict[int, set[int]] = {}
    for index, label in enumerate(membership):
        groups.setdefault(int(label), set()).add(index)
    return float(
        nx.algorithms.community.modularity(
            graph, list(groups.values()), weight="weight", resolution=resolution
        )
    )


def leiden_membership(
    n_bits: int,
    edges: list[tuple[int, int]],
    weights: list[float],
    resolution: float,
    seed: int,
) -> tuple[list[int], float]:
    """Fit one weighted Leiden partition and make isolates singleton communities.

    `RBConfigurationVertexPartition` is the weighted configuration-model
    modularity objective with a tunable resolution parameter.  Passing the
    PMI vector as the named `weights` argument is deliberately explicit: no
    frequency or phi statistic enters the optimisation objective.
    """
    if not edges:
        # Modularity is defined here as zero on the edgeless graph.  Singleton
        # communities retain every isolated bit in the result rather than
        # treating an empty graph as a failed fit.
        return list(range(n_bits)), 0.0
    if len(edges) != len(weights) or not np.isfinite(weights).all() or not np.all(np.asarray(weights) > 0):
        raise ValueError("Leiden requires one finite, strictly positive PMI weight per edge")
    graph = ig.Graph(n=n_bits, edges=edges, directed=False)
    graph.es["weight"] = weights
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=seed,
    )
    membership = list(map(int, partition.membership))
    degree = np.zeros(n_bits, dtype=int)
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    # A zero-degree node has no positive-PMI evidence linking it to another
    # bit.  Force it to a singleton after optimisation; this leaves weighted
    # modularity unchanged but prevents arbitrary attachment to a community.
    next_label = max(membership, default=-1) + 1
    for index in np.flatnonzero(degree == 0):
        membership[int(index)] = next_label
        next_label += 1
    membership = canonical_membership(membership)
    return membership, weighted_modularity(n_bits, edges, weights, membership, resolution)


def community_self_checks() -> dict[str, bool]:
    """Small deterministic checks for the two edge cases documented in the report."""
    isolated_membership, isolated_modularity = leiden_membership(
        n_bits=3, edges=[], weights=[], resolution=1.0, seed=1234
    )
    return {
        "edgeless_graph_keeps_singleton_nodes": isolated_membership == [0, 1, 2],
        "edgeless_graph_modularity_is_zero": isolated_modularity == 0.0,
    }


def positive_pmi_graph(bits: list[str], edges: pd.DataFrame) -> nx.Graph:
    """Construct the complete positive-PMI graph, retaining isolated bit nodes."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(bits)))
    for row in edges.itertuples(index=False):
        graph.add_edge(
            bit_key(row.bit_a), bit_key(row.bit_b), weight=float(row.pmi_bits)
        )
    return graph


def save_community_graph(
    bits: list[str],
    prevalence: np.ndarray,
    edges: pd.DataFrame,
    membership: list[int],
    resolution: float,
    modularity: float,
    path: Path,
) -> dict[int, np.ndarray]:
    """Plot every positive-PMI edge with reproducible force layout."""
    graph = positive_pmi_graph(bits, edges)
    layout = nx.spring_layout(
        graph, seed=COMMUNITY_LAYOUT_SEED, weight="weight", iterations=500
    )
    fig, ax = plt.subplots(figsize=(10.8, 8.8))
    if graph.number_of_edges():
        values = np.asarray([graph[left][right]["weight"] for left, right in graph.edges()], dtype=float)
        widths = 0.55 + 3.9 * values / values.max()
        nx.draw_networkx_edges(
            graph, layout, ax=ax, width=widths, alpha=0.36, edge_color="#4c566a"
        )
    communities = np.asarray(membership, dtype=int)
    palette = plt.get_cmap("tab20", max(1, int(communities.max()) + 1))
    node_sizes = 420 + 2300 * prevalence
    nx.draw_networkx_nodes(
        graph,
        layout,
        ax=ax,
        node_size=node_sizes,
        node_color=[palette(label) for label in communities],
        edgecolors="#202020",
        linewidths=0.8,
        alpha=0.96,
    )
    nx.draw_networkx_labels(
        graph,
        layout,
        labels={index: name for index, name in enumerate(bits)},
        font_size=8.5,
        font_weight="bold",
        ax=ax,
    )
    ax.set_title(
        "Positive-PMI hard-bit communities\n"
        f"Leiden weighted modularity, resolution {resolution:g}; Q = {modularity:.3f}"
    )
    ax.text(
        0.01,
        0.01,
        "node area ∝ activation prevalence; edge width ∝ PMI (bits)",
        transform=ax.transAxes,
        fontsize=8,
        color="#303030",
        bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.9, "pad": 2.5},
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return layout


def load_posthoc_category_overlay(
    path: Path, run_name: str, bits: list[str]
) -> tuple[pd.DataFrame, list[str], str, Path]:
    """Load exactly the categories marked as selected in the existing heatmap JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if run_name not in payload.get("runs", {}):
        raise ValueError(f"Run {run_name!r} is not present in {path}")
    association = payload["runs"][run_name].get("strongest_category_associations", {})
    entries = association.get("bits", [])
    by_bit = {entry.get("bit"): entry for entry in entries}
    if set(by_bit) != set(bits):
        missing = sorted(set(bits) - set(by_bit))
        unexpected = sorted(set(by_bit) - set(bits))
        raise ValueError(
            "Post-hoc category source does not match the encoded bit columns; "
            f"missing={missing}, unexpected={unexpected}"
        )
    category_order = [str(category) for category in payload.get("classes", [])]
    order = {category: index for index, category in enumerate(category_order)}
    rows: list[dict[str, str]] = []
    for bit in bits:
        selected = [
            str(row["category"])
            for row in by_bit[bit].get("selected_categories", [])
        ]
        selected = sorted(set(selected), key=lambda category: order.get(category, len(order)))
        display = " / ".join(selected) if selected else "no selected category"
        rows.append(
            {"bit": bit, "selected_categories": display, "colour_category": display}
        )
    label_column = str(payload.get("label_column", "external category"))
    label_path = Path(payload.get("labels", {}).get("path", ""))
    if not label_path.is_file():
        raise ValueError(
            "The post-hoc category JSON does not point to a readable label file: "
            f"{label_path}"
        )
    return pd.DataFrame(rows), category_order, label_column, label_path


def save_posthoc_category_graph(
    bits: list[str],
    prevalence: np.ndarray,
    edges: pd.DataFrame,
    layout: dict[int, np.ndarray],
    overlay: pd.DataFrame,
    category_order: list[str],
    label_column: str,
    path: Path,
) -> None:
    """Draw the same graph geometry with colours from a declared external heatmap."""
    graph = positive_pmi_graph(bits, edges)
    category_by_bit = overlay.set_index("bit")["colour_category"].to_dict()
    shown_categories = [category_by_bit[bit] for bit in bits]
    base_colours = plt.get_cmap("tab10", max(1, len(category_order)))
    colours = {
        category: base_colours(index)
        for index, category in enumerate(category_order)
    }
    colours.setdefault("no selected category", "#9e9e9e")
    for category in shown_categories:
        if category not in colours:  # tied maxima are displayed neutrally, not arbitrarily broken.
            colours[category] = "#7f7f7f"
    fig, ax = plt.subplots(figsize=(10.8, 8.8))
    if graph.number_of_edges():
        values = np.asarray([graph[left][right]["weight"] for left, right in graph.edges()], dtype=float)
        widths = 0.55 + 3.9 * values / values.max()
        nx.draw_networkx_edges(
            graph, layout, ax=ax, width=widths, alpha=0.36, edge_color="#4c566a"
        )
    nx.draw_networkx_nodes(
        graph,
        layout,
        ax=ax,
        node_size=420 + 2300 * prevalence,
        node_color=[colours[category] for category in shown_categories],
        edgecolors="#202020",
        linewidths=0.8,
        alpha=0.96,
    )
    nx.draw_networkx_labels(
        graph,
        layout,
        labels={index: name for index, name in enumerate(bits)},
        font_size=8.5,
        font_weight="bold",
        ax=ax,
    )
    legend_categories = [
        category for category in category_order if category in set(shown_categories)
    ] + [
        category for category in sorted(set(shown_categories)) if category not in category_order
    ]
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", markersize=8,
                markerfacecolor=colours[category], markeredgecolor="#202020", label=category,
            )
            for category in legend_categories
        ],
        title=f"Selected {label_column} category",
        frameon=True,
        loc="upper left",
        fontsize=8,
        title_fontsize=8,
    )
    ax.set_title(
        "Positive-PMI hard-bit graph: post-hoc category overlay\n"
        "Same graph/layout as the community figure; colours are external heatmap maxima"
    )
    ax.text(
        0.01,
        0.01,
        "node area ∝ activation prevalence; edge width ∝ PMI (bits)",
        transform=ax.transAxes,
        fontsize=8,
        color="#303030",
        bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.9, "pad": 2.5},
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def posthoc_category_colours(
    category_order: list[str], observed: list[str] | set[str]
) -> dict[str, Any]:
    """Use one stable categorical palette in every ex-post category artifact."""
    palette = plt.get_cmap("tab10", max(1, len(category_order)))
    colours: dict[str, Any] = {
        category: palette(index) for index, category in enumerate(category_order)
    }
    colours.setdefault("no selected category", "#9e9e9e")
    for category in observed:
        if category not in colours:  # tied maxima are not arbitrarily broken.
            colours[category] = "#7f7f7f"
    return colours


def save_joint_profile_label_sankey(
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
    category_order: list[str],
    label_column: str,
    path: Path,
) -> None:
    """Draw counts from jointly satisfied Leiden-community profiles to labels."""
    communities = summary["community"].astype(int).tolist()
    labels = [
        category for category in category_order
        if int(distribution.loc[distribution["label"] == category, "documents"].sum()) > 0
    ]
    if not communities or not labels:
        raise ValueError("Cannot draw a community-label Sankey without non-empty communities and labels")
    flows = np.asarray(
        [
            [
                int(
                    distribution.loc[
                        (distribution["community"] == community)
                        & (distribution["label"] == label),
                        "documents",
                    ].sum()
                )
                for label in labels
            ]
            for community in communities
        ],
        dtype=float,
    )
    total = float(flows.sum())
    if total <= 0:
        raise ValueError("Cannot draw a community-label Sankey with zero document flow")
    margin, gap = 0.035, 0.025
    usable_left = 1.0 - 2 * margin - gap * (len(communities) - 1)
    usable_right = 1.0 - 2 * margin - gap * (len(labels) - 1)
    left_heights = flows.sum(axis=1) / total * usable_left
    right_heights = flows.sum(axis=0) / total * usable_right

    def bottoms(heights: np.ndarray) -> np.ndarray:
        out = np.zeros_like(heights)
        top = 1.0 - margin
        for index, height in enumerate(heights):
            out[index] = top - height
            top = out[index] - gap
        return out

    left_bottom = bottoms(left_heights)
    right_bottom = bottoms(right_heights)
    left_offset = left_bottom.copy()
    right_offset = right_bottom.copy()
    left_palette = plt.get_cmap("tab20", max(1, len(communities)))
    category_colours = posthoc_category_colours(labels, labels)
    figure, axis = plt.subplots(figsize=(14.5, max(7.0, 1.65 * max(len(communities), len(labels)))))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    left_x0, left_x1 = 0.09, 0.12
    right_x0, right_x1 = 0.87, 0.90
    for community_index, community in enumerate(communities):
        for label_index, _label in enumerate(labels):
            height = flows[community_index, label_index] / total * usable_left
            if height <= 0:
                continue
            source_y, target_y = left_offset[community_index], right_offset[label_index]
            vertices = [
                (left_x1, source_y), (0.39, source_y), (0.61, target_y), (right_x0, target_y),
                (right_x0, target_y + height), (0.61, target_y + height),
                (0.39, source_y + height), (left_x1, source_y + height), (left_x1, source_y),
            ]
            codes = [
                MatplotlibPath.MOVETO, MatplotlibPath.CURVE4, MatplotlibPath.CURVE4,
                MatplotlibPath.CURVE4, MatplotlibPath.LINETO, MatplotlibPath.CURVE4,
                MatplotlibPath.CURVE4, MatplotlibPath.CURVE4, MatplotlibPath.CLOSEPOLY,
            ]
            axis.add_patch(
                PathPatch(
                    MatplotlibPath(vertices, codes),
                    facecolor=left_palette(community_index), edgecolor="none", alpha=0.60, zorder=1,
                )
            )
            left_offset[community_index] += height
            right_offset[label_index] += height
    for index, community in enumerate(communities):
        row = summary.loc[summary["community"] == community].iloc[0]
        axis.add_patch(
            Rectangle(
                (left_x0, left_bottom[index]), left_x1 - left_x0, left_heights[index],
                facecolor=left_palette(index), edgecolor="white", linewidth=0.7, zorder=3,
            )
        )
        axis.text(
            left_x0 - 0.012, left_bottom[index] + left_heights[index] / 2,
            f"Community {community}\n{row.bits}\n{row.documents_with_all_community_bits:,} documents",
            ha="right", va="center", fontsize=8,
        )
    for index, label in enumerate(labels):
        count = int(flows[:, index].sum())
        axis.add_patch(
            Rectangle(
                (right_x0, right_bottom[index]), right_x1 - right_x0, right_heights[index],
                facecolor=category_colours[label], edgecolor="white", linewidth=0.7, zorder=3,
            )
        )
        axis.text(
            right_x1 + 0.012, right_bottom[index] + right_heights[index] / 2,
            f"{label}\n{count:,} flows",
            ha="left", va="center", fontsize=9,
        )
    axis.text((left_x0 + left_x1) / 2, 1.012, "Leiden communities", ha="center", va="bottom", fontsize=11)
    axis.text((right_x0 + right_x1) / 2, 1.012, label_column, ha="center", va="bottom", fontsize=11)
    axis.set_title(
        "Jointly active community profiles → post-hoc labels\n"
        "Ribbon width = number of documents; ribbons are coloured by source community",
        fontsize=14,
        pad=24,
    )
    figure.text(
        0.5, 0.008,
        "A document contributes to a community only if every bit in that community is active. "
        "If a document satisfies multiple community conjunctions, it contributes once to each flow.",
        ha="center", va="bottom", fontsize=8.5,
    )
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def build_posthoc_profile_label_distribution(
    encoded_frame: pd.DataFrame,
    bits: list[str],
    node_table: pd.DataFrame,
    category_overlay: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Describe labels among profiles that activate every bit of a community jointly.

    A community is matched only where all of its member bits are one.  A profile can
    satisfy more than one community's conjunction, so the resulting subsets may still
    overlap; this is recorded rather than resolved by an arbitrary document partition.
    """
    if "id" not in encoded_frame:
        raise ValueError("Post-hoc profile analysis requires an id column in the encoded CSV")
    label_column = str(category_overlay["label_column"])
    labels = pd.read_csv(
        category_overlay["label_path"],
        usecols=["id", label_column],
        dtype={"id": str},
    )
    if labels["id"].duplicated().any() or encoded_frame["id"].duplicated().any():
        raise ValueError("Post-hoc label distribution requires unique encoded and label IDs")
    code_columns = ["id"] + (["text"] if "text" in encoded_frame else []) + bits
    codes = encoded_frame[code_columns].copy()
    codes["id"] = codes["id"].astype(str)
    joined = codes.merge(labels, on="id", how="inner", validate="one_to_one")
    if len(joined) != len(codes):
        raise ValueError(
            f"Post-hoc labels align to {len(joined):,} of {len(codes):,} encoded documents"
        )
    joined[label_column] = joined[label_column].astype(str)
    category_order = list(category_overlay["categories"])
    category_order.extend(
        category
        for category in sorted(joined[label_column].unique())
        if category not in category_order
    )
    bit_to_index = {bit: index for index, bit in enumerate(bits)}
    hard_matrix = joined[bits].to_numpy(dtype=np.int8)
    patterns = profile_strings(hard_matrix)
    community_bits = (
        node_table.sort_values("community_resolution_1")
        .groupby("community_resolution_1", sort=True)["bit"]
        .apply(list)
    )
    membership_matrix = np.zeros((len(joined), len(community_bits)), dtype=bool)
    summaries: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    joint_profiles: list[dict[str, Any]] = []
    text_samples: list[dict[str, Any]] = []
    for position, (community, members) in enumerate(community_bits.items()):
        bit_columns = [bit_to_index[bit] for bit in members]
        present = hard_matrix[:, bit_columns].all(axis=1)
        membership_matrix[:, position] = present
        subset = joined.loc[present, label_column]
        counts = subset.value_counts().reindex(category_order, fill_value=0)
        rates = counts / len(subset) if len(subset) else counts.astype(float)
        dominant = rates[rates == rates.max()].index.tolist() if len(subset) else []
        summaries.append(
            {
                "community": int(community),
                "bits": "; ".join(members),
                "documents_with_all_community_bits": int(present.sum()),
                "share_of_all_documents": float(present.mean()),
                "distinct_exact_profiles": int(pd.Series(patterns[present]).nunique()),
                "dominant_label": " / ".join(dominant),
                "dominant_label_share": float(rates.max()) if len(subset) else np.nan,
            }
        )
        for category, count, rate in zip(category_order, counts, rates):
            distributions.append(
                {
                    "community": int(community),
                    "bits": "; ".join(members),
                    "label_column": label_column,
                    "label": category,
                    "documents": int(count),
                    "within_community_document_share": float(rate),
                }
            )
        profile_frame = pd.DataFrame(
            {"profile": patterns[present], "label": subset.to_numpy()}
        )
        for profile, profile_subset in profile_frame.groupby("profile", sort=False):
            profile_counts = profile_subset["label"].value_counts().reindex(category_order, fill_value=0)
            profile_rates = profile_counts / len(profile_subset)
            profile_dominant = profile_rates[profile_rates == profile_rates.max()].index.tolist()
            joint_profiles.append(
                {
                    "community": int(community),
                    "bits_required_jointly": "; ".join(members),
                    "profile": profile,
                    "documents": int(len(profile_subset)),
                    "share_of_community_documents": float(len(profile_subset) / present.sum()),
                    "dominant_label": " / ".join(profile_dominant),
                    "dominant_label_share": float(profile_rates.max()),
                    "label_counts_json": json.dumps(
                        {category: int(count) for category, count in profile_counts.items()}
                    ),
                }
            )
        if "text" in joined:
            for category_index, category in enumerate(category_order):
                eligible = np.flatnonzero(
                    present & (joined[label_column].to_numpy() == category)
                )
                if not len(eligible):
                    continue
                generator = np.random.default_rng(1234 + 1009 * int(community) + category_index)
                selected = generator.choice(eligible, size=min(10, len(eligible)), replace=False)
                for rank, document_index in enumerate(selected, start=1):
                    text_samples.append(
                        {
                            "community": int(community),
                            "bits_required_jointly": "; ".join(members),
                            "label_column": label_column,
                            "label": category,
                            "sample_rank": rank,
                            "sample_seed": 1234 + 1009 * int(community) + category_index,
                            "id": joined.iloc[document_index]["id"],
                            "profile": patterns[document_index],
                            "text": joined.iloc[document_index]["text"],
                        }
                    )
    summary = pd.DataFrame(summaries)
    distribution = pd.DataFrame(distributions)
    joint_profile_table = pd.DataFrame(joint_profiles).sort_values(
        ["community", "documents", "profile"], ascending=[True, False, True]
    )
    overlap = pd.DataFrame(
        {"jointly_satisfied_leiden_communities_per_document": membership_matrix.sum(axis=1)}
    ).value_counts().rename("documents").reset_index()
    overlap["share_of_all_documents"] = overlap["documents"] / len(joined)
    summary.to_csv(output_dir / "bit_community_profile_label_summary.csv", index=False)
    distribution.to_csv(output_dir / "bit_community_profile_label_distribution.csv", index=False)
    joint_profile_table.to_csv(output_dir / "bit_community_joint_profiles.csv", index=False)
    text_sample_table = pd.DataFrame(text_samples)
    if not text_sample_table.empty:
        text_sample_table.to_csv(
            output_dir / "bit_community_joint_profile_text_samples.csv",
            index=False,
            encoding="utf-8",
        )
    overlap.to_csv(output_dir / "bit_community_profile_overlap.csv", index=False)

    save_joint_profile_label_sankey(
        summary,
        distribution,
        category_order,
        label_column,
        output_dir / "bit_community_profile_label_sankey.png",
    )
    return {
        "summary": summary,
        "distribution": distribution,
        "overlap": overlap,
        "joint_profiles": joint_profile_table,
        "text_samples": text_sample_table,
        "categories": category_order,
        "label_column": label_column,
        "matched_documents": int(len(joined)),
    }


def build_community_analysis(
    bits: list[str],
    prevalence: np.ndarray,
    pair_table: pd.DataFrame,
    output_dir: Path,
    run_dir: Path,
    encoded_path: Path,
    posthoc_category_json: Path | None = None,
    posthoc_category_run: str | None = None,
) -> dict[str, Any]:
    """Build the complete label-free weighted positive-PMI community analysis."""
    positive = pair_table[
        np.isfinite(pair_table["pmi_bits"]) & (pair_table["pmi_bits"] > 0)
    ].copy()
    positive = positive.sort_values(["bit_a", "bit_b"], key=lambda col: col.map(bit_key))
    positive["weight_pmi_bits"] = positive["pmi_bits"].astype(float)
    positive.to_csv(output_dir / "bit_community_edges.csv", index=False)
    bit_to_index = {bit: index for index, bit in enumerate(bits)}
    edges = [
        (bit_to_index[row.bit_a], bit_to_index[row.bit_b])
        for row in positive.itertuples(index=False)
    ]
    weights = positive["pmi_bits"].to_numpy(dtype=float).tolist()
    if not np.allclose(np.asarray(weights), positive["weight_pmi_bits"].to_numpy(dtype=float)):
        raise AssertionError("Graph weights must equal pmi_bits exactly")
    if len({tuple(sorted(edge)) for edge in edges}) != len(edges):
        raise AssertionError("Positive-PMI graph contains duplicate edges")
    if any(left == right for left, right in edges):
        raise AssertionError("Positive-PMI graph must not contain self loops")

    degree = np.zeros(len(bits), dtype=int)
    weighted_degree = np.zeros(len(bits), dtype=float)
    for (left, right), weight in zip(edges, weights):
        degree[left] += 1
        degree[right] += 1
        weighted_degree[left] += weight
        weighted_degree[right] += weight
    rows: list[dict[str, Any]] = []
    partitions: dict[tuple[float, int], list[int]] = {}
    for resolution in COMMUNITY_RESOLUTIONS:
        for seed in COMMUNITY_SEEDS:
            membership, modularity = leiden_membership(
                len(bits), edges, weights, resolution, seed
            )
            partitions[(resolution, seed)] = membership
            rows.append(
                {
                    "resolution": resolution,
                    "seed": seed,
                    "weighted_modularity": modularity,
                    "n_communities": len(set(membership)),
                    "membership_by_bit_order": json.dumps(membership),
                }
            )
    seed_results = pd.DataFrame(rows).sort_values(
        ["resolution", "weighted_modularity", "seed"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    seed_results.to_csv(output_dir / "bit_community_seed_results.csv", index=False)
    best_by_resolution: dict[float, pd.Series] = {}
    ari_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for resolution in COMMUNITY_RESOLUTIONS:
        current = seed_results[seed_results["resolution"] == resolution].sort_values(
            ["weighted_modularity", "seed"], ascending=[False, True]
        )
        best = current.iloc[0]
        best_by_resolution[resolution] = best
        aris: list[float] = []
        current_seeds = current["seed"].astype(int).tolist()
        for position, first_seed in enumerate(current_seeds):
            for second_seed in current_seeds[position + 1:]:
                ari = float(
                    adjusted_rand_score(
                        partitions[(resolution, first_seed)],
                        partitions[(resolution, second_seed)],
                    )
                )
                aris.append(ari)
                ari_rows.append(
                    {
                        "comparison_type": "within_resolution_seed_pair",
                        "resolution_a": resolution,
                        "resolution_b": resolution,
                        "seed_a": first_seed,
                        "seed_b": second_seed,
                        "adjusted_rand_index": ari,
                    }
                )
        stability_rows.append(
            {
                "resolution": resolution,
                "best_seed": int(best.seed),
                "best_weighted_modularity": float(best.weighted_modularity),
                "best_n_communities": int(best.n_communities),
                "mean_seed_pair_ari": float(np.mean(aris)) if aris else 1.0,
                "min_seed_pair_ari": float(np.min(aris)) if aris else 1.0,
                "max_seed_pair_ari": float(np.max(aris)) if aris else 1.0,
                "seed_pairs": len(aris),
            }
        )
    resolution_ari_rows: list[dict[str, Any]] = []
    for left_index, left_resolution in enumerate(COMMUNITY_RESOLUTIONS):
        for right_resolution in COMMUNITY_RESOLUTIONS[left_index + 1:]:
            left_best = best_by_resolution[left_resolution]
            right_best = best_by_resolution[right_resolution]
            ari = float(
                adjusted_rand_score(
                    partitions[(left_resolution, int(left_best.seed))],
                    partitions[(right_resolution, int(right_best.seed))],
                )
            )
            resolution_ari_rows.append(
                {
                    "resolution_a": left_resolution,
                    "best_seed_a": int(left_best.seed),
                    "resolution_b": right_resolution,
                    "best_seed_b": int(right_best.seed),
                    "adjusted_rand_index": ari,
                }
            )
    seed_ari = pd.DataFrame(ari_rows)
    seed_ari.to_csv(output_dir / "bit_community_seed_ari.csv", index=False)
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output_dir / "bit_community_resolution_stability.csv", index=False)
    resolution_ari = pd.DataFrame(resolution_ari_rows)
    resolution_ari.to_csv(output_dir / "bit_community_resolution_ari.csv", index=False)

    principal = best_by_resolution[1.0]
    principal_membership = partitions[(1.0, int(principal.seed))]
    node_table = pd.DataFrame(
        {
            "bit": bits,
            "activation_prevalence": prevalence,
            "degree_positive_pmi": degree,
            "incident_pmi_weight_sum": weighted_degree,
            "is_isolated": degree == 0,
            "community_resolution_1": principal_membership,
        }
    ).sort_values(["community_resolution_1", "bit"], key=lambda col: col.map(bit_key) if col.name == "bit" else col)
    node_table.to_csv(output_dir / "bit_community_nodes.csv", index=False)
    layout = save_community_graph(
        bits,
        prevalence,
        positive,
        principal_membership,
        1.0,
        float(principal.weighted_modularity),
        output_dir / "bit_community_graph.png",
    )
    category_overlay: dict[str, Any] | None = None
    if posthoc_category_json is not None:
        if not posthoc_category_run:
            raise ValueError("--posthoc_category_run is required with --posthoc_category_json")
        overlay, category_order, label_column, label_path = load_posthoc_category_overlay(
            posthoc_category_json, posthoc_category_run, bits
        )
        overlay.to_csv(output_dir / "bit_community_posthoc_categories.csv", index=False)
        save_posthoc_category_graph(
            bits,
            prevalence,
            positive,
            layout,
            overlay,
            category_order,
            label_column,
            output_dir / "bit_community_graph_posthoc_categories.png",
        )
        category_overlay = {
            "source_json": str(posthoc_category_json.resolve()),
            "run": posthoc_category_run,
            "label_column": label_column,
            "label_path": str(label_path.resolve()),
            "categories": category_order,
            "artifact": "bit_community_graph_posthoc_categories.png",
        }
    repeat_membership, repeat_modularity = leiden_membership(
        len(bits), edges, weights, 1.0, int(principal.seed)
    )
    checks = community_self_checks()
    checks.update(
        {
            "graph_weights_equal_pmi_bits": bool(
                np.allclose(np.asarray(weights), positive["pmi_bits"].to_numpy(dtype=float))
            ),
            "primary_seed_reproducible": repeat_membership == principal_membership
            and np.isclose(repeat_modularity, float(principal.weighted_modularity)),
            "all_bits_retained_as_nodes": len(node_table) == len(bits),
        }
    )
    if not all(checks.values()):
        raise AssertionError(f"Community analysis self-check failed: {checks}")
    metadata = {
        "schema_version": 1,
        "purpose": "label-free positive-PMI hard-bit community analysis",
        "source_run_dir": str(run_dir),
        "source_encoded_csv": str(encoded_path),
        "source_pair_table": str(output_dir / "bit_coactivation.csv"),
        "n_nodes": len(bits),
        "n_possible_pairs": len(pair_table),
        "n_positive_finite_pmi_edges": len(positive),
        "edge_weight_column": "pmi_bits",
        "objective": "weighted RBConfigurationVertexPartition modularity",
        "principal_resolution": 1.0,
        "resolutions": list(COMMUNITY_RESOLUTIONS),
        "seeds": list(COMMUNITY_SEEDS),
        "layout": {"algorithm": "networkx.spring_layout", "seed": COMMUNITY_LAYOUT_SEED, "weight": "pmi_bits"},
        "libraries": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "igraph": ig.__version__,
            "leidenalg": leidenalg.__version__,
            "networkx": nx.__version__,
        },
        "weight_sha256": hashlib.sha256(np.asarray(weights, dtype=np.float64).tobytes()).hexdigest(),
        "self_checks": {name: bool(value) for name, value in checks.items()},
        "posthoc_category_overlay": category_overlay,
    }
    (output_dir / "bit_community_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "positive_edges": len(positive),
        "isolated_nodes": int((degree == 0).sum()),
        "principal_seed": int(principal.seed),
        "principal_modularity": float(principal.weighted_modularity),
        "principal_communities": int(principal.n_communities),
        "stability": stability,
        "resolution_ari": resolution_ari,
        "nodes": node_table,
        "checks": checks,
        "category_overlay": category_overlay,
    }


def write_community_paper_handoff(
    output_dir: Path,
    bits: list[str],
    community: dict[str, Any],
    posthoc_profile_analysis: dict[str, Any] | None = None,
) -> None:
    """Write a run-specific, evidence-only handoff for later manuscript drafting."""
    principal_stability = community["stability"].loc[
        community["stability"]["resolution"] == 1.0
    ].iloc[0]
    memberships = community["nodes"].groupby("community_resolution_1", sort=True)["bit"].apply(list)
    lines = [
        "# Paper handoff: label-free hard-bit co-activation communities",
        "",
        "## What was estimated",
        "",
        "The core graph analysis treats the fitted hard bits as nodes in an undirected graph. "
        "An edge is retained exactly when the empirical pairwise PMI is finite and strictly "
        "positive; its weight is that PMI in bits. Thus the graph records co-activation above "
        "the rate expected under marginal independence. The graph fit does not use an external "
        "label, and it is not a test of semantic significance.",
        "",
        "The graph has "
        f"**{len(bits)} nodes**, **{community['positive_edges']} positive-PMI edges**, and "
        f"**{community['isolated_nodes']} isolated nodes**. The principal partition uses weighted "
        "Leiden (RB configuration-model modularity) at resolution 1.0. The optimizer receives "
        "only the PMI edge weights; joint frequency and phi are retained as descriptive columns, "
        "not as clustering weights.",
        "",
        "## Principal result (resolution 1.0)",
        "",
        f"- Best of 20 fixed seeds: `{community['principal_seed']}`.",
        f"- Weighted modularity: **{community['principal_modularity']:.4f}**.",
        f"- Communities: **{community['principal_communities']}**.",
        f"- Across the 190 seed-pair comparisons, ARI mean/min/max: "
        f"**{principal_stability.mean_seed_pair_ari:.4f} / {principal_stability.min_seed_pair_ari:.4f} / {principal_stability.max_seed_pair_ari:.4f}**.",
        "",
        "Seed-to-seed ARI quantifies optimization stability for this fixed empirical graph. "
        "It is not robustness to a new document sample and should not be described as such.",
        "",
        "## Memberships",
        "",
        "| Community | Bits |",
        "|---:|---|",
    ]
    for label, members in memberships.items():
        lines.append(f"| {label} | {', '.join(members)} |")
    if posthoc_profile_analysis is not None:
        lines.extend(
            [
                "",
                "## Ex-post label composition of jointly active community profiles",
                "",
                "For each community, documents were retained only when **all** its member bits "
                "were active. These are post-hoc descriptive results using the declared external "
                f"`{posthoc_profile_analysis['label_column']}` column; labels did not enter the "
                "graph or the Leiden partition.",
                "",
                "| Community | Documents | Share of corpus | Exact profiles | Dominant label | Dominant-label share |",
                "|---:|---:|---:|---:|---|---:|",
            ]
        )
        for row in posthoc_profile_analysis["summary"].itertuples(index=False):
            lines.append(
                f"| {row.community} | {row.documents_with_all_community_bits:,} | "
                f"{row.share_of_all_documents:.3%} | {row.distinct_exact_profiles:,} | "
                f"{row.dominant_label} | {row.dominant_label_share:.3%} |"
            )
        lines.extend(
            [
                "",
                "The resulting subsets can overlap if a profile jointly satisfies more than one "
                "community; the overlap artifact documents this rather than forcing a document into "
                "a single community.",
            ]
        )
    lines.extend(
        [
            "",
            "## Required interpretation boundary",
            "",
            "A positive PMI edge means only that two bits jointly activate more often than "
            "marginal independence predicts in this encoded corpus. The positive-only graph "
            "intentionally omits negative associations. No edge therefore does not prove "
            "independence, and a community is a co-activation structure rather than a semantic "
            "topic or a causal mechanism. Semantic interpretation requires later document-level "
            "inspection or a separately declared analysis.",
            "",
            "## Artifacts",
            "",
            "- `bit_community_graph.png` and `bit_community_graph.pdf`: principal graph figure.",
            "- `bit_community_edges.csv`: every retained positive-PMI edge, including joint "
            "frequency and phi.",
            "- `bit_community_nodes.csv`: all bits, including isolates, with prevalence, degree, "
            "incident PMI strength, and principal community.",
            "- `bit_community_seed_results.csv`, `bit_community_seed_ari.csv`, "
            "`bit_community_resolution_stability.csv`, and `bit_community_resolution_ari.csv`: "
            "seed and resolution sensitivity.",
            "- `bit_community_metadata.json`: data provenance, exact configuration, package "
            "versions, edge-weight checksum, and checks.",
            "- When supplied, `bit_community_graph_posthoc_categories.*`, "
            "`bit_community_posthoc_categories.csv`, and `bit_community_profile_label_*` "
            "document the strictly post-hoc category overlay and jointly active profile subsets; "
            "`bit_community_joint_profiles.csv` lists every extracted profile and its label counts, "
            "and `bit_community_joint_profile_text_samples.csv` supplies reproducible reading samples.",
        ]
    )
    (output_dir / "paper_handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_encoded_csv(run_dir: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        return supplied
    candidates = sorted(run_dir.glob("encoded_*_best.csv"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one encoded_*_best.csv in {run_dir}; found {len(candidates)}. "
            "Pass --encoded_csv explicitly."
        )
    return candidates[0]


def build_report(
    run_dir: Path,
    encoded_path: Path,
    output_dir: Path,
    top_profiles: int,
    posthoc_category_json: Path | None = None,
    posthoc_category_run: str | None = None,
) -> None:
    header = pd.read_csv(encoded_path, nrows=0).columns.tolist()
    bits = sorted((column for column in header if BIT_RE.match(column)), key=bit_key)
    if not bits:
        raise ValueError(f"No hard-bit columns in {encoded_path}")
    usecols = (
        (["id"] if "id" in header else [])
        + (["text"] if posthoc_category_json is not None and "text" in header else [])
        + bits
        + (["n_active"] if "n_active" in header else [])
    )
    frame = pd.read_csv(
        encoded_path, usecols=usecols, dtype={"id": str} if "id" in header else None
    )
    matrix = frame[bits].to_numpy(dtype=np.int8, copy=True)
    if not np.isin(matrix, [0, 1]).all():
        raise ValueError("Hard-code file contains values other than 0 and 1")
    n_documents, n_bits = matrix.shape
    active = matrix.sum(axis=1)
    prevalence = matrix.mean(axis=0)
    bit_entropy = binary_entropy(prevalence)
    statuses = np.select(
        [
            prevalence == 0,
            prevalence == 1,
            prevalence <= 0.05,
            prevalence >= 0.95,
            (prevalence >= 0.40) & (prevalence <= 0.60),
        ],
        ["dead", "universal", "rare (≤5%)", "near-universal (≥95%)", "near-balanced (40–60%)"],
        default="selective",
    )
    bit_stats = pd.DataFrame(
        {
            "bit": bits,
            "activation_count": matrix.sum(axis=0),
            "activation_share": prevalence,
            "activation_pct": prevalence * 100.0,
            "binary_entropy_bits": bit_entropy,
            "structural_status": statuses,
        }
    ).sort_values("activation_share", ascending=False, ignore_index=True)

    patterns = profile_strings(matrix)
    profile_counts = pd.Series(patterns).value_counts(sort=True)
    profile_probabilities = profile_counts.to_numpy(dtype=float) / n_documents
    profile_entropy = float(-np.sum(profile_probabilities * np.log2(profile_probabilities)))
    profile_table = pd.DataFrame(
        {
            "rank": np.arange(1, min(top_profiles, len(profile_counts)) + 1),
            "profile": profile_counts.index[:top_profiles],
            "count": profile_counts.to_numpy()[:top_profiles],
        }
    )
    profile_table["share"] = profile_table["count"] / n_documents
    profile_table["share_pct"] = profile_table["share"] * 100.0

    joint = matrix.T.astype(float) @ matrix.astype(float) / n_documents
    expected = np.outer(prevalence, prevalence)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(joint / expected)
    # A zero observed joint activation is a genuine -infinity plug-in PMI. It
    # stays in the tabular artifacts but is masked later for the finite heatmap.
    pmi[expected == 0] = np.nan
    np.fill_diagonal(pmi, 0.0)
    centered = matrix.astype(float) - prevalence
    denom = np.sqrt(np.outer(prevalence * (1.0 - prevalence), prevalence * (1.0 - prevalence)))
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (joint - expected) / denom
    phi[~np.isfinite(phi)] = np.nan
    np.fill_diagonal(phi, 1.0)
    pairs = []
    for first in range(n_bits):
        for second in range(first + 1, n_bits):
            pairs.append(
                {
                    "bit_a": bits[first],
                    "bit_b": bits[second],
                    "joint_activation_count": int(
                        matrix[:, first].astype(np.int64)
                        @ matrix[:, second].astype(np.int64)
                    ),
                    "joint_activation_share": joint[first, second],
                    "expected_if_independent": expected[first, second],
                    "pmi_bits": pmi[first, second],
                    "phi_correlation": phi[first, second],
                }
            )
    pair_table = pd.DataFrame(pairs).sort_values("pmi_bits", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    bit_stats.to_csv(output_dir / "bit_statistics.csv", index=False)
    profile_table.to_csv(output_dir / "profile_prevalence_top20.csv", index=False)
    pair_table.to_csv(output_dir / "bit_coactivation.csv", index=False)
    community = build_community_analysis(
        bits,
        prevalence,
        pair_table,
        output_dir,
        run_dir,
        encoded_path,
        posthoc_category_json,
        posthoc_category_run,
    )
    posthoc_profile_analysis: dict[str, Any] | None = None
    if community["category_overlay"] is not None:
        posthoc_profile_analysis = build_posthoc_profile_label_distribution(
            frame,
            bits,
            community["nodes"],
            community["category_overlay"],
            output_dir,
        )
    write_community_paper_handoff(
        output_dir, bits, community, posthoc_profile_analysis
    )
    save_activation_plot(bit_stats, output_dir / "bit_activation_prevalence.png")
    save_heatmap(
        phi,
        bits,
        output_dir / "bit_coactivation_phi.png",
        "Hard-bit co-activation: phi correlation",
        -1,
        1,
        "coolwarm",
        "Phi correlation",
    )
    pmi_for_plot = pmi.copy()
    pmi_for_plot[~np.isfinite(pmi_for_plot)] = np.nan
    finite_pmi = pmi_for_plot[np.isfinite(pmi_for_plot)]
    bound = max(1.0, float(np.nanquantile(np.abs(finite_pmi), 0.95))) if finite_pmi.size else 1.0
    save_heatmap(
        pmi_for_plot,
        bits,
        output_dir / "bit_coactivation_pmi.png",
        "Hard-bit co-activation: pointwise mutual information",
        -bound,
        bound,
        PMI_CMAP,
        "PMI (bits)",
    )
    save_pmi_phi_scatter(pair_table, output_dir / "pmi_vs_phi_associations.png")
    save_pmi_phi_scatter(
        pair_table,
        output_dir / "pmi_vs_phi_joint_activation_share.png",
        color_by="joint_share",
    )
    save_profile_plot(profile_table, output_dir / "profile_prevalence_top20.png")

    metadata = source_metadata(run_dir)
    target_active = metadata.get("target_active")
    model_name = metadata.get("model_name", run_dir.name)
    diagnostics = run_dir / "training_convergence_diagnostics.png"
    diagnostics_link = (
        f"[training diagnostic](../{diagnostics.name})"
        if diagnostics.is_file()
        else "No compact training diagnostic was available."
    )
    top_share = float(profile_table["share"].sum()) if not profile_table.empty else 0.0
    positive_pairs = pair_table[pair_table["pmi_bits"] > 0].copy()
    # Phi remains finite when a pair has no observed joint activations, whereas
    # its PMI is -infinity. Sort on phi so complete exclusions remain visible.
    low_pairs = pair_table.dropna(subset=["phi_correlation"]).sort_values(
        "phi_correlation", ascending=True
    ).head(5)
    principal_stability = community["stability"].loc[
        community["stability"]["resolution"] == 1.0
    ].iloc[0]
    principal_nodes = community["nodes"]

    def pmi_text(value: float) -> str:
        if value == -math.inf:
            return "−∞ (no joint activation)"
        return f"{value:.3f}"
    scope = (
        [
            "The core bit-structure and Leiden-community analysis uses no outcome, taxonomy, "
            "or annotation column. This run also includes an explicitly marked ex-post category "
            "overlay and profile-label composition section; those external labels are never used "
            "to construct the PMI graph or fit Leiden. A bit is called *selective* here when it "
            "fires on a restricted subset of documents; this is not by itself a claim that it "
            "predicts any particular label.",
        ]
        if community["category_overlay"] is not None
        else [
            "This report deliberately uses no outcome, taxonomy, or annotation column. It "
            "describes only the internal organization of one fitted binary code. A bit is "
            "called *selective* here when it fires on a restricted subset of documents; this "
            "is not a claim that it predicts any particular label.",
        ]
    )
    lines = [
        "# Label-free binary-code structure report",
        "",
        "## Scope",
        "",
        *scope,
        "",
        "## Definitions used in the tables",
        "",
        "- **Hard activation.** A document activates a bit when its thresholded code is "
        "exactly `1`; all reported rates use the complete encoded corpus.",
        "- **Target active bits/document.** This is the optimizer's requested average, "
        "not a post-hoc constraint. The observed mean assesses whether the fitted code "
        "reached it.",
        "- **Binary entropy.** For a bit with activation share `p`, "
        "`h(p) = −p log2(p) − (1−p) log2(1−p)`. It ranges from 0 bits (always off or "
        "on) to 1 bit (50% activation). Higher means more marginal variation, not "
        "automatically more semantic value.",
        "- **Exact profile.** The complete ordered hard-code string for one document. "
        "Observed profiles count distinct strings; profile entropy is the categorical "
        "entropy of their empirical frequency distribution. `2^H(B)` is the number of "
        "equally frequent profiles that would have the same entropy, a summary of "
        "effective rather than merely observed profile diversity.",
        "- **Structural status.** `dead` and `universal` mean exactly 0% and 100% "
        "activation. `rare` and `near-universal` use descriptive 5% and 95% cutoffs. "
        "`near-balanced` means 40–60%. These are reporting labels, not tests of "
        "semantic quality.",
        "- **PMI.** `log2[P(A=1,B=1)/(P(A=1)P(B=1))]`. Positive values indicate more "
        "joint activation than independence predicts; negative values indicate less. "
        "`−∞` means the pair never jointly activated in this finite sample. There is no "
        "universal numerical cutoff for a ‘strong’ PMI: its magnitude depends on the "
        "marginal activation rates. The pair tables rank observed values only and are "
        "not significance tests.",
        "- **Phi correlation.** The Pearson correlation of two binary activation vectors, "
        "ranging from −1 (mutual exclusion) through 0 (no linear association) to +1 "
        "(co-activation). It remains finite when observed joint activation is zero.",
        "",
        "## Code overview",
        "",
        "| Property | Value |",
        "|---|---:|",
        f"| Backbone | {model_name} |",
        f"| Documents encoded | {n_documents:,} |",
        f"| Binary bits | {n_bits} |",
        f"| Target active bits/document | {target_active if target_active is not None else 'not recorded'} |",
        f"| Mean active bits/document | {active.mean():.3f} |",
        f"| Median active bits/document | {np.median(active):.0f} |",
        f"| Observed exact profiles | {len(profile_counts):,} |",
        f"| Profile entropy | {profile_entropy:.3f} bits |",
        f"| Effective equally frequent profiles, `2^H(B)` | {2 ** profile_entropy:,.1f} |",
        f"| Exact dead / universal bits | {(prevalence == 0).sum()} / {(prevalence == 1).sum()} |",
        f"| Near-universal / rare bits | {(prevalence >= 0.95).sum()} / {(prevalence <= 0.05).sum()} |",
        "",
        "## Bit prevalence and intrinsic selectivity",
        "",
        "A bit near 50% activation has high marginal binary entropy; a bit near 0% or "
        "100% has little marginal variation. Neither property by itself establishes a "
        "semantic interpretation. Blue bars are rare (≤5%), red bars are near-universal "
        "(≥95%), and green bars lie between these ranges.",
        "",
        "![Hard-bit activation prevalence](bit_activation_prevalence.png)",
        "",
        "| Bit | Active documents | Activation | Binary entropy (bits) | Structural status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in bit_stats.itertuples(index=False):
        lines.append(
            f"| {row.bit} | {row.activation_count:,} | {row.activation_pct:.2f}% | "
            f"{row.binary_entropy_bits:.3f} | {row.structural_status} |"
        )
    lines.extend(
        [
            "",
            "## Co-activation and mutual exclusion",
            "",
            "The phi heatmap is the ordinary Pearson correlation of two binary activation "
            "vectors. The PMI heatmap compares the observed joint activation with that "
            "expected under independent firing: positive values indicate co-activation, "
            "negative values mutual exclusion, and values near zero approximate independence. "
            "The PMI heatmap masks a `−∞` value (no observed joint activation), whereas "
            "the phi heatmap retains its finite correlation. "
            "These are descriptive dependence measures, not evidence of causality or of "
            "distinct semantic concepts.",
            "",
            "![Phi co-activation matrix](bit_coactivation_phi.png)",
            "",
            "![PMI co-activation matrix](bit_coactivation_pmi.png)",
            "",
            "### All positive-PMI associations",
            "",
            "Every pair whose observed PMI exceeds zero is listed below. The scatter places "
            "the same pairwise observations by PMI (x-axis) and phi (y-axis). Blue marks "
            "negative PMI, grey values near zero, and red increasingly positive PMI; labels use "
            "the run-local bit indices `(i,j)`, for example `(0,12)` for `bit_00` and "
            "`bit_12`. To keep the static graph readable, black repelled labels and leader "
            "segments identify the 20 largest and 10 smallest finite PMI values. This table "
            "remains the exhaustive record of positive-PMI pairs.",
            "",
            "![PMI versus phi, labelled positive associations](pmi_vs_phi_associations.png)",
            "",
            "### Alternative colour encoding: observed joint share",
            "",
            "This second scatter has identical axes and the same repelled labels, but its "
            "blue-to-red sequential colour scale is the raw share of all documents for which both bits "
            "are active, `P(A=1,B=1)`. It emphasizes how common a pair is, rather than "
            "whether it occurs more than independence predicts. The scale is linear from "
            "smallest to the largest observed joint share in this run; read it from the colourbar.",
            "",
            "![PMI versus phi, dot colour is joint activation share](pmi_vs_phi_joint_activation_share.png)",
            "",
            "| Bit A | Bit B | Joint documents | Joint share | PMI (bits) | Phi |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in positive_pairs.itertuples(index=False):
        lines.append(
            f"| {row.bit_a} | {row.bit_b} | {row.joint_activation_count:,} | "
            f"{row.joint_activation_share:.3%} | {pmi_text(row.pmi_bits)} | "
            f"{row.phi_correlation:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Five most negative observed phi correlations",
            "",
            "| Bit A | Bit B | Joint documents | PMI (bits) | Phi |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in low_pairs.itertuples(index=False):
        lines.append(
            f"| {row.bit_a} | {row.bit_b} | {row.joint_activation_count:,} | "
            f"{pmi_text(row.pmi_bits)} | {row.phi_correlation:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Positive-PMI co-activation communities",
            "",
            "The following graph is a label-free view of positive co-activation structure. "
            "Every bit is a node, including isolated nodes. Every finite pair with strictly "
            "positive PMI is an edge, with its base-2 PMI as the edge weight; no further edge "
            "thresholding or strongest-edge selection is applied. Joint activation frequency and "
            "phi remain in the edge artifact but do not enter clustering. Leiden optimizes weighted "
            "configuration-model modularity (`RBConfigurationVertexPartition`) at the shown resolution. "
            "Node area is activation prevalence, node colour is the selected community, and edge "
            "width is PMI. The force layout has a fixed seed and is separate from the clustering seed.",
            "",
            f"This run has **{n_bits} nodes**, **{len(pair_table):,} possible pairs**, "
            f"**{community['positive_edges']} finite positive-PMI edges**, and "
            f"**{community['isolated_nodes']} isolated nodes**. At principal resolution 1.0, the "
            f"best of 20 seeds is `{community['principal_seed']}`, with weighted modularity "
            f"**{community['principal_modularity']:.4f}** and "
            f"**{community['principal_communities']} communities**.",
            "",
            "![Positive-PMI weighted Leiden communities](bit_community_graph.png)",
            "",
            "PMI > 0 means association relative to marginal independence, **not** statistical "
            "significance. The graph omits negative-PMI associations by design; consequently, the "
            "absence of an edge does not establish independence. Communities describe this fitted "
            "co-activation graph only. They are not semantic topics or causal entities without a "
            "separate analysis of documents or labels.",
            "",
            "### Principal partition: node statistics",
            "",
            "| Bit | Prevalence | Positive-PMI degree | Incident PMI weight | Isolated | Community (resolution 1) |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in principal_nodes.itertuples(index=False):
        lines.append(
            f"| {row.bit} | {row.activation_prevalence:.3%} | {row.degree_positive_pmi} | "
            f"{row.incident_pmi_weight_sum:.3f} | {bool(row.is_isolated)} | "
            f"{row.community_resolution_1} |"
        )
    if community["category_overlay"] is not None:
        overlay = community["category_overlay"]
        lines.extend(
            [
                "",
                "### Post-hoc category-coloured view",
                "",
                "This is the **same positive-PMI graph and fixed layout** as the community figure "
                "above. It changes only node colour: every bit receives the category marked as "
                "selected in the external conditional-activation heatmap (the largest "
                "`P(bit=1 | category)`). The overlay comes from `"
                f"{Path(overlay['source_json']).name}` for run `{overlay['run']}` and label "
                f"column `{overlay['label_column']}`. It is therefore ex-post and label-aware; "
                "it neither enters the PMI graph nor the Leiden optimization.",
                "",
                "![Positive-PMI graph coloured by post-hoc category](bit_community_graph_posthoc_categories.png)",
            ]
        )
    lines.extend(
        [
            "",
            "### Resolution and optimizer-seed sensitivity",
            "",
            "For each resolution, 20 fixed Leiden seeds are run and the largest weighted modularity "
            "partition is retained (ties break on the smaller seed). ARI compares partitions without "
            "external labels. Within-resolution ARI measures only optimizer stability on this same "
            "empirical graph—not robustness to resampling documents. The cross-resolution ARI file "
            "compares the retained partition at each pair of resolutions.",
            "",
            "| Resolution | Best seed | Best weighted modularity | Communities | Mean seed-pair ARI | Min ARI | Max ARI |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in community["stability"].itertuples(index=False):
        lines.append(
            f"| {row.resolution:g} | {row.best_seed} | {row.best_weighted_modularity:.4f} | "
            f"{row.best_n_communities} | {row.mean_seed_pair_ari:.4f} | "
            f"{row.min_seed_pair_ari:.4f} | {row.max_seed_pair_ari:.4f} |"
        )
    lines.extend(
        [
            "",
            "The principal-resolution seed-pair ARI mean/min/max are "
            f"**{principal_stability.mean_seed_pair_ari:.4f} / "
            f"{principal_stability.min_seed_pair_ari:.4f} / "
            f"{principal_stability.max_seed_pair_ari:.4f}**.",
            "",
            "## Exact profile distribution",
            "",
            "A profile is the complete hard-code string, in `bit_00` through final-bit "
            "order. The plot ranks profiles by observed frequency, rather than by their "
            "arbitrary binary identifier. The top 20 account for "
            f"**{top_share * 100:.2f}%** of all encoded documents.",
            "",
            "![Top exact-profile prevalence](profile_prevalence_top20.png)",
            "",
            "| Rank | Profile | Documents | Share |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in profile_table.itertuples(index=False):
        lines.append(f"| {row.rank} | `{row.profile}` | {row.count:,} | {row.share_pct:.3f}% |")
    if posthoc_profile_analysis is not None:
        lines.extend(
            [
                "",
                "### Profiles jointly satisfying Leiden communities: post-hoc label composition",
                "",
                "For each principal Leiden community, this section extracts only exact profiles "
                "whose documents activate **every bit in that community jointly**. The table and "
                "Sankey then show the conditional distribution of the external label within "
                "that subset. The Sankey ribbon width is the corresponding document count. This is not a new clustering criterion. A profile may satisfy more "
                "than one community conjunction, so rows can overlap; `bit_community_profile_overlap.csv` "
                "records how many conjunctions each document satisfies. `bit_community_joint_profiles.csv` "
                "contains every extracted exact profile, its frequency, and its within-profile label counts. "
                "`bit_community_joint_profile_text_samples.csv` contains deterministic samples (up to ten per "
                "community-label cell) for document reading.",
                "",
                "![Post-hoc label composition among jointly satisfied communities](bit_community_profile_label_sankey.png)",
                "",
                "| Community | Bits required jointly | Documents | Share of corpus | Exact profiles | Dominant label | Dominant-label share |",
                "|---:|---|---:|---:|---:|---|---:|",
            ]
        )
        for row in posthoc_profile_analysis["summary"].itertuples(index=False):
            lines.append(
                f"| {row.community} | {row.bits} | {row.documents_with_all_community_bits:,} | "
                f"{row.share_of_all_documents:.3%} | {row.distinct_exact_profiles:,} | "
                f"{row.dominant_label} | {row.dominant_label_share:.3%} |"
            )
    lines.extend(
        [
            "",
            "## Training convergence",
            "",
            "The fitted code is assessed here independently of labels. The diagnostic tracks "
            "whether active bits per document settle near the requested target and shows the "
            "weighted contributions of the adopted loss terms during optimization.",
            "",
            f"![Compact training diagnostic](../{diagnostics.name})" if diagnostics.is_file() else diagnostics_link,
            "",
            "## Artifacts and interpretation boundary",
            "",
            "- `bit_statistics.csv` contains every bit's activation rate, binary entropy, "
            "and structural status.",
            "- `bit_coactivation.csv` contains all pairwise joint rates, PMI, and phi "
            "correlations.",
            "- `bit_community_edges.csv` retains every finite positive-PMI graph edge with its "
            "PMI weight, joint frequency, and phi; `bit_community_nodes.csv` retains all nodes, "
            "including isolates.",
            "- `bit_community_seed_results.csv`, `bit_community_seed_ari.csv`, "
            "`bit_community_resolution_stability.csv`, and `bit_community_resolution_ari.csv` "
            "document optimizer and resolution comparisons; `bit_community_metadata.json` records "
            "the exact graph, library, and reproducibility configuration.",
            "- `paper_handoff.md` contains run-specific results and interpretation boundaries for a "
            "later manuscript-writing step.",
            "- `profile_prevalence_top20.csv` contains the plotted profile distribution.",
            "- If a `--posthoc_category_json` overlay is explicitly supplied, "
            "`bit_community_posthoc_categories.csv`, `bit_community_graph_posthoc_categories.*`, "
            "`bit_community_profile_label_*`, and `bit_community_joint_profiles.csv` are label-aware "
            "descriptive overlays; `bit_community_joint_profile_text_samples.csv` holds reproducible "
            "document samples. They are "
            "kept separate from the label-free graph and Leiden fit.",
            "- Otherwise, this report intentionally does **not** use labels. The subsequent "
            "label-aware and cross-backbone reports test external association and robustness separately.",
        ]
    )
    (output_dir / "report.md").write_text(contents("\n".join(lines)), encoding="utf-8")
    record = {
        "schema_version": 2,
        "purpose": "label-free single-run binary-code characterization",
        "run_dir": str(run_dir),
        "encoded_csv": str(encoded_path),
        "model_name": model_name,
        "documents": int(n_documents),
        "n_bits": int(n_bits),
        "target_active_bits": target_active,
        "top_profiles": int(top_profiles),
        "label_columns_used": [],
        "community_analysis": {
            "edge_weight_column": "pmi_bits",
            "principal_resolution": 1.0,
            "positive_edges": int(community["positive_edges"]),
            "isolated_nodes": int(community["isolated_nodes"]),
        },
        "posthoc_category_overlay": community["category_overlay"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    encoded = resolve_encoded_csv(run_dir, args.encoded_csv).resolve()
    output = (args.output_dir or run_dir / "bit_structure").resolve()
    if args.top_profiles <= 0:
        raise ValueError("--top_profiles must be positive")
    if (args.posthoc_category_json is None) != (args.posthoc_category_run is None):
        raise ValueError(
            "--posthoc_category_json and --posthoc_category_run must be provided together"
        )
    build_report(
        run_dir,
        encoded,
        output,
        args.top_profiles,
        args.posthoc_category_json,
        args.posthoc_category_run,
    )
    print(f"Wrote label-free bit-structure report to {output}")


if __name__ == "__main__":
    main()
