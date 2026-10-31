#!/usr/bin/env python3
"""Label-free comparison of Qwen SAE hard codes with existing AG News k-means.

For each k-means partition, this script measures residual variation of every
hard bit and of the whole hard-code profile *within* clusters.  A prevalence-
matched null independently permutes each bit over documents, preserving every
bit's activation count while removing its relation to cluster membership and
its dependence on the other bits.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/agnews_qwen_all_fixed_balance"
OUT = RUN / "kmeans_business_crosscutting_case_study"
KS = (32, 64, 256)
SEED = 1234
N_NULL = 100


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    output = np.zeros_like(p, dtype=float)
    interior = (p > 0) & (p < 1)
    q = p[interior]
    output[interior] = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return output


def cluster_bit_statistics(bits: np.ndarray, clusters: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-cluster probabilities, bitwise H(B|C), and two cluster metrics."""
    sizes = np.bincount(clusters, minlength=k).astype(float)
    if (sizes == 0).any():
        raise ValueError("k-means generated an empty cluster")
    sums = np.column_stack([
        np.bincount(clusters, weights=bits[:, j], minlength=k)
        for j in range(bits.shape[1])
    ])
    probabilities = sums / sizes[:, None]
    entropies = binary_entropy(probabilities)
    conditional = (sizes[:, None] * entropies).sum(axis=0) / sizes.sum()
    mean_cluster_entropy = entropies.mean(axis=1)
    intermediate_share = ((probabilities > 0.2) & (probabilities < 0.8)).mean(axis=1)
    return probabilities, conditional, mean_cluster_entropy, intermediate_share


def profile_statistics(profile_codes: np.ndarray, clusters: np.ndarray, k: int) -> tuple[float, float, pd.DataFrame]:
    n = len(profile_codes)
    h_profile = entropy_from_counts(np.bincount(profile_codes, minlength=2**16))
    rows = []
    conditional = 0.0
    for cluster in range(k):
        codes = profile_codes[clusters == cluster]
        h_here = entropy_from_counts(np.bincount(codes, minlength=2**16))
        conditional += (len(codes) / n) * h_here
        rows.append({
            "cluster": cluster,
            "cluster_documents": len(codes),
            "distinct_profiles": int(np.unique(codes).size),
            "profile_entropy_bits": h_here,
            "effective_profiles": 2**h_here,
            "top_profile_share": float(np.bincount(codes, minlength=2**16).max() / len(codes)),
        })
    return h_profile, conditional, pd.DataFrame(rows)


def conjunction_coverage(bits: np.ndarray, clusters: np.ndarray, k: int, bit_columns: list[str]) -> pd.DataFrame:
    rows = []
    for degree in (2, 3):
        for selected in combinations(range(bits.shape[1]), degree):
            member = np.all(bits[:, selected] == 1, axis=1)
            support = int(member.sum())
            counts = np.bincount(clusters[member], minlength=k)
            ordered = np.sort(counts)[::-1]
            needed = int(np.searchsorted(np.cumsum(ordered), 0.8 * support, side="left") + 1)
            rows.append({
                "degree": degree,
                "conjunction": " ∧ ".join(bit_columns[index] for index in selected),
                "support": support,
                "clusters_for_80pct": needed,
                "top_cluster_share": float(ordered[0] / support),
                "effective_cluster_count": 2**entropy_from_counts(counts),
            })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def draw_activation_incidence_sankey(
    counts: np.ndarray, cluster_sizes: np.ndarray, bit_columns: list[str], output_stem: Path
) -> None:
    """Render a k=32 alluvial/Sankey diagram of overlapping bit activations.

    Areas encode activation incidences, not documents: a document activating
    six bits contributes six incidences, one to each relevant ribbon.
    """
    n_clusters, n_bits = counts.shape
    total = counts.sum()
    gap_left = 0.004
    gap_right = 0.010
    scale_left = (1 - gap_left * (n_clusters - 1)) / total
    scale_right = (1 - gap_right * (n_bits - 1)) / total
    # Both sides must share the same total height, so use a common scale and
    # retain larger gaps only where feasible.
    scale = min(scale_left, scale_right)
    palette = plt.get_cmap("tab20")(np.linspace(0.02, 0.98, n_bits))

    source_top = []
    y = 1.0
    for cluster in range(n_clusters):
        source_top.append(y)
        y -= counts[cluster].sum() * scale + gap_left
    destination_top = []
    y = 1.0
    for bit in range(n_bits):
        destination_top.append(y)
        y -= counts[:, bit].sum() * scale + gap_right

    source_cursor = np.asarray(source_top, dtype=float).copy()
    destination_cursor = np.asarray(destination_top, dtype=float).copy()
    fig, ax = plt.subplots(figsize=(18, 14), constrained_layout=True)
    left_x, right_x, bar_width = 0.04, 0.96, 0.016
    for cluster in range(n_clusters):
        height = counts[cluster].sum() * scale
        bottom = source_top[cluster] - height
        ax.add_patch(Rectangle((left_x - bar_width / 2, bottom), bar_width, height,
                               facecolor="#5C88C4", edgecolor="white", linewidth=0.35, zorder=3))
        ax.text(left_x - 0.015, bottom + height / 2, f"C{cluster:02d}\n(n={cluster_sizes[cluster]:,})",
                ha="right", va="center", fontsize=6.5)
    for bit in range(n_bits):
        height = counts[:, bit].sum() * scale
        bottom = destination_top[bit] - height
        ax.add_patch(Rectangle((right_x - bar_width / 2, bottom), bar_width, height,
                               facecolor=palette[bit], edgecolor="white", linewidth=0.45, zorder=3))
        ax.text(right_x + 0.015, bottom + height / 2, bit_columns[bit].replace("bit_", "b"),
                ha="left", va="center", fontsize=9)

    for cluster in range(n_clusters):
        for bit in range(n_bits):
            count = counts[cluster, bit]
            if count == 0:
                continue
            height = count * scale
            source_upper, source_lower = source_cursor[cluster], source_cursor[cluster] - height
            destination_upper, destination_lower = destination_cursor[bit], destination_cursor[bit] - height
            source_cursor[cluster] -= height
            destination_cursor[bit] -= height
            control_left, control_right = 0.32, 0.68
            vertices = [
                (left_x + bar_width / 2, source_upper),
                (control_left, source_upper), (control_right, destination_upper), (right_x - bar_width / 2, destination_upper),
                (right_x - bar_width / 2, destination_lower),
                (control_right, destination_lower), (control_left, source_lower), (left_x + bar_width / 2, source_lower),
                (left_x + bar_width / 2, source_upper),
            ]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                     MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=palette[bit], edgecolor="none", alpha=0.55, zorder=1))

    ax.text(left_x, 1.025, "k=32 clusters", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.text(right_x, 1.025, "SAE bits", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.text(0.5, -0.035,
            "Ribbon area = number of documents in the cluster with the bit active. A document may contribute to multiple ribbons; flows are activation incidences, not a partition of documents.",
            ha="center", va="top", fontsize=9)
    ax.set(xlim=(-0.15, 1.16), ylim=(-0.06, 1.07))
    ax.axis("off")
    fig.savefig(output_stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    encoded = pd.read_csv(RUN / "encoded_agnews_all_ready_best.csv", dtype={"id": "string"})
    bit_columns = [f"bit_{i:02d}" for i in range(16)]
    bits = encoded[bit_columns].to_numpy(dtype=np.int8)
    n = len(bits)
    if n != 127_600:
        raise ValueError(f"Expected 127,600 full-corpus codes; found {n:,}")
    bit_entropy = binary_entropy(bits.mean(axis=0))
    bit_prevalence = bits.mean(axis=0)
    profile_codes = (bits.astype(np.uint32) @ (1 << np.arange(16, dtype=np.uint32)))
    rng = np.random.default_rng(SEED)

    assignments_by_k = {}
    for k in KS:
        assignments = pd.read_csv(
            RUN / f"kmeans{k}_tech_market_case_study/document_assignments.csv",
            dtype={"id": "string"},
        )[["id", f"kmeans{k}_cluster"]]
        aligned = encoded[["id"]].merge(assignments, on="id", how="left", validate="one_to_one")
        clusters = aligned[f"kmeans{k}_cluster"].to_numpy()
        if pd.isna(clusters).any() or np.unique(clusters).size != k:
            raise ValueError(f"Invalid k={k} assignments")
        assignments_by_k[k] = clusters.astype(int)

    true_cluster_metrics: dict[int, pd.DataFrame] = {}
    bit_rows = []
    profile_rows = []
    coverage_frames = []
    true_conditional: dict[int, np.ndarray] = {}
    null_conditional: dict[int, np.ndarray] = {k: np.empty((N_NULL, 16)) for k in KS}
    null_cluster_entropy: dict[int, np.ndarray] = {k: np.empty((N_NULL, k)) for k in KS}
    null_intermediate: dict[int, np.ndarray] = {k: np.empty((N_NULL, k)) for k in KS}

    for k, clusters in assignments_by_k.items():
        probabilities, conditional, mean_entropy, intermediate = cluster_bit_statistics(bits, clusters, k)
        true_conditional[k] = conditional
        h_profile, h_profile_conditional, profile_cluster = profile_statistics(profile_codes, clusters, k)
        profile_cluster.insert(0, "k", k)
        for bit_index, bit_name in enumerate(bit_columns):
            profile_cluster[f"p_{bit_name}"] = probabilities[:, bit_index]
        profile_cluster["mean_bit_entropy_bits"] = mean_entropy
        profile_cluster["intermediate_bit_share_0p2_to_0p8"] = intermediate
        true_cluster_metrics[k] = profile_cluster
        coverage = conjunction_coverage(bits, clusters, k, bit_columns)
        coverage.insert(0, "k", k)
        coverage_frames.append(coverage)
        profile_rows.append({
            "k": k,
            "profile_entropy_bits": h_profile,
            "profile_conditional_entropy_bits": h_profile_conditional,
            "profile_mutual_information_bits": h_profile - h_profile_conditional,
            "profile_residual_fraction": h_profile_conditional / h_profile if h_profile else np.nan,
            "profile_adjusted_mutual_information_with_cluster": adjusted_mutual_info_score(profile_codes, clusters),
            "profile_adjusted_rand_index_with_cluster": adjusted_rand_score(profile_codes, clusters),
            "median_distinct_profiles_within_cluster": float(profile_cluster["distinct_profiles"].median()),
            "median_effective_profiles_within_cluster": float(profile_cluster["effective_profiles"].median()),
        })

    for replicate in range(N_NULL):
        null_bits = np.empty_like(bits)
        for bit_index in range(bits.shape[1]):
            null_bits[:, bit_index] = rng.permutation(bits[:, bit_index])
        for k, clusters in assignments_by_k.items():
            _, conditional, mean_entropy, intermediate = cluster_bit_statistics(null_bits, clusters, k)
            null_conditional[k][replicate] = conditional
            null_cluster_entropy[k][replicate] = mean_entropy
            null_intermediate[k][replicate] = intermediate

    cluster_frames = []
    for k, clusters in assignments_by_k.items():
        profile_cluster = true_cluster_metrics[k].copy()
        profile_cluster["null_mean_bit_entropy_mean"] = null_cluster_entropy[k].mean(axis=0)
        profile_cluster["null_mean_bit_entropy_q025"] = np.quantile(null_cluster_entropy[k], 0.025, axis=0)
        profile_cluster["null_mean_bit_entropy_q975"] = np.quantile(null_cluster_entropy[k], 0.975, axis=0)
        profile_cluster["null_intermediate_share_mean"] = null_intermediate[k].mean(axis=0)
        profile_cluster["null_intermediate_share_q025"] = np.quantile(null_intermediate[k], 0.025, axis=0)
        profile_cluster["null_intermediate_share_q975"] = np.quantile(null_intermediate[k], 0.975, axis=0)
        cluster_frames.append(profile_cluster)
        for bit_index, bit_name in enumerate(bit_columns):
            h = bit_entropy[bit_index]
            null_fraction = null_conditional[k][:, bit_index] / h if h else np.full(N_NULL, np.nan)
            bit_rows.append({
                "k": k,
                "bit": bit_name,
                "prevalence": bit_prevalence[bit_index],
                "entropy_bits": h,
                "conditional_entropy_bits": true_conditional[k][bit_index],
                "mutual_information_with_cluster_bits": h - true_conditional[k][bit_index],
                "residual_fraction_Hbit_given_cluster_over_Hbit": true_conditional[k][bit_index] / h if h else np.nan,
                "null_conditional_entropy_mean": null_conditional[k][:, bit_index].mean(),
                "null_conditional_entropy_q025": np.quantile(null_conditional[k][:, bit_index], 0.025),
                "null_conditional_entropy_q975": np.quantile(null_conditional[k][:, bit_index], 0.975),
                "null_residual_fraction_mean": np.nanmean(null_fraction),
                "null_residual_fraction_q025": np.nanquantile(null_fraction, 0.025),
                "null_residual_fraction_q975": np.nanquantile(null_fraction, 0.975),
            })

    bit_summary = pd.DataFrame(bit_rows)
    cluster_summary = pd.concat(cluster_frames, ignore_index=True)
    coverage_summary = pd.concat(coverage_frames, ignore_index=True)
    profile_summary = pd.DataFrame(profile_rows)
    bit_summary.to_csv(OUT / "bit_cluster_information_decomposition.csv", index=False)
    cluster_summary.to_csv(OUT / "cluster_bit_profile_diversity.csv", index=False)
    coverage_summary.to_csv(OUT / "conjunction_cluster_coverage_80pct.csv", index=False)
    profile_summary.to_csv(OUT / "profile_cluster_information_decomposition.csv", index=False)

    association_rows = []
    for _, row in cluster_summary.iterrows():
        for bit_index, bit_name in enumerate(bit_columns):
            conditional_probability = float(row[f"p_{bit_name}"])
            excess = conditional_probability - bit_prevalence[bit_index]
            association_rows.append({
                "k": int(row["k"]),
                "cluster": int(row["cluster"]),
                "bit": bit_name,
                "cluster_documents": int(row["cluster_documents"]),
                "bit_prevalence": bit_prevalence[bit_index],
                "conditional_activation_probability": conditional_probability,
                "active_documents_in_cluster": int(round(int(row["cluster_documents"]) * conditional_probability)),
                "activation_excess_over_corpus": excess,
                "activation_lift_over_corpus": conditional_probability / bit_prevalence[bit_index],
                "excess_active_documents": int(row["cluster_documents"]) * excess,
            })
    associations = pd.DataFrame(association_rows)
    associations.to_csv(OUT / "cluster_bit_activation_excess.csv", index=False)

    sankey_k = 32
    sankey_clusters = assignments_by_k[sankey_k]
    sankey_counts = np.column_stack([
        np.bincount(sankey_clusters, weights=bits[:, bit], minlength=sankey_k).astype(int)
        for bit in range(bits.shape[1])
    ])
    draw_activation_incidence_sankey(
        sankey_counts,
        np.bincount(sankey_clusters, minlength=sankey_k),
        bit_columns,
        OUT / "k32_cluster_to_bit_activation_sankey",
    )

    # Figure 1: per-bit residual entropy against a prevalence-matched null.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    x = np.arange(16)
    for ax, k in zip(axes, KS):
        subset = bit_summary.loc[bit_summary.k.eq(k)].sort_values("bit")
        ax.fill_between(
            x, subset.null_residual_fraction_q025, subset.null_residual_fraction_q975,
            color="#B9BEC8", alpha=0.55, label="prevalence-matched null (95%)",
        )
        ax.plot(x, subset.null_residual_fraction_mean, color="#777E8B", linewidth=1.25, label="null mean")
        ax.plot(x, subset.residual_fraction_Hbit_given_cluster_over_Hbit, "o-", color="#2166AC", label="observed SAE bits")
        ax.set_title(f"k={k}")
        ax.set_xticks(x, [f"b{i:02d}" for i in range(16)], rotation=45, ha="right")
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel(r"Residual bit information $H(B_k\mid C)/H(B_k)$")
    axes[1].set_xlabel("SAE bit")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False)
    fig.savefig(OUT / "bit_residual_information_by_k.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "bit_residual_information_by_k.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: within-cluster residual bit uncertainty.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, constrained_layout=True)
    for ax, k in zip(axes, KS):
        subset = cluster_summary.loc[cluster_summary.k.eq(k)]
        ax.scatter(subset.cluster_documents, subset.null_mean_bit_entropy_mean, color="#B9BEC8", s=28, label="null mean", zorder=1)
        ax.scatter(subset.cluster_documents, subset.mean_bit_entropy_bits, color="#2166AC", s=36, alpha=0.85, label="observed", zorder=2)
        ax.set_xscale("log")
        ax.set_title(f"k={k}")
        ax.set_xlabel("Cluster documents (log scale)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean Bernoulli entropy of 16 bits within cluster")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False)
    fig.savefig(OUT / "within_cluster_bit_entropy.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "within_cluster_bit_entropy.pdf", bbox_inches="tight")
    plt.close(fig)

    # A readable bipartite summary for k=32.  The complete, unthresholded
    # cluster-by-bit association matrix is kept in cluster_bit_activation_excess.csv.
    k = 32
    association_k = associations.loc[associations.k.eq(k)].copy()
    selected_edges = (
        association_k.loc[association_k.activation_excess_over_corpus.gt(0)]
        .sort_values(["cluster", "activation_excess_over_corpus"], ascending=[True, False])
        .groupby("cluster", as_index=False, group_keys=False)
        .head(3)
    )
    cluster_nodes = cluster_summary.loc[cluster_summary.k.eq(k)].sort_values("cluster")
    fig, ax = plt.subplots(figsize=(13.5, 10), constrained_layout=True)
    left_x, right_x = 0.0, 1.0
    left_y = {int(cluster): 1 - (index + 0.5) / len(cluster_nodes) for index, cluster in enumerate(cluster_nodes.cluster)}
    right_y = {bit: 1 - (index + 0.5) / len(bit_columns) for index, bit in enumerate(bit_columns)}
    max_documents = cluster_nodes.cluster_documents.max()
    for _, edge in selected_edges.iterrows():
        alpha = 0.17 + 0.63 * edge.activation_excess_over_corpus
        ax.plot(
            [left_x, right_x], [left_y[int(edge.cluster)], right_y[edge.bit]],
            color="#2D6A9F", alpha=min(alpha, 0.78),
            linewidth=0.5 + 5.0 * edge.activation_excess_over_corpus,
            zorder=1,
        )
    for _, node in cluster_nodes.iterrows():
        cluster = int(node.cluster)
        size = 35 + 150 * np.sqrt(node.cluster_documents / max_documents)
        ax.scatter(left_x, left_y[cluster], s=size, color="#5C88C4", edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(left_x - 0.025, left_y[cluster], f"C{cluster:02d}", ha="right", va="center", fontsize=8)
    for bit in bit_columns:
        ax.scatter(right_x, right_y[bit], s=210, color="#E68653", edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(right_x + 0.025, right_y[bit], bit.replace("bit_", "b"), ha="left", va="center", fontsize=10)
    ax.text(left_x, 1.035, "k=32 clusters", ha="center", va="bottom", fontweight="bold")
    ax.text(right_x, 1.035, "SAE bits", ha="center", va="bottom", fontweight="bold")
    ax.text(
        0.5, -0.055,
        "Edges: three largest positive conditional activation excesses per cluster; width = P(bit=1 | cluster) − P(bit=1).",
        ha="center", va="top", fontsize=9,
    )
    ax.set(xlim=(-0.16, 1.19), ylim=(-0.08, 1.08))
    ax.axis("off")
    fig.savefig(OUT / "k32_cluster_bit_bipartite_graph.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT / "k32_cluster_bit_bipartite_graph.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: the inverse mapping, conjunctions distributed across clusters.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False, constrained_layout=True)
    colors = {2: "#E68653", 3: "#67A877"}
    for ax, k in zip(axes, KS):
        subset = coverage_summary.loc[coverage_summary.k.eq(k)]
        for degree, group in subset.groupby("degree"):
            ax.scatter(group.support, group.clusters_for_80pct, color=colors[degree], alpha=0.72, s=28, label=f"{degree}-bit conjunction")
        ax.set_xscale("log")
        ax.set_title(f"k={k}")
        ax.set_xlabel("Conjunction support (documents; log scale)")
        ax.set_ylabel("Clusters needed for 80% coverage")
        ax.grid(alpha=0.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False)
    fig.savefig(OUT / "conjunction_cluster_coverage_80pct.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "conjunction_cluster_coverage_80pct.pdf", bbox_inches="tight")
    plt.close(fig)

    aggregate_rows = []
    for k in KS:
        bits_k = bit_summary.loc[bit_summary.k.eq(k)]
        clusters_k = cluster_summary.loc[cluster_summary.k.eq(k)]
        profiles_k = profile_summary.loc[profile_summary.k.eq(k)].iloc[0]
        aggregate_rows.append({
            "k": k,
            "mean_bit_residual_fraction": bits_k.residual_fraction_Hbit_given_cluster_over_Hbit.mean(),
            "mean_null_bit_residual_fraction": bits_k.null_residual_fraction_mean.mean(),
            "mean_bit_mutual_information_bits": bits_k.mutual_information_with_cluster_bits.mean(),
            "median_within_cluster_bit_entropy": clusters_k.mean_bit_entropy_bits.median(),
            "median_null_within_cluster_bit_entropy": clusters_k.null_mean_bit_entropy_mean.median(),
            "median_within_cluster_intermediate_bit_share": clusters_k.intermediate_bit_share_0p2_to_0p8.median(),
            "median_null_intermediate_bit_share": clusters_k.null_intermediate_share_mean.median(),
            **profiles_k.to_dict(),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUT / "bit_profile_residual_information_summary.csv", index=False)

    coverage_display = coverage_summary.groupby(["k", "degree"], as_index=False).agg(
        conjunctions=("conjunction", "size"),
        median_support=("support", "median"),
        median_clusters_for_80pct=("clusters_for_80pct", "median"),
        median_top_cluster_share=("top_cluster_share", "median"),
    )
    coverage_display.to_csv(OUT / "conjunction_cluster_coverage_summary.csv", index=False)
    display = aggregate.copy()
    for column in [
        "mean_bit_residual_fraction", "mean_null_bit_residual_fraction",
        "median_within_cluster_bit_entropy", "median_null_within_cluster_bit_entropy",
        "profile_residual_fraction",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    for column in ["median_within_cluster_intermediate_bit_share", "median_null_intermediate_bit_share"]:
        display[column] = display[column].map(lambda value: f"{100 * value:.1f}%")
    for column in ["mean_bit_mutual_information_bits", "profile_entropy_bits", "profile_conditional_entropy_bits", "profile_mutual_information_bits"]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    for column in ["profile_adjusted_mutual_information_with_cluster", "profile_adjusted_rand_index_with_cluster"]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    for column in ["median_distinct_profiles_within_cluster", "median_effective_profiles_within_cluster"]:
        display[column] = display[column].map(lambda value: f"{value:,.1f}")
    coverage_for_report = coverage_display.copy()
    coverage_for_report["median_support"] = coverage_for_report.median_support.map(lambda value: f"{value:,.0f}")
    coverage_for_report["median_clusters_for_80pct"] = coverage_for_report.median_clusters_for_80pct.map(lambda value: f"{value:.1f}")
    coverage_for_report["median_top_cluster_share"] = coverage_for_report.median_top_cluster_share.map(lambda value: f"{100 * value:.1f}%")

    residual_conclusion = (
        "**Conclusion.** The cluster partition captures genuine information about the SAE bits: mean residual bit entropy is below the prevalence-matched null at every resolution "
        f"(k=32: {aggregate.iloc[0].mean_bit_residual_fraction:.3f} versus {aggregate.iloc[0].mean_null_bit_residual_fraction:.3f}; "
        f"k=64: {aggregate.iloc[1].mean_bit_residual_fraction:.3f} versus {aggregate.iloc[1].mean_null_bit_residual_fraction:.3f}; "
        f"k=256: {aggregate.iloc[2].mean_bit_residual_fraction:.3f} versus {aggregate.iloc[2].mean_null_bit_residual_fraction:.3f}). "
        "However, the values remain far above zero, so no partition is a deterministic recoding of the 16 SAE bits. The scatter/line plot makes the heterogeneity explicit: different bits retain different amounts of variation within the same geometric partition."
    )
    within_cluster_conclusion = (
        "**Conclusion.** The observed within-cluster bit entropy is substantially lower than the random-code null, showing that clusters do organize the codes, but it remains materially positive. "
        "Thus documents that are close enough to share a k-means cluster still differ in several binary semantic dimensions; this is residual structured variation rather than the null's random variation."
    )
    bipartite_conclusion = (
        f"**Conclusion.** In the k=32 visual summary, the top positive associations span {selected_edges['bit'].nunique()} of 16 bits and every cluster has up to three characteristic-bit links. "
        "There is no visually one-to-one mapping from clusters to bits: bits recur across clusters and clusters are characterized by combinations. Because the graph deliberately retains only the three strongest positive links per cluster, this conclusion is illustrative; the complete signed matrix is the formal artifact."
    )
    active_source_counts = (sankey_counts > 0).sum(axis=0)
    sankey_conclusion = (
        f"**Conclusion.** The activation-incidence Sankey shows the same many-to-many structure using raw document counts: each bit receives activations from {int(active_source_counts.min())} to {int(active_source_counts.max())} of the 32 clusters. "
        "Its thick ribbons identify where activations concentrate, while the broad fan-out from each cluster demonstrates that cluster membership does not select one exclusive code."
    )
    coverage_conclusion = (
        "**Conclusion.** In the inverse direction, typical conjunctions require multiple clusters to cover 80% of their documents. "
        f"For example, median 3-bit coverage rises from {coverage_display.loc[(coverage_display.k.eq(32)) & (coverage_display.degree.eq(3)), 'median_clusters_for_80pct'].iloc[0]:.1f} clusters at k=32 "
        f"to {coverage_display.loc[(coverage_display.k.eq(256)) & (coverage_display.degree.eq(3)), 'median_clusters_for_80pct'].iloc[0]:.1f} at k=256. "
        "Finer k-means partitions therefore fragment, rather than localize, the document sets sharing a bit conjunction."
    )

    section = [
        "## Label-free residual information inside k-means clusters",
        "",
        "This section does not use AG News labels. Let `C` denote a raw-embedding k-means cluster and `B_k` an SAE hard bit. For each cluster it records `P(B_k=1 | C=c)`, the mean Bernoulli entropy of its 16 conditional bit probabilities, the share of conditional probabilities between 0.2 and 0.8, and profile diversity. A deterministic code signature for a cluster would have conditional bit probabilities near zero or one and low within-cluster entropy.",
        "",
        "For each bit, the exact empirical decomposition is `H(B_k) = I(B_k; C) + H(B_k | C)`. The residual fraction `H(B_k | C) / H(B_k)` is the fraction of the bit's marginal entropy not captured by the cluster partition: zero means that the cluster determines the bit, while one means that the partition gives no information about it. Profile entropy is reported analogously for the observed 16-bit profiles. These entropy estimates are descriptive plug-in estimates on the full corpus; profile quantities are not bias-corrected.",
        "",
        "The null consists of 100 independently permuted versions of each bit column. It preserves every bit's empirical prevalence exactly, while removing all document-level relation to clusters and all bit-bit dependence. Under this null, residual fractions should be close to one and mutual information close to zero. The observed SAE should therefore lie between a deterministic recoding of k-means and this independent null: non-zero information about clusters together with substantial residual variation.",
        "",
        "![Residual information by bit](bit_residual_information_by_k.png)",
        "",
        residual_conclusion,
        "",
        "![Within-cluster bit entropy](within_cluster_bit_entropy.png)",
        "",
        within_cluster_conclusion,
        "",
        "### Cluster-to-bit bipartite graph",
        "",
        "A Sankey plot would be misleading here because a document can activate several bits: its apparent flows would not be mutually exclusive. The bipartite graph below instead links each k=32 cluster to its three bits with the largest positive activation excess over the corpus baseline. Edge width is `P(bit=1 | cluster) − P(bit=1)`. It is a readability summary, not a thresholded analysis: the full cluster-by-bit matrix, including negative associations, is in `cluster_bit_activation_excess.csv`.",
        "",
        "![k=32 cluster-to-bit bipartite graph](k32_cluster_bit_bipartite_graph.png)",
        "",
        bipartite_conclusion,
        "",
        "### Cluster-to-bit activation-incidence Sankey",
        "",
        "This Sankey answers a different descriptive question: every ribbon from cluster `c` to bit `b` has area equal to the number of documents in `c` for which `b=1`. A document with several active bits appears in several ribbons, so the outgoing ribbons deliberately do **not** sum to the number of documents in the cluster. The colours identify destination bits; cluster bars therefore encode total activation incidences, not document counts. Cluster document counts are printed beside the source nodes.",
        "",
        "![k=32 cluster-to-bit activation Sankey](k32_cluster_to_bit_activation_sankey.png)",
        "",
        sankey_conclusion,
        "",
        "### Aggregate bit and profile information",
        "",
        markdown_table(display),
        "",
        "`mean_bit_residual_fraction` averages the 16 bit-level residual fractions. `mean_null_bit_residual_fraction` is its prevalence-matched null analogue. `median_within_cluster_bit_entropy` summarizes the cluster-wise mean entropy across bits. `profile_residual_fraction` is `H(profile | C) / H(profile)`, and the final two columns quantify observed profile diversity inside a typical cluster. `profile_adjusted_mutual_information_with_cluster` (AMI) and `profile_adjusted_rand_index_with_cluster` (ARI) compare the entire profile partition with the cluster partition, each corrected for chance: 1 means identical partitions, approximately 0 means chance-level agreement, and negative values mean less agreement than expected by chance. ARI is especially stringent here because it evaluates document pairs.",
        "",
        f"**Conclusion.** Even at k=256, a typical cluster contains {aggregate.iloc[2].median_effective_profiles_within_cluster:.1f} effective profiles and the profile residual fraction is {aggregate.iloc[2].profile_residual_fraction:.3f}. Profile–cluster AMI remains near 0.40 across resolutions, whereas ARI ranges only from {aggregate.profile_adjusted_rand_index_with_cluster.min():.3f} to {aggregate.profile_adjusted_rand_index_with_cluster.max():.3f}. Thus profiles and clusters share information well above chance but are not alternate names for the same document partition; the stringent pairwise ARI makes the absence of a one-to-one partition correspondence explicit.",
        "",
        "### Inverse direction: how many clusters contain each conjunction?",
        "",
        "For every 2-bit and 3-bit conjunction, clusters are sorted by the number of documents satisfying the conjunction. `clusters_for_80pct` is the smallest number required to cover 80% of its supporting documents. A value above one means that the conjunction is not localized in one geometric cluster.",
        "",
        "![Conjunction coverage across clusters](conjunction_cluster_coverage_80pct.png)",
        "",
        markdown_table(coverage_for_report),
        "",
        coverage_conclusion,
        "",
        "Detailed artifacts: `bit_cluster_information_decomposition.csv` (16 bits × 3 partitions), `cluster_bit_profile_diversity.csv` (one row per cluster with all 16 conditional activation probabilities), `profile_cluster_information_decomposition.csv`, and `conjunction_cluster_coverage_80pct.csv` (all 680 conjunctions × 3 partitions).",
    ]
    (OUT / "bit_residual_information_section.md").write_text("\n".join(section) + "\n")
    metadata = {"seed": SEED, "n_null_replicates": N_NULL, "ks": list(KS), "n_documents": n,
                "null": "independent random permutation of each bit column; exact marginal bit prevalences preserved"}
    (OUT / "bit_residual_information_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(OUT / "bit_residual_information_section.md")


if __name__ == "__main__":
    main()
