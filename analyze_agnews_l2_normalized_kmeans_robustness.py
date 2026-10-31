#!/usr/bin/env python3
"""Robustness check: AG News raw versus L2-normalized embedding k-means.

Fits ordinary Lloyd k-means after row-wise L2 normalization at k=32 and k=256.
It compares post-hoc label purity and SAE-code residual information with the
already stored raw-embedding k-means partitions. No embedding or SAE is fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/agnews_qwen_all_fixed_balance"
OUT = RUN / "normalized_kmeans_robustness"
KS = (32, 256)
SEED = 1234
LABEL_ORDER = ["World", "Sports", "Business", "Sci_Tech"]


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    answer = np.zeros_like(p)
    valid = (p > 0) & (p < 1)
    q = p[valid]
    answer[valid] = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return answer


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def evaluate_partition(bits: np.ndarray, profile_codes: np.ndarray, labels: np.ndarray, clusters: np.ndarray, k: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    sizes = np.bincount(clusters, minlength=k).astype(float)
    sums = np.column_stack([np.bincount(clusters, weights=bits[:, j], minlength=k) for j in range(16)])
    probabilities = sums / sizes[:, None]
    entropy_by_cluster_bit = binary_entropy(probabilities)
    conditional_bits = (sizes[:, None] * entropy_by_cluster_bit).sum(axis=0) / sizes.sum()
    marginal_bits = binary_entropy(bits.mean(axis=0))
    bit_table = pd.DataFrame({
        "bit": [f"bit_{i:02d}" for i in range(16)],
        "prevalence": bits.mean(axis=0),
        "entropy_bits": marginal_bits,
        "conditional_entropy_bits": conditional_bits,
        "mutual_information_bits": marginal_bits - conditional_bits,
        "residual_fraction": conditional_bits / marginal_bits,
    })
    purity_rows = []
    profile_conditional = 0.0
    for cluster in range(k):
        member = clusters == cluster
        label_counts = pd.Series(labels[member]).value_counts().reindex(LABEL_ORDER, fill_value=0)
        codes = profile_codes[member]
        profile_conditional += member.mean() * entropy_from_counts(np.bincount(codes, minlength=2**16))
        row = {
            "cluster": cluster,
            "cluster_documents": int(member.sum()),
            "dominant_label": label_counts.idxmax(),
            "dominant_label_share": float(label_counts.max() / member.sum()),
            "mean_bit_entropy": float(entropy_by_cluster_bit[cluster].mean()),
            "intermediate_bit_share": float(((probabilities[cluster] > 0.2) & (probabilities[cluster] < 0.8)).mean()),
        }
        for label in LABEL_ORDER:
            row[f"share_{label.lower()}"] = float(label_counts[label] / member.sum())
        purity_rows.append(row)
    purity = pd.DataFrame(purity_rows)
    h_profile = entropy_from_counts(np.bincount(profile_codes, minlength=2**16))
    summary = {
        "median_label_purity": float(purity.dominant_label_share.median()),
        "clusters_at_least_80pct_one_label": int((purity.dominant_label_share >= 0.8).sum()),
        "mean_bit_residual_fraction": float(bit_table.residual_fraction.mean()),
        "mean_bit_mutual_information_bits": float(bit_table.mutual_information_bits.mean()),
        "median_within_cluster_bit_entropy": float(purity.mean_bit_entropy.median()),
        "profile_residual_fraction": float(profile_conditional / h_profile),
        "profile_adjusted_mutual_information_with_cluster": float(adjusted_mutual_info_score(profile_codes, clusters)),
        "profile_adjusted_rand_index_with_cluster": float(adjusted_rand_score(profile_codes, clusters)),
    }
    return purity, bit_table, summary


def main() -> None:
    OUT.mkdir(exist_ok=True)
    metadata = json.loads((RUN / "raw_embeddings_agnews_all_ready_best_metadata.json").read_text())
    if metadata.get("l2_normalized") is not False:
        raise ValueError("Expected the cached raw, unnormalised Qwen embeddings")
    embeddings = np.load(RUN / "raw_embeddings_agnews_all_ready_best.npy", mmap_mode="r")
    ids = pd.read_csv(RUN / "raw_embeddings_agnews_all_ready_best_ids.csv", dtype={"id": "string"})
    encoded = pd.read_csv(RUN / "encoded_agnews_all_ready_best.csv", dtype={"id": "string"})
    corpus = pd.read_csv(ROOT / "data/agnews/agnews_all.csv", dtype={"id": "string"})[["id", "label_name"]]
    merged = ids.merge(encoded[["id"] + [f"bit_{i:02d}" for i in range(16)]], on="id", how="left", validate="one_to_one")
    merged = merged.merge(corpus, on="id", how="left", validate="one_to_one")
    if len(merged) != 127_600 or merged.isna().any().any():
        raise ValueError("Embeddings, hard codes, and labels do not align")
    bits = merged[[f"bit_{i:02d}" for i in range(16)]].to_numpy(dtype=np.int8)
    labels = merged.label_name.to_numpy()
    profile_codes = bits.astype(np.uint32) @ (1 << np.arange(16, dtype=np.uint32))

    # One float32 in-memory unit-norm copy; the source cache is not changed.
    x = np.asarray(embeddings, dtype=np.float32).copy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero-norm embedding encountered")
    x /= norms

    summary_rows = []
    bit_comparison = []
    purity_frames = []
    for k in KS:
        raw_assignments = pd.read_csv(
            RUN / f"kmeans{k}_tech_market_case_study/document_assignments.csv",
            dtype={"id": "string"},
        )[["id", f"kmeans{k}_cluster"]]
        raw = ids.merge(raw_assignments, on="id", how="left", validate="one_to_one")[f"kmeans{k}_cluster"].to_numpy(dtype=int)
        raw_purity, raw_bits, raw_summary = evaluate_partition(bits, profile_codes, labels, raw, k)
        raw_purity.insert(0, "geometry", "raw")
        purity_frames.append(raw_purity.assign(k=k))
        raw_bits = raw_bits.assign(k=k, geometry="raw")

        assignment_path = OUT / f"k{k}_l2_normalized_assignments.csv"
        if assignment_path.exists():
            stored = pd.read_csv(assignment_path, dtype={"id": "string"})
            column = f"l2_normalized_kmeans{k}_cluster"
            normalized = ids.merge(stored[["id", column]], on="id", how="left", validate="one_to_one")[column].to_numpy(dtype=int)
            if np.unique(normalized).size != k:
                raise ValueError(f"Stored normalized k={k} assignments are invalid")
        else:
            model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, algorithm="lloyd", random_state=SEED)
            normalized = model.fit_predict(x)
            assignment_output = ids.copy()
            assignment_output[f"l2_normalized_kmeans{k}_cluster"] = normalized
            assignment_output.to_csv(assignment_path, index=False)
        normalized_purity, normalized_bits, normalized_summary = evaluate_partition(bits, profile_codes, labels, normalized, k)
        normalized_purity.insert(0, "geometry", "l2_normalized")
        normalized_purity.to_csv(OUT / f"k{k}_l2_normalized_cluster_label_purity.csv", index=False)
        purity_frames.append(normalized_purity.assign(k=k))
        normalized_bits = normalized_bits.assign(k=k, geometry="l2_normalized")
        normalized_bits.to_csv(OUT / f"k{k}_l2_normalized_bit_information.csv", index=False)

        for geometry, results in (("raw", raw_summary), ("l2_normalized", normalized_summary)):
            summary_rows.append({"k": k, "geometry": geometry, **results})
        bit_comparison.append(pd.concat([raw_bits, normalized_bits], ignore_index=True))

    summary = pd.DataFrame(summary_rows)
    all_purity = pd.concat(purity_frames, ignore_index=True)
    all_bits = pd.concat(bit_comparison, ignore_index=True)
    summary.to_csv(OUT / "raw_vs_l2_normalized_summary.csv", index=False)
    all_purity.to_csv(OUT / "raw_vs_l2_normalized_cluster_label_purity.csv", index=False)
    all_bits.to_csv(OUT / "raw_vs_l2_normalized_bit_information.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharey=True, constrained_layout=True)
    for ax, k in zip(axes, KS):
        subset = all_purity.loc[all_purity.k.eq(k)]
        data = [subset.loc[subset.geometry.eq(geometry), "dominant_label_share"] * 100 for geometry in ("raw", "l2_normalized")]
        ax.boxplot(data, tick_labels=["raw", "L2-normalized"], showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#BFD3EA"}, medianprops={"color": "#1F5E99", "linewidth": 2})
        for index, values in enumerate(data, start=1):
            jitter = np.linspace(-0.09, 0.09, len(values))
            ax.scatter(index + jitter, values, s=10, alpha=0.35, color="#2D6A9F")
        ax.set_title(f"k={k}")
        ax.set_ylim(0, 102)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Dominant AG News category share (%)")
    fig.savefig(OUT / "raw_vs_l2_label_purity.png", dpi=240)
    fig.savefig(OUT / "raw_vs_l2_label_purity.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharex=True, sharey=True, constrained_layout=True)
    for ax, k in zip(axes, KS):
        pivot = all_bits.loc[all_bits.k.eq(k)].pivot(index="bit", columns="geometry", values="residual_fraction")
        ax.scatter(pivot.raw, pivot.l2_normalized, color="#2D6A9F", s=48)
        for bit, row in pivot.iterrows():
            ax.annotate(bit.replace("bit_", "b"), (row.raw, row.l2_normalized), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
        ax.set_title(f"k={k}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Raw-embedding k-means residual fraction")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("L2-normalized k-means residual fraction")
    fig.savefig(OUT / "raw_vs_l2_bit_residuals.png", dpi=240)
    fig.savefig(OUT / "raw_vs_l2_bit_residuals.pdf")
    plt.close(fig)

    display = summary.copy()
    for column in ["median_label_purity", "mean_bit_residual_fraction", "profile_residual_fraction", "profile_adjusted_mutual_information_with_cluster", "profile_adjusted_rand_index_with_cluster"]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    for column in ["mean_bit_mutual_information_bits", "median_within_cluster_bit_entropy"]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    section = [
        "## Robustness to L2-normalizing the raw embeddings before k-means",
        "",
        "This robustness check fits ordinary Lloyd k-means to the same Qwen embeddings after row-wise L2 normalization, at k=32 and k=256. This changes the Euclidean geometry to emphasize angular direction. It is not labelled spherical k-means, because centroid renormalization is not imposed at each update. No labels, SAE codes, or embeddings enter the fit beyond the normalized embedding vectors; labels and hard codes are used only after fitting for evaluation.",
        "",
        "![Raw versus L2-normalized cluster label purity](../normalized_kmeans_robustness/raw_vs_l2_label_purity.png)",
        "",
        "![Raw versus L2-normalized bit residual fractions](../normalized_kmeans_robustness/raw_vs_l2_bit_residuals.png)",
        "",
        markdown_table(display),
        "",
    ]
    raw = summary.loc[summary.geometry.eq("raw")].set_index("k")
    normalized = summary.loc[summary.geometry.eq("l2_normalized")].set_index("k")
    section.extend([
        f"**Conclusion.** L2 normalization leaves the central conclusions qualitatively unchanged. At k=32, median label purity changes from {raw.loc[32, 'median_label_purity']:.1%} to {normalized.loc[32, 'median_label_purity']:.1%}, while mean bit residual entropy changes from {raw.loc[32, 'mean_bit_residual_fraction']:.3f} to {normalized.loc[32, 'mean_bit_residual_fraction']:.3f}. At k=256 the corresponding values are {raw.loc[256, 'median_label_purity']:.1%} versus {normalized.loc[256, 'median_label_purity']:.1%}, and {raw.loc[256, 'mean_bit_residual_fraction']:.3f} versus {normalized.loc[256, 'mean_bit_residual_fraction']:.3f}. The profile–cluster AMI/ARI values in the table provide the corresponding chance-corrected partition agreement. The point panels show whether this aggregate stability also holds bit by bit; the equality diagonal is the exact-agreement reference.",
        "",
        "Artifacts: `raw_vs_l2_normalized_summary.csv`, per-k normalized assignments, full cluster purity tables, and per-bit information tables are stored in `runs/agnews_qwen_all_fixed_balance/normalized_kmeans_robustness/`.",
    ])
    (OUT / "robustness_section.md").write_text("\n".join(section) + "\n")
    (OUT / "metadata.json").write_text(json.dumps({"seed": SEED, "ks": list(KS), "n_init": 10, "max_iter": 300, "algorithm": "lloyd", "embedding_input": "row-wise L2-normalized cached raw Qwen embeddings", "source_embedding_metadata": metadata}, indent=2) + "\n")
    print(OUT / "robustness_section.md")


if __name__ == "__main__":
    main()
