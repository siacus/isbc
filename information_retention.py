#!/usr/bin/env python3
"""Rate--distortion--geometry diagnostics for sparse binary factor reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_reporting import file_record
from information_retention_extended import (
    categorical_entropy_bits,
    fano_metrics,
    paired_cross_entropy_contrasts,
    pca_and_quantizer_codes,
    posthoc_linear_decoder_metrics_many,
    profile_label_association,
    simple_code_rate,
)


EPSILON = 1e-12


def binary_entropy(probability: np.ndarray | float) -> np.ndarray:
    """Bernoulli entropy in bits, with the conventional 0 log 0 = 0."""
    p = np.asarray(probability, dtype=float)
    entropy = np.zeros_like(p, dtype=float)
    interior = (p > 0.0) & (p < 1.0)
    q = p[interior]
    entropy[interior] = -(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q))
    return entropy


def profile_strings(bits: np.ndarray) -> np.ndarray:
    return np.asarray(["".join(map(str, row.astype(np.uint8))) for row in bits])


def code_rate_metrics(hard: np.ndarray, soft: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    hard = np.asarray(hard, dtype=np.uint8)
    soft = np.asarray(soft, dtype=float)
    if hard.shape != soft.shape or hard.ndim != 2:
        raise ValueError("Hard and soft codes must be two-dimensional and have equal shape")
    n_rows, n_bits = hard.shape
    hard_rate = hard.mean(axis=0)
    soft_rate = soft.mean(axis=0)
    hard_entropy = binary_entropy(hard_rate)
    soft_bernoulli_entropy = binary_entropy(soft).mean(axis=0)
    profiles = profile_strings(hard)
    counts = pd.Series(profiles).value_counts(sort=True)
    shares = counts.to_numpy(dtype=float) / n_rows
    joint_entropy = float(-(shares * np.log2(shares)).sum())
    miller_madow_joint_entropy = min(
        float(n_bits),
        joint_entropy + (len(counts) - 1) / (2.0 * n_rows * np.log(2.0)),
    )
    marginal_entropy_sum = float(hard_entropy.sum())
    observed_states = ((hard_rate > 0.0) & (hard_rate < 1.0)).astype(float) + 1.0
    miller_madow_marginal_sum = float(
        np.minimum(
            1.0,
            hard_entropy + (observed_states - 1.0) / (2.0 * n_rows * np.log(2.0)),
        ).sum()
    )
    total_correlation = max(0.0, marginal_entropy_sum - joint_entropy)
    miller_madow_total_correlation = max(
        0.0, miller_madow_marginal_sum - miller_madow_joint_entropy
    )
    per_bit = pd.DataFrame(
        {
            "bit": [f"bit_{index:02d}" for index in range(n_bits)],
            "hard_activation_rate": hard_rate,
            "soft_mean_probability": soft_rate,
            "hard_marginal_entropy_bits": hard_entropy,
            "mean_soft_bernoulli_entropy_bits": soft_bernoulli_entropy,
        }
    )
    summary = {
        "n_documents": int(n_rows),
        "n_bits": int(n_bits),
        "mean_hard_active_bits": float(hard.sum(axis=1).mean()),
        "sd_hard_active_bits": float(hard.sum(axis=1).std(ddof=0)),
        "mean_soft_active_bits": float(soft.sum(axis=1).mean()),
        "mean_marginal_bit_entropy_bits": float(hard_entropy.mean()),
        "sum_marginal_bit_entropy_bits": marginal_entropy_sum,
        "joint_profile_entropy_bits_per_document": joint_entropy,
        "miller_madow_profile_entropy_bits_per_document": miller_madow_joint_entropy,
        "miller_madow_profile_entropy_correction_bits_per_document": float(
            miller_madow_joint_entropy - joint_entropy
        ),
        "effective_profile_count": float(2.0**joint_entropy),
        "miller_madow_effective_profile_count": float(
            2.0**miller_madow_joint_entropy
        ),
        "observed_profile_count": int(len(counts)),
        "observed_profile_fraction_of_sample": float(len(counts) / n_rows),
        "most_common_profile_share": float(shares[0]),
        "top_10_profile_share": float(shares[:10].sum()),
        "pairwise_profile_collision_probability": float(np.square(shares).sum()),
        "total_correlation_bits_per_document": float(total_correlation),
        "miller_madow_total_correlation_bits_per_document": float(
            miller_madow_total_correlation
        ),
        "normalized_redundancy": (
            float(total_correlation / marginal_entropy_sum)
            if marginal_entropy_sum > 0
            else 0.0
        ),
        "mean_soft_bernoulli_entropy_bits_per_bit": float(
            soft_bernoulli_entropy.mean()
        ),
    }
    return summary, per_bit


def _decoder_parameters(run_dir: Path, checkpoint: str) -> tuple[np.ndarray, np.ndarray] | None:
    model_path = run_dir / checkpoint / "model.pt"
    config_path = run_dir / checkpoint / "run_config.json"
    if not model_path.is_file() or not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("objective", "consistency") != "sparse_ae":
        return None
    import torch

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    weight = state.get("decoder.linear.weight")
    bias = state.get("decoder.linear.bias")
    if weight is None or bias is None:
        return None
    return weight.detach().float().numpy(), bias.detach().float().numpy()


def reconstruction_metrics(
    raw: np.ndarray,
    code: np.ndarray,
    decoder: tuple[np.ndarray, np.ndarray],
    *,
    chunk_size: int = 2048,
) -> dict[str, float]:
    """Evaluate the saved linear decoder without loading the embedding backbone."""
    raw = np.asarray(raw)
    code = np.asarray(code)
    weight, bias = decoder
    if code.shape[1] != weight.shape[1] or raw.shape[1] != weight.shape[0]:
        raise ValueError(
            f"Decoder shape mismatch: raw={raw.shape}, code={code.shape}, weight={weight.shape}"
        )
    mean = np.asarray(raw.mean(axis=0, dtype=np.float64), dtype=np.float64)
    baseline_sse = 0.0
    reconstruction_sse = 0.0
    cosine_sum = 0.0
    cosine_count = 0
    for start in range(0, len(raw), chunk_size):
        stop = min(len(raw), start + chunk_size)
        observed = np.asarray(raw[start:stop], dtype=np.float64)
        reconstructed = (
            np.asarray(code[start:stop], dtype=np.float64) @ weight.T.astype(np.float64)
            + bias.astype(np.float64)
        )
        residual = observed - reconstructed
        reconstruction_sse += float(np.square(residual).sum())
        baseline_sse += float(np.square(observed - mean).sum())
        denominator = np.linalg.norm(observed, axis=1) * np.linalg.norm(
            reconstructed, axis=1
        )
        valid = denominator > EPSILON
        if valid.any():
            cosine_sum += float(
                np.sum(np.einsum("ij,ij->i", observed[valid], reconstructed[valid]) / denominator[valid])
            )
            cosine_count += int(valid.sum())
    n_values = int(np.prod(raw.shape))
    normalized_mse = reconstruction_sse / baseline_sse if baseline_sse > 0 else float("nan")
    return {
        "mse": float(reconstruction_sse / n_values),
        "mean_baseline_mse": float(baseline_sse / n_values),
        "normalized_mse": float(normalized_mse),
        "reconstruction_r2": float(1.0 - normalized_mse),
        "mean_cosine_similarity": float(cosine_sum / cosine_count) if cosine_count else float("nan"),
    }


def _ranks_from_distances(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(distances, axis=1, kind="stable")
    order = np.asarray([row[row != index] for index, row in enumerate(order)])
    ranks = np.zeros((len(order), len(order)), dtype=np.int64)
    row_indices = np.arange(len(order))[:, None]
    ranks[row_indices, order] = np.arange(1, order.shape[1] + 1)[None, :]
    return order, ranks


def _trustworthiness_continuity_from_ranks(
    original_order: np.ndarray,
    original_rank: np.ndarray,
    representation_order: np.ndarray,
    representation_rank: np.ndarray,
    k: int,
) -> tuple[float, float, float]:
    n_rows = len(original_order)
    if n_rows <= 2 * k + 1:
        return float("nan"), float("nan"), float("nan")
    original_neighbors = original_order[:, :k]
    representation_neighbors = representation_order[:, :k]
    overlap = np.mean(
        [
            len(set(original_neighbors[index]).intersection(representation_neighbors[index])) / k
            for index in range(n_rows)
        ]
    )
    trust_penalty = 0.0
    continuity_penalty = 0.0
    for index in range(n_rows):
        original_set = set(original_neighbors[index])
        representation_set = set(representation_neighbors[index])
        intrusions = np.fromiter(representation_set - original_set, dtype=np.int64)
        extrusions = np.fromiter(original_set - representation_set, dtype=np.int64)
        if len(intrusions):
            trust_penalty += float((original_rank[index, intrusions] - k).sum())
        if len(extrusions):
            continuity_penalty += float((representation_rank[index, extrusions] - k).sum())
    scale = 2.0 / (n_rows * k * (2 * n_rows - 3 * k - 1))
    return float(overlap), float(1.0 - scale * trust_penalty), float(
        1.0 - scale * continuity_penalty
    )


def geometry_metrics(
    raw: np.ndarray,
    hard: np.ndarray,
    soft: np.ndarray,
    *,
    seed: int,
    max_documents: int = 2000,
    max_pairs: int = 100_000,
    neighbor_ks: tuple[int, ...] = (10, 25, 50),
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare cosine embedding geometry with hard and soft code geometry."""
    n_rows = len(raw)
    rng = np.random.default_rng(seed)
    sample_size = min(max_documents, n_rows)
    sample_indices = np.sort(rng.choice(n_rows, sample_size, replace=False))
    raw_sample = np.asarray(raw[sample_indices], dtype=np.float32)
    hard_sample = np.asarray(hard[sample_indices], dtype=bool)
    soft_sample = np.asarray(soft[sample_indices], dtype=np.float32)
    raw_distances = pairwise_distances(raw_sample, metric="cosine", n_jobs=-1)
    hard_jaccard = pairwise_distances(hard_sample, metric="jaccard", n_jobs=-1)
    hard_hamming = pairwise_distances(hard_sample, metric="hamming", n_jobs=-1)
    soft_distances = pairwise_distances(soft_sample, metric="cosine", n_jobs=-1)
    raw_order, raw_rank = _ranks_from_distances(raw_distances)
    code_rankings = {
        "hard_jaccard": _ranks_from_distances(hard_jaccard),
        "soft_cosine": _ranks_from_distances(soft_distances),
    }

    upper_i, upper_j = np.triu_indices(sample_size, k=1)
    if len(upper_i) > max_pairs:
        selected = rng.choice(len(upper_i), max_pairs, replace=False)
        upper_i, upper_j = upper_i[selected], upper_j[selected]
    distance_rows = []
    for representation, matrix in (
        ("hard_jaccard", hard_jaccard),
        ("hard_hamming", hard_hamming),
        ("soft_cosine", soft_distances),
    ):
        statistic = spearmanr(
            raw_distances[upper_i, upper_j], matrix[upper_i, upper_j]
        ).statistic
        distance_rows.append(
            {
                "representation": representation,
                "metric": "pairwise_distance_spearman",
                "k": np.nan,
                "value": float(statistic),
            }
        )

    neighborhood_rows = []
    for representation, matrix in (
        ("hard_jaccard", hard_jaccard),
        ("soft_cosine", soft_distances),
    ):
        representation_order, representation_rank = code_rankings[representation]
        for k in neighbor_ks:
            if sample_size <= 2 * k + 1:
                continue
            overlap, trustworthiness, continuity = _trustworthiness_continuity_from_ranks(
                raw_order,
                raw_rank,
                representation_order,
                representation_rank,
                k,
            )
            for metric, value in (
                ("knn_overlap", overlap),
                ("trustworthiness", trustworthiness),
                ("continuity", continuity),
            ):
                neighborhood_rows.append(
                    {
                        "representation": representation,
                        "metric": metric,
                        "k": int(k),
                        "value": value,
                    }
                )
    detail = pd.DataFrame(distance_rows + neighborhood_rows)
    return {
        "seed": int(seed),
        "sample_documents": int(sample_size),
        "population_documents": int(n_rows),
        "sample_pairs": int(len(upper_i)),
        "raw_distance": "cosine",
        "hard_primary_distance": "Jaccard",
        "hard_secondary_distance": "Hamming",
        "soft_distance": "cosine",
        "neighbor_ks": [int(value) for value in neighbor_ks if sample_size > 2 * value + 1],
    }, detail


def _utility_rows(
    modeling: dict[str, Any] | None,
    predictions: pd.DataFrame | None = None,
    cross_entropy_contrasts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not modeling:
        return pd.DataFrame()
    frame = pd.DataFrame(modeling.get("metrics", []))
    if frame.empty:
        return frame
    keep = [
        column
        for column in (
            "run",
            "representation",
            "classifier",
            "test_accuracy",
            "balanced_accuracy",
            "balanced_accuracy_ci_low",
            "balanced_accuracy_ci_high",
            "macro_f1",
            "test_cross_entropy_nats",
            "test_cross_entropy_bits_per_document",
            "predictive_information_over_training_prior_bits_per_document",
        )
        if column in frame
    ]
    frame = frame[keep].copy()
    if "test_cross_entropy_bits_per_document" in frame:
        raw = frame[frame["representation"] == "raw_embeddings"].set_index(
            ["run", "classifier"]
        )["test_cross_entropy_bits_per_document"]
        frame["cross_entropy_loss_vs_raw_bits_per_document"] = [
            (
                float(value - raw.loc[(run, classifier)])
                if (run, classifier) in raw.index
                else np.nan
            )
            for run, classifier, value in zip(
                frame["run"],
                frame["classifier"],
                frame["test_cross_entropy_bits_per_document"],
            )
        ]
    raw_balanced = frame[frame["representation"] == "raw_embeddings"].set_index(
        ["run", "classifier"]
    )["balanced_accuracy"]
    frame["balanced_accuracy_loss_vs_raw"] = [
        (
            float(raw_balanced.loc[(run, classifier)] - value)
            if (run, classifier) in raw_balanced.index
            else np.nan
        )
        for run, classifier, value in zip(
            frame["run"], frame["classifier"], frame["balanced_accuracy"]
        )
    ]
    if predictions is not None and not predictions.empty:
        unique_truth = predictions[["id", "true_label"]].drop_duplicates()
        conflicts = unique_truth.groupby("id")["true_label"].nunique()
        if (conflicts > 1).any():
            raise ValueError("Predictive artifacts contain conflicting true labels")
        truth = unique_truth.drop_duplicates("id")["true_label"].astype(str).to_numpy()
        counts = pd.Series(truth).value_counts().to_numpy(dtype=float)
        probabilities = counts / counts.sum()
        label_entropy = float(-(probabilities * np.log2(probabilities)).sum())
        n_classes = int(len(counts))
        frame["label_entropy_bits_per_document"] = label_entropy
        fano = [
            fano_metrics(value, label_entropy, n_classes)
            for value in frame["test_accuracy"]
        ]
        frame["fano_mutual_information_lower_bound_bits_per_document"] = [
            row["fano_mutual_information_lower_bound_bits_per_document"]
            for row in fano
        ]
        frame["fano_fraction_label_entropy_resolved"] = [
            row["fano_fraction_label_entropy_resolved"] for row in fano
        ]
        if "predictive_information_over_training_prior_bits_per_document" in frame:
            frame["probe_fraction_label_entropy_resolved"] = (
                frame["predictive_information_over_training_prior_bits_per_document"]
                / label_entropy
                if label_entropy > 0
                else np.nan
            )
    if cross_entropy_contrasts is not None and not cross_entropy_contrasts.empty:
        contrast = cross_entropy_contrasts.set_index(
            ["run", "representation", "classifier"]
        )
        for name, source in (
            ("cross_entropy_loss_vs_raw_ci_low", "ci_low"),
            ("cross_entropy_loss_vs_raw_ci_high", "ci_high"),
        ):
            frame[name] = [
                (
                    float(contrast.loc[(run, representation, classifier), source])
                    if (run, representation, classifier) in contrast.index
                    else np.nan
                )
                for run, representation, classifier in zip(
                    frame["run"], frame["representation"], frame["classifier"]
                )
            ]
    return frame


def _plot_overview(summary: pd.DataFrame, geometry: pd.DataFrame, output: Path) -> None:
    runs = summary["run"].tolist()
    x = np.arange(len(runs))
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    axes[0].bar(x - 0.18, summary["joint_profile_entropy_bits_per_document"], 0.36, label="joint profile entropy")
    axes[0].bar(x + 0.18, summary["sum_marginal_bit_entropy_bits"], 0.36, label="sum marginal entropies")
    axes[0].set_ylabel("Bits per document")
    axes[0].set_title("Effective code rate and redundancy")
    axes[0].legend(fontsize=8)

    axes[1].scatter(
        x, summary["hard_reconstruction_r2"], marker="o", s=46, label="hard bits"
    )
    axes[1].scatter(
        x,
        summary["soft_reconstruction_r2"],
        marker="s",
        s=46,
        label="soft probabilities",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Decoder reconstruction $R^2$")
    axes[1].set_title("Embedding reconstructability")
    axes[1].legend(fontsize=8)

    chosen = geometry[
        ((geometry["metric"] == "pairwise_distance_spearman") & geometry["representation"].isin(["hard_jaccard", "soft_cosine"]))
        | ((geometry["metric"] == "knn_overlap") & (geometry["k"] == 25))
    ]
    labels = ["Hard distance", "Soft distance", "Hard kNN@25", "Soft kNN@25"]
    selectors = [
        ("pairwise_distance_spearman", "hard_jaccard"),
        ("pairwise_distance_spearman", "soft_cosine"),
        ("knn_overlap", "hard_jaccard"),
        ("knn_overlap", "soft_cosine"),
    ]
    width = 0.18
    for index, ((metric, representation), label) in enumerate(zip(selectors, labels)):
        values = []
        for run in runs:
            selected = chosen[(chosen["run"] == run) & (chosen["metric"] == metric) & (chosen["representation"] == representation)]
            values.append(float(selected.iloc[0]["value"]) if not selected.empty else np.nan)
        axes[2].bar(x + (index - 1.5) * width, values, width, label=label)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Agreement with raw embedding geometry")
    axes[2].set_title("Global distances and local neighbors")
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.set_xticks(x, runs, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_rate_distortion(summary: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    for _, row in summary.iterrows():
        rate = row["joint_profile_entropy_bits_per_document"]
        axis.scatter(rate, row["hard_normalized_mse"], marker="o", s=70, color="#4472C4")
        axis.scatter(rate, row["soft_normalized_mse"], marker="s", s=70, color="#ED7D31")
        axis.annotate(str(row["run"]), (rate, row["hard_normalized_mse"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.9, label="mean-vector baseline")
    axis.set_xlabel("Empirical hard-profile entropy $H(B)$ (bits/document)")
    axis.set_ylabel("Normalized embedding reconstruction MSE (lower is better)")
    axis.set_title("Empirical rate--distortion view")
    axis.grid(alpha=0.22)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color="#4472C4", label="hard bits"),
        plt.Line2D([0], [0], marker="s", linestyle="none", color="#ED7D31", label="soft probabilities"),
        plt.Line2D([0], [0], linestyle="--", color="black", label="mean-vector baseline"),
    ]
    axis.legend(handles=handles, fontsize=8)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_linear_references(frame: pd.DataFrame, output: Path) -> None:
    runs = frame["run"].tolist()
    x = np.arange(len(runs))
    figure, axis = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    series = (
        ("pca_reconstruction_r2_ceiling", "PCA-K reference (estimated)", "D", "#222222"),
        ("soft_posthoc_linear_decoder_r2", "soft bits + optimal linear decoder", "s", "#ED7D31"),
        ("hard_posthoc_linear_decoder_r2", "hard bits + optimal linear decoder", "o", "#4472C4"),
        ("soft_saved_decoder_r2", "soft bits + learned decoder", "^", "#F4A261"),
        ("hard_saved_decoder_r2", "hard bits + learned decoder", "v", "#4F81BD"),
    )
    for column, label, marker, color in series:
        axis.scatter(x, frame[column], label=label, marker=marker, s=62, color=color)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_xticks(x, runs, rotation=30, ha="right")
    axis.set_ylabel("In-sample reconstruction $R^2$ (higher is better)")
    axis.set_title("Linear ceiling, representation content, and learned decoder")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_cross_entropy_contrasts(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[frame["representation"] != "raw_embeddings"].copy()
    runs = list(dict.fromkeys(selected["run"].tolist()))
    representations = list(dict.fromkeys(selected["representation"].tolist()))
    figure, axis = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    offsets = np.linspace(-0.18, 0.18, max(1, len(representations)))
    colors = {"hard_bits": "#4472C4", "head_log_odds": "#ED7D31"}
    for offset, representation in zip(offsets, representations):
        part = selected[selected["representation"] == representation].set_index("run")
        values = np.asarray([part.loc[run, "cross_entropy_loss_vs_raw_bits_per_document"] for run in runs])
        low = np.asarray([part.loc[run, "ci_low"] for run in runs])
        high = np.asarray([part.loc[run, "ci_high"] for run in runs])
        axis.errorbar(
            np.arange(len(runs)) + offset,
            values,
            yerr=np.vstack([values - low, high - values]),
            fmt="o",
            capsize=3,
            label=representation.replace("_", " "),
            color=colors.get(representation),
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axis.set_xticks(np.arange(len(runs)), runs, rotation=30, ha="right")
    axis.set_ylabel("CE(code) - CE(raw), bits/document (lower is better)")
    axis.set_title("Paired held-out cross-entropy loss with 95% bootstrap intervals")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_quantizer_references(frame: pd.DataFrame, output: Path) -> None:
    runs = list(dict.fromkeys(frame["run"].tolist()))
    methods = list(dict.fromkeys(frame["method"].tolist()))
    figure, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
    markers = ["o", "s", "^", "D", "P"]
    for index, method in enumerate(methods):
        part = frame[frame["method"] == method].set_index("run")
        xs = np.arange(len(runs)) + (index - (len(methods) - 1) / 2) * 0.09
        axes[0].scatter(
            xs,
            [part.loc[run, "posthoc_linear_decoder_r2"] for run in runs],
            marker=markers[index % len(markers)],
            s=45,
            label=method.replace("_", " "),
        )
        if "balanced_accuracy" in frame and part["balanced_accuracy"].notna().any():
            axes[1].scatter(
                xs,
                [part.loc[run, "balanced_accuracy"] for run in runs],
                marker=markers[index % len(markers)],
                s=45,
                label=method.replace("_", " "),
            )
    axes[0].set_ylabel("Optimal linear reconstruction $R^2$ (higher is better)")
    axes[0].set_title("Reconstruction from nonlearned binary controls")
    axes[1].set_ylabel("Held-out balanced accuracy (higher is better)")
    axes[1].set_title("Taxonomy utility of nonlearned binary controls")
    for axis in axes:
        axis.set_xticks(np.arange(len(runs)), runs, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.22)
    axes[1].legend(fontsize=7, loc="best")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compute_information_retention(
    *,
    runs: dict[str, dict[str, Any]],
    raw_embeddings: dict[str, np.ndarray],
    output_dir: Path,
    checkpoint: str,
    seed: int,
    predictive_modeling: dict[str, Any] | None,
    labels: np.ndarray | None = None,
    ids: np.ndarray | None = None,
    split_assignment: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute and persist every information-retention diagnostic for one report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    bit_frames: list[pd.DataFrame] = []
    geometry_frames: list[pd.DataFrame] = []
    geometry_protocols: dict[str, Any] = {}
    profile_label_rows: list[dict[str, Any]] = []
    linear_reference_rows: list[dict[str, Any]] = []
    quantizer_rows: list[dict[str, Any]] = []
    reference_protocols: dict[str, Any] = {}
    learned_logistic = {
        (str(row.get("run")), str(row.get("representation"))): row
        for row in (predictive_modeling or {}).get("metrics", [])
        if row.get("classifier") == "logistic_regression"
    }
    y = None if labels is None else np.asarray(labels).astype(str)
    aligned_ids = None if ids is None else np.asarray(ids).astype(str)
    if y is not None and len(y) != len(next(iter(runs.values()))["matrix"]):
        raise ValueError("Information-retention labels are not aligned to code rows")
    if aligned_ids is not None and len(aligned_ids) != len(next(iter(runs.values()))["matrix"]):
        raise ValueError("Information-retention ids are not aligned to code rows")
    train_indices = test_indices = None
    if split_assignment is not None:
        split = np.asarray(split_assignment).astype(str)
        train_indices = np.flatnonzero(split == "train")
        test_indices = np.flatnonzero(split == "test")
        if len(split) != len(next(iter(runs.values()))["matrix"]):
            raise ValueError("Information-retention split is not aligned to code rows")

    for offset, (name, run) in enumerate(runs.items()):
        hard = np.asarray(run["matrix"], dtype=np.uint8)
        soft = np.asarray(run["probability_matrix"], dtype=np.float32)
        rate, per_bit = code_rate_metrics(hard, soft)
        row: dict[str, Any] = {"run": name, **rate}
        per_bit.insert(0, "run", name)
        bit_frames.append(per_bit)
        learned_association = None
        if y is not None:
            learned_association = profile_label_association(hard, y)
            profile_label_rows.append({"run": name, **learned_association})
        raw = raw_embeddings.get(name)
        if raw is not None:
            embedding_dim = int(raw.shape[1])
            row["embedding_dim"] = embedding_dim
            row["nominal_raw_float32_bits_per_document"] = int(32 * embedding_dim)
            row["nominal_binary_compression_ratio"] = float(32 * embedding_dim / hard.shape[1])
            decoder = _decoder_parameters(Path(run["run_dir"]), checkpoint)
            if decoder is not None:
                for prefix, code in (("hard", hard), ("soft", soft)):
                    metrics = reconstruction_metrics(raw, code, decoder)
                    row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
            protocol, detail = geometry_metrics(
                raw, hard, soft, seed=seed + offset
            )
            detail.insert(0, "run", name)
            geometry_frames.append(detail)
            geometry_protocols[name] = protocol

            if aligned_ids is not None:
                target_active = int(
                    round(
                        float(
                            run.get("training", {}).get(
                                "target_active", rate["mean_hard_active_bits"]
                            )
                        )
                    )
                )
                pca_metrics, reference_codes, reference_protocol = (
                    pca_and_quantizer_codes(
                        raw,
                        aligned_ids,
                        n_bits=hard.shape[1],
                        target_active=target_active,
                        seed=seed + offset,
                        cache_dir=(
                            output_dir.parent
                            / "_information_reference_cache"
                            / name
                        ),
                        raw_cache_key=str(run.get("raw_cache_key", "")),
                    )
                )
                all_codes = {"hard": hard, "soft": soft, **reference_codes}
                posthoc = posthoc_linear_decoder_metrics_many(raw, all_codes)
                linear_reference_rows.append(
                    {
                        "run": name,
                        "n_bits": int(hard.shape[1]),
                        **pca_metrics,
                        "hard_posthoc_linear_decoder_r2": posthoc["hard"][
                            "posthoc_linear_decoder_r2"
                        ],
                        "soft_posthoc_linear_decoder_r2": posthoc["soft"][
                            "posthoc_linear_decoder_r2"
                        ],
                        "hard_saved_decoder_r2": row.get("hard_reconstruction_r2"),
                        "soft_saved_decoder_r2": row.get("soft_reconstruction_r2"),
                        "hard_binary_sparse_gap_from_pca": (
                            pca_metrics["pca_reconstruction_r2_ceiling"]
                            - posthoc["hard"]["posthoc_linear_decoder_r2"]
                        ),
                        "soft_representation_gap_from_pca": (
                            pca_metrics["pca_reconstruction_r2_ceiling"]
                            - posthoc["soft"]["posthoc_linear_decoder_r2"]
                        ),
                        "hard_decoder_optimization_gap": (
                            posthoc["hard"]["posthoc_linear_decoder_r2"]
                            - row.get("hard_reconstruction_r2", np.nan)
                        ),
                        "soft_decoder_optimization_gap": (
                            posthoc["soft"]["posthoc_linear_decoder_r2"]
                            - row.get("soft_reconstruction_r2", np.nan)
                        ),
                    }
                )
                reference_protocols[name] = {
                    **reference_protocol,
                    "target_active_bits": target_active,
                    "pca_scope": "same complete aligned corpus as the in-sample SAE reconstruction",
                    "posthoc_decoder": "affine ordinary least squares fitted in-sample from sufficient statistics",
                }
                learned_row: dict[str, Any] = {
                    "run": name,
                    "method": "learned_sae_hard",
                    "n_bits": int(hard.shape[1]),
                    "target_active_bits": target_active,
                    **simple_code_rate(hard),
                    **posthoc["hard"],
                }
                if learned_association is not None:
                    learned_row.update(
                        {
                            "adjusted_mutual_information": learned_association[
                                "adjusted_mutual_information"
                            ],
                            "chance_corrected_mutual_information_bits_per_document": learned_association[
                                "chance_corrected_mutual_information_bits_per_document"
                            ],
                        }
                    )
                learned_score = learned_logistic.get((name, "hard_bits"))
                if learned_score is not None:
                    learned_row.update(learned_score)
                    if y is not None and test_indices is not None and len(test_indices):
                        label_entropy = categorical_entropy_bits(y[test_indices])
                        learned_row.update(
                            fano_metrics(
                                learned_score["test_accuracy"],
                                label_entropy,
                                len(np.unique(y[test_indices])),
                            )
                        )
                        learned_row["label_entropy_bits_per_document"] = label_entropy
                        learned_row["probe_fraction_label_entropy_resolved"] = (
                            learned_score.get(
                                "predictive_information_over_training_prior_bits_per_document",
                                np.nan,
                            )
                            / label_entropy
                            if label_entropy > 0
                            else np.nan
                        )
                quantizer_rows.append(learned_row)
                for method, code in reference_codes.items():
                    quantizer_row: dict[str, Any] = {
                        "run": name,
                        "method": method,
                        "n_bits": int(code.shape[1]),
                        "target_active_bits": target_active,
                        **simple_code_rate(code),
                        **posthoc[method],
                    }
                    if y is not None:
                        association = profile_label_association(code, y)
                        quantizer_row.update(
                            {
                                "adjusted_mutual_information": association[
                                    "adjusted_mutual_information"
                                ],
                                "chance_corrected_mutual_information_bits_per_document": association[
                                    "chance_corrected_mutual_information_bits_per_document"
                                ],
                            }
                        )
                    if (
                        y is not None
                        and train_indices is not None
                        and test_indices is not None
                        and len(train_indices)
                        and len(test_indices)
                    ):
                        from predictive_modeling import _fit_one

                        fitted, _, _, _, _ = _fit_one(
                            code,
                            y,
                            train_indices,
                            test_indices,
                            "logistic_regression",
                            seed,
                        )
                        quantizer_row.update(fitted)
                        label_entropy = categorical_entropy_bits(y[test_indices])
                        quantizer_row.update(
                            fano_metrics(
                                fitted["test_accuracy"],
                                label_entropy,
                                len(np.unique(y[test_indices])),
                            )
                        )
                        quantizer_row["label_entropy_bits_per_document"] = label_entropy
                        quantizer_row["probe_fraction_label_entropy_resolved"] = (
                            fitted.get(
                                "predictive_information_over_training_prior_bits_per_document",
                                np.nan,
                            )
                            / label_entropy
                            if label_entropy > 0
                            else np.nan
                        )
                    quantizer_rows.append(quantizer_row)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    bits = pd.concat(bit_frames, ignore_index=True)
    geometry = (
        pd.concat(geometry_frames, ignore_index=True)
        if geometry_frames
        else pd.DataFrame(columns=["run", "representation", "metric", "k", "value"])
    )
    predictions = None
    cross_entropy_contrasts = pd.DataFrame()
    if predictive_modeling:
        prediction_record = predictive_modeling.get("artifacts", {}).get("predictions")
        if prediction_record:
            prediction_path = Path(prediction_record)
            if not prediction_path.is_absolute():
                prediction_path = output_dir / prediction_path
            if prediction_path.is_file():
                predictions = pd.read_csv(prediction_path, dtype={"id": str})
                cross_entropy_contrasts = paired_cross_entropy_contrasts(
                    predictions, seed=seed
                )
    utility = _utility_rows(
        predictive_modeling,
        predictions=predictions,
        cross_entropy_contrasts=cross_entropy_contrasts,
    )
    profile_label = pd.DataFrame(profile_label_rows)
    linear_references = pd.DataFrame(linear_reference_rows)
    quantizers = pd.DataFrame(quantizer_rows)
    summary_path = output_dir / "information_retention_summary.csv"
    bits_path = output_dir / "information_retention_per_bit.csv"
    geometry_path = output_dir / "information_retention_geometry.csv"
    utility_path = output_dir / "information_retention_utility.csv"
    profile_label_path = output_dir / "information_retention_profile_label.csv"
    linear_references_path = output_dir / "information_retention_linear_references.csv"
    quantizers_path = output_dir / "information_retention_quantizer_baselines.csv"
    cross_entropy_path = output_dir / "information_retention_cross_entropy_contrasts.csv"
    summary.to_csv(summary_path, index=False)
    bits.to_csv(bits_path, index=False)
    geometry.to_csv(geometry_path, index=False)
    utility.to_csv(utility_path, index=False)
    profile_label.to_csv(profile_label_path, index=False)
    linear_references.to_csv(linear_references_path, index=False)
    quantizers.to_csv(quantizers_path, index=False)
    cross_entropy_contrasts.to_csv(cross_entropy_path, index=False)
    overview_path = output_dir / "information_retention_overview.png"
    rate_distortion_path = output_dir / "information_retention_rate_distortion.png"
    linear_references_plot_path = output_dir / "information_retention_linear_references.png"
    quantizers_plot_path = output_dir / "information_retention_quantizer_baselines.png"
    cross_entropy_plot_path = output_dir / "information_retention_cross_entropy_contrasts.png"
    if geometry_frames:
        _plot_overview(summary, geometry, overview_path)
    if {"hard_normalized_mse", "soft_normalized_mse"}.issubset(summary.columns):
        _plot_rate_distortion(summary, rate_distortion_path)
    if not linear_references.empty:
        _plot_linear_references(linear_references, linear_references_plot_path)
    if not quantizers.empty:
        _plot_quantizer_references(quantizers, quantizers_plot_path)
    if not cross_entropy_contrasts.empty:
        _plot_cross_entropy_contrasts(
            cross_entropy_contrasts, cross_entropy_plot_path
        )

    metadata = {
        "schema_version": 2,
        "framework": "rate--distortion--geometry--utility",
        "seed": int(seed),
        "checkpoint": checkpoint,
        "reconstruction_scope": (
            "descriptive reconstruction on the same complete corpus used to fit the unsupervised SAE; "
            "not an out-of-sample generalization estimate"
        ),
        "rate_definitions": {
            "joint_profile_entropy": "empirical Shannon entropy H(B) of complete hard profiles, in bits/document",
            "miller_madow_profile_entropy": "first-order finite-sample correction H_plugin + (observed_profiles - 1)/(2 n ln 2), capped at hard width; a sensitivity estimate, not ground truth",
            "effective_profiles": "2^H(B)",
            "total_correlation": "sum_j H(B_j) - H(B)",
            "normalized_redundancy": "total correlation divided by sum_j H(B_j)",
            "collision_probability": "sum_b p(b)^2, the probability that two independent documents share a profile",
            "nominal_compression_ratio": "32 times raw embedding dimension divided by hard bit width; excludes model/decoder storage",
        },
        "distortion_definitions": {
            "normalized_mse": "decoder MSE divided by MSE of predicting the corpus mean embedding",
            "reconstruction_r2": "1 - normalized MSE; can be negative",
            "cosine_similarity": "mean cosine similarity between raw and decoder-reconstructed embeddings",
            "pca_k_ceiling": "theoretical maximum empirical in-sample R-squared attainable by any K-dimensional affine linear reconstruction; the reported value is a deterministic high-accuracy randomized-SVD numerical estimate on the same aligned corpus",
            "posthoc_linear_decoder": "optimal in-sample affine least-squares decoder for a fixed hard or soft code",
        },
        "geometry_definitions": {
            "distance_spearman": "Spearman correlation across sampled pairwise distances",
            "knn_overlap": "mean fraction of raw-embedding k-nearest neighbors retained by the code",
            "trustworthiness": "penalizes code-space neighbors that were not close in raw embedding space",
            "continuity": "penalizes raw-space neighbors that are lost in code space",
            "tie_rule": (
                "equal distances are ordered deterministically by aligned document order via stable sorting"
            ),
        },
        "utility_definition": (
            "held-out taxonomy prediction on the common 80/20 split; balanced accuracy is task-specific. "
            "Cross-entropy differences in bits/document are probe-dependent estimates, not model-free mutual information."
        ),
        "usable_information_definition": (
            "held-out training-prior cross-entropy minus logistic-probe cross-entropy; "
            "an empirical probe-dependent usable-information estimate, not classical mutual information"
        ),
        "fano_definition": (
            "max(0, H(Y)-h2(error)-error*log2(C-1)) using ordinary held-out classification error; "
            "a lower bound on classical mutual information, not based on balanced accuracy"
        ),
        "profile_label_definition": (
            "plug-in profile-label mutual information plus exact expected MI under fixed contingency-table margins "
            "and adjusted mutual information; all use the complete aligned corpus"
        ),
        "binary_reference_definition": (
            "the learned SAE hard code is shown beside equal-width PCA-sign, ITQ, random-hyperplane, "
            "PCA top-s, and random top-s references; top-s controls match the configured activity count"
        ),
        "geometry_protocols": geometry_protocols,
        "reference_protocols": reference_protocols,
        "artifacts": {
            "summary": file_record(summary_path),
            "per_bit": file_record(bits_path),
            "geometry": file_record(geometry_path),
            "utility": file_record(utility_path),
            "profile_label": file_record(profile_label_path),
            "linear_references": file_record(linear_references_path),
            "quantizer_baselines": file_record(quantizers_path),
            "cross_entropy_contrasts": file_record(cross_entropy_path),
            "overview_plot": file_record(overview_path) if overview_path.is_file() else None,
            "rate_distortion_plot": file_record(rate_distortion_path) if rate_distortion_path.is_file() else None,
            "linear_references_plot": file_record(linear_references_plot_path) if linear_references_plot_path.is_file() else None,
            "quantizer_baselines_plot": file_record(quantizers_plot_path) if quantizers_plot_path.is_file() else None,
            "cross_entropy_contrasts_plot": file_record(cross_entropy_plot_path) if cross_entropy_plot_path.is_file() else None,
        },
    }
    metadata_path = output_dir / "information_retention_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["artifacts"]["metadata"] = file_record(metadata_path)
    clean_summary = summary.astype(object).where(pd.notna(summary), None)
    metadata["summary_rows"] = clean_summary.to_dict(orient="records")
    clean_utility = utility.astype(object).where(pd.notna(utility), None)
    metadata["utility_rows"] = clean_utility.to_dict(orient="records")
    clean_geometry = geometry.astype(object).where(pd.notna(geometry), None)
    metadata["geometry_rows"] = clean_geometry.to_dict(orient="records")
    metadata["profile_label_rows"] = profile_label.astype(object).where(
        pd.notna(profile_label), None
    ).to_dict(orient="records")
    metadata["linear_reference_rows"] = linear_references.astype(object).where(
        pd.notna(linear_references), None
    ).to_dict(orient="records")
    metadata["quantizer_rows"] = quantizers.astype(object).where(
        pd.notna(quantizers), None
    ).to_dict(orient="records")
    metadata["cross_entropy_contrast_rows"] = cross_entropy_contrasts.astype(object).where(
        pd.notna(cross_entropy_contrasts), None
    ).to_dict(orient="records")
    return metadata


def information_retention_markdown(manifest: dict[str, Any]) -> list[str]:
    """Render a self-contained explanation and compact results table."""
    def number(value: Any, digits: int = 3) -> str:
        return "—" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"

    rows = manifest.get("summary_rows", [])
    artifacts = manifest["artifacts"]
    lines = [
        "## Information retained by the binary bottleneck",
        "",
        "There is no single, model-free percentage of *semantic information retained*. "
        "An embedding contains many possible signals, and every measure asks a different "
        "question. We therefore use a **rate--distortion--geometry--utility** framework. "
        "The four families must be read together; none is a substitute for the others.",
        "In the reading guides below, **higher is better** or **lower is better** means "
        "better for the stated diagnostic only. A metric can improve while a different "
        "kind of information worsens; metrics marked **descriptive/trade-off** have no "
        "universally optimal direction.",
        "",
        "### 1. Code rate, diversity, and redundancy",
        "",
        "The empirical hard-code rate is the joint Shannon entropy "
        "`H(B) = -sum_b p(b) log2 p(b)` of complete binary profiles. It is the average "
        "number of bits per document that an ideal entropy coder would need for the "
        "observed profile distribution—not the nominal width of the code. `2^H(B)` is "
        "the effective number of equally common profiles. The sum of marginal bit "
        "entropies ignores dependence; subtracting joint entropy gives total correlation, "
        "or redundancy shared across bits. Normalized redundancy divides this quantity "
        "by the marginal-entropy sum. Profile collision probability is the chance that "
        "two independently sampled documents receive exactly the same hard profile.",
        "The per-bit artifact also distinguishes hard marginal entropy from mean soft "
        "Bernoulli entropy: the former measures whether a bit is used across documents, "
        "whereas the latter measures head uncertainty within documents (zero is confident "
        "and one bit corresponds to probability 0.5). All profile entropies are empirical "
        "plug-in estimates and therefore depend on corpus size; they are not mutual "
        "information with document meaning. The report also gives the Miller--Madow "
        "first-order correction as a finite-sample sensitivity check. A small correction "
        "supports the plug-in estimate; a large one warns that many rare profiles make "
        "the estimated rate sample-sensitive.",
        "",
        "The nominal compression ratio compares `32 × embedding dimension` float32 bits "
        "with the hard-code width. It describes per-document storage only: it does not "
        "include the encoder or decoder parameters and is not an empirical information "
        "measure.",
        "",
        "**How to read the code-rate statistics**",
        "",
        "| Statistic | What it measures | Direction for information/compression |",
        "|---|---|---|",
        "| Raw dimension | Number of continuous embedding coordinates. | Descriptive: larger is not automatically more informative; it only increases nominal storage. |",
        "| Hard width | Number of available binary coordinates. | A design choice, not a quality score. At equal quality, fewer bits mean stronger compression. |",
        "| Mean active bits | Average number of 1s per document. | Best read against the configured training target, not maximized or minimized. Near 0 or the full width indicates collapse. |",
        "| Joint profile entropy `H(B)` | Empirical information rate/diversity of complete hard profiles, in bits per document. | Higher means more code diversity and utilized capacity; lower means stronger compressibility. It does **not** by itself mean more semantic information. |",
        "| Miller--Madow `H(B)` | First-order bias-corrected profile entropy, capped by the nominal hard width. | Read it as a sensitivity estimate: closeness to plug-in `H(B)` is reassuring; a large upward correction signals undersampling, not necessarily useful information. |",
        "| Effective profiles `2^H(B)` | Number of equally frequent profiles that would have the same entropy. | Higher means greater effective diversity; compare with observed profiles to diagnose concentration. Not a standalone quality score. |",
        "| Observed profiles | Distinct hard profiles present in the corpus. | Higher means more observed variety, but can also reflect rare/noisy patterns; descriptive. |",
        "| Total correlation | Dependence/redundancy among bits: `sum_j H(B_j) - H(B)`. | Lower means a less redundant use of marginal bit capacity; higher means more shared/duplicated structure. Zero is independent bits, not necessarily the most interpretable solution. |",
        "| Normalized redundancy | Total correlation divided by summed marginal entropy. | Lower means less redundancy; 0 is no measured dependence and 1 approaches fully redundant marginal information. |",
        "| Top-10 profile share | Fraction of documents assigned to the ten most common profiles. | Lower means less concentration and usually richer code use; high values warn that a few profiles dominate. |",
        "| Collision probability | Probability that two randomly drawn documents have the same hard profile. | Lower means greater document-level discrimination; higher means more collisions/concentration. |",
        "| Nominal compression ratio | Float32 embedding storage divided by hard-code storage. | Higher means more nominal storage compression, **not** more retained semantic information; encoder/decoder storage is excluded. |",
        "| Per-bit hard activation rate | Fraction of documents activating a bit. | Values near 0 or 1 imply a nearly dead or universal bit; 0.5 maximizes that bit's marginal entropy, but need not maximize usefulness. |",
        "| Per-bit hard marginal entropy | Across-document variability of a thresholded bit. | Higher means the bit separates the corpus more evenly (maximum 1 at 50% activation); low means little variable information. |",
        "| Mean soft Bernoulli entropy | Average uncertainty of the bit probability within documents. | **Lower is more confident** (probabilities near 0/1); 1 means maximal uncertainty at 0.5. Confidence is not the same as semantic usefulness. |",
        "",
        "| Backbone | Raw dim | Hard width | Mean active | H(profile), plug-in/MM | Effective/observed profiles | Total correlation | Redundancy | Top-10 share | Collision prob. | Nominal ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row.get('embedding_dim', '—')} | {row['n_bits']} | "
            f"{row['mean_hard_active_bits']:.2f} | {row['joint_profile_entropy_bits_per_document']:.3f}/"
            f"{number(row.get('miller_madow_profile_entropy_bits_per_document'))} | "
            f"{row['effective_profile_count']:.1f}/{row['observed_profile_count']} | "
            f"{row['total_correlation_bits_per_document']:.3f} | "
            f"{row['normalized_redundancy']:.3f} | {row['top_10_profile_share']:.3f} | "
            f"{row['pairwise_profile_collision_probability']:.4f} | "
            + (f"{row['nominal_binary_compression_ratio']:.0f}:1 |" if row.get('nominal_binary_compression_ratio') is not None else "— |")
        )
    lines.extend(
        [
            "",
            "### 2. Embedding reconstruction distortion",
            "",
            "The saved SAE decoder reconstructs the raw, non-L2-normalized pooled "
            "embedding from either thresholded hard bits or soft activation probabilities. "
            "Normalized MSE divides decoder MSE by the MSE of the trivial corpus-mean "
            "predictor; `R² = 1 - normalized MSE`, so 1 is perfect, 0 matches the mean "
            "baseline, and negative values are worse than that baseline. Mean cosine "
            "similarity measures directional rather than scale agreement.",
            "",
            "These are **descriptive, in-sample reconstructability measures** because the "
            "unsupervised SAE was deliberately learned on the complete corpus. They show "
            "what the fitted bottleneck and decoder preserved on those documents; they do "
            "not estimate reconstruction on unseen documents.",
            "",
            "**How to read the reconstruction statistics**",
            "",
            "| Statistic | What it measures | Better direction |",
            "|---|---|---|",
            "| Raw MSE | Mean squared coordinate error between the raw embedding and decoder reconstruction. | Lower is better, but values are backbone-scale dependent and should not be compared across differently scaled embedding models. |",
            "| Mean-baseline MSE | Error from always predicting the corpus-mean embedding. | Reference denominator, not a model score to optimize. |",
            "| Normalized MSE | Decoder MSE divided by mean-baseline MSE. | **Lower is better**: 0 is exact reconstruction, 1 ties the mean baseline, and values above 1 are worse than the baseline. |",
            "| Reconstruction `R²` | `1 - normalized MSE`, the fraction of embedding variance reconstructed relative to the mean baseline. | **Higher is better**: 1 is exact, 0 ties the baseline, and negative values are worse than the baseline. |",
            "| Mean cosine similarity | Average directional agreement between raw and reconstructed embedding vectors, ignoring overall magnitude. | **Higher is better**: 1 is the same direction, 0 is orthogonal, and -1 is opposite. |",
            "| Hard versus soft reconstruction | Reconstruction from thresholded 0/1 bits versus activation probabilities. | At the same metric, the better hard score shows what survives binarization; a better soft score quantifies information lost specifically at thresholding. |",
            "",
            "| Backbone | Hard normalized MSE | Hard R² | Hard cosine | Soft normalized MSE | Soft R² | Soft cosine |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['run']} | {number(row.get('hard_normalized_mse'))} | "
            f"{number(row.get('hard_reconstruction_r2'))} | "
            f"{number(row.get('hard_mean_cosine_similarity'))} | "
            f"{number(row.get('soft_normalized_mse'))} | "
            f"{number(row.get('soft_reconstruction_r2'))} | "
            f"{number(row.get('soft_mean_cosine_similarity'))} |"
        )
    lines.extend(
        [
            "",
            f"![Empirical rate-distortion view]({Path(artifacts['rate_distortion_plot']['path']).name})" if artifacts.get("rate_distortion_plot") else "",
            "",
            "**Reading the rate--distortion plot.** The x-axis is empirical profile "
            "entropy: moving right means the code uses a higher information rate and is "
            "less compressible, not automatically that its semantics are better. The "
            "y-axis is normalized reconstruction MSE, so moving down is better. There is "
            "no single preferred corner when rate and fidelity are both objectives: one "
            "point dominates another only if it attains lower distortion at no greater "
            "rate, or lower rate at no greater distortion. The dashed line at 1 is the "
            "corpus-mean baseline; points below it beat that baseline and points above it "
            "do not. Circles are hard bits and squares are soft probabilities; a vertical "
            "hard--soft gap isolates thresholding loss for the same backbone.",
        ]
    )
    linear_rows = manifest.get("linear_reference_rows", [])
    if linear_rows:
        lines.extend(
            [
                "",
                "### 2a. PCA-K reference and decoder decomposition",
                "",
                "Because the decoder is affine and the code has `K` coordinates, every "
                "reconstruction lies in an affine subspace of dimension at most `K`. "
                "The Eckart--Young result therefore makes PCA-`K` the theoretical "
                "best in-sample rank-`K` linear reconstruction. The table reports a "
                "deterministic high-accuracy randomized-SVD estimate of that reference; "
                "it is a numerical estimate, not a certified exact bound.",
                "",
                "We additionally refit an optimal affine least-squares decoder while "
                "holding each learned code fixed. The gap from PCA to this post-hoc "
                "decoder measures the combined cost of the learned subspace, nonlinear "
                "encoding, sparsity and (for hard codes) binarization. It must **not** be "
                "called a pure binarization cost. The gap between post-hoc and saved "
                "decoder isolates imperfect decoder fitting for that fixed code. Comparing "
                "hard and soft post-hoc scores is the cleanest available thresholding "
                "diagnostic, although the hard and soft regressions are separately refit.",
                "",
                "| Quantity | Better direction |",
                "|---|---|",
                "| PCA-`K` reconstruction R² | Higher is better; it is the linear dimensional reference at the same `K`. |",
                "| Post-hoc hard/soft R² | Higher is better; it measures linearly decodable embedding variance in the fixed representation. |",
                "| PCA minus post-hoc gap | Lower is better for reconstruction retention; zero approaches the rank-`K` reference. |",
                "| Post-hoc minus saved-decoder gap | Lower is better; a positive value is decoder/optimization headroom rather than representation loss. |",
                "",
                "| Backbone | PCA-K R² (estimated) | Hard post-hoc R² | Soft post-hoc R² | Hard gap to PCA | Soft gap to PCA | Hard decoder gap | Soft decoder gap |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in linear_rows:
            lines.append(
                f"| {row['run']} | {number(row.get('pca_reconstruction_r2_ceiling'))} | "
                f"{number(row.get('hard_posthoc_linear_decoder_r2'))} | "
                f"{number(row.get('soft_posthoc_linear_decoder_r2'))} | "
                f"{number(row.get('hard_binary_sparse_gap_from_pca'))} | "
                f"{number(row.get('soft_representation_gap_from_pca'))} | "
                f"{number(row.get('hard_decoder_optimization_gap'))} | "
                f"{number(row.get('soft_decoder_optimization_gap'))} |"
            )
        if artifacts.get("linear_references_plot"):
            lines.extend(
                [
                    "",
                    f"![PCA-K and decoder references]({Path(artifacts['linear_references_plot']['path']).name})",
                    "",
                    "**Reading the linear-reference plot.** `Hard bits + learned decoder` "
                    "uses the decoder trained jointly with the SAE: during training its "
                    "forward pass saw Bernoulli-sampled hard bits through the straight-through "
                    "estimator. The reported hard code is instead the deterministic 0.5 "
                    "threshold used at encoding. `Hard bits "
                    "+ optimal linear decoder` keeps those exact learned bits fixed but "
                    "refits the best affine OLS map to the raw embeddings. Their vertical "
                    "gap is decoder/training headroom, not information absent from the "
                    "bits. The PCA-K point is the estimated best rank-K linear reference "
                    "over any K-dimensional representation; its gap to hard-bit OLS is a "
                    "combined representation cost and cannot be attributed to binarization "
                    "alone. Higher R² is better for every marker.",
                ]
            )
    lines.extend(
        [
            "",
            "### 3. Geometry preservation",
            "",
            "Pairwise-distance Spearman correlation asks whether document pairs retain "
            "their global ordering from near to far. Raw embeddings use cosine distance; "
            "hard codes use Jaccard distance as the primary measure (with Hamming also "
            "saved), and soft codes use cosine distance. This is rank agreement, not "
            "equality of distances.",
            "",
            "For local structure, k-nearest-neighbor overlap is the fraction of each "
            "document's raw-embedding neighbors that remain neighbors after compression. "
            "Trustworthiness penalizes *intrusions*—documents made artificially close by "
            "the code—whereas continuity penalizes *extrusions*—raw neighbors lost after "
            "compression. Both range toward 1 when local neighborhoods are preserved. "
            "All geometry results use the deterministic document and pair samples recorded "
            "in the metadata; they are not label-dependent.",
            "Equal distances, which are common for binary codes, are resolved by a "
            "stable deterministic ordering of aligned document IDs; the same tie rule "
            "is applied to every backbone.",
            "",
            "**How to read the geometry statistics**",
            "",
            "| Statistic | What it measures | Better direction |",
            "|---|---|---|",
            "| Pairwise-distance Spearman `rho` | Global agreement in the rank ordering of sampled document-pair distances. | **Higher is better**: 1 is identical ordering, 0 is no monotonic agreement, and negative values indicate reversed ordering. |",
            "| kNN overlap at `k` | Fraction of each document's raw-embedding nearest neighbors also retained by the code. | **Higher is better**: 1 is complete overlap and 0 is none. Values at different `k` answer different neighborhood-scale questions. |",
            "| Trustworthiness at `k` | Whether code-space neighbors are genuine raw-space neighbors; it penalizes false neighbors (intrusions). | **Higher is better**: 1 means no neighborhood intrusions. |",
            "| Continuity at `k` | Whether raw-space neighbors remain neighbors after compression; it penalizes lost neighbors (extrusions). | **Higher is better**: 1 means no raw neighbors are lost. |",
            "| Hard Jaccard versus hard Hamming | Two distance definitions for binary codes; Jaccard ignores shared zeros, Hamming counts every mismatch. | Compare correlations under the same definition; higher is better. Jaccard is primary, while Hamming is a robustness check in the artifact. |",
            "",
            f"![Code rate, reconstruction, and geometry overview]({Path(artifacts['overview_plot']['path']).name})" if artifacts.get("overview_plot") else "",
            "",
            "**Reading the three-panel overview.** In the left panel, taller joint "
            "entropy means greater profile diversity, while the vertical gap up to the "
            "sum of marginal entropies is total correlation (redundancy); entropy itself "
            "is a rate--diversity trade-off, whereas a smaller gap means less redundant "
            "capacity use. In the middle panel, higher reconstruction `R²` is better; the "
            "dashed zero line is the corpus-mean baseline and negative values are worse. "
            "In the right panel, every bar is an agreement score and higher is better: "
            "distance bars summarize global rank geometry and kNN bars summarize local "
            "neighbor retention. Backbone positions on the x-axis are categories, not a "
            "continuous path or ordering.",
        ]
    )
    geometry_rows = manifest.get("geometry_rows", [])
    if geometry_rows:
        def geometry_value(run: str, representation: str, metric: str, k: int | None = None) -> float | None:
            for item in geometry_rows:
                if (
                    item.get("run") == run
                    and item.get("representation") == representation
                    and item.get("metric") == metric
                    and (k is None or item.get("k") == k)
                ):
                    return item.get("value")
            return None

        lines.extend(
            [
                "",
                "The compact table uses `k=25`; the linked geometry artifact also stores "
                "all values at `k=10` and `k=50` plus hard-code Hamming distance correlation.",
                "",
                "| Backbone | Distance rho hard/soft | kNN overlap hard/soft | Trustworthiness hard/soft | Continuity hard/soft |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for run in [row["run"] for row in rows]:
            pairs = []
            for metric, k in (
                ("pairwise_distance_spearman", None),
                ("knn_overlap", 25),
                ("trustworthiness", 25),
                ("continuity", 25),
            ):
                values = [
                    geometry_value(run, "hard_jaccard", metric, k),
                    geometry_value(run, "soft_cosine", metric, k),
                ]
                pairs.append(
                    "/".join("—" if value is None else f"{value:.3f}" for value in values)
                )
            lines.append(f"| {run} | " + " | ".join(pairs) + " |")
    lines.extend(
        [
            "",
            "### 4. Task-specific predictive utility",
            "",
            "Held-out balanced accuracy asks how much information useful for predicting "
            "the report's observed taxonomy survives in hard or soft bits. It gives each "
            "class equal weight and is therefore the primary task-fidelity measure. Its "
            "raw-minus-code difference is **not** loss of all semantic information: it is "
            "loss for one label system, classifier family, and train/test split.",
            "",
            "Where probability predictions are available, cross-entropy adds a confidence-"
            "sensitive measure. `CE(code) - CE(raw)`, divided by `ln(2)`, is reported in "
            "bits per document: positive values mean the code needs more predictive "
            "surprisal than raw embeddings for that probe. Improvement over a predictor "
            "using only training-set class frequencies is also reported. These are "
            "**probe-dependent predictive-information estimates**, not exact mutual "
            "information between representation and label; poor calibration or a weak "
            "probe can change them.",
            "",
            "**How to read the predictive-utility statistics**",
            "",
            "| Statistic | What it measures | Better direction |",
            "|---|---|---|",
            "| Balanced accuracy | Mean recall across taxonomy classes on the common held-out split. | **Higher is better**: 1 is perfect; chance depends on the number of classes. |",
            "| Balanced-accuracy loss vs raw | `balanced accuracy(raw) - balanced accuracy(code)` for the same backbone and probe. | **Lower is better**: 0 means no loss, positive means the code is worse, and negative means the code outperforms raw embeddings. |",
            "| Cross-entropy (bits/document) | Average held-out surprisal assigned to the true class by the probability probe. | **Lower is better**: 0 is perfect confident prediction; unlike accuracy, it penalizes confident errors. |",
            "| CE loss vs raw (bits/document) | `CE(code) - CE(raw)` for the same backbone and probe. | **Lower is better**: 0 is equal, positive means information/calibration loss relative to raw, and negative means the code predicts better. |",
            "| Predictive gain vs class prior (bits/document) | `CE(training-prior predictor) - CE(representation)` converted to bits. | **Higher is better**: positive means the representation beats class-frequency-only prediction, 0 ties it, and negative is worse. |",
            "| Probe fraction of label entropy resolved | Predictive gain over the training prior divided by held-out label entropy. | **Higher is better for this probe**, but it is not model-free mutual information and can be negative or exceed intuitive bounds when the prior/probe is misspecified. |",
            "| Fano lower bound | `max(0, H(Y)-h2(error)-error log2(C-1))`, using ordinary held-out error. | **Higher is a stronger certified lower bound** on classical label information carried by the classifier output. It can be loose; balanced accuracy cannot be substituted into Fano's formula. |",
            "| Fano fraction of label entropy | Fano lower bound divided by `H(Y)`. | Higher is a stronger lower-bound fraction; zero means only that Fano is uninformative, not that the representation contains no label information. |",
            "| Paired CE 95% interval | Class-stratified document bootstrap interval for `CE(code)-CE(raw)`. | An interval wholly above 0 supports worse probabilistic prediction than raw; wholly below 0 supports better prediction; overlap with 0 is inconclusive at this interval level. |",
            "| Macro-F1 (artifact) | Unweighted mean of per-class F1, balancing precision and recall. | **Higher is better**; it is complementary to balanced accuracy and remains task/probe specific. |",
        ]
    )
    utility_rows = [
        row
        for row in manifest.get("utility_rows", [])
        if row.get("classifier") == "logistic_regression"
    ]
    if utility_rows:
        has_cross_entropy = any(
            row.get("test_cross_entropy_bits_per_document") is not None
            for row in utility_rows
        )
        header = "| Backbone | Representation | Balanced accuracy | Loss vs raw |"
        separator = "|---|---|---:|---:|"
        if has_cross_entropy:
            header += " Cross-entropy | CE loss vs raw [95% CI] | Gain vs prior | Probe fraction H(Y) | Fano lower bound | Fano fraction H(Y) |"
            separator += "---:|---:|---:|---:|---:|---:|"
        lines.extend(["", "**Common logistic-regression probe**", "", header, separator])
        for row in utility_rows:
            line = (
                f"| {row['run']} | {row['representation']} | "
                f"{row['balanced_accuracy']:.3f} | "
                f"{number(row.get('balanced_accuracy_loss_vs_raw'))} |"
            )
            if has_cross_entropy:
                values = [
                    row.get("test_cross_entropy_bits_per_document"),
                    None,
                    row.get("predictive_information_over_training_prior_bits_per_document"),
                    row.get("probe_fraction_label_entropy_resolved"),
                    row.get("fano_mutual_information_lower_bound_bits_per_document"),
                    row.get("fano_fraction_label_entropy_resolved"),
                ]
                contrast = number(row.get("cross_entropy_loss_vs_raw_bits_per_document"))
                if row.get("cross_entropy_loss_vs_raw_ci_low") is not None:
                    contrast += (
                        f" [{number(row.get('cross_entropy_loss_vs_raw_ci_low'))}, "
                        f"{number(row.get('cross_entropy_loss_vs_raw_ci_high'))}]"
                    )
                values[1] = contrast
                line += " " + " | ".join(
                    value if isinstance(value, str) else number(value) for value in values
                ) + " |"
            lines.append(line)
    if artifacts.get("cross_entropy_contrasts_plot"):
        lines.extend(
            [
                "",
                f"![Paired cross-entropy contrasts]({Path(artifacts['cross_entropy_contrasts_plot']['path']).name})",
                "",
                "**Reading the paired CE plot.** The zero line means equal held-out "
                "cross-entropy to raw embeddings. Points below zero favor the code; "
                "points above zero favor raw embeddings. Bars are paired, class-"
                "stratified 95% bootstrap intervals over the same test documents and "
                "therefore quantify sampling uncertainty in the contrast, not uncertainty "
                "from relearning the encoder, bits, split, or probe family.",
            ]
        )
    association_rows = manifest.get("profile_label_rows", [])
    if association_rows:
        lines.extend(
            [
                "",
                "### 5. Profile--label association corrected for chance",
                "",
                "The ordinary plug-in mutual information between complete hard profiles "
                "and the report taxonomy is optimistic when there are many rare profiles. "
                "We therefore compute the exact expected mutual information under the "
                "fixed contingency-table margins and subtract it, and also report adjusted "
                "mutual information (AMI). This correction addresses chance agreement due "
                "to the observed numbers and sizes of profiles and classes; it does not "
                "turn the in-sample taxonomy association into an out-of-sample estimate.",
                "",
                "| Statistic | What it measures | Better direction |",
                "|---|---|---|",
                "| Plug-in profile--label MI | Empirical reduction in label uncertainty from knowing the complete profile. | Higher means stronger in-sample association, but rare profiles can inflate it. |",
                "| Expected MI under fixed margins | Chance association expected from partitions with the same cluster/class sizes. | Lower means less finite-partition chance bias; it is a correction term, not a quality score. |",
                "| Chance-corrected MI | Plug-in MI minus expected MI, in bits/document. | Higher means more association above the fixed-margin chance expectation; values near or below 0 provide no positive evidence by this correction. |",
                "| AMI | Chance-adjusted MI normalized by the arithmetic mean of profile and label entropies. | Higher is better: 1 is identical partitions, expected random agreement is approximately 0, and negative values indicate below-chance agreement. |",
                "| Corrected fraction of H(Y) | Chance-corrected MI divided by label entropy. | Higher means a larger above-chance share of taxonomy entropy is associated with profiles; it remains descriptive and in-sample. |",
                "",
                "| Backbone | Profiles | H(Y) | Plug-in MI | Expected MI | Corrected MI | Corrected fraction H(Y) | AMI |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in association_rows:
            lines.append(
                f"| {row['run']} | {row['n_profiles']} | "
                f"{number(row.get('label_entropy_bits_per_document'))} | "
                f"{number(row.get('plugin_mutual_information_bits_per_document'))} | "
                f"{number(row.get('expected_mutual_information_under_fixed_margins_bits_per_document'))} | "
                f"{number(row.get('chance_corrected_mutual_information_bits_per_document'))} | "
                f"{number(row.get('chance_corrected_fraction_label_entropy'))} | "
                f"{number(row.get('adjusted_mutual_information'))} |"
            )
    quantizer_rows = manifest.get("quantizer_rows", [])
    if quantizer_rows:
        lines.extend(
            [
                "",
                "### 6. Nonlearned and lightly learned binary references",
                "",
                "A learned SAE should be interpreted against equal-width controls. "
                "`pca_sign` thresholds centered PCA scores at zero; `itq` rotates PCA "
                "scores before thresholding; `random_hyperplane` signs seeded random "
                "projections. These standard sign codes are not sparsity matched and "
                "typically activate about half their bits. `pca_top_s` and "
                "`random_top_s` are additional study-specific controls that activate "
                "exactly the configured `s` largest scores per document. They isolate "
                "some consequences of fixed activity, but are not standard ITQ/PCA-sign "
                "algorithms.",
                "The `learned_sae_hard` row is the fitted sparse binary bottleneck itself; "
                "it appears in the same table and plot so every control can be compared "
                "directly with the adopted representation.",
                "",
                "All PCA/ITQ transformations are fit without labels on the same complete "
                "corpus as the SAE; taxonomy utility uses the identical held-out 80/20 "
                "split and logistic probe. Thus reconstruction R² is descriptive and "
                "in-sample, whereas balanced accuracy is held out. A control beating the "
                "SAE on one metric does not automatically beat it on interpretability, "
                "sparsity, geometry, or another taxonomy.",
                "",
                "| Metric | Better direction |",
                "|---|---|",
                "| Optimal linear reconstruction R² | Higher preserves more raw-embedding variance through that fixed binary code. |",
                "| Held-out balanced accuracy | Higher retains more information usable for the named taxonomy under the common probe. |",
                "| Joint profile entropy | Descriptive rate trade-off: higher uses more diverse profiles; lower compresses more. |",
                "| Mean active bits | Compare with the SAE target; closeness is relevant only for sparsity-matched controls. |",
                "| AMI | Higher means stronger chance-adjusted profile--taxonomy association. |",
                "",
                "| Backbone | Reference | Mean active | H(profile) | Linear R² | Balanced accuracy | AMI |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in quantizer_rows:
            lines.append(
                f"| {row['run']} | {row['method']} | "
                f"{number(row.get('mean_active_bits'), 2)} | "
                f"{number(row.get('joint_profile_entropy_bits_per_document'))} | "
                f"{number(row.get('posthoc_linear_decoder_r2'))} | "
                f"{number(row.get('balanced_accuracy'))} | "
                f"{number(row.get('adjusted_mutual_information'))} |"
            )
        if artifacts.get("quantizer_baselines_plot"):
            lines.extend(
                [
                    "",
                    f"![Nonlearned binary references]({Path(artifacts['quantizer_baselines_plot']['path']).name})",
                    "",
                    "**Reading the reference plot.** Higher is better in both panels: "
                    "the left compares optimal linear reconstruction of the raw embedding, "
                    "and the right compares held-out balanced taxonomy accuracy. Compare "
                    "methods within a backbone; raw embedding scales and task difficulty "
                    "differ across backbones and reports.",
                ]
            )
    artifact_links = [
        f"[summary]({Path(artifacts['summary']['path']).name})",
        f"[per-bit entropy table]({Path(artifacts['per_bit']['path']).name})",
        f"[geometry table]({Path(artifacts['geometry']['path']).name})",
        f"[task-utility table]({Path(artifacts['utility']['path']).name})",
    ]
    for key, label in (
        ("profile_label", "profile--label association"),
        ("linear_references", "linear references"),
        ("quantizer_baselines", "binary controls"),
        ("cross_entropy_contrasts", "paired CE contrasts"),
    ):
        if artifacts.get(key):
            artifact_links.append(
                f"[{label}]({Path(artifacts[key]['path']).name})"
            )
    artifact_links.append(
        f"[protocol metadata]({Path(artifacts['metadata']['path']).name})"
    )
    lines.extend(
        [
            "",
            "The complete numeric artifacts are " + ", ".join(artifact_links) + ".",
        ]
    )
    return lines
