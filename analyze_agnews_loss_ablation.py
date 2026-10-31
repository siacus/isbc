#!/usr/bin/env python3
"""Summarize the seeded AG News Qwen K=16/target-6 loss ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS: dict[str, dict[str, Any]] = {
    "full": {
        "label": "Full objective",
        "omitted": "none",
        "weights": (1.0, 1.0, 1.0, 0.05, 0.1),
    },
    "no_activity": {
        "label": "No activity control",
        "omitted": "adaptive per-document sparsity",
        "weights": (1.0, 0.0, 1.0, 0.05, 0.1),
    },
    "no_balance": {
        "label": "No balance",
        "omitted": "per-bit balance",
        "weights": (1.0, 1.0, 0.0, 0.05, 0.1),
    },
    "no_decorrelation": {
        "label": "No decorrelation",
        "omitted": "off-diagonal covariance penalty",
        "weights": (1.0, 1.0, 1.0, 0.0, 0.1),
    },
    "no_entropy": {
        "label": "No entropy sharpening",
        "omitted": "conditional Bernoulli entropy",
        "weights": (1.0, 1.0, 1.0, 0.05, 0.0),
    },
    "no_reconstruction": {
        "label": "No reconstruction",
        "omitted": "normalized reconstruction",
        "weights": (0.0, 1.0, 1.0, 0.05, 0.1),
    },
}

COLORS = {
    "full": "#111111",
    "no_activity": "#D62728",
    "no_balance": "#FF7F0E",
    "no_decorrelation": "#9467BD",
    "no_entropy": "#2CA02C",
    "no_reconstruction": "#1F77B4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2025, 31415])
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "runs/agnews_embedding_comparison/qwen_loss_ablation_k16_target6"
        ),
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("data/agnews/agnews_all.csv")
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path(
            "runs/agnews_embedding_comparison/all_fixed_balance_label_name/"
            "predictive_split.csv"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_dir(root: Path, variant: str, seed: int) -> Path:
    return root / f"agnews_qwen_loss_ablation_{variant}_seed{seed}"


def binary_entropy(values: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(values, dtype=float), 1e-12, 1.0 - 1e-12)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def reconstruction_metrics(
    embeddings: np.ndarray,
    code: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    variance: float,
) -> tuple[float, float]:
    squared_error = 0.0
    total = 0
    for start in range(0, len(code), 512):
        stop = min(start + 512, len(code))
        target = np.asarray(embeddings[start:stop], dtype=np.float32)
        reconstructed = code[start:stop].astype(np.float32) @ weight.T + bias
        difference = target - reconstructed
        squared_error += float(np.square(difference, dtype=np.float64).sum())
        total += difference.size
    mse = squared_error / total
    return mse, mse / variance


def optimal_linear_mse(
    embeddings: np.ndarray, code: np.ndarray, variance: float
) -> float:
    design = np.column_stack(
        [np.ones(len(code), dtype=np.float64), code.astype(np.float64)]
    )
    target = np.asarray(embeddings, dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return float(np.square(residual).mean() / variance)


def r2_from_mse(mse: float, embeddings: np.ndarray) -> float:
    mean = np.asarray(embeddings, dtype=np.float64).mean(axis=0)
    baseline_mse = float(
        np.square(np.asarray(embeddings, dtype=np.float64) - mean).mean()
    )
    return 1.0 - mse / baseline_mse


def profile_strings(bits: np.ndarray) -> np.ndarray:
    return np.asarray(
        ["".join(str(int(value)) for value in row) for row in bits], dtype=str
    )


def predictive_metrics(
    features: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> dict[str, float]:
    model = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=seed
    )
    model.fit(features[train], y[train])
    train_prediction = model.predict(features[train])
    prediction = model.predict(features[test])
    return {
        "train_accuracy": float(accuracy_score(y[train], train_prediction)),
        "accuracy": float(accuracy_score(y[test], prediction)),
        "train_balanced_accuracy": float(
            balanced_accuracy_score(y[train], train_prediction)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], prediction)),
        "macro_f1": float(f1_score(y[test], prediction, average="macro")),
    }


def pattern_majority_metrics(
    patterns: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> dict[str, float]:
    training = pd.DataFrame({"pattern": patterns[train], "label": y[train]})
    global_majority = str(training["label"].value_counts().index[0])
    lookup: dict[str, str] = {}
    for pattern, group in training.groupby("pattern", sort=False):
        counts = group["label"].value_counts()
        lookup[str(pattern)] = str(sorted(counts[counts == counts.max()].index)[0])
    prediction = np.asarray(
        [lookup.get(str(pattern), global_majority) for pattern in patterns[test]]
    )
    return {
        "profile_majority_accuracy": float(accuracy_score(y[test], prediction)),
        "profile_majority_balanced_accuracy": float(
            balanced_accuracy_score(y[test], prediction)
        ),
        "profile_majority_macro_f1": float(
            f1_score(y[test], prediction, average="macro")
        ),
        "profile_majority_fallback_rate": float(
            np.mean([str(pattern) not in lookup for pattern in patterns[test]])
        ),
    }


def evaluate_run(
    directory: Path,
    variant: str,
    seed: int,
    labels: pd.DataFrame,
    split: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    encoded_path = directory / "encoded_agnews_all_ready_best.csv"
    cache_path = directory / "training_embeddings_agnews_all_ready.npy"
    checkpoint_path = directory / "best/model.pt"
    calibration_path = directory / "reconstruction_calibration.json"
    metadata_path = directory / "metadata.json"
    diagnostic_path = directory / "training_diagnostic_metrics.csv"
    required = (
        encoded_path,
        cache_path,
        checkpoint_path,
        calibration_path,
        metadata_path,
        diagnostic_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete {variant}/seed={seed}: {missing}")

    encoded = pd.read_csv(encoded_path, dtype={"id": str}, float_precision="round_trip")
    bit_columns = sorted(column for column in encoded if column.startswith("bit_"))
    probability_columns = sorted(
        column for column in encoded if column.startswith("prob_")
    )
    if len(bit_columns) != 16 or len(probability_columns) != 16:
        raise ValueError(f"{directory}: expected exactly 16 bit/probability columns")
    bits = encoded[bit_columns].to_numpy(dtype=np.uint8)
    probabilities = encoded[probability_columns].to_numpy(dtype=np.float64)
    embeddings = np.load(cache_path, mmap_mode="r")
    if embeddings.shape != (len(encoded), 1024):
        raise ValueError(f"{directory}: unexpected embedding shape {embeddings.shape}")

    aligned = encoded[["id"]].merge(labels[["id", "label_name"]], on="id", how="left")
    aligned = aligned.merge(split, on="id", how="left")
    if aligned.isna().any().any() or len(aligned) != len(encoded):
        raise ValueError(f"{directory}: label/split alignment failed")
    y = aligned["label_name"].astype(str).to_numpy()
    train = np.flatnonzero(aligned["predictive_split"].eq("train").to_numpy())
    test = np.flatnonzero(aligned["predictive_split"].eq("test").to_numpy())

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weight = state["decoder.linear.weight"].detach().cpu().numpy().astype(np.float32)
    bias = state["decoder.linear.bias"].detach().cpu().numpy().astype(np.float32)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    variance = float(calibration["coordinate_variance_mean"])
    hard_mse, hard_mse_normalized = reconstruction_metrics(
        embeddings, bits, weight, bias, variance
    )
    soft_mse, soft_mse_normalized = reconstruction_metrics(
        embeddings, probabilities, weight, bias, variance
    )

    patterns = profile_strings(bits)
    _, counts = np.unique(patterns, return_counts=True)
    shares = counts / counts.sum()
    profile_entropy = float(-(shares * np.log2(shares)).sum())
    prevalence = bits.mean(axis=0)
    bit_entropies = binary_entropy(prevalence)
    active = bits.sum(axis=1)
    target_active = 6.0
    rho = target_active / bits.shape[1]
    centered = probabilities - probabilities.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(probabilities) - 1)
    off_diagonal = covariance[~np.eye(covariance.shape[0], dtype=bool)]
    nonconstant = np.flatnonzero(bits.std(axis=0) > 0)
    if len(nonconstant) >= 2:
        correlation = np.corrcoef(bits[:, nonconstant], rowvar=False)
        correlations = correlation[~np.eye(len(nonconstant), dtype=bool)]
        mean_abs_correlation = float(np.mean(np.abs(correlations)))
        max_abs_correlation = float(np.max(np.abs(correlations)))
    else:
        mean_abs_correlation = float("nan")
        max_abs_correlation = float("nan")

    hard_prediction = predictive_metrics(bits, y, train, test, seed)
    soft_prediction = predictive_metrics(probabilities, y, train, test, seed)
    majority = pattern_majority_metrics(patterns, y, train, test)
    test_patterns = patterns[test]
    test_labels = y[test]

    row: dict[str, Any] = {
        "variant": variant,
        "variant_label": VARIANTS[variant]["label"],
        "omitted_term": VARIANTS[variant]["omitted"],
        "seed": seed,
        "documents": len(bits),
        "hard_accuracy": hard_prediction["accuracy"],
        "hard_balanced_accuracy": hard_prediction["balanced_accuracy"],
        "hard_macro_f1": hard_prediction["macro_f1"],
        "hard_train_balanced_accuracy": hard_prediction["train_balanced_accuracy"],
        "soft_accuracy": soft_prediction["accuracy"],
        "soft_balanced_accuracy": soft_prediction["balanced_accuracy"],
        "soft_macro_f1": soft_prediction["macro_f1"],
        "soft_train_balanced_accuracy": soft_prediction["train_balanced_accuracy"],
        **majority,
        "hard_decoder_mse": hard_mse,
        "hard_decoder_mse_normalized": hard_mse_normalized,
        "hard_decoder_r2": r2_from_mse(hard_mse, embeddings),
        "soft_decoder_mse": soft_mse,
        "soft_decoder_mse_normalized": soft_mse_normalized,
        "soft_decoder_r2": r2_from_mse(soft_mse, embeddings),
        "hard_optimal_linear_mse_normalized": optimal_linear_mse(
            embeddings, bits, variance
        ),
        "soft_optimal_linear_mse_normalized": optimal_linear_mse(
            embeddings, probabilities, variance
        ),
        "mean_active_bits": float(active.mean()),
        "sd_active_bits": float(active.std()),
        "mean_absolute_active_target_error": float(
            np.mean(np.abs(active - target_active))
        ),
        "exact_target_activity_rate": float(np.mean(active == target_active)),
        "observed_profiles": int(len(counts)),
        "profile_entropy_bits": profile_entropy,
        "effective_profiles": float(2.0**profile_entropy),
        "mean_bit_entropy_bits": float(bit_entropies.mean()),
        "sum_bit_entropy_bits": float(bit_entropies.sum()),
        "total_correlation_bits": float(bit_entropies.sum() - profile_entropy),
        "bit_prevalence_mean": float(prevalence.mean()),
        "bit_prevalence_sd": float(prevalence.std()),
        "bit_prevalence_min": float(prevalence.min()),
        "bit_prevalence_max": float(prevalence.max()),
        "dead_bits_below_1pct": int(np.sum(prevalence < 0.01)),
        "saturated_bits_above_99pct": int(np.sum(prevalence > 0.99)),
        "mean_abs_hard_bit_correlation": mean_abs_correlation,
        "max_abs_hard_bit_correlation": max_abs_correlation,
        "soft_activity_loss": float(
            np.square(probabilities.mean(axis=1) - rho).mean()
        ),
        "soft_balance_loss": float(
            np.square(probabilities.mean(axis=0) - rho).mean()
        ),
        "soft_decorrelation_loss": float(np.square(off_diagonal).sum() / 16**2),
        "soft_conditional_entropy_nats": float(
            -(
                np.clip(probabilities, 1e-6, 1 - 1e-6)
                * np.log(np.clip(probabilities, 1e-6, 1 - 1e-6))
                + (1 - np.clip(probabilities, 1e-6, 1 - 1e-6))
                * np.log(1 - np.clip(probabilities, 1e-6, 1 - 1e-6))
            ).mean()
        ),
        "mean_distance_to_binary": float(
            np.minimum(probabilities, 1.0 - probabilities).mean()
        ),
        "profile_label_nmi_test": float(
            normalized_mutual_info_score(test_labels, test_patterns)
        ),
        "profile_label_ami_test": float(
            adjusted_mutual_info_score(test_labels, test_patterns)
        ),
        "encoded_sha256": sha256_file(encoded_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }

    bit_rows = []
    for index, column in enumerate(bit_columns):
        bit_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "bit": column,
                "prevalence": float(prevalence[index]),
                "entropy_bits": float(bit_entropies[index]),
                "mean_probability": float(probabilities[:, index].mean()),
            }
        )
    return row, bit_rows


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "documents"}
    ]
    records = []
    full = frame[frame["variant"] == "full"].set_index("seed")
    for variant in VARIANTS:
        subset = frame[frame["variant"] == variant]
        record: dict[str, Any] = {
            "variant": variant,
            "variant_label": VARIANTS[variant]["label"],
            "omitted_term": VARIANTS[variant]["omitted"],
            "n_seeds": int(len(subset)),
        }
        for column in numeric:
            record[f"{column}_mean"] = float(subset[column].mean())
            record[f"{column}_sd"] = float(subset[column].std(ddof=1))
            paired = subset.set_index("seed")[column] - full[column]
            record[f"{column}_delta_vs_full_mean"] = float(paired.mean())
            record[f"{column}_delta_vs_full_sd"] = float(paired.std(ddof=1))
        records.append(record)
    return pd.DataFrame(records)


def mean_sd(row: pd.Series, metric: str, digits: int = 3) -> str:
    mean = float(row[f"{metric}_mean"])
    sd = float(row[f"{metric}_sd"])
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def plot_effects(summary: pd.DataFrame, output: Path) -> None:
    panels = [
        ("hard_balanced_accuracy", "Held-out hard-bit balanced accuracy", True),
        ("hard_decoder_mse_normalized", "Hard-code normalized reconstruction MSE", False),
        ("mean_active_bits", "Mean active bits (target = 6)", None),
        ("mean_absolute_active_target_error", "Mean |active bits - 6|", False),
        ("effective_profiles", "Effective number of profiles", None),
        ("bit_prevalence_sd", "SD of per-bit prevalence", False),
        ("mean_abs_hard_bit_correlation", "Mean absolute bit correlation", False),
        ("soft_conditional_entropy_nats", "Conditional Bernoulli entropy", False),
    ]
    positions = np.arange(len(summary))
    figure, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    for axis, (metric, title, direction) in zip(axes.flat, panels):
        means = summary[f"{metric}_mean"].to_numpy(dtype=float)
        errors = summary[f"{metric}_sd"].to_numpy(dtype=float)
        axis.errorbar(
            positions,
            means,
            yerr=errors,
            fmt="o",
            color="#222222",
            ecolor="#666666",
            capsize=4,
        )
        axis.scatter(
            positions,
            means,
            s=60,
            color=[COLORS[value] for value in summary["variant"]],
            zorder=3,
        )
        if metric == "mean_active_bits":
            axis.axhline(6, color="#D62728", linestyle="--", linewidth=1)
        axis.set_xticks(
            positions,
            [VARIANTS[value]["label"].replace(" ", "\n", 1) for value in summary["variant"]],
            fontsize=8,
        )
        suffix = "higher is better" if direction is True else "lower is better" if direction is False else "descriptive"
        axis.set_title(f"{title}\n({suffix})", fontsize=10)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("AG News Qwen 16/6: leave-one-loss-term-out ablation (mean ± SD across seeds)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def aggregate_training_paths(
    runs_root: Path, seeds: list[int], output_dir: Path
) -> pd.DataFrame:
    columns = [
        "hard_active_mean",
        "loss_recon_normalized",
        "loss_tweet_sparse",
        "loss_balance",
        "loss_decor",
        "loss_entropy",
    ]
    frames = []
    for variant in VARIANTS:
        for seed in seeds:
            path = run_dir(runs_root, variant, seed) / "training_diagnostic_metrics.csv"
            frame = pd.read_csv(path, usecols=["global_step", *columns])
            frame["variant"] = variant
            frame["seed"] = seed
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    grouped = raw.groupby(["variant", "global_step"], sort=False)[columns].agg(
        ["mean", "std"]
    )
    grouped.columns = [f"{name}_{stat}" for name, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(output_dir / "training_trajectories.csv", index=False)

    panels = [
        ("hard_active_mean", "Hard active bits", 6.0),
        ("loss_recon_normalized", "Normalized reconstruction term", None),
        ("loss_tweet_sparse", "Raw activity-control term", None),
        ("loss_balance", "Raw balance term", None),
        ("loss_decor", "Raw decorrelation term", None),
        ("loss_entropy", "Raw entropy term", None),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for axis, (metric, title, target) in zip(axes.flat, panels):
        for variant in VARIANTS:
            subset = grouped[grouped["variant"] == variant]
            x = subset["global_step"].to_numpy(dtype=float)
            mean = subset[f"{metric}_mean"].to_numpy(dtype=float)
            sd = subset[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
            axis.plot(x, mean, color=COLORS[variant], label=VARIANTS[variant]["label"])
            axis.fill_between(x, mean - sd, mean + sd, color=COLORS[variant], alpha=0.08)
        if target is not None:
            axis.axhline(target, color="#D62728", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    figure.suptitle("Training consequences of omitting each loss component")
    figure.savefig(output_dir / "training_trajectories.png", dpi=180)
    plt.close(figure)
    return grouped


def write_report(
    summary: pd.DataFrame, output_dir: Path, seeds: list[int], documents: int
) -> None:
    lookup = summary.set_index("variant")
    full = lookup.loc["full"]

    def delta(variant: str, metric: str) -> float:
        return float(lookup.loc[variant, f"{metric}_delta_vs_full_mean"])

    lines = [
        "# AG News Qwen 16/6 loss-component ablation",
        "",
        "## TL;DR",
        "",
        (
            f"This targeted experiment holds all {documents:,} AG News documents, cached raw Qwen embeddings, "
            f"16-bit width, target activity of six, predictive split, optimizer settings, and "
            f"three seeds ({', '.join(map(str, seeds))}) fixed. Each variant removes exactly one "
            "loss component. The experiment tests marginal contributions; it does not assume in "
            "advance that every term improves every outcome."
        ),
        "",
        (
            f"The full objective reaches hard-bit balanced accuracy "
            f"{full['hard_balanced_accuracy_mean']:.3f} ± {full['hard_balanced_accuracy_sd']:.3f}, "
            f"mean activity {full['mean_active_bits_mean']:.2f}, and "
            f"{full['effective_profiles_mean']:.1f} effective profiles. Removing activity control "
            f"changes mean activity by {delta('no_activity', 'mean_active_bits'):+.2f} bits and "
            f"changes the mean absolute target error by "
            f"{delta('no_activity', 'mean_absolute_active_target_error'):+.2f}. Removing balance "
            f"changes the SD of bit prevalence by {delta('no_balance', 'bit_prevalence_sd'):+.3f}; "
            f"removing decorrelation changes mean absolute bit correlation by "
            f"{delta('no_decorrelation', 'mean_abs_hard_bit_correlation'):+.3f}; removing entropy "
            f"changes conditional Bernoulli entropy by "
            f"{delta('no_entropy', 'soft_conditional_entropy_nats'):+.3f}; and removing reconstruction "
            f"changes normalized hard-code MSE by "
            f"{delta('no_reconstruction', 'hard_decoder_mse_normalized'):+.3f}."
        ),
        "",
        "## Synthesis of results",
        "",
        "Values are mean ± sample SD across the three paired seeds. `Δ BA` is the paired change "
        "in hard-bit balanced accuracy relative to the full objective; positive is better. Profile "
        "count and entropy are descriptive rather than monotone quality scores.",
        "",
        "| Variant | Hard accuracy | Hard BA | Hard macro-F1 | Δ BA | Soft BA | Hard MSE / variance | Mean active | Mean abs. activity error | Observed profiles | Effective profiles | Profile entropy | Bit prevalence SD | Mean abs. bit correlation | Soft entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['variant_label']} | {mean_sd(row, 'hard_accuracy')} | "
            f"{mean_sd(row, 'hard_balanced_accuracy')} | "
            f"{mean_sd(row, 'hard_macro_f1')} | "
            f"{row['hard_balanced_accuracy_delta_vs_full_mean']:+.3f} | "
            f"{mean_sd(row, 'soft_balanced_accuracy')} | "
            f"{mean_sd(row, 'hard_decoder_mse_normalized')} | "
            f"{mean_sd(row, 'mean_active_bits', 2)} | "
            f"{mean_sd(row, 'mean_absolute_active_target_error', 2)} | "
            f"{mean_sd(row, 'observed_profiles', 1)} | "
            f"{mean_sd(row, 'effective_profiles', 1)} | "
            f"{mean_sd(row, 'profile_entropy_bits')} | "
            f"{mean_sd(row, 'bit_prevalence_sd')} | "
            f"{mean_sd(row, 'mean_abs_hard_bit_correlation')} | "
            f"{mean_sd(row, 'soft_conditional_entropy_nats')} |"
        )
    lines.extend(
        [
            "",
            "![Ablation effects](ablation_effects.png)",
            "",
            "## What each diagnostic establishes",
            "",
            "- **Held-out balanced accuracy, accuracy, and macro-F1:** usable AG News category "
            "information in the hard or soft representation; higher is better. The main table uses "
            "a common class-balanced logistic probe on the immutable 80/20 split.",
            "- **Decoder MSE divided by raw-embedding variance and decoder R²:** distortion relative "
            "to the original Qwen vectors; lower normalized MSE and higher R² are better. Learned and "
            "post-hoc optimal linear decoders are both stored.",
            "- **Mean active bits and |active-6|:** whether the per-document activity constraint does "
            "the job it was designed to do; mean activity close to six and smaller absolute error are "
            "better for the specified 16/6 code.",
            "- **Per-bit prevalence SD, dead bits, and saturated bits:** whether capacity is distributed "
            "across bits instead of collapsing onto a few coordinates; lower dispersion and fewer "
            "dead/saturated bits are better.",
            "- **Mean absolute bit correlation and the raw decorrelation term:** redundancy among "
            "coordinates; lower values indicate more distinct bits.",
            "- **Conditional Bernoulli entropy and distance to {0,1}:** softness of the encoder output; "
            "lower values mean cleaner binarization.",
            "- **Observed/effective profiles and profile entropy:** diversity of used codewords. Higher "
            "values indicate a richer partition, but are not intrinsically better because excessive "
            "fragmentation can reduce stability or predictive generalization.",
            "- **Profile-label AMI/NMI and profile-majority prediction:** category structure retained by "
            "complete patterns. AMI corrects the chance agreement that can inflate raw NMI when many "
            "rare profiles occur.",
            "",
            "## Training dynamics",
            "",
            "Every curve is the mean across seeds with a ±1 SD band. Raw terms are plotted even when "
            "their coefficient is zero, which shows the consequence of removing the corresponding "
            "pressure rather than merely drawing its weighted contribution as zero.",
            "",
            "![Training trajectories](training_trajectories.png)",
            "",
            "## Design and interpretation",
            "",
            "This is a paired leave-one-term-out design. Each seed gives every variant the same head "
            "initialization and document-order RNG stream; only one coefficient changes. The frozen "
            "Qwen vectors are hard-linked from the definitive AG News cache and validated before "
            "training, so no backbone embedding is recomputed. The full model is rerun for all seeds "
            "instead of borrowing a single historical checkpoint, allowing paired deltas and seed "
            "variation to be reported fairly.",
            "",
            "A leave-one-out result identifies a term's marginal role conditional on the other four "
            "terms. It does not isolate higher-order interactions, establish that the coefficient is "
            "globally optimal, or generalize beyond AG News, Qwen, and the 16/6 operating point.",
            "",
            "## Numeric artifacts",
            "",
            "- `ablation_metrics_by_seed.csv`: every corpus, reconstruction, and predictive metric.",
            "- `ablation_summary.csv`: means, SDs, and paired deltas versus the full objective.",
            "- `ablation_per_bit.csv`: per-bit prevalence, entropy, and mean probability.",
            "- `training_trajectories.csv`: seed-aggregated optimizer-step diagnostics.",
            "- `metadata.json`: cache hashes, seeds, coefficients, split, code revision, and device.",
            "- [Paper-writing handoff](paper_handoff.md): concise design, results, and claim limits.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_handoff(
    summary: pd.DataFrame, output_dir: Path, seeds: list[int], documents: int
) -> None:
    lookup = summary.set_index("variant")
    full = lookup.loc["full"]

    def value(variant: str, metric: str) -> float:
        return float(lookup.loc[variant, f"{metric}_mean"])

    lines = [
        "# Paper handoff: evidence for the regularization choices",
        "",
        "## Reviewer-facing question",
        "",
        "> Evidence for the regularization choices. Targeted ablations should establish what "
        "activity control, balance, and the remaining penalties contribute. A different objective "
        "is more persuasive when its consequences are demonstrated.",
        "",
        "## Experimental design",
        "",
        f"AG News ({documents:,} documents), Qwen, K=16, target activity=6, seeds={seeds}. The frozen unnormalised Qwen "
        "embeddings and the downstream 80/20 split are identical across variants. The full "
        "objective is rerun at every seed. Five paired variants set exactly one coefficient to zero: "
        "activity control, balance, decorrelation, entropy sharpening, or reconstruction.",
        "",
        "## Results to report",
        "",
        f"The full objective yields hard-bit balanced accuracy "
        f"{full['hard_balanced_accuracy_mean']:.3f} ± {full['hard_balanced_accuracy_sd']:.3f}, "
        f"mean activity {full['mean_active_bits_mean']:.2f} ± {full['mean_active_bits_sd']:.2f}, "
        f"and {full['effective_profiles_mean']:.1f} ± {full['effective_profiles_sd']:.1f} "
        "effective profiles.",
        "",
        "| Removed term | Intended property | Full | Ablated | Paired change |",
        "|---|---|---:|---:|---:|",
        f"| Activity control | Mean absolute distance from six active bits | {value('full','mean_absolute_active_target_error'):.3f} | {value('no_activity','mean_absolute_active_target_error'):.3f} | {lookup.loc['no_activity','mean_absolute_active_target_error_delta_vs_full_mean']:+.3f} |",
        f"| Balance | SD of bit prevalence | {value('full','bit_prevalence_sd'):.3f} | {value('no_balance','bit_prevalence_sd'):.3f} | {lookup.loc['no_balance','bit_prevalence_sd_delta_vs_full_mean']:+.3f} |",
        f"| Decorrelation | Mean absolute hard-bit correlation | {value('full','mean_abs_hard_bit_correlation'):.3f} | {value('no_decorrelation','mean_abs_hard_bit_correlation'):.3f} | {lookup.loc['no_decorrelation','mean_abs_hard_bit_correlation_delta_vs_full_mean']:+.3f} |",
        f"| Entropy sharpening | Conditional Bernoulli entropy | {value('full','soft_conditional_entropy_nats'):.3f} | {value('no_entropy','soft_conditional_entropy_nats'):.3f} | {lookup.loc['no_entropy','soft_conditional_entropy_nats_delta_vs_full_mean']:+.3f} |",
        f"| Reconstruction | Normalized hard decoder MSE | {value('full','hard_decoder_mse_normalized'):.3f} | {value('no_reconstruction','hard_decoder_mse_normalized'):.3f} | {lookup.loc['no_reconstruction','hard_decoder_mse_normalized_delta_vs_full_mean']:+.3f} |",
        "",
        "## Suggested prose",
        "",
        (
            "We evaluated the role of each objective component in a paired leave-one-term-out "
            "ablation on AG News. All variants used the same frozen Qwen embeddings, 16-bit width, "
            "six-bit activity target, predictive split, optimizer settings, and three random seeds. "
            "This design links each regularizer to its intended observable consequence while also "
            "measuring reconstruction, code capacity, and held-out category utility. [Insert the "
            "term-specific numerical changes from the table above.] The results should be described "
            "as marginal contributions conditional on the remaining terms, rather than as proof that "
            "the selected coefficients are globally optimal."
        ),
        "",
        "## Claim limits",
        "",
        "- Do not say that every term improves every metric; state its intended effect and trade-offs.",
        "- Do not treat more profiles or higher profile entropy as automatically better.",
        "- Do not generalize beyond AG News/Qwen/K=16/target-6 without another experiment.",
        "- Do not call the coefficients optimal; this is component necessity, not hyperparameter search.",
        "",
        "Full evidence: [report.md](report.md), `ablation_metrics_by_seed.csv`, and "
        "`ablation_summary.csv`.",
    ]
    (output_dir / "paper_handoff.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels, dtype={"id": str})
    split = pd.read_csv(args.split, dtype={"id": str})
    if set(split["predictive_split"]) != {"train", "test"}:
        raise ValueError("Predictive split must contain train and test")

    rows: list[dict[str, Any]] = []
    per_bit: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in args.seeds:
            print(f"evaluating {variant}, seed={seed}", flush=True)
            row, bit_rows = evaluate_run(
                run_dir(args.runs_root, variant, seed),
                variant,
                seed,
                labels,
                split,
            )
            rows.append(row)
            per_bit.extend(bit_rows)
    metrics = pd.DataFrame(rows)
    documents = int(metrics["documents"].iloc[0])
    if not metrics["documents"].eq(documents).all():
        raise ValueError("All ablation variants must encode the same number of documents")
    metrics.to_csv(output_dir / "ablation_metrics_by_seed.csv", index=False)
    pd.DataFrame(per_bit).to_csv(output_dir / "ablation_per_bit.csv", index=False)
    summary = aggregate_metrics(metrics)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    plot_effects(summary, output_dir / "ablation_effects.png")
    aggregate_training_paths(args.runs_root, args.seeds, output_dir)
    write_report(summary, output_dir, args.seeds, documents)
    write_handoff(summary, output_dir, args.seeds, documents)

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    source_cache = (
        args.runs_root
        / "agnews_qwen_all_fixed_balance"
        / "training_embeddings_agnews_all_ready.npy"
    )
    metadata = {
        "schema_version": 1,
        "experiment": "paired leave-one-loss-term-out ablation",
        "dataset": "AG News",
        "backbone": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_normalization": "none",
        "k_factors": 16,
        "target_active": 6,
        "documents": documents,
        "seeds": args.seeds,
        "device": "mps",
        "epochs": 3,
        "predictive_split": str(args.split.resolve()),
        "predictive_split_sha256": sha256_file(args.split),
        "source_embedding_cache": str(source_cache.resolve()),
        "source_embedding_cache_sha256": sha256_file(source_cache),
        "cache_reuse": "hard-linked into every run; require_cached_embeddings enforced",
        "variants": {
            key: {
                "label": value["label"],
                "omitted": value["omitted"],
                "lambda_recon": value["weights"][0],
                "lambda_activity_initial": value["weights"][1],
                "lambda_balance": value["weights"][2],
                "lambda_decorrelation": value["weights"][3],
                "lambda_entropy": value["weights"][4],
            }
            for key, value in VARIANTS.items()
        },
        "git_revision": revision,
        "artifacts": {
            "report": "report.md",
            "paper_handoff": "paper_handoff.md",
            "metrics_by_seed": "ablation_metrics_by_seed.csv",
            "summary": "ablation_summary.csv",
            "per_bit": "ablation_per_bit.csv",
            "effect_plot": "ablation_effects.png",
            "training_paths": "training_trajectories.csv",
            "training_plot": "training_trajectories.png",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Ablation report complete: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
