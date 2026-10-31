#!/usr/bin/env python3
"""Permutation-invariant comparison of two binary-factor runs.

The bit numbers produced by independently trained runs are arbitrary.  This tool
therefore never compares bit_00 with bit_00 by position.  It aligns the same rows
by ``id`` and computes two optimal (Hungarian) matchings:

* semantic matching: similarity of label-conditional activation profiles
  P(bit=1 | label), allowing a factor to match the complement of another factor;
* activation matching: normalized mutual information between the two binary
  activation vectors on the same documents (already invariant to complementation).

It also compares code diversity, label information and held-out predictive
performance.  The output is a machine-readable JSON file and a Markdown report
whose first section is a short TL;DR.

Example:
    python compare_factor_runs.py \
        --run_a runs/agnews_qwen_all_fixed_balance \
        --run_b runs/agnews_embeddinggemma2_all_fixed_balance \
        --input_stem agnews_all_ready \
        --labels data/agnews/agnews_all.csv \
        --label_col label_name \
        --output_dir runs/agnews_embedding_comparison/qwen_vs_gemma
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MatplotlibPath
from matplotlib.patches import PathPatch, Rectangle
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split

from analysis_reporting import file_record, find_training_metadata, write_json
from training_diagnostic_plots import bundle_compact_training_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the information in two binary-factor runs without using bit numbers."
    )
    parser.add_argument("--run_a", required=True, help="First run directory")
    parser.add_argument("--run_b", required=True, help="Second run directory")
    parser.add_argument(
        "--input_stem",
        required=True,
        help="Stem used by encoded_<stem>_<checkpoint>.csv, e.g. agnews_all_ready",
    )
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--labels", required=True, help="CSV containing id and the selected label")
    parser.add_argument("--label_col", default="label_name")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--predictive_test_size",
        type=float,
        default=0.2,
        help="Fraction reserved for the seeded stratified predictive test split.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def bit_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("bit_")]
    if not columns:
        raise ValueError("No bit_XX columns found in encoded file")
    return sorted(columns)


def load_aligned(args: argparse.Namespace):
    path_a = Path(args.run_a) / f"encoded_{args.input_stem}_{args.checkpoint}.csv"
    path_b = Path(args.run_b) / f"encoded_{args.input_stem}_{args.checkpoint}.csv"
    labels_path = Path(args.labels)

    frames = []
    for name, path in (("run_a", path_a), ("run_b", path_b)):
        frame = pd.read_csv(path, dtype={"id": str})
        if frame["id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate ids: {path}")
        frames.append(frame)
    labels = pd.read_csv(labels_path, dtype={"id": str})
    if args.label_col not in labels.columns:
        raise ValueError(
            f"--label_col {args.label_col!r} not found; available: {list(labels.columns)}"
        )
    if labels["id"].duplicated().any():
        raise ValueError(f"Labels contain duplicate ids: {labels_path}")

    ids_a, ids_b = set(frames[0]["id"]), set(frames[1]["id"])
    if ids_a != ids_b:
        raise ValueError(
            "Runs do not contain exactly the same ids: "
            f"only run_a={len(ids_a - ids_b)}, only run_b={len(ids_b - ids_a)}"
        )
    missing_labels = ids_a - set(labels["id"])
    if missing_labels:
        raise ValueError(f"Labels are missing {len(missing_labels)} run ids")

    cols_a, cols_b = bit_columns(frames[0]), bit_columns(frames[1])
    a = frames[0].set_index("id").sort_index()
    b = frames[1].set_index("id").reindex(a.index)
    label_series = labels.set_index("id")[args.label_col].astype(str).str.strip().reindex(a.index)
    if label_series.isna().any():
        raise ValueError("Label alignment produced missing values")
    X_a = a[cols_a].to_numpy(dtype=np.int8)
    X_b = b[cols_b].to_numpy(dtype=np.int8)
    for name, matrix in (("run_a", X_a), ("run_b", X_b)):
        if not np.isin(matrix, (0, 1)).all():
            raise ValueError(f"{name} bit columns contain values other than 0 and 1")
    return (
        path_a,
        path_b,
        X_a,
        X_b,
        label_series.to_numpy(),
        cols_a,
        cols_b,
        a.index.to_numpy(dtype=str),
    )


def entropy_from_counts(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0] / counts.sum()
    return max(0.0, float(-(probabilities * np.log2(probabilities)).sum()))


def code_summary(X: np.ndarray, y: np.ndarray, bit_names: list[str]) -> dict[str, Any]:
    _, pattern_ids, pattern_counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    active = X.sum(axis=1)
    per_bit = []
    for index, name in enumerate(bit_names):
        prevalence = float(X[:, index].mean())
        per_bit.append(
            {
                "bit": name,
                "prevalence": prevalence,
                "label_mutual_information_nats": float(mutual_info_score(y, X[:, index])),
                "label_normalized_mutual_information": float(
                    normalized_mutual_info_score(y, X[:, index])
                ),
                "constant": bool(prevalence in (0.0, 1.0)),
            }
        )
    return {
        "n_rows": int(X.shape[0]),
        "n_bits": int(X.shape[1]),
        "n_unique_patterns": int(len(pattern_counts)),
        "pattern_entropy_bits": entropy_from_counts(pattern_counts),
        "pattern_label_mutual_information_nats": float(mutual_info_score(y, pattern_ids)),
        "pattern_label_normalized_mutual_information": float(
            normalized_mutual_info_score(y, pattern_ids)
        ),
        "active_bits": {
            "mean": float(active.mean()),
            "median": float(np.median(active)),
            "min": int(active.min()),
            "max": int(active.max()),
        },
        "dead_bits": [bit_names[i] for i in np.flatnonzero(X.mean(axis=0) == 0)],
        "universal_bits": [bit_names[i] for i in np.flatnonzero(X.mean(axis=0) == 1)],
        "per_bit": per_bit,
    }


def class_profiles(X: np.ndarray, y: np.ndarray, classes: list[str]) -> np.ndarray:
    return np.vstack([X[y == label].mean(axis=0) for label in classes]).T


def centered_profile_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    """Absolute Pearson similarity and whether the best orientation is complemented."""
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denominator == 0:
        return 0.0, False
    correlation = float(np.dot(a_centered, b_centered) / denominator)
    return abs(correlation), correlation < 0


def binary_activation_nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Return document-level NMI, treating constant factors as uninformative."""
    if np.unique(a).size <= 1 or np.unique(b).size <= 1:
        return 0.0
    return float(normalized_mutual_info_score(a, b))


def semantic_matching(
    profiles_a: np.ndarray,
    profiles_b: np.ndarray,
    names_a: list[str],
    names_b: list[str],
    label_nmi_a: np.ndarray,
    label_nmi_b: np.ndarray,
    X_a: np.ndarray,
    X_b: np.ndarray,
) -> dict[str, Any]:
    scores = np.zeros((len(names_a), len(names_b)), dtype=float)
    complemented = np.zeros_like(scores, dtype=bool)
    for i in range(len(names_a)):
        for j in range(len(names_b)):
            scores[i, j], complemented[i, j] = centered_profile_similarity(
                profiles_a[i], profiles_b[j]
            )
    rows, columns = linear_sum_assignment(-scores)
    pairs = []
    for i, j in zip(rows, columns):
        stronger = max(label_nmi_a[i], label_nmi_b[j])
        strength_ratio = min(label_nmi_a[i], label_nmi_b[j]) / stronger if stronger > 0 else 0.0
        pairs.append({
            "bit_a": names_a[i],
            "bit_b": names_b[j],
            "similarity": float(scores[i, j]),
            "complemented": bool(complemented[i, j]),
            "label_nmi_a": float(label_nmi_a[i]),
            "label_nmi_b": float(label_nmi_b[j]),
            "information_strength_ratio": float(strength_ratio),
            "strength_aware_similarity": float(scores[i, j] * strength_ratio),
            "activation_nmi": binary_activation_nmi(X_a[:, i], X_b[:, j]),
        })
    return {
        "method": "Hungarian matching on absolute Pearson correlation of class profiles",
        "mean_similarity": float(np.mean([pair["similarity"] for pair in pairs])) if pairs else 0.0,
        "mean_information_strength_ratio": float(
            np.mean([pair["information_strength_ratio"] for pair in pairs])
        ) if pairs else 0.0,
        "mean_strength_aware_similarity": float(
            np.mean([pair["strength_aware_similarity"] for pair in pairs])
        ) if pairs else 0.0,
        "mean_activation_nmi_on_semantic_assignment": float(
            np.mean([pair["activation_nmi"] for pair in pairs])
        ) if pairs else 0.0,
        "pairs": sorted(pairs, key=lambda pair: pair["similarity"], reverse=True),
    }


def activation_matching(
    X_a: np.ndarray, X_b: np.ndarray, names_a: list[str], names_b: list[str]
) -> dict[str, Any]:
    scores = np.zeros((len(names_a), len(names_b)), dtype=float)
    for i in range(len(names_a)):
        for j in range(len(names_b)):
            scores[i, j] = binary_activation_nmi(X_a[:, i], X_b[:, j])
    rows, columns = linear_sum_assignment(-scores)
    pairs = [
        {"bit_a": names_a[i], "bit_b": names_b[j], "normalized_mutual_information": float(scores[i, j])}
        for i, j in zip(rows, columns)
    ]
    return {
        "method": "Activation-optimal Hungarian matching on per-document normalized mutual information",
        "mean_normalized_mutual_information": float(
            np.mean([pair["normalized_mutual_information"] for pair in pairs])
        ) if pairs else 0.0,
        "pairs": sorted(
            pairs, key=lambda pair: pair["normalized_mutual_information"], reverse=True
        ),
    }


def plot_activation_sankey(
    X_a: np.ndarray,
    X_b: np.ndarray,
    names_a: list[str],
    names_b: list[str],
    semantic: dict[str, Any],
    label_a: str,
    label_b: str,
    output_path: Path,
) -> dict[str, Any]:
    """Plot equal-document-mass active/active flows for one semantic assignment."""
    active_a = X_a.sum(axis=1)
    active_b = X_b.sum(axis=1)
    valid = (active_a > 0) & (active_b > 0)
    weights = np.zeros(X_a.shape[0], dtype=float)
    weights[valid] = 1.0 / (active_a[valid] * active_b[valid])
    flows = (X_a.astype(float) * weights[:, None]).T @ X_b.astype(float)

    source_index = {name: index for index, name in enumerate(names_a)}
    target_index = {name: index for index, name in enumerate(names_b)}
    matched_target = {pair["bit_a"]: pair["bit_b"] for pair in semantic["pairs"]}
    complemented = {pair["bit_b"]: pair["complemented"] for pair in semantic["pairs"]}
    ordered_a = list(names_a)
    ordered_b = [matched_target[name] for name in ordered_a]
    source_order = [source_index[name] for name in ordered_a]
    target_order = [target_index[name] for name in ordered_b]
    ordered_flows = flows[np.ix_(source_order, target_order)]

    total_flow = float(ordered_flows.sum())
    if total_flow <= 0:
        raise ValueError("Cannot draw Sankey: no document has active bits in both runs")

    margin = 0.02
    gap = 0.012
    usable = 1.0 - (2 * margin) - gap * (len(ordered_a) - 1)
    row_heights = ordered_flows.sum(axis=1) / total_flow * usable
    column_heights = ordered_flows.sum(axis=0) / total_flow * usable

    def node_bottoms(heights: np.ndarray) -> np.ndarray:
        result = np.zeros_like(heights)
        top = 1.0 - margin
        for index, height in enumerate(heights):
            result[index] = top - height
            top = result[index] - gap
        return result

    source_bottom = node_bottoms(row_heights)
    target_bottom = node_bottoms(column_heights)
    source_offsets = source_bottom.copy()
    target_offsets = target_bottom.copy()
    links: list[tuple[bool, int, int, float, float, float]] = []
    for i in range(len(ordered_a)):
        for j in range(len(ordered_b)):
            height = ordered_flows[i, j] / total_flow * usable
            if height <= 0:
                continue
            links.append(
                (i == j, i, j, source_offsets[i], target_offsets[j], height)
            )
            source_offsets[i] += height
            target_offsets[j] += height

    figure, axis = plt.subplots(figsize=(15, 11), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    source_x0, source_x1 = 0.08, 0.105
    target_x0, target_x1 = 0.895, 0.92
    cmap = plt.get_cmap("turbo")
    colors = [cmap(index / max(1, len(ordered_a) - 1)) for index in range(len(ordered_a))]

    def draw_link(item: tuple[bool, int, int, float, float, float]) -> None:
        is_semantic, i, _j, source_y, target_y, height = item
        x_curve_left, x_curve_right = 0.38, 0.62
        vertices = [
            (source_x1, source_y),
            (x_curve_left, source_y),
            (x_curve_right, target_y),
            (target_x0, target_y),
            (target_x0, target_y + height),
            (x_curve_right, target_y + height),
            (x_curve_left, source_y + height),
            (source_x1, source_y + height),
            (source_x1, source_y),
        ]
        codes = [
            MatplotlibPath.MOVETO,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.LINETO,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CURVE4,
            MatplotlibPath.CLOSEPOLY,
        ]
        color = colors[i] if is_semantic else (0.45, 0.48, 0.52, 1.0)
        alpha = 0.72 if is_semantic else 0.075
        axis.add_patch(
            PathPatch(
                MatplotlibPath(vertices, codes),
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=2 if is_semantic else 1,
            )
        )

    for link in sorted(links, key=lambda item: item[0]):
        draw_link(link)

    for index, name in enumerate(ordered_a):
        color = colors[index]
        axis.add_patch(
            Rectangle(
                (source_x0, source_bottom[index]),
                source_x1 - source_x0,
                row_heights[index],
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        )
        axis.text(
            source_x0 - 0.008,
            source_bottom[index] + row_heights[index] / 2,
            name,
            ha="right",
            va="center",
            fontsize=8,
        )

    for index, name in enumerate(ordered_b):
        color = colors[index]
        axis.add_patch(
            Rectangle(
                (target_x0, target_bottom[index]),
                target_x1 - target_x0,
                column_heights[index],
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        )
        suffix = " †" if complemented.get(name, False) else ""
        axis.text(
            target_x1 + 0.008,
            target_bottom[index] + column_heights[index] / 2,
            f"{name}{suffix}",
            ha="left",
            va="center",
            fontsize=8,
        )

    axis.text((source_x0 + source_x1) / 2, 1.005, label_a, ha="center", va="bottom", fontsize=12)
    axis.text((target_x0 + target_x1) / 2, 1.005, label_b, ha="center", va="bottom", fontsize=12)
    axis.set_title(
        "Document-weighted bit activation flows\n"
        "colored ribbons = semantic Hungarian pairs; gray ribbons = other active/active flows",
        fontsize=14,
        pad=22,
    )
    figure.text(
        0.5,
        0.002,
        "Each document contributes one total flow unit, divided equally across all active "
        "A×B bit pairs. † = complemented semantic match; ribbons still show raw 1/1 activations.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    if output_path.suffix.lower() == ".svg":
        svg = output_path.read_text(encoding="utf-8")
        output_path.write_text(
            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
            encoding="utf-8",
        )
    return {
        "filename": output_path.name,
        "path": str(output_path.resolve()),
        "n_rows": int(X_a.shape[0]),
        "n_rows_with_flow": int(valid.sum()),
        "total_flow_units": total_flow,
        "flow_definition": (
            "Each document contributes one unit split equally over all active A-by-B bit pairs"
        ),
    }


def stratified_holdout_indices(
    y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create one reproducible label-stratified split shared by all compared runs."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("--predictive_test_size must be strictly between 0 and 1")
    counts = Counter(y)
    if min(counts.values()) < 2:
        raise ValueError(
            "Every retained class needs at least two rows for a stratified holdout"
        )
    indices = np.arange(len(y))
    train, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    return np.sort(train), np.sort(test)


def predictive_holdout(
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Fit one classifier on 80% of hard codes and evaluate it once on 20%."""
    model = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed
    )
    model.fit(X[train_indices], y[train_indices])
    prediction = model.predict(X[test_indices])
    return {
        "classifier": "class-balanced logistic regression",
        "protocol": "single seeded stratified random holdout",
        "seed": int(seed),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "accuracy": float(accuracy_score(y[test_indices], prediction)),
        "macro_f1": float(
            f1_score(y[test_indices], prediction, average="macro", zero_division=0)
        ),
    }


def prediction_accuracy(result: dict[str, Any]) -> float:
    """Read current holdout scores while retaining legacy-report compatibility."""
    return float(result.get("accuracy", result.get("accuracy_mean")))


def prediction_macro_f1(result: dict[str, Any]) -> float:
    return float(result.get("macro_f1", result.get("macro_f1_mean")))


def run_metadata(encoded_path: Path) -> dict[str, Any]:
    metadata, source = find_training_metadata(encoded_path)
    return {"source": source, **metadata}


def verdict(summary_a: dict[str, Any], summary_b: dict[str, Any], pred_a, pred_b) -> dict[str, str]:
    degenerate_a = summary_a["n_unique_patterns"] == 1
    degenerate_b = summary_b["n_unique_patterns"] == 1
    if degenerate_a and degenerate_b:
        return {
            "status": "degenerate_no_information",
            "reason": (
                "Both runs have a single hard pattern. Their codes contain no varying "
                "factor information to match, regardless of bit permutation."
            ),
        }
    if degenerate_a != degenerate_b:
        collapsed = "run_a" if degenerate_a else "run_b"
        return {
            "status": "not_equivalent",
            "reason": (
                f"{collapsed} has a single hard pattern, while the other run does not; "
                "no permutation or bit complementation can restore missing label information."
            ),
        }
    accuracy_gap = abs(prediction_accuracy(pred_a) - prediction_accuracy(pred_b))
    information_gap = abs(
        summary_a["pattern_label_normalized_mutual_information"]
        - summary_b["pattern_label_normalized_mutual_information"]
    )
    if accuracy_gap > 0.05 or information_gap > 0.05:
        return {
            "status": "not_equivalent",
            "reason": (
                f"Predictive accuracy differs by {accuracy_gap:.3f} and pattern-label NMI "
                f"differs by {information_gap:.3f}."
            ),
        }
    return {
        "status": "compatible_information_content",
        "reason": (
            "Predictive accuracy and pattern-label NMI are within 0.05. Inspect the matching "
            "scores before claiming factor-level equivalence."
        ),
    }


def markdown_report(result: dict[str, Any]) -> str:
    a, b = result["runs"]["run_a"], result["runs"]["run_b"]
    name_a = a["training"].get("model_name", "run_a")
    name_b = b["training"].get("model_name", "run_b")
    sa, sb = a["code_summary"], b["code_summary"]
    pa = a.get("predictive_holdout", a.get("predictive_cv"))
    pb = b.get("predictive_holdout", b.get("predictive_cv"))
    if pa is None or pb is None:
        raise ValueError("Report is missing predictive evaluation results")
    holdout_protocol = "predictive_holdout" in a and "predictive_holdout" in b
    accuracy_a, accuracy_b = prediction_accuracy(pa), prediction_accuracy(pb)
    macro_f1_a, macro_f1_b = prediction_macro_f1(pa), prediction_macro_f1(pb)
    semantic = result["matching"]["semantic"]
    activation = result["matching"]["activation"]
    verdict_data = result["verdict"]

    lines = [
        f"# Permutation-invariant factor comparison: {name_a} vs {name_b}",
        "",
        "## TL;DR",
        "",
        f"**Verdict: `{verdict_data['status']}`.** {verdict_data['reason']}",
        "",
        f"- Run A (`{name_a}`): {sa['n_unique_patterns']} patterns, {sa['active_bits']['mean']:.2f} active bits/document, "
        f"accuracy {accuracy_a:.3f}, macro-F1 {macro_f1_a:.3f}.",
        f"- Run B (`{name_b}`): {sb['n_unique_patterns']} patterns, {sb['active_bits']['mean']:.2f} active bits/document, "
        f"accuracy {accuracy_b:.3f}, macro-F1 {macro_f1_b:.3f}.",
        f"- Optimal semantic matching: mean directional similarity {semantic['mean_similarity']:.3f}, "
        f"strength-aware similarity {semantic['mean_strength_aware_similarity']:.3f}, and "
        f"same-pair activation NMI {semantic['mean_activation_nmi_on_semantic_assignment']:.3f}; "
        f"activation-optimal matching: mean NMI {activation['mean_normalized_mutual_information']:.3f}.",
        "- Bit names and indices were not used for alignment; complements are allowed in semantic matching.",
        "- Each full Hungarian match scores every cross-run bit pair, then selects the "
        "one-to-one assignment with maximum total score. Semantic and activation matching "
        "are optimized separately.",
        "",
        "## Run-level evidence",
        "",
        f"| Metric | Run A: {name_a} | Run B: {name_b} |",
        "|---|---:|---:|",
        f"| Bits | {sa['n_bits']} | {sb['n_bits']} |",
        f"| Unique hard patterns | {sa['n_unique_patterns']} | {sb['n_unique_patterns']} |",
        f"| Pattern entropy (bits) | {sa['pattern_entropy_bits']:.3f} | {sb['pattern_entropy_bits']:.3f} |",
        f"| Pattern-label NMI | {sa['pattern_label_normalized_mutual_information']:.3f} | {sb['pattern_label_normalized_mutual_information']:.3f} |",
        (
            f"| 80/20 holdout accuracy | {accuracy_a:.3f} | {accuracy_b:.3f} |"
            if holdout_protocol
            else f"| CV accuracy | {accuracy_a:.3f} ± {pa['accuracy_std']:.3f} | "
            f"{accuracy_b:.3f} ± {pb['accuracy_std']:.3f} |"
        ),
        (
            f"| 80/20 holdout macro-F1 | {macro_f1_a:.3f} | {macro_f1_b:.3f} |"
            if holdout_protocol
            else f"| CV macro-F1 | {macro_f1_a:.3f} ± {pa['macro_f1_std']:.3f} | "
            f"{macro_f1_b:.3f} ± {pb['macro_f1_std']:.3f} |"
        ),
        f"| Dead bits | {len(sa['dead_bits'])} | {len(sb['dead_bits'])} |",
        f"| Universal bits | {len(sa['universal_bits'])} | {len(sb['universal_bits'])} |",
        "",
        "### Scope of the predictive rows",
        "",
        (
            f"These scores come from a single seeded stratified split: {pa['train_rows']:,} "
            f"documents (80%) fit the class-balanced logistic regression and "
            f"{pa['test_rows']:,} documents (20%) are evaluated once. Both runs use the "
            f"same document assignment (seed {pa['seed']}). The classifier uses only the "
            "run's hard `bit_XX` columns; the SAE itself was fitted beforehand on the full "
            "corpus, without labels."
            if holdout_protocol
            else f"Legacy report: these scores come from {pa['folds']}-fold stratified, "
            "class-balanced logistic regression using only the run's hard `bit_XX` columns."
        ),
        "",
        "## Document-level activation flow",
        "",
        "Each document contributes one total flow unit, divided equally across all of its "
        "active Run A × Run B bit pairs. Ribbon width therefore measures document mass "
        "without overweighting documents that activate more bits. Colored ribbons identify "
        "the semantic Hungarian pairs; gray ribbons show all other active/active connections. "
        "A dagger marks a complemented semantic match, while the ribbons themselves retain "
        "the observed 1/1 activations.",
        "",
        f"![Document-weighted activation Sankey]({result['sankey']['filename']})",
        "",
        "## Optimal semantic matches",
        "",
        "Directional similarity is the absolute correlation between label-conditional activation profiles. "
        "`complemented=yes` means the profiles align after reversing the orientation of run B's bit.",
        "The strength ratio compares the two bits' label NMI; strength-aware similarity is their product, "
        "so a weak trace with the right direction is not treated as equivalent information.",
        "Activation NMI is calculated on each of these exact semantic pairs. It tests whether "
        "two bits that behave similarly across classes also activate on the same individual documents.",
        "High direction and high same-pair NMI indicate both aggregate semantic and document-level "
        "agreement. High direction but low NMI means similar class-level behavior is implemented on "
        "different documents; low direction but high NMI means the bits co-select documents without "
        "following the same class profile.",
        "",
        "| Run A bit | Run B bit | Direction | Label NMI A | Label NMI B | Strength ratio | Strength-aware | Activation NMI (same pair) | Complemented |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for pair in semantic["pairs"]:
        lines.append(
            f"| {pair['bit_a']} | {pair['bit_b']} | {pair['similarity']:.3f} | "
            f"{pair['label_nmi_a']:.3f} | {pair['label_nmi_b']:.3f} | "
            f"{pair['information_strength_ratio']:.3f} | {pair['strength_aware_similarity']:.3f} | "
            f"{pair['activation_nmi']:.3f} | "
            f"{'yes' if pair['complemented'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Activation-optimal matches",
        "",
        "This is a second, independent Hungarian assignment that maximizes document-level NMI. "
        "It measures the strongest document-level correspondence available anywhere in the two "
        "codes and is not forced to reuse the semantic pairs above.",
        "This matching uses the same documents and is invariant to swapping 0 and 1.",
        "Its score is normalized mutual information, `2 MI / (H(A) + H(B))`; constant bits score zero.",
        "",
        "| Run A bit | Run B bit | Activation NMI |",
        "|---|---|---:|",
    ])
    for pair in activation["pairs"]:
        lines.append(
            f"| {pair['bit_a']} | {pair['bit_b']} | {pair['normalized_mutual_information']:.3f} |"
        )
    lines.extend([
        "",
        "## Provenance",
        "",
        f"- Run A encoded file: `{a['encoded']['path']}`",
        f"- Run B encoded file: `{b['encoded']['path']}`",
            f"- Labels: `{result['labels']['path']}` (`{result['label_column']}`)",
            f"- Shared rows: {result['n_aligned_rows']}",
            (
                "- Predictive split: [assignment](predictive_split.csv)"
                if "predictive_split" in result
                else "- Predictive split: legacy cross-validation"
            ),
            "",
    ])
    lines.extend([
        "## Training convergence and loss balance",
        "",
        "Each compact diagnostic keeps only soft/hard active factors per document versus "
        "their target and the weighted contributions of the adopted loss terms. Step-level "
        "series use a rolling window of 12; older runs may provide epoch means for loss.",
    ])
    for run_key, label in (("run_a", f"Run A: {name_a}"), ("run_b", f"Run B: {name_b}")):
        lines.extend(["", f"### {label}", ""])
        diagnostic = result["runs"][run_key].get("training_diagnostics")
        if not diagnostic:
            lines.append("No compact training diagnostic could be reconstructed for this run.")
            continue
        lines.extend([
            f"Activity granularity: **{diagnostic['activity_granularity']}**; "
            f"loss granularity: **{diagnostic['loss_granularity']}**.",
            "",
            f"![{label}: active-bit and weighted-loss convergence]"
            f"({diagnostic['plot_relative_path']})",
            "",
            "Numeric artifacts: "
            f"[active-bit series]({diagnostic['active_bits_metrics_relative_path']}), "
            f"[weighted-loss series]({diagnostic['weighted_loss_metrics_relative_path']}), "
            f"[raw per-batch metrics]({diagnostic['batch_metrics_relative_path']}), "
            f"[summary JSON]({diagnostic['summary_relative_path']}), and "
            f"[run metadata]({diagnostic['run_metadata_relative_path']}).",
        ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    path_a, path_b, X_a, X_b, y, names_a, names_b, aligned_ids = load_aligned(args)
    classes = sorted(np.unique(y).tolist())

    train_indices, test_indices = stratified_holdout_indices(
        y, args.predictive_test_size, args.seed
    )

    summary_a = code_summary(X_a, y, names_a)
    summary_b = code_summary(X_b, y, names_b)
    prediction_a = predictive_holdout(
        X_a, y, train_indices, test_indices, args.seed
    )
    prediction_b = predictive_holdout(
        X_b, y, train_indices, test_indices, args.seed
    )
    semantic = semantic_matching(
        class_profiles(X_a, y, classes),
        class_profiles(X_b, y, classes),
        names_a,
        names_b,
        np.array([row["label_normalized_mutual_information"] for row in summary_a["per_bit"]]),
        np.array([row["label_normalized_mutual_information"] for row in summary_b["per_bit"]]),
        X_a,
        X_b,
    )
    activation = activation_matching(X_a, X_b, names_a, names_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_assignment = np.full(len(y), "train", dtype=object)
    split_assignment[test_indices] = "test"
    split_path = output_dir / "predictive_split.csv"
    pd.DataFrame(
        {"id": aligned_ids, "predictive_split": split_assignment}
    ).to_csv(split_path, index=False)
    sankey = plot_activation_sankey(
        X_a,
        X_b,
        names_a,
        names_b,
        semantic,
        Path(args.run_a).name,
        Path(args.run_b).name,
        output_dir / "factor_activation_sankey.svg",
    )
    def training_diagnostics(run_dir: str) -> dict[str, Any] | None:
        directory = Path(run_dir)
        return bundle_compact_training_diagnostics(
            directory, output_dir, directory.name
        )

    result = {
        "schema_version": 3,
        "comparison": "permutation-invariant binary-factor information",
        "label_column": args.label_col,
        "classes": classes,
        "n_aligned_rows": int(len(y)),
        "labels": file_record(args.labels),
        "predictive_split": {
            "method": "seeded stratified random holdout",
            "requested_test_fraction": float(args.predictive_test_size),
            "seed": int(args.seed),
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "assignment": file_record(split_path),
        },
        "runs": {
            "run_a": {
                "run_dir": str(Path(args.run_a).resolve()),
                "encoded": file_record(path_a),
                "training": run_metadata(path_a),
                "code_summary": summary_a,
                "predictive_holdout": prediction_a,
                "training_diagnostics": training_diagnostics(args.run_a),
            },
            "run_b": {
                "run_dir": str(Path(args.run_b).resolve()),
                "encoded": file_record(path_b),
                "training": run_metadata(path_b),
                "code_summary": summary_b,
                "predictive_holdout": prediction_b,
                "training_diagnostics": training_diagnostics(args.run_b),
            },
        },
        "matching": {"semantic": semantic, "activation": activation},
        "sankey": sankey,
        "verdict": verdict(summary_a, summary_b, prediction_a, prediction_b),
    }

    write_json(output_dir / "factor_run_comparison.json", result)
    (output_dir / "factor_run_comparison.md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'factor_run_comparison.json'}")
    print(f"Wrote {output_dir / 'factor_run_comparison.md'}")
    print(f"Verdict: {result['verdict']['status']} — {result['verdict']['reason']}")


if __name__ == "__main__":
    main()
