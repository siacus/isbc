#!/usr/bin/env python3
"""Comparable downstream models for hard bits, head log-odds, and raw embeddings."""

from __future__ import annotations

import inspect
import hashlib
import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import binomtest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyze_holdout_accuracy import PatternMajorityClassifier


COMMON_CLASSIFIERS = ("logistic_regression", "elastic_net", "random_forest")
HARD_BIT_CLASSIFIERS = COMMON_CLASSIFIERS + ("knn_jaccard", "pattern_majority")


def _elastic_net(seed: int) -> LogisticRegression:
    kwargs: dict[str, Any] = {
        "solver": "saga",
        "l1_ratio": 0.5,
        "C": 1.0,
        "max_iter": 5000,
        "class_weight": "balanced",
        "random_state": seed,
    }
    penalty = inspect.signature(LogisticRegression).parameters.get("penalty")
    if penalty is not None and penalty.default != "deprecated":
        kwargs["penalty"] = "elasticnet"
    return LogisticRegression(**kwargs)


def build_classifier(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=seed
        )
    if name == "elastic_net":
        return _elastic_net(seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "knn_jaccard":
        return KNeighborsClassifier(
            n_neighbors=5, metric="jaccard", algorithm="brute", n_jobs=-1
        )
    if name == "pattern_majority":
        return PatternMajorityClassifier(random_state=seed)
    raise ValueError(f"Unknown classifier: {name}")


def probability_log_odds(probabilities: np.ndarray) -> np.ndarray:
    """Recover stable head log-odds from the probabilities stored during encoding."""
    clipped = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def wilson_accuracy_interval(correct: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval for an observed accuracy."""
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return float(center - margin), float(center + margin)


def _fit_one(
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    classifier: str,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], Any, np.ndarray | None]:
    model = build_classifier(classifier, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=DataConversionWarning)
        model.fit(X[train_indices], y[train_indices])
    train_prediction = model.predict(X[train_indices])
    test_prediction = model.predict(X[test_indices])
    test_correct = int(np.sum(test_prediction == y[test_indices]))
    accuracy_low, accuracy_high = wilson_accuracy_interval(
        test_correct, len(test_indices)
    )
    train_accuracy = float(accuracy_score(y[train_indices], train_prediction))
    test_accuracy = float(accuracy_score(y[test_indices], test_prediction))
    train_balanced_accuracy = float(
        balanced_accuracy_score(y[train_indices], train_prediction)
    )
    test_balanced_accuracy = float(
        balanced_accuracy_score(y[test_indices], test_prediction)
    )
    report = classification_report(
        y[test_indices], test_prediction, output_dict=True, zero_division=0
    )
    metrics = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "accuracy_loss_pp": 100.0 * (train_accuracy - test_accuracy),
        "test_accuracy_ci_low": accuracy_low,
        "test_accuracy_ci_high": accuracy_high,
        "train_balanced_accuracy": train_balanced_accuracy,
        "balanced_accuracy": test_balanced_accuracy,
        "balanced_accuracy_loss_pp": 100.0
        * (train_balanced_accuracy - test_balanced_accuracy),
        "macro_f1": float(
            f1_score(y[test_indices], test_prediction, average="macro", zero_division=0)
        ),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "weighted_f1": float(
            f1_score(y[test_indices], test_prediction, average="weighted", zero_division=0)
        ),
        "fallback_rate": (
            float(model.fallback_rate(X[test_indices]))
            if isinstance(model, PatternMajorityClassifier)
            else None
        ),
        "tie_rate": (
            float(model.tie_rate(X[test_indices]))
            if isinstance(model, PatternMajorityClassifier)
            else None
        ),
    }
    true_class_probability: np.ndarray | None = None
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probabilities = np.asarray(model.predict_proba(X[test_indices]), dtype=float)
        classes = np.asarray(model.classes_).astype(str)
        positions = {label: index for index, label in enumerate(classes)}
        true_positions = np.asarray(
            [positions[str(label)] for label in y[test_indices]], dtype=np.int64
        )
        true_class_probability = np.clip(
            probabilities[np.arange(len(test_indices)), true_positions], 1e-15, 1.0
        )
        test_cross_entropy = float(-np.log(true_class_probability).mean())
        training_counts = pd.Series(y[train_indices].astype(str)).value_counts()
        training_priors = training_counts / training_counts.sum()
        baseline_probability = np.asarray(
            [
                max(float(training_priors.get(str(label), 0.0)), 1e-15)
                for label in y[test_indices]
            ]
        )
        baseline_cross_entropy = float(-np.log(baseline_probability).mean())
        metrics.update(
            {
                "test_cross_entropy_nats": test_cross_entropy,
                "test_cross_entropy_bits_per_document": test_cross_entropy
                / np.log(2.0),
                "training_prior_cross_entropy_nats": baseline_cross_entropy,
                "predictive_information_over_training_prior_bits_per_document": (
                    baseline_cross_entropy - test_cross_entropy
                )
                / np.log(2.0),
            }
        )
    return (
        metrics,
        np.asarray(test_prediction, dtype=object),
        report,
        model,
        true_class_probability,
    )


def _export_pattern_table(
    model: PatternMajorityClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    output_path: Path,
) -> None:
    train_patterns = model.pattern_strings(X_train)
    test_counts = Counter(model.pattern_strings(X_test))
    labels_by_pattern: dict[str, list[str]] = {}
    for pattern, label in zip(train_patterns, y_train):
        labels_by_pattern.setdefault(pattern, []).append(str(label))
    rows = []
    for pattern in sorted(labels_by_pattern):
        counts = Counter(labels_by_pattern[pattern])
        maximum = max(counts.values())
        tied = sorted(label for label, count in counts.items() if count == maximum)
        rows.append(
            {
                "pattern_string": pattern,
                "train_count": sum(counts.values()),
                "test_count": test_counts.get(pattern, 0),
                "majority_label": str(model.table_[pattern]),
                "majority_count": maximum,
                "tied_majority_labels": "; ".join(tied),
                "is_tied": len(tied) > 1,
                "training_label_counts": json.dumps(dict(sorted(counts.items()))),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_accuracy_interval_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot test accuracy as points with 95% Wilson score intervals."""
    runs = list(dict.fromkeys(metrics["run"].tolist()))
    figure, axes = plt.subplots(
        len(runs), 1, figsize=(13, max(4.0, 2.8 * len(runs))), squeeze=False,
        constrained_layout=True,
    )
    colors = {
        "hard_bits": "#4472C4",
        "head_log_odds": "#ED7D31",
        "raw_embeddings": "#70AD47",
    }
    classifiers = [
        name for name in HARD_BIT_CLASSIFIERS if name in set(metrics["classifier"])
    ]
    x = np.arange(len(classifiers))
    offsets = {"hard_bits": -0.20, "head_log_odds": 0.0, "raw_embeddings": 0.20}
    for row_index, run in enumerate(runs):
        axis = axes[row_index, 0]
        subset = metrics[metrics["run"] == run]
        observed_bounds: list[float] = []
        for representation in colors:
            centers: list[float] = []
            values: list[float] = []
            lower_errors: list[float] = []
            upper_errors: list[float] = []
            for classifier_index, classifier in enumerate(classifiers):
                selected = subset[
                    (subset["representation"] == representation)
                    & (subset["classifier"] == classifier)
                ]
                if selected.empty:
                    continue
                row = selected.iloc[0]
                value = float(row["test_accuracy"])
                low = float(row["test_accuracy_ci_low"])
                high = float(row["test_accuracy_ci_high"])
                centers.append(float(classifier_index) + offsets[representation])
                values.append(value)
                lower_errors.append(value - low)
                upper_errors.append(high - value)
                observed_bounds.extend((low, high))
            if values:
                axis.errorbar(
                    centers,
                    values,
                    yerr=np.asarray([lower_errors, upper_errors]),
                    fmt="o",
                    markersize=5,
                    capsize=3,
                    linewidth=1.4,
                    label=representation.replace("_", " "),
                    color=colors[representation],
                )
        if observed_bounds:
            lower = max(0.0, min(observed_bounds) - 0.04)
            upper = min(1.0, max(observed_bounds) + 0.04)
            if upper - lower < 0.15:
                midpoint = (upper + lower) / 2.0
                lower = max(0.0, midpoint - 0.075)
                upper = min(1.0, midpoint + 0.075)
            axis.set_ylim(lower, upper)
        axis.set_xticks(
            x, [name.replace("_", "\n") for name in classifiers], fontsize=8
        )
        axis.set_ylabel("Test accuracy (95% CI)")
        axis.set_title(f"{run}: accuracy with Wilson 95% intervals")
        axis.grid(axis="y", alpha=0.25)
        if row_index == 0:
            axis.legend(fontsize=8, ncol=3)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def add_balanced_accuracy_intervals(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    seed: int,
    bootstrap_draws: int = 10_000,
) -> pd.DataFrame:
    """Add class-stratified bootstrap intervals for balanced accuracy."""
    enriched = metrics.copy()
    enriched["balanced_accuracy_ci_low"] = np.nan
    enriched["balanced_accuracy_ci_high"] = np.nan
    frame = predictions.copy()
    for column in ("run", "representation", "classifier", "true_label", "predicted_label"):
        frame[column] = frame[column].astype(str)
    for index, row in enriched.iterrows():
        selected = frame[
            (frame["run"] == str(row["run"]))
            & (frame["representation"] == str(row["representation"]))
            & (frame["classifier"] == str(row["classifier"]))
        ]
        if selected.empty:
            continue
        truth = selected["true_label"].to_numpy()
        correct = selected["predicted_label"].to_numpy() == truth
        labels = sorted(set(truth))
        digest = hashlib.sha256(
            (
                f"{seed}|{row['run']}|{row['representation']}|"
                f"{row['classifier']}|balanced-accuracy"
            ).encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        bootstrap = np.zeros(bootstrap_draws, dtype=float)
        for label in labels:
            label_correct = correct[truth == label]
            successes = int(label_correct.sum())
            total = int(len(label_correct))
            bootstrap += rng.binomial(
                total, successes / total, size=bootstrap_draws
            ) / total
        bootstrap /= len(labels)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        enriched.at[index, "balanced_accuracy_ci_low"] = float(low)
        enriched.at[index, "balanced_accuracy_ci_high"] = float(high)
    return enriched


def write_balanced_accuracy_interval_plot(
    metrics: pd.DataFrame, output_path: Path
) -> None:
    """Plot balanced accuracy with class-stratified bootstrap intervals."""
    runs = list(dict.fromkeys(metrics["run"].tolist()))
    figure, axes = plt.subplots(
        len(runs), 1, figsize=(13, max(4.0, 2.8 * len(runs))), squeeze=False,
        constrained_layout=True,
    )
    colors = {
        "hard_bits": "#4472C4",
        "head_log_odds": "#ED7D31",
        "raw_embeddings": "#70AD47",
    }
    classifiers = [
        name for name in HARD_BIT_CLASSIFIERS if name in set(metrics["classifier"])
    ]
    x = np.arange(len(classifiers))
    offsets = {"hard_bits": -0.20, "head_log_odds": 0.0, "raw_embeddings": 0.20}
    for row_index, run in enumerate(runs):
        axis = axes[row_index, 0]
        subset = metrics[metrics["run"] == run]
        observed_bounds: list[float] = []
        for representation in colors:
            centers: list[float] = []
            values: list[float] = []
            lower_errors: list[float] = []
            upper_errors: list[float] = []
            for classifier_index, classifier in enumerate(classifiers):
                selected = subset[
                    (subset["representation"] == representation)
                    & (subset["classifier"] == classifier)
                ]
                if selected.empty:
                    continue
                row = selected.iloc[0]
                value = float(row["balanced_accuracy"])
                low = float(row["balanced_accuracy_ci_low"])
                high = float(row["balanced_accuracy_ci_high"])
                centers.append(float(classifier_index) + offsets[representation])
                values.append(value)
                lower_errors.append(value - low)
                upper_errors.append(high - value)
                observed_bounds.extend((low, high))
            if values:
                axis.errorbar(
                    centers,
                    values,
                    yerr=np.asarray([lower_errors, upper_errors]),
                    fmt="o",
                    markersize=5,
                    capsize=3,
                    linewidth=1.4,
                    label=representation.replace("_", " "),
                    color=colors[representation],
                )
        if observed_bounds:
            lower = max(0.0, min(observed_bounds) - 0.04)
            upper = min(1.0, max(observed_bounds) + 0.04)
            if upper - lower < 0.15:
                midpoint = (upper + lower) / 2.0
                lower = max(0.0, midpoint - 0.075)
                upper = min(1.0, midpoint + 0.075)
            axis.set_ylim(lower, upper)
        axis.set_xticks(
            x, [name.replace("_", "\n") for name in classifiers], fontsize=8
        )
        axis.set_ylabel("Balanced accuracy (95% CI)")
        axis.set_title(f"{run}: balanced accuracy with stratified bootstrap intervals")
        axis.grid(axis="y", alpha=0.25)
        if row_index == 0:
            axis.legend(fontsize=8, ncol=3)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_confusion_artifacts(
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    run_order: list[str],
    class_labels: list[str],
) -> dict[str, Any]:
    """Write row-normalized confusion grids and their underlying counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"run", "representation", "classifier", "true_label", "predicted_label"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table lacks columns: {sorted(missing)}")

    frame = predictions.copy()
    for column in ("run", "representation", "classifier", "true_label", "predicted_label"):
        frame[column] = frame[column].astype(str)
    representations = list(dict.fromkeys(frame["representation"].tolist()))
    matrix_rows: list[dict[str, Any]] = []
    plot_files: dict[str, str] = {}

    for representation in representations:
        subset = frame[frame["representation"] == representation]
        preferred = HARD_BIT_CLASSIFIERS if representation == "hard_bits" else COMMON_CLASSIFIERS
        classifiers = [name for name in preferred if name in set(subset["classifier"])]
        if not classifiers:
            continue
        runs = [name for name in run_order if name in set(subset["run"])]
        figure, axes = plt.subplots(
            len(runs),
            len(classifiers),
            figsize=(max(8.0, 3.15 * len(classifiers)), max(4.0, 2.9 * len(runs))),
            squeeze=False,
            constrained_layout=True,
        )
        image = None
        for row_index, run in enumerate(runs):
            for column_index, classifier in enumerate(classifiers):
                axis = axes[row_index, column_index]
                selected = subset[
                    (subset["run"] == run) & (subset["classifier"] == classifier)
                ]
                if selected.empty:
                    axis.axis("off")
                    continue
                counts = confusion_matrix(
                    selected["true_label"],
                    selected["predicted_label"],
                    labels=class_labels,
                )
                denominators = counts.sum(axis=1, keepdims=True)
                normalized = np.divide(
                    counts,
                    denominators,
                    out=np.zeros_like(counts, dtype=float),
                    where=denominators != 0,
                )
                for true_index, true_label in enumerate(class_labels):
                    for predicted_index, predicted_label in enumerate(class_labels):
                        matrix_rows.append(
                            {
                                "representation": representation,
                                "run": run,
                                "classifier": classifier,
                                "true_label": true_label,
                                "predicted_label": predicted_label,
                                "count": int(counts[true_index, predicted_index]),
                                "row_share": float(normalized[true_index, predicted_index]),
                            }
                        )
                image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
                accuracy = float(np.trace(counts) / counts.sum()) if counts.sum() else 0.0
                if row_index == 0:
                    axis.set_title(classifier.replace("_", " "), fontsize=9)
                axis.text(
                    0.98,
                    0.02,
                    f"acc={accuracy:.3f}",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color="black",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
                )
                if len(class_labels) <= 12:
                    ticks = np.arange(len(class_labels))
                    axis.set_xticks(ticks)
                    axis.set_yticks(ticks)
                    axis.set_xticklabels(
                        class_labels if row_index == len(runs) - 1 else [],
                        rotation=45,
                        ha="right",
                        fontsize=6,
                    )
                    axis.set_yticklabels(
                        class_labels if column_index == 0 else [], fontsize=6
                    )
                else:
                    axis.set_xticks([])
                    axis.set_yticks([])
                if len(class_labels) <= 8:
                    for true_index in range(len(class_labels)):
                        for predicted_index in range(len(class_labels)):
                            value = normalized[true_index, predicted_index]
                            axis.text(
                                predicted_index,
                                true_index,
                                f"{value:.0%}",
                                ha="center",
                                va="center",
                                fontsize=5.5,
                                color="white" if value >= 0.55 else "black",
                            )
                if column_index == 0:
                    axis.set_ylabel(f"{run}\nTrue label", fontsize=8)
                if row_index == len(runs) - 1:
                    axis.set_xlabel("Predicted label", fontsize=8)

        figure.suptitle(
            f"{representation.replace('_', ' ')}: row-normalized confusion matrices",
            fontsize=13,
        )
        if image is not None:
            figure.colorbar(image, ax=axes, shrink=0.55, label="Share within true class")
        plot_path = output_dir / f"predictive_confusion_{representation}.png"
        figure.savefig(plot_path, dpi=180)
        plt.close(figure)
        plot_files[representation] = plot_path.name

    table_path = output_dir / "predictive_confusion_matrices.csv"
    pd.DataFrame(matrix_rows).to_csv(table_path, index=False)
    return {
        "normalization": "rows sum to one within each observed true class",
        "class_order": class_labels,
        "table": table_path.name,
        "plots": plot_files,
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise-error adjustment without an additional dependency."""
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def write_paired_accuracy_contrasts(
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    run_order: list[str],
    seed: int,
    bootstrap_draws: int = 10_000,
) -> dict[str, Any] | None:
    """Compare raw embeddings with hard bits on the same held-out documents."""
    frame = predictions.copy()
    for column in ("id", "run", "representation", "classifier", "true_label", "predicted_label"):
        frame[column] = frame[column].astype(str)
    classifiers = [
        name
        for name in COMMON_CLASSIFIERS
        if name in set(frame["classifier"])
    ]
    rows: list[dict[str, Any]] = []
    for run in run_order:
        for classifier in classifiers:
            hard = frame[
                (frame["run"] == run)
                & (frame["representation"] == "hard_bits")
                & (frame["classifier"] == classifier)
            ].set_index("id")
            raw = frame[
                (frame["run"] == run)
                & (frame["representation"] == "raw_embeddings")
                & (frame["classifier"] == classifier)
            ].set_index("id")
            if hard.empty or raw.empty:
                continue
            ids = hard.index.intersection(raw.index, sort=False)
            hard = hard.loc[ids]
            raw = raw.loc[ids]
            if not np.array_equal(hard["true_label"].to_numpy(), raw["true_label"].to_numpy()):
                raise ValueError(f"{run}/{classifier}: paired predictions have different labels")
            truth = hard["true_label"].to_numpy()
            hard_correct = hard["predicted_label"].to_numpy() == truth
            raw_correct = raw["predicted_label"].to_numpy() == truth
            difference = raw_correct.astype(np.int8) - hard_correct.astype(np.int8)

            digest = hashlib.sha256(
                f"{seed}|{run}|{classifier}|raw_embeddings-hard_bits".encode("utf-8")
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            bootstrap_totals = np.zeros(bootstrap_draws, dtype=np.int64)
            for label in sorted(set(truth)):
                label_difference = difference[truth == label]
                counts = np.asarray(
                    [
                        np.sum(label_difference == -1),
                        np.sum(label_difference == 0),
                        np.sum(label_difference == 1),
                    ],
                    dtype=np.int64,
                )
                sampled = rng.multinomial(
                    int(counts.sum()), counts / counts.sum(), size=bootstrap_draws
                )
                bootstrap_totals += sampled[:, 2] - sampled[:, 0]
            bootstrap_differences = bootstrap_totals / len(difference)
            ci_low, ci_high = np.quantile(
                bootstrap_differences, [0.025, 0.975]
            )
            raw_only = int(np.sum((raw_correct == 1) & (hard_correct == 0)))
            hard_only = int(np.sum((raw_correct == 0) & (hard_correct == 1)))
            discordant = raw_only + hard_only
            p_value = (
                float(binomtest(raw_only, discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            rows.append(
                {
                    "run": run,
                    "classifier": classifier,
                    "contrast": "raw_embeddings - hard_bits",
                    "test_rows": int(len(difference)),
                    "hard_bits_accuracy": float(hard_correct.mean()),
                    "raw_embeddings_accuracy": float(raw_correct.mean()),
                    "accuracy_difference": float(difference.mean()),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "raw_only_correct": raw_only,
                    "hard_bits_only_correct": hard_only,
                    "mcnemar_exact_p": p_value,
                }
            )
    if not rows:
        return None
    adjusted = _holm_adjust([float(row["mcnemar_exact_p"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["mcnemar_holm_p"] = value

    table_path = output_dir / "predictive_accuracy_contrasts.csv"
    contrast_frame = pd.DataFrame(rows)
    contrast_frame.to_csv(table_path, index=False)

    present_classifiers = [
        name for name in classifiers if name in set(contrast_frame["classifier"])
    ]
    figure, axes = plt.subplots(
        1,
        len(present_classifiers),
        figsize=(max(9.0, 4.4 * len(present_classifiers)), max(4.0, 0.62 * len(run_order) + 2.0)),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
    )
    maximum = max(
        abs(float(contrast_frame["ci_low"].min())),
        abs(float(contrast_frame["ci_high"].max())),
    )
    extent = max(0.01, maximum * 1.12) * 100.0
    for column_index, classifier in enumerate(present_classifiers):
        axis = axes[0, column_index]
        subset = contrast_frame.set_index(["classifier", "run"])
        for row_index, run in enumerate(run_order):
            key = (classifier, run)
            if key not in subset.index:
                continue
            row = subset.loc[key]
            point = float(row["accuracy_difference"]) * 100.0
            low = float(row["ci_low"]) * 100.0
            high = float(row["ci_high"]) * 100.0
            axis.errorbar(
                point,
                row_index,
                xerr=np.asarray([[point - low], [high - point]]),
                fmt="o",
                color="#2E7D32",
                capsize=3,
                markersize=5,
                linewidth=1.5,
            )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xlim(-extent, extent)
        axis.set_yticks(np.arange(len(run_order)))
        axis.set_yticklabels(run_order if column_index == 0 else [])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
        axis.set_title(classifier.replace("_", " "))
        axis.set_xlabel("Accuracy difference (percentage points)")
        if column_index == 0:
            axis.set_ylabel("Embedding backbone")
    figure.suptitle(
        "Paired accuracy contrast: raw embeddings − hard bits (95% bootstrap CI)",
        fontsize=13,
    )
    plot_path = output_dir / "predictive_accuracy_difference_forest.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return {
        "contrast": "raw_embeddings - hard_bits",
        "positive_direction": "raw_embeddings",
        "confidence_interval": (
            "paired stratified nonparametric bootstrap percentile interval"
        ),
        "bootstrap_draws": int(bootstrap_draws),
        "seed": int(seed),
        "test": "exact McNemar with Holm adjustment across all reported contrasts",
        "table": table_path.name,
        "plot": plot_path.name,
    }


def write_paired_balanced_accuracy_contrasts(
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    run_order: list[str],
    seed: int,
    bootstrap_draws: int = 10_000,
) -> dict[str, Any] | None:
    """Compare raw embeddings with hard bits using equal weight per class."""
    frame = predictions.copy()
    for column in ("id", "run", "representation", "classifier", "true_label", "predicted_label"):
        frame[column] = frame[column].astype(str)
    classifiers = [
        name for name in COMMON_CLASSIFIERS if name in set(frame["classifier"])
    ]
    rows: list[dict[str, Any]] = []
    for run in run_order:
        for classifier in classifiers:
            hard = frame[
                (frame["run"] == run)
                & (frame["representation"] == "hard_bits")
                & (frame["classifier"] == classifier)
            ].set_index("id")
            raw = frame[
                (frame["run"] == run)
                & (frame["representation"] == "raw_embeddings")
                & (frame["classifier"] == classifier)
            ].set_index("id")
            if hard.empty or raw.empty:
                continue
            ids = hard.index.intersection(raw.index, sort=False)
            hard = hard.loc[ids]
            raw = raw.loc[ids]
            if not np.array_equal(hard["true_label"].to_numpy(), raw["true_label"].to_numpy()):
                raise ValueError(f"{run}/{classifier}: paired predictions have different labels")
            truth = hard["true_label"].to_numpy()
            labels = sorted(set(truth))
            hard_correct = hard["predicted_label"].to_numpy() == truth
            raw_correct = raw["predicted_label"].to_numpy() == truth
            difference = raw_correct.astype(np.int8) - hard_correct.astype(np.int8)
            hard_balanced = float(
                np.mean([hard_correct[truth == label].mean() for label in labels])
            )
            raw_balanced = float(
                np.mean([raw_correct[truth == label].mean() for label in labels])
            )

            digest = hashlib.sha256(
                f"{seed}|{run}|{classifier}|raw-hard-balanced-accuracy".encode("utf-8")
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            bootstrap_differences = np.zeros(bootstrap_draws, dtype=float)
            for label in labels:
                label_difference = difference[truth == label]
                counts = np.asarray(
                    [
                        np.sum(label_difference == -1),
                        np.sum(label_difference == 0),
                        np.sum(label_difference == 1),
                    ],
                    dtype=np.int64,
                )
                sampled = rng.multinomial(
                    int(counts.sum()), counts / counts.sum(), size=bootstrap_draws
                )
                bootstrap_differences += (
                    sampled[:, 2] - sampled[:, 0]
                ) / counts.sum()
            bootstrap_differences /= len(labels)
            ci_low, ci_high = np.quantile(
                bootstrap_differences, [0.025, 0.975]
            )
            rows.append(
                {
                    "run": run,
                    "classifier": classifier,
                    "contrast": "raw_embeddings - hard_bits",
                    "test_rows": int(len(difference)),
                    "n_classes": int(len(labels)),
                    "hard_bits_balanced_accuracy": hard_balanced,
                    "raw_embeddings_balanced_accuracy": raw_balanced,
                    "balanced_accuracy_difference": raw_balanced - hard_balanced,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                }
            )
    if not rows:
        return None
    contrast_frame = pd.DataFrame(rows)
    table_path = output_dir / "predictive_balanced_accuracy_contrasts.csv"
    contrast_frame.to_csv(table_path, index=False)

    present_classifiers = [
        name for name in classifiers if name in set(contrast_frame["classifier"])
    ]
    figure, axes = plt.subplots(
        1,
        len(present_classifiers),
        figsize=(max(9.0, 4.4 * len(present_classifiers)), max(4.0, 0.62 * len(run_order) + 2.0)),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
    )
    maximum = max(
        abs(float(contrast_frame["ci_low"].min())),
        abs(float(contrast_frame["ci_high"].max())),
    )
    extent = max(0.01, maximum * 1.12) * 100.0
    indexed = contrast_frame.set_index(["classifier", "run"])
    for column_index, classifier in enumerate(present_classifiers):
        axis = axes[0, column_index]
        for row_index, run in enumerate(run_order):
            key = (classifier, run)
            if key not in indexed.index:
                continue
            row = indexed.loc[key]
            point = float(row["balanced_accuracy_difference"]) * 100.0
            low = float(row["ci_low"]) * 100.0
            high = float(row["ci_high"]) * 100.0
            axis.errorbar(
                point,
                row_index,
                xerr=np.asarray([[point - low], [high - point]]),
                fmt="o",
                color="#2E7D32",
                capsize=3,
                markersize=5,
                linewidth=1.5,
            )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xlim(-extent, extent)
        axis.set_yticks(np.arange(len(run_order)))
        axis.set_yticklabels(run_order if column_index == 0 else [])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
        axis.set_title(classifier.replace("_", " "))
        axis.set_xlabel("Balanced-accuracy difference (percentage points)")
        if column_index == 0:
            axis.set_ylabel("Embedding backbone")
    figure.suptitle(
        "Paired balanced-accuracy contrast: raw embeddings − hard bits (95% bootstrap CI)",
        fontsize=13,
    )
    plot_path = output_dir / "predictive_balanced_accuracy_difference_forest.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return {
        "contrast": "raw_embeddings - hard_bits",
        "metric": "balanced accuracy (macro-average recall)",
        "positive_direction": "raw_embeddings",
        "confidence_interval": "paired class-stratified nonparametric bootstrap percentile interval",
        "bootstrap_draws": int(bootstrap_draws),
        "seed": int(seed),
        "table": table_path.name,
        "plot": plot_path.name,
    }


def write_balanced_accuracy_tradeoff_scatter(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Plot test balanced accuracy against its train-to-test loss."""
    frame = metrics.copy()
    required = {"train_balanced_accuracy", "balanced_accuracy"}
    if not required.issubset(frame.columns):
        return None
    frame["balanced_accuracy_loss_pp"] = 100.0 * (
        frame["train_balanced_accuracy"] - frame["balanced_accuracy"]
    )
    representation_order = ("raw_embeddings", "head_log_odds", "hard_bits")
    tradeoff = frame[
        frame["representation"].isin(representation_order)
        & frame["classifier"].isin(COMMON_CLASSIFIERS)
    ].copy()
    if tradeoff.empty:
        return None
    tradeoff["test_balanced_accuracy_pct"] = 100.0 * tradeoff["balanced_accuracy"]
    table_columns = [
        "run",
        "classifier",
        "representation",
        "train_balanced_accuracy",
        "balanced_accuracy",
        "test_balanced_accuracy_pct",
        "balanced_accuracy_loss_pp",
    ]
    tradeoff = tradeoff[table_columns].sort_values(
        ["representation", "classifier", "run"], kind="stable"
    )
    table_path = output_dir / "predictive_balanced_accuracy_tradeoff.csv"
    tradeoff.to_csv(table_path, index=False)

    representation_colors = {
        "raw_embeddings": "#2E7D32",
        "head_log_odds": "#E67E22",
        "hard_bits": "#6A51A3",
    }
    representation_labels = {
        "raw_embeddings": "Raw embeddings",
        "head_log_odds": "Soft bits",
        "hard_bits": "Hard bits",
    }
    markers = {
        "logistic_regression": "o",
        "elastic_net": "s",
        "random_forest": "^",
    }
    abbreviations = {
        "qwen": "QW",
        "minilm": "ML",
        "nvidia": "NV",
        "harrier": "HA",
        "gemma": "GE",
    }
    figure, axis = plt.subplots(figsize=(11.5, 7.5), constrained_layout=True)
    for _, row in tradeoff.iterrows():
        x_value = row["test_balanced_accuracy_pct"]
        y_value = row["balanced_accuracy_loss_pp"]
        color = representation_colors[row["representation"]]
        axis.scatter(
            x_value,
            y_value,
            color=color,
            marker=markers[row["classifier"]],
            s=74,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.88,
            zorder=3,
        )
        axis.annotate(
            abbreviations.get(row["run"], str(row["run"])[:2].upper()),
            (x_value, y_value),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
            color=color,
            alpha=0.95,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.grid(alpha=0.22)
    axis.set_xlabel("Test balanced accuracy (%)")
    axis.set_ylabel("Balanced-accuracy loss: train − test (percentage points)")
    x_values = tradeoff["test_balanced_accuracy_pct"].to_numpy(dtype=float)
    y_values = tradeoff["balanced_accuracy_loss_pp"].to_numpy(dtype=float)
    x_padding = max(1.0, 0.10 * (x_values.max() - x_values.min()))
    y_padding = max(1.0, 0.10 * (y_values.max() - y_values.min()))
    axis.set_xlim(x_values.min() - x_padding, x_values.max() + x_padding)
    axis.set_ylim(min(0.0, y_values.min()) - y_padding, max(0.0, y_values.max()) + y_padding)
    axis.text(
        0.985,
        0.015,
        "Higher held-out balanced accuracy\nand smaller train−test loss",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#2E7D32",
    )
    representation_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", color=color,
            label=representation_labels[representation],
            markersize=7,
        )
        for representation, color in representation_colors.items()
    ]
    classifier_handles = [
        Line2D(
            [0], [0], marker=markers[name], linestyle="none", color="black",
            label=name.replace("_", " "), markersize=7,
        )
        for name in COMMON_CLASSIFIERS
    ]
    first_legend = axis.legend(
        handles=representation_handles,
        loc="upper left",
        title="Representation",
        fontsize=8,
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=classifier_handles,
        loc="upper right",
        title="Classifier",
        fontsize=8,
    )
    axis.set_title(
        "Backbone annotations: QW Qwen · ML MiniLM · NV NVIDIA · "
        "HA Harrier · GE Gemma",
        fontsize=8,
        color="#444444",
        pad=7,
    )
    figure.suptitle(
        "Held-out balanced accuracy versus train−test balanced-accuracy loss",
        fontsize=13,
    )
    plot_path = output_dir / "predictive_balanced_accuracy_tradeoff_scatter.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "x_definition": "absolute test balanced accuracy",
        "y_definition": "train balanced accuracy minus test balanced accuracy",
        "representations": list(representation_colors),
        "x_units": "percent",
        "y_units": "percentage points",
        "table": table_path.name,
        "plot": plot_path.name,
    }


def _sequence_sha256(values: np.ndarray) -> str:
    """Hash an ordered string-like vector without ambiguous concatenation."""
    digest = hashlib.sha256()
    for value in np.asarray(values).astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _array_sample_signature(values: np.ndarray) -> dict[str, Any]:
    """Cheaply identify a representation for safe local fit-cache reuse."""
    array = np.asarray(values)
    indices = sorted({0, max(0, len(array) // 2), max(0, len(array) - 1)})
    sample = np.ascontiguousarray(array[indices]) if len(array) else array
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sample_sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
    }


def _progress_cache(
    *,
    representations: dict[str, dict[str, np.ndarray]],
    y: np.ndarray,
    ids: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    split_record: dict[str, Any],
    target_name: str | None,
    output_dir: Path,
) -> tuple[Path, str]:
    identity = {
        "schema_version": 1,
        "seed": int(seed),
        "target_name": target_name,
        "split_sha256": split_record.get("sha256"),
        "ids_sha256": _sequence_sha256(ids),
        "labels_sha256": _sequence_sha256(y),
        "train_indices_sha256": hashlib.sha256(
            np.asarray(train_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "test_indices_sha256": hashlib.sha256(
            np.asarray(test_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "representations": {
            run: {
                name: _array_sample_signature(values)
                for name, values in run_representations.items()
            }
            for run, run_representations in representations.items()
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_dir = output_dir / ".predictive_modeling_progress" / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    return cache_dir, fingerprint


def _fit_cache_paths(cache_dir: Path, key: tuple[str, str, str]) -> tuple[Path, Path]:
    token = hashlib.sha256("\0".join(key).encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{token}.json", cache_dir / f"{token}.npz"


def _load_fit_cache(
    cache_dir: Path,
    key: tuple[str, str, str],
    expected_predictions: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], np.ndarray | None] | None:
    record_path, predictions_path = _fit_cache_paths(cache_dir, key)
    if not record_path.is_file() or not predictions_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if tuple(record["key"]) != key:
            return None
        if "train_balanced_accuracy" not in record.get("metrics", {}):
            return None
        with np.load(predictions_path, allow_pickle=False) as payload:
            prediction = payload["predicted"].astype(str)
            true_probability = (
                payload["true_class_probability"].astype(float)
                if "true_class_probability" in payload.files
                else None
            )
        if len(prediction) != expected_predictions:
            return None
        return (
            record["metrics"],
            prediction,
            record["classification_report"],
            true_probability,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_fit_cache(
    cache_dir: Path,
    key: tuple[str, str, str],
    metrics: dict[str, Any],
    prediction: np.ndarray,
    report: dict[str, Any],
    true_class_probability: np.ndarray | None,
) -> None:
    record_path, predictions_path = _fit_cache_paths(cache_dir, key)
    predictions_temporary = predictions_path.with_suffix(".npz.tmp")
    with predictions_temporary.open("wb") as stream:
        payload = {"predicted": np.asarray(prediction).astype(str)}
        if true_class_probability is not None:
            payload["true_class_probability"] = np.asarray(
                true_class_probability, dtype=np.float64
            )
        np.savez_compressed(stream, **payload)
    predictions_temporary.replace(predictions_path)
    record_temporary = record_path.with_suffix(".json.tmp")
    record_temporary.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "key": list(key),
                "metrics": metrics,
                "classification_report": report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record_temporary.replace(record_path)


def run_modeling_benchmark(
    *,
    representations: dict[str, dict[str, np.ndarray]],
    y: np.ndarray,
    ids: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    output_dir: Path,
    split_record: dict[str, Any],
    target_name: str | None = None,
) -> dict[str, Any]:
    """Fit every downstream model on one immutable document split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Balanced accuracy is the adopted publication metric.  A report directory may
    # predate that decision, so remove superseded accuracy-only artifacts before
    # writing the definitive bundle; otherwise stale files can be mistaken for
    # current outputs even though they are no longer referenced by the manifest.
    for obsolete_name in (
        "predictive_accuracy_comparison.png",
        "predictive_accuracy_contrasts.csv",
        "predictive_accuracy_difference_forest.png",
        "predictive_accuracy_tradeoff.csv",
        "predictive_accuracy_tradeoff_scatter.png",
    ):
        obsolete_path = output_dir / obsolete_name
        if obsolete_path.is_file():
            obsolete_path.unlink()
    majority_label = pd.Series(y[train_indices]).value_counts().idxmax()
    majority_accuracy = float(np.mean(y[test_indices] == majority_label))
    metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    progress_dir, progress_fingerprint = _progress_cache(
        representations=representations,
        y=y,
        ids=ids,
        train_indices=train_indices,
        test_indices=test_indices,
        seed=seed,
        split_record=split_record,
        target_name=target_name,
        output_dir=output_dir,
    )

    for run, run_representations in representations.items():
        for representation, X in run_representations.items():
            if len(X) != len(y):
                raise ValueError(f"{run}/{representation}: row count does not match labels")
            classifiers = (
                HARD_BIT_CLASSIFIERS if representation == "hard_bits" else COMMON_CLASSIFIERS
            )
            for classifier in classifiers:
                key = (run, representation, classifier)
                cached = _load_fit_cache(progress_dir, key, len(test_indices))
                if (
                    cached is not None
                    and classifier == "logistic_regression"
                    and cached[3] is None
                ):
                    # Refit only the designated probabilistic probe when upgrading
                    # a legacy cache. Other classifiers keep their valid cached scores.
                    cached = None
                pattern_path = output_dir / f"predictive_pattern_profiles_{run}.csv"
                if classifier == "pattern_majority" and not pattern_path.is_file():
                    cached = None
                if cached is None:
                    print(f"modeling: {'/'.join(key)}", flush=True)
                    started = time.perf_counter()
                    metrics, prediction, report, model, true_probability = _fit_one(
                        X, y, train_indices, test_indices, classifier, seed
                    )
                    if isinstance(model, PatternMajorityClassifier):
                        _export_pattern_table(
                            model,
                            X[train_indices],
                            y[train_indices],
                            X[test_indices],
                            pattern_path,
                        )
                    _write_fit_cache(
                        progress_dir,
                        key,
                        metrics,
                        prediction,
                        report,
                        true_probability,
                    )
                    elapsed = time.perf_counter() - started
                    print(
                        f"completed: {'/'.join(key)} ({elapsed:.1f}s)", flush=True
                    )
                else:
                    metrics, prediction, report, true_probability = cached
                    print(f"modeling: {'/'.join(key)} (cached)", flush=True)
                metric_rows.append(
                    {
                        "run": run,
                        "representation": representation,
                        "n_features": int(X.shape[1]),
                        "classifier": classifier,
                        "train_rows": int(len(train_indices)),
                        "test_rows": int(len(test_indices)),
                        "majority_baseline_accuracy": majority_accuracy,
                        **metrics,
                    }
                )
                for label, values in report.items():
                    if label in {"accuracy", "macro avg", "weighted avg"}:
                        continue
                    per_class_rows.append(
                        {
                            "run": run,
                            "representation": representation,
                            "classifier": classifier,
                            "class": label,
                            "precision": values["precision"],
                            "recall": values["recall"],
                            "f1": values["f1-score"],
                            "support": int(values["support"]),
                        }
                    )
                for position, (index, predicted) in enumerate(
                    zip(test_indices, prediction)
                ):
                    prediction_rows.append(
                        {
                            "id": str(ids[index]),
                            "true_label": str(y[index]),
                            "run": run,
                            "representation": representation,
                            "classifier": classifier,
                            "predicted_label": str(predicted),
                            "true_class_probability": (
                                float(true_probability[position])
                                if true_probability is not None
                                else np.nan
                            ),
                        }
                    )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "predictive_modeling_metrics.csv"
    per_class_path = output_dir / "predictive_modeling_per_class.csv"
    predictions_path = output_dir / "predictive_modeling_predictions.csv"
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)
    predictions_frame = pd.DataFrame(prediction_rows)
    predictions_frame.to_csv(predictions_path, index=False)
    metrics_frame = add_balanced_accuracy_intervals(
        metrics_frame,
        predictions_frame,
        seed=seed,
    )
    for metric_row, (_, enriched_row) in zip(metric_rows, metrics_frame.iterrows()):
        metric_row["balanced_accuracy_ci_low"] = float(
            enriched_row["balanced_accuracy_ci_low"]
        )
        metric_row["balanced_accuracy_ci_high"] = float(
            enriched_row["balanced_accuracy_ci_high"]
        )
    metrics_frame.to_csv(metrics_path, index=False)
    balanced_plot_path = output_dir / "predictive_balanced_accuracy_comparison.png"
    write_balanced_accuracy_interval_plot(metrics_frame, balanced_plot_path)
    class_labels = sorted({str(value) for value in y})
    confusion_artifacts = write_confusion_artifacts(
        predictions_frame,
        output_dir=output_dir,
        run_order=list(representations),
        class_labels=class_labels,
    )
    paired_balanced_contrasts = write_paired_balanced_accuracy_contrasts(
        predictions_frame,
        output_dir=output_dir,
        run_order=list(representations),
        seed=seed,
    )
    balanced_accuracy_tradeoff = write_balanced_accuracy_tradeoff_scatter(
        metrics_frame,
        output_dir=output_dir,
    )
    metadata = {
        "schema_version": 7,
        "protocol": "single common seeded stratified 80/20 holdout",
        "seed": int(seed),
        "split": split_record,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "majority_training_label": str(majority_label),
        "majority_test_accuracy": majority_accuracy,
        "target": {
            "column": target_name,
            "n_classes": len(class_labels),
            "classes": class_labels,
            "accuracy_definition": "share of test documents assigned their exact true category",
        },
        "raw_embedding_definition": (
            "unnormalised pooled backbone output used by the SAE; no L2 normalization"
        ),
        "head_log_odds_definition": (
            "log(p/(1-p)) recovered from stored head probabilities with clipping at 1e-7"
        ),
        "classifier_preprocessing": "none; raw embeddings remain raw and unnormalised",
        "resumable_fit_cache": {
            "schema_version": 1,
            "fingerprint": progress_fingerprint,
            "local_directory": ".predictive_modeling_progress",
            "policy": (
                "one atomic cache record per backbone/representation/classifier fit; "
                "validated against the split, labels, ids, seed, shapes, dtypes, and "
                "sampled representation content"
            ),
        },
        "accuracy_interval": (
            "two-sided 95% Wilson score interval for exact-match test accuracy"
        ),
        "balanced_accuracy_interval": (
            "two-sided 95% class-stratified nonparametric bootstrap percentile interval"
        ),
        "cross_entropy": {
            "units": "natural logarithm in test_cross_entropy_nats; bits/document after division by ln(2)",
            "designated_probe": "logistic_regression",
            "predictive_information_definition": (
                "training-prior cross-entropy minus representation-probe cross-entropy, divided by ln(2)"
            ),
            "interpretation_limit": (
                "probe-dependent held-out predictive information, not model-free mutual information"
            ),
        },
        "classifier_scope": {
            "common": list(COMMON_CLASSIFIERS),
            "hard_bits_only": ["knn_jaccard", "pattern_majority"],
            "pattern_majority": (
                "canonical bit string to training-set majority label; unseen strings fall "
                "back to the training-set majority label"
            ),
        },
        "artifacts": {
            "metrics": metrics_path.name,
            "per_class": per_class_path.name,
            "predictions": predictions_path.name,
            "balanced_accuracy_plot": balanced_plot_path.name,
            "confusion_matrices": confusion_artifacts,
            "paired_balanced_accuracy_contrasts": paired_balanced_contrasts,
            "balanced_accuracy_tradeoff": balanced_accuracy_tradeoff,
            "pattern_profiles": {
                run: f"predictive_pattern_profiles_{run}.csv" for run in representations
            },
        },
    }
    metadata_path = output_dir / "predictive_modeling_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["artifacts"]["metadata"] = metadata_path.name
    metadata["metrics"] = metric_rows
    return metadata


def modeling_markdown(modeling: dict[str, Any]) -> list[str]:
    rows = modeling["metrics"]
    best = max(rows, key=lambda row: row["macro_f1"])
    best_balanced = max(rows, key=lambda row: row["balanced_accuracy"])
    representations = list(dict.fromkeys(row["representation"] for row in rows))
    raw_available = "raw_embeddings" in representations
    train_balanced_available = any(
        row.get("train_balanced_accuracy") is not None for row in rows
    )
    target = modeling.get("target", {})
    target_name = target.get("column")
    target_classes = [str(value) for value in target.get("classes", [])]
    if len(target_classes) <= 20:
        class_text = ", ".join(f"`{value}`" for value in target_classes)
    else:
        class_text = ", ".join(f"`{value}`" for value in target_classes[:20]) + ", …"
    lines = [
        "## Downstream predictive modeling",
        "",
        "Every score below uses the **same document-level stratified 80/20 split** as the "
        "run-level predictive columns. The unsupervised SAE was trained on the complete "
        "corpus; only these downstream classifiers are fitted on the 80% partition. "
        + (
            "Raw embeddings are the unnormalised pooled backbone vectors supplied to the "
            "SAE, with no L2 normalization or classifier-side rescaling."
            if raw_available
            else "This report does not yet contain raw-embedding caches; its available "
            "representations are listed in the table."
        ),
        "",
        (
            f"**Prediction target:** column `{target_name}` with {len(target_classes)} observed "
            f"categor{'y' if len(target_classes) == 1 else 'ies'}: {class_text}. Accuracy is the "
            "share of test documents assigned their exact true category."
            if target_name and target_classes
            else "The prediction target and its observed categories are recorded in the protocol metadata."
        ),
        "",
        f"Best balanced accuracy: **{best_balanced['run']} / "
        f"{best_balanced['representation']} / {best_balanced['classifier']} = "
        f"{best_balanced['balanced_accuracy']:.3f}**. Best macro-F1: "
        f"**{best['run']} / {best['representation']} / {best['classifier']} = "
        f"{best['macro_f1']:.3f}**. The majority-class accuracy "
        f"baseline is {modeling['majority_test_accuracy']:.3f}.",
        "",
        "Logistic regression, elastic net, and random forest are evaluated on every "
        "available representation. Jaccard k-NN and exact-pattern majority are defined only for hard "
        "binary profiles. Pattern majority treats each code as a canonical string such as "
        "`001101`; a previously unseen test string falls back to the training-set majority "
        "class. Its exported profile table reports counts, ties, and test coverage.",
        "",
        "Balanced accuracy is the primary graphical comparison: it is macro-average "
        "recall, so every observed category receives equal weight even when the target is "
        "imbalanced. Points have class-stratified bootstrap 95% intervals.",
        "",
        "![Downstream predictive balanced accuracy with 95% confidence intervals]"
        "(predictive_balanced_accuracy_comparison.png)",
        "",
        "Exact-match accuracy remains in the numeric table as a population-weighted "
        "complement, but is not used for the comparison plots.",
        "",
        "| Embedding model | Representation | Features | Classifier | Test accuracy (95% CI) | Balanced accuracy (95% CI)"
        + (" | Train balanced accuracy" if train_balanced_available else "")
        + " | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | Train accuracy | Fallback | Tie rate |",
        "|---|---|---:|---|---:|---:|"
        + ("---:|" if train_balanced_available else "")
        + "---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fallback = "—" if row["fallback_rate"] is None else f"{row['fallback_rate']:.3f}"
        tie = "—" if row["tie_rate"] is None else f"{row['tie_rate']:.3f}"
        accuracy = f"{row['test_accuracy']:.3f}"
        if row.get("test_accuracy_ci_low") is not None:
            accuracy += (
                f" [{row['test_accuracy_ci_low']:.3f}, "
                f"{row['test_accuracy_ci_high']:.3f}]"
            )
        balanced_accuracy = f"{row['balanced_accuracy']:.3f}"
        if row.get("balanced_accuracy_ci_low") is not None:
            balanced_accuracy += (
                f" [{row['balanced_accuracy_ci_low']:.3f}, "
                f"{row['balanced_accuracy_ci_high']:.3f}]"
            )
        train_balanced_accuracy = (
            "—"
            if row.get("train_balanced_accuracy") is None
            else f"{row['train_balanced_accuracy']:.3f}"
        )
        train_balanced_cell = (
            f"{train_balanced_accuracy} | " if train_balanced_available else ""
        )
        lines.append(
            f"| {row['run']} | {row['representation']} | {row['n_features']} | "
            f"{row['classifier']} | {accuracy} | "
            f"{balanced_accuracy} | "
            + train_balanced_cell
            + f"{row['macro_precision']:.3f} | "
            f"{row['macro_recall']:.3f} | {row['macro_f1']:.3f} | "
            f"{row['weighted_f1']:.3f} | {row['train_accuracy']:.3f} | "
            f"{fallback} | {tie} |"
        )
    shared_lookup = {
        (row["run"], row["classifier"], row["representation"]): row
        for row in rows
        if row["classifier"] in COMMON_CLASSIFIERS
    }
    run_order = list(dict.fromkeys(row["run"] for row in rows))
    shared_rows = [
        (run, classifier)
        for run in run_order
        for classifier in COMMON_CLASSIFIERS
        if all(
            (run, classifier, representation) in shared_lookup
            for representation in ("raw_embeddings", "head_log_odds", "hard_bits")
        )
    ]
    balanced_shared_rows = [
        (run, classifier)
        for run, classifier in shared_rows
        if all(
            shared_lookup[(run, classifier, representation)].get(
                "train_balanced_accuracy"
            )
            is not None
            for representation in ("raw_embeddings", "head_log_odds", "hard_bits")
        )
    ]
    if balanced_shared_rows:
        lines.extend(
            [
                "",
                "### Train-to-test balanced-accuracy loss",
                "",
                "The loss is `100 × (training balanced accuracy − test balanced "
                "accuracy)`, in percentage points. Every category therefore has equal "
                "weight on both sides of the comparison. **Soft bits** are the continuous "
                "binary-head log-odds before the 0.5 hard threshold. Only classifiers "
                "available for all three "
                "representations are shown, so every row is directly comparable.",
                "",
                "| Backbone | Classifier | Loss raw embeddings (pp) | Loss soft bits (pp) | Loss hard bits (pp) |",
                "|---|---|---:|---:|---:|",
            ]
        )
        gaps: dict[str, list[float]] = {
            "raw_embeddings": [],
            "head_log_odds": [],
            "hard_bits": [],
        }
        linear_gaps = {key: [] for key in gaps}
        forest_gaps = {key: [] for key in gaps}
        for run, classifier in balanced_shared_rows:
            values: dict[str, float] = {}
            for representation in gaps:
                row = shared_lookup[(run, classifier, representation)]
                value = float(
                    row.get(
                        "balanced_accuracy_loss_pp",
                        100.0
                        * (
                            row["train_balanced_accuracy"]
                            - row["balanced_accuracy"]
                        ),
                    )
                )
                values[representation] = value
                gaps[representation].append(value)
                target = forest_gaps if classifier == "random_forest" else linear_gaps
                target[representation].append(value)
            lines.append(
                f"| {run} | {classifier.replace('_', ' ')} | "
                f"{values['raw_embeddings']:.2f} | {values['head_log_odds']:.2f} | "
                f"{values['hard_bits']:.2f} |"
            )
        mean_gap = {key: float(np.mean(value)) for key, value in gaps.items()}
        mean_linear = {
            key: float(np.mean(value)) for key, value in linear_gaps.items() if value
        }
        mean_forest = {
            key: float(np.mean(value)) for key, value in forest_gaps.items() if value
        }
        lines.extend(
            [
                "",
                "A smaller loss means less resubstitution optimism, not necessarily a "
                "better representation. Hard bits strongly constrain classifier capacity "
                "and can therefore have nearly identical train and test balanced accuracy "
                "while still attaining lower held-out balanced accuracy. Raw embeddings retain more predictive "
                "information but give the classifier more opportunity to overfit. Soft bits "
                "separate the effect of the 16-dimensional SAE bottleneck from the final "
                "binary threshold.",
                "",
                f"Across all shared backbone/classifier combinations, the mean losses are "
                f"**{mean_gap['raw_embeddings']:.2f} pp for raw embeddings**, "
                f"**{mean_gap['head_log_odds']:.2f} pp for soft bits**, and "
                f"**{mean_gap['hard_bits']:.2f} pp for hard bits**. For logistic and "
                f"elastic-net models alone they are {mean_linear['raw_embeddings']:.2f}, "
                f"{mean_linear['head_log_odds']:.2f}, and {mean_linear['hard_bits']:.2f} "
                f"pp, respectively; for random forests they are "
                f"{mean_forest['raw_embeddings']:.2f}, "
                f"{mean_forest['head_log_odds']:.2f}, and "
                f"{mean_forest['hard_bits']:.2f} pp. Small negative values mean test "
                "balanced accuracy was marginally higher than its training value on this split.",
                "",
                "Read this table together with the absolute test balanced accuracies and paired "
                "forest plot: a near-zero loss can indicate effective regularization, but "
                "it can also indicate that a compressed representation underfits both "
                "partitions.",
                "",
            ]
        )
    tradeoff = modeling.get("artifacts", {}).get("balanced_accuracy_tradeoff")
    if tradeoff:
        lines.extend(
            [
                "",
                "### Test balanced accuracy versus train−test balanced-accuracy loss",
                "",
                "This scatterplot combines the two descriptive quantities without "
                "pretending that the train−test gap is a formal variance estimate. The "
                "horizontal axis is absolute held-out test balanced accuracy; the vertical "
                "axis is the loss from training to test balanced accuracy, calculated as "
                "`train balanced accuracy − test balanced accuracy`. Raw embeddings, soft "
                "bits, and hard bits share the same "
                "coordinate system. Color identifies the representation, marker shape the "
                "classifier, and the two-letter annotation the backbone.",
                "",
                "Points farther right have higher held-out balanced accuracy; points lower down have "
                "a smaller train−test loss. The most favorable region is therefore the "
                "bottom-right. A small gap can still arise from underfitting, but this plot "
                "makes that visible because absolute test balanced accuracy is shown directly.",
                "",
                f"![Test balanced accuracy versus train-test balanced-accuracy loss]({tradeoff['plot']})",
                "",
                "The exact coordinates and component balanced accuracies are available in the "
                f"[trade-off table]({tradeoff['table']}).",
                "",
            ]
        )
    paired_balanced = modeling.get("artifacts", {}).get(
        "paired_balanced_accuracy_contrasts"
    )
    if paired_balanced:
        lines.extend(
            [
                "",
                "### Forest plot of paired balanced-accuracy differences",
                "",
                "This is the primary paired comparison when category frequencies are "
                "unequal. Each point is `balanced accuracy(raw embeddings) - balanced "
                "accuracy(hard bits)`, with every true category receiving equal weight. "
                "Whiskers are paired, class-stratified 95% bootstrap intervals computed "
                "from predictions on the same held-out documents. Points to the right of "
                "zero favor raw embeddings; points to the left favor hard bits; an interval "
                "containing zero does not establish a difference on this split.",
                "",
                f"![Forest plot of paired balanced-accuracy differences]({paired_balanced['plot']})",
                "",
                "Exact balanced accuracies, paired differences, and interval endpoints "
                f"are available in the [balanced-accuracy contrast table]({paired_balanced['table']}).",
                "",
            ]
        )
    confusion = modeling.get("artifacts", {}).get("confusion_matrices", {})
    plots = confusion.get("plots", {})
    if plots:
        lines.extend(
            [
                "",
                "### Confusion matrices",
                "",
                "Each panel is row-normalized: every row conditions on the true category and "
                "therefore sums to one. Backbone models are arranged by row and downstream "
                "classifiers by column; all panels use the same category order and color scale. "
                "Diagonal cells are correct predictions, while off-diagonal cells show which "
                "categories are confused. Exact counts and row shares are available in the "
                f"[confusion-matrix table]({confusion['table']}).",
                "",
            ]
        )
        for representation in representations:
            if representation not in plots:
                continue
            lines.extend(
                [
                    f"#### {representation.replace('_', ' ').title()}",
                    "",
                    f"![{representation.replace('_', ' ')} confusion matrices]({plots[representation]})",
                    "",
                ]
            )
    lines.extend(
        [
            "",
            "Artifacts: [complete metrics](predictive_modeling_metrics.csv), "
            "[per-class precision/recall/F1](predictive_modeling_per_class.csv), "
            "[test predictions](predictive_modeling_predictions.csv), and "
            "[protocol metadata](predictive_modeling_metadata.json).",
        ]
    )
    return lines
