#!/usr/bin/env python3
"""Compact, report-facing diagnostics for sparse-autoencoder convergence."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOSS_COLUMNS = [
    ("weighted_recon", "reconstruction"),
    ("weighted_tweet_sparse", "document sparsity"),
    ("weighted_balance", "balance"),
    ("weighted_decor", "decorrelation"),
    ("weighted_entropy", "binary entropy"),
]


def _run_config(run_dir: Path) -> dict[str, Any]:
    for path in (
        run_dir / "metadata.json",
        run_dir / "best" / "run_config.json",
        run_dir / "last" / "run_config.json",
    ):
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def _batch_metrics(run_dir: Path) -> pd.DataFrame:
    for filename in ("training_diagnostic_metrics.csv", "training_batch_metrics.csv"):
        path = run_dir / filename
        if path.is_file():
            return pd.read_csv(path)
    return pd.DataFrame()


def _activity_frame(run_dir: Path, batch: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    required = {"global_step", "soft_active_mean", "hard_active_mean"}
    if not batch.empty and required.issubset(batch.columns):
        columns = ["global_step", "soft_active_mean", "hard_active_mean"]
        if "epoch" in batch:
            columns.append("epoch")
        return batch[columns].dropna().copy(), "optimizer step (all batches)"

    path = run_dir / "training_coefficient_metrics.csv"
    if not path.is_file():
        return pd.DataFrame(), "optimizer step"
    metrics = pd.read_csv(path)
    required = {"global_step", "soft_activation_batch", "hard_activation_batch"}
    if not required.issubset(metrics.columns):
        return pd.DataFrame(), "optimizer step"
    metrics = metrics.dropna(subset=["soft_activation_batch", "hard_activation_batch"])
    aggregation = {
        "soft_activation_batch": "sum",
        "hard_activation_batch": "sum",
    }
    if "epoch" in metrics:
        aggregation["epoch"] = "first"
    activity = metrics.groupby("global_step", as_index=False).agg(aggregation).rename(
        columns={
            "soft_activation_batch": "soft_active_mean",
            "hard_activation_batch": "hard_active_mean",
        }
    )
    return activity, "optimizer step (monitored batches)"


def _loss_frame(run_dir: Path, batch: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    weighted = [column for column, _ in LOSS_COLUMNS]
    if not batch.empty and "global_step" in batch and any(
        column in batch for column in weighted
    ):
        columns = ["global_step", *[column for column in weighted if column in batch]]
        if "epoch" in batch:
            columns.append("epoch")
        return batch[columns].dropna(subset=["global_step"]).copy(), "global_step", "optimizer step (all batches)"

    path = run_dir / "train_history.csv"
    if path.is_file():
        history = pd.read_csv(path)
        if "epoch" in history and any(column in history for column in weighted):
            columns = ["epoch", *[column for column in weighted if column in history]]
            return history[columns].copy(), "epoch", "epoch mean"

    legacy = run_dir / "training_epoch_metrics.csv"
    if legacy.is_file():
        history = pd.read_csv(legacy)
        if "epoch" in history and any(column in history for column in weighted):
            columns = ["epoch", *[column for column in weighted if column in history]]
            return history[columns].copy(), "epoch", "epoch mean"
    return pd.DataFrame(), "epoch", "epoch"


def _epoch_boundaries(frame: pd.DataFrame) -> list[float]:
    if "epoch" not in frame or "global_step" not in frame:
        return []
    boundaries = frame.groupby("epoch")["global_step"].min().sort_index().tolist()
    return [float(value) for value in boundaries[1:]]


def write_compact_training_diagnostics(
    run_dir: str | Path,
    *,
    rolling_window: int = 12,
) -> dict[str, Any] | None:
    """Render active-bit convergence and weighted loss contributions for one run."""
    run_dir = Path(run_dir)
    batch = _batch_metrics(run_dir)
    activity, activity_granularity = _activity_frame(run_dir, batch)
    losses, loss_x, loss_granularity = _loss_frame(run_dir, batch)
    if activity.empty and losses.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = _run_config(run_dir)
    target_active = float(config.get("target_active", 6.0))
    artifacts: dict[str, Any] = {
        "rolling_window": int(rolling_window),
        "target_active": target_active,
        "activity_granularity": activity_granularity,
        "loss_granularity": loss_granularity,
    }
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)

    activity_axis = axes[0]
    if activity.empty:
        activity_axis.text(0.5, 0.5, "Active-bit history unavailable", ha="center", va="center")
        activity_axis.set_axis_off()
    else:
        activity = activity.sort_values("global_step")
        window = min(rolling_window, max(1, len(activity)))
        soft = activity["soft_active_mean"].rolling(window, min_periods=1).mean()
        hard = activity["hard_active_mean"].rolling(window, min_periods=1).mean()
        activity_export = activity.copy()
        activity_export["soft_active_rolling"] = soft
        activity_export["hard_active_rolling"] = hard
        activity_export["target_active"] = target_active
        activity_export["granularity"] = activity_granularity
        activity_path = run_dir / "training_active_bits_diagnostics.csv"
        activity_export.to_csv(activity_path, index=False)
        artifacts["active_bits_metrics"] = activity_path.name
        artifacts["final_soft_active_rolling"] = float(soft.iloc[-1])
        artifacts["final_hard_active_rolling"] = float(hard.iloc[-1])
        activity_axis.plot(activity["global_step"], soft, label="soft active", linewidth=2)
        activity_axis.plot(activity["global_step"], hard, label="hard active", linewidth=2)
        activity_axis.axhline(
            target_active, color="black", linestyle="--", linewidth=1.5,
            label=f"target = {target_active:g}",
        )
        for boundary in _epoch_boundaries(activity):
            activity_axis.axvline(boundary, color="0.75", linewidth=0.8, linestyle=":")
        activity_axis.set_title("Active factors per document")
        activity_axis.set_xlabel(activity_granularity)
        activity_axis.set_ylabel("mean active factors")
        activity_axis.legend(fontsize=8)
        activity_axis.grid(alpha=0.2)

    loss_axis = axes[1]
    if losses.empty:
        loss_axis.text(0.5, 0.5, "Weighted loss history unavailable", ha="center", va="center")
        loss_axis.set_axis_off()
    else:
        losses = losses.sort_values(loss_x)
        available = [(column, label) for column, label in LOSS_COLUMNS if column in losses]
        smoothed: dict[str, pd.Series] = {}
        for column, _label in available:
            values = losses[column].fillna(0.0)
            if loss_x == "global_step":
                window = min(rolling_window, max(1, len(values)))
                values = values.rolling(window, min_periods=1).mean()
            smoothed[column] = values
        final_total = sum(float(values.iloc[-1]) for values in smoothed.values())
        positive_values = []
        loss_export = losses.copy()
        row_total = sum(smoothed.values())
        for column, label in available:
            values = smoothed[column]
            final_share = 100.0 * float(values.iloc[-1]) / final_total if final_total > 0 else 0.0
            loss_export[f"smoothed_{column}"] = values
            loss_export[f"share_{column}"] = np.where(
                row_total > 0, values / row_total, 0.0
            )
            loss_axis.plot(
                losses[loss_x], values, linewidth=2, label=f"{label} ({final_share:.1f}% final)",
            )
            positive_values.extend(values[values > 0].tolist())
        loss_export["smoothed_weighted_total"] = row_total
        loss_export["granularity"] = loss_granularity
        loss_path = run_dir / "training_weighted_loss_diagnostics.csv"
        loss_export.to_csv(loss_path, index=False)
        artifacts["weighted_loss_metrics"] = loss_path.name
        artifacts["final_weighted_contributions"] = {
            label: {
                "value": float(smoothed[column].iloc[-1]),
                "share": (
                    float(smoothed[column].iloc[-1]) / final_total
                    if final_total > 0 else 0.0
                ),
            }
            for column, label in available
        }
        if positive_values and max(positive_values) / max(min(positive_values), 1e-12) > 100:
            loss_axis.set_yscale("log")
        for boundary in _epoch_boundaries(losses):
            loss_axis.axvline(boundary, color="0.75", linewidth=0.8, linestyle=":")
        loss_axis.set_title("Weighted contributions to total loss")
        loss_axis.set_xlabel(loss_granularity)
        loss_axis.set_ylabel("weighted loss contribution")
        loss_axis.legend(fontsize=8)
        loss_axis.grid(alpha=0.2)

    figure.suptitle(
        f"Training convergence diagnostics — {run_dir.name}\n"
        f"rolling window={rolling_window} for step-level series",
        fontsize=14,
    )
    output = run_dir / "training_convergence_diagnostics.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    artifacts["plot"] = output.name
    summary_path = run_dir / "training_convergence_diagnostics.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, indent=2)
        handle.write("\n")
    artifacts["summary"] = summary_path.name
    return artifacts


def bundle_compact_training_diagnostics(
    run_dir: str | Path,
    comparison_dir: str | Path,
    run_name: str,
    *,
    rolling_window: int = 12,
) -> dict[str, Any] | None:
    """Copy every report-facing training artifact into a comparison bundle."""
    run_dir = Path(run_dir)
    comparison_dir = Path(comparison_dir)
    diagnostic = write_compact_training_diagnostics(
        run_dir, rolling_window=rolling_window
    )
    if diagnostic is None:
        return None

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("._") or "run"
    destination = comparison_dir / "training_diagnostics" / safe_name
    destination.mkdir(parents=True, exist_ok=True)
    public = dict(diagnostic)

    for key in ("plot", "active_bits_metrics", "weighted_loss_metrics", "summary"):
        source = run_dir / diagnostic[key]
        target = destination / source.name
        shutil.copy2(source, target)
        public[f"{key}_relative_path"] = target.relative_to(comparison_dir).as_posix()

    extras = {
        "batch_metrics": run_dir / "training_diagnostic_metrics.csv",
        "run_metadata": run_dir / "metadata.json",
        "reconstruction_calibration": run_dir / "reconstruction_calibration.json",
    }
    for key, source in extras.items():
        if source.is_file():
            target = destination / source.name
            shutil.copy2(source, target)
            public[f"{key}_relative_path"] = target.relative_to(comparison_dir).as_posix()

    return public
