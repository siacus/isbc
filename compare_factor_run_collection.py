#!/usr/bin/env python3
"""Compare a collection of binary-factor runs without assuming shared bit indices."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

from analysis_reporting import file_record, file_sha256, write_json
from training_diagnostic_plots import bundle_compact_training_diagnostics
from information_retention import (
    compute_information_retention,
    information_retention_markdown,
)
from predictive_modeling import (
    modeling_markdown,
    probability_log_odds,
    run_modeling_benchmark,
)
from compare_factor_runs import (
    activation_matching,
    bit_columns,
    class_profiles,
    code_summary,
    prediction_accuracy,
    prediction_macro_f1,
    predictive_holdout,
    plot_activation_sankey,
    run_metadata,
    semantic_matching,
    stratified_holdout_indices,
    verdict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare three or more binary-factor runs using permutation- and "
            "complement-invariant bit matching."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Short unique name and run directory; repeat for every run.",
    )
    parser.add_argument("--input_stem", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--label_col", default="label_name")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--predictive_test_size",
        type=float,
        default=0.2,
        help="Fraction reserved for the seeded stratified predictive test split.",
    )
    parser.add_argument(
        "--predictive_group_col",
        default=None,
        help=(
            "Optional labels-table column that must not cross the predictive "
            "train/test boundary (for example speech_id for chunked speeches)."
        ),
    )
    parser.add_argument(
        "--min_class_count",
        type=int,
        default=2,
        help=(
            "Exclude label classes with fewer observations than this before all "
            "comparison metrics (default: 2, the minimum required for CV)."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--skip_profile_matching",
        action="store_true",
        help="Do not add the integrated profile analysis (enabled when all runs have the same bit width).",
    )
    parser.add_argument(
        "--profile_null_draws",
        type=int,
        default=1000,
        help="Null draws for integrated profile matching (default: 1000).",
    )
    parser.add_argument(
        "--profile_second_taxonomy",
        default=None,
        help=(
            "Optional second label taxonomy displayed in integrated per-profile "
            "graphs; it does not affect profile matching."
        ),
    )
    parser.add_argument(
        "--profile_min_share",
        type=float,
        default=None,
        help="Deprecated compatibility option; integrated profile matching is rank-based.",
    )
    parser.add_argument(
        "--profile_top_reference_profiles",
        type=int,
        default=20,
        help="Most frequent observed profiles retained per reference (default: 20).",
    )
    parser.add_argument(
        "--profile_tail_reference_profiles",
        type=int,
        default=10,
        help="Least frequent observed profiles retained per reference (default: 10).",
    )
    parser.add_argument(
        "--profile_additional_label_summary",
        action="store_true",
        help="Add prevalent-class matrices for every taxonomy beyond the two profile taxonomies.",
    )
    parser.add_argument(
        "--profile_missing_label_value",
        action="append",
        default=[],
        help="Value treated as unavailable in additional-label matrices; repeat as needed.",
    )
    parser.add_argument(
        "--skip_extended_modeling",
        action="store_true",
        help="Skip the full classifier-by-representation benchmark.",
    )
    parser.add_argument(
        "--extract_raw_embeddings",
        action="store_true",
        help=(
            "Generate missing raw pooled-embedding caches before downstream modeling. "
            "Existing caches are always reused."
        ),
    )
    return parser.parse_args()


def parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --run {value!r}; expected NAME=DIR")
        name, directory = value.split("=", 1)
        name = name.strip()
        if not name or name in seen:
            raise ValueError(f"Run names must be non-empty and unique: {name!r}")
        seen.add(name)
        runs.append((name, Path(directory)))
    if len(runs) < 2:
        raise ValueError("At least two --run arguments are required")
    return runs


def load_runs(args: argparse.Namespace, specs: list[tuple[str, Path]]):
    labels_path = Path(args.labels)
    labels = pd.read_csv(labels_path, dtype={"id": str})
    if args.label_col not in labels or labels["id"].duplicated().any():
        raise ValueError("Labels must contain unique id values and the selected label column")
    labels = labels.set_index("id")

    expected_ids: set[str] | None = None
    ordered_ids: list[str] | None = None
    loaded: dict[str, dict[str, Any]] = {}
    for name, directory in specs:
        encoded_path = directory / f"encoded_{args.input_stem}_{args.checkpoint}.csv"
        frame = pd.read_csv(encoded_path, dtype={"id": str})
        if frame["id"].duplicated().any():
            raise ValueError(f"Run {name!r} contains duplicate ids")
        ids = set(frame["id"])
        if expected_ids is None:
            expected_ids = ids
            ordered_ids = sorted(ids)
        elif ids != expected_ids:
            raise ValueError(
                f"Run {name!r} does not contain the same ids: "
                f"missing={len(expected_ids - ids)}, extra={len(ids - expected_ids)}"
            )
        missing_labels = ids - set(labels.index)
        if missing_labels:
            raise ValueError(f"Labels are missing {len(missing_labels)} ids for {name!r}")
        names = bit_columns(frame)
        prob_names = sorted(column for column in frame if column.startswith("prob_"))
        if len(prob_names) != len(names):
            raise ValueError(
                f"Run {name!r} has {len(names)} bit columns but {len(prob_names)} probabilities"
            )
        matrix = (
            frame.set_index("id").reindex(ordered_ids)[names].to_numpy(dtype=np.int8)
        )
        probability_matrix = (
            frame.set_index("id")
            .reindex(ordered_ids)[prob_names]
            .to_numpy(dtype=np.float32)
        )
        if not np.isin(matrix, (0, 1)).all():
            raise ValueError(f"Run {name!r} has non-binary bit values")
        loaded[name] = {
            "run_dir": str(directory.resolve()),
            "encoded": file_record(encoded_path),
            "training": run_metadata(encoded_path),
            "bit_names": names,
            "matrix": matrix,
            "probability_matrix": probability_matrix,
            "encoded_path": str(encoded_path.resolve()),
        }

    assert ordered_ids is not None
    y_series = (
        labels.reindex(ordered_ids)[args.label_col]
        .astype(str)
        .str.strip()
    )
    class_counts = y_series.value_counts()
    retained_mask = y_series.map(class_counts).ge(args.min_class_count).to_numpy()
    excluded_counts = class_counts[class_counts < args.min_class_count]
    y = y_series.to_numpy()[retained_mask]
    for run in loaded.values():
        run["matrix"] = run["matrix"][retained_mask]
        run["probability_matrix"] = run["probability_matrix"][retained_mask]
    label_filter = {
        "minimum_class_count": int(args.min_class_count),
        "rows_before_filtering": int(len(y_series)),
        "rows_after_filtering": int(retained_mask.sum()),
        "excluded_rows": int((~retained_mask).sum()),
        "excluded_classes": {
            str(label): int(count)
            for label, count in excluded_counts.sort_index().items()
        },
    }
    filtered_ids = np.asarray(ordered_ids, dtype=str)[retained_mask]
    predictive_groups = None
    if args.predictive_group_col is not None:
        if args.predictive_group_col not in labels:
            raise ValueError(
                f"Labels do not contain predictive group column "
                f"{args.predictive_group_col!r}"
            )
        group_series = labels.reindex(ordered_ids)[args.predictive_group_col]
        if group_series.isna().any() or group_series.astype(str).str.strip().eq("").any():
            raise ValueError("Predictive group column contains missing or empty values")
        predictive_groups = group_series.astype(str).to_numpy()[retained_mask]
    return loaded, y, label_filter, filtered_ids, predictive_groups


def stratified_group_holdout_indices(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a reproducible stratified fold without splitting any group."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("--predictive_test_size must be between zero and one")
    n_splits = max(2, int(round(1.0 / test_size)))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    classes, encoded_y = np.unique(y, return_inverse=True)
    overall = np.bincount(encoded_y, minlength=len(classes)) / len(y)
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for train_indices, test_indices in splitter.split(
        np.zeros(len(y)), y, groups=groups
    ):
        test_distribution = (
            np.bincount(encoded_y[test_indices], minlength=len(classes))
            / len(test_indices)
        )
        score = abs(len(test_indices) / len(y) - test_size) + float(
            np.abs(test_distribution - overall).mean()
        )
        candidates.append((score, train_indices, test_indices))
    _, train_indices, test_indices = min(candidates, key=lambda item: item[0])
    if set(groups[train_indices]) & set(groups[test_indices]):
        raise RuntimeError("Predictive group leakage detected")
    return train_indices, test_indices


def raw_embedding_paths(run: dict[str, Any], input_stem: str, checkpoint: str):
    run_dir = Path(run["run_dir"])
    prefix = f"raw_embeddings_{input_stem}_{checkpoint}"
    return (
        run_dir / f"{prefix}.npy",
        run_dir / f"{prefix}_ids.csv",
        run_dir / f"{prefix}_metadata.json",
    )


@lru_cache(maxsize=None)
def cached_file_sha256(path: str) -> str:
    return file_sha256(Path(path))


def load_raw_embeddings(
    run: dict[str, Any],
    aligned_ids: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    matrix_path, ids_path, metadata_path = raw_embedding_paths(
        run, args.input_stem, args.checkpoint
    )
    present = matrix_path.is_file() and ids_path.is_file() and metadata_path.is_file()
    if present:
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(cached_metadata.get("schema_version", 1)) >= 2:
            # Schema 2 is keyed to the source corpus/backbone configuration. The raw
            # vectors remain valid if only the SAE head, threshold, or encoded CSV
            # changes, so do not force another backbone pass in that case.
            source_input = Path(cached_metadata.get("source_input", ""))
            source_input_hash = cached_metadata.get("source_input_sha256")
            config_path = Path(run["run_dir"]) / args.checkpoint / "run_config.json"
            current_config = (
                json.loads(config_path.read_text(encoding="utf-8"))
                if config_path.is_file()
                else {}
            )
            same_backbone = all(
                cached_metadata.get(metadata_key) == current_config.get(config_key, default)
                for metadata_key, config_key, default in (
                    ("model_name", "model_name", None),
                    ("pooling_mode", "pooling_mode", "mean"),
                    ("text_prefix", "text_prefix", ""),
                    ("max_length", "max_length", 128),
                    ("backbone_precision", "precision", "bf16"),
                )
            )
            present = bool(
                source_input_hash
                and source_input.is_file()
                and source_input_hash == cached_file_sha256(str(source_input.resolve()))
                and same_backbone
            )
        else:
            source_hash = cached_metadata.get("source_encoded_sha256")
            present = bool(
                source_hash
                and source_hash
                == cached_file_sha256(str(Path(run["encoded_path"]).resolve()))
            )
    if not present and args.extract_raw_embeddings:
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "extract_raw_embeddings.py"),
            "--run_dir",
            run["run_dir"],
            "--encoded",
            run["encoded_path"],
            "--checkpoint",
            args.checkpoint,
        ]
        subprocess.run(command, check=True)
        present = matrix_path.is_file() and ids_path.is_file() and metadata_path.is_file()
    if not present:
        return None, None

    raw = np.load(matrix_path, mmap_mode="r")
    raw_ids = pd.read_csv(ids_path, dtype={"id": str})["id"].to_numpy(dtype=str)
    if len(raw) != len(raw_ids):
        raise ValueError(f"Raw embedding matrix/id mismatch in {matrix_path}")
    positions = {document_id: index for index, document_id in enumerate(raw_ids)}
    missing = [document_id for document_id in aligned_ids if document_id not in positions]
    if missing:
        raise ValueError(f"Raw embedding cache is missing {len(missing)} aligned ids")
    indices = np.fromiter(
        (positions[document_id] for document_id in aligned_ids),
        dtype=np.int64,
        count=len(aligned_ids),
    )
    if np.array_equal(raw_ids, aligned_ids):
        matrix = raw
    else:
        matrix = np.asarray(raw[indices], dtype=np.float32)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return matrix, metadata


def matrix_markdown(names: list[str], values: dict[str, dict[str, float]]) -> list[str]:
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for row in names:
        cells = ["—" if row == column else f"{values[row][column]:.3f}" for column in names]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return lines


def strongest_category_associations(
    matrix: np.ndarray,
    y: np.ndarray,
    bit_names: list[str],
    classes: list[str],
) -> dict[str, Any]:
    """Return the class with maximum conditional activation for every run-local bit."""
    marginal_rates = matrix.mean(axis=0)
    class_rates = np.array(
        [matrix[y == label].mean(axis=0) for label in classes], dtype=float
    )
    bits = []
    for bit_index, bit_name in enumerate(bit_names):
        rates = class_rates[:, bit_index]
        maximum = float(rates.max())
        selected = np.flatnonzero(np.isclose(rates, maximum, rtol=0.0, atol=1e-12))
        marginal = float(marginal_rates[bit_index])
        bits.append(
            {
                "bit": bit_name,
                "marginal_activation_rate": marginal,
                "maximum_class_activation_rate": maximum,
                "category_activation_rates": [
                    {
                        "category": str(classes[class_index]),
                        "activation_rate": float(rates[class_index]),
                        "activation_lift_over_marginal": float(
                            rates[class_index] - marginal
                        ),
                    }
                    for class_index in range(len(classes))
                ],
                "selected_categories": [
                    {
                        "category": str(classes[class_index]),
                        "activation_rate": float(rates[class_index]),
                        "activation_lift_over_marginal": float(
                            rates[class_index] - marginal
                        ),
                    }
                    for class_index in selected
                ],
            }
        )
    return {
        "definition": "category maximizing P(bit=1 | category)",
        "tie_rule": "all exact maxima within absolute tolerance 1e-12",
        "bits": bits,
    }


def strongest_category_matrix_markdown(result: dict[str, Any]) -> list[str]:
    names = result["run_order"]
    first = result["runs"][names[0]]["strongest_category_associations"]["bits"]
    by_run = {
        name: {
            row["bit"]: row
            for row in result["runs"][name]["strongest_category_associations"]["bits"]
        }
        for name in names
    }
    lines = [
        "| Run-local bit | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for bit_row in first:
        bit_name = bit_row["bit"]
        cells = []
        for name in names:
            categories = [
                selected["category"].replace("|", "\\|")
                for selected in by_run[name][bit_name]["selected_categories"]
            ]
            cells.append("; ".join(categories))
        lines.append(f"| {bit_name} | " + " | ".join(cells) + " |")
    return lines


def markdown_report(
    result: dict[str, Any], profile_matching_markdown: str | None = None
) -> str:
    names = result["run_order"]
    bit_counts = [result["runs"][name]["code_summary"]["n_bits"] for name in names]
    shared_bit_count = bit_counts[0] if len(set(bit_counts)) == 1 else None
    predictions = {
        name: result["runs"][name].get(
            "predictive_holdout", result["runs"][name].get("predictive_cv")
        )
        for name in names
    }
    if any(prediction is None for prediction in predictions.values()):
        raise ValueError("Report is missing predictive evaluation results")
    holdout_protocol = all(
        "predictive_holdout" in result["runs"][name] for name in names
    )
    ranked = sorted(
        names,
        key=lambda name: prediction_macro_f1(predictions[name]),
        reverse=True,
    )
    best = ranked[0]
    best_f1 = prediction_macro_f1(predictions[best])
    lines = [
        "# Multi-run permutation-invariant factor comparison",
        "",
        "## TL;DR",
        "",
        f"- Compared {len(names)} runs on {result['n_aligned_rows']:,} identical documents; "
        "bit indices were ignored and semantic complements were allowed.",
        (
            f"- Highest 80/20 held-out macro-F1: **{best} ({best_f1:.3f})**."
            if holdout_protocol
            else f"- Highest cross-validated macro-F1: **{best} ({best_f1:.3f})**."
        ),
        "- Predictive strength, pattern diversity, semantic alignment, and document-level "
        "bit alignment are distinct criteria; inspect all matrices before calling two "
        "representations equivalent.",
        "- A full Hungarian match means that all pairwise bit scores are computed first, "
        "then a one-to-one assignment is chosen to maximize the total score. "
        + (
            f"With {shared_bit_count} bits in every run, every bit appears exactly once "
            "in each matching."
            if shared_bit_count is not None
            else "When bit counts differ, every bit in the smaller run appears exactly "
            "once and the larger run can have unmatched bits."
        ),
    ]
    label_filter = result["label_filter"]
    if label_filter["excluded_rows"]:
        excluded = ", ".join(
            f"{label} ({count})"
            for label, count in label_filter["excluded_classes"].items()
        )
        lines.append(
            f"- Label-frequency filter: excluded {label_filter['excluded_rows']:,} of "
            f"{label_filter['rows_before_filtering']:,} rows because their classes had "
            f"fewer than {label_filter['minimum_class_count']} observations: {excluded}. "
            "SAE training and encoding still used the complete dataset."
        )
    if not result["semantic_matching_identifiable"]:
        lines.append(
            "- **Binary-target caveat:** semantic direction is not uniquely identifiable. "
            "After "
            "centering, every non-constant two-class profile is collinear, so absolute "
            "Pearson correlation is mechanically 1. The displayed 1.000 values are retained "
            "for consistency but do not demonstrate semantic equivalence, and the resulting "
            "semantic Hungarian assignment is not unique."
        )
    lines.extend(
        [
            "",
            "## Run-level evidence",
            "",
            "| Run | Model | Dim | Patterns | Pattern entropy | Active bits | Pattern-label NMI | Predictive accuracy | Predictive macro-F1 | Dead/universal |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ranked:
        run = result["runs"][name]
        summary = run["code_summary"]
        prediction = predictions[name]
        training = run["training"]
        lines.append(
            f"| {name} | {training.get('model_name', '')} | "
            f"{training.get('embedding_dim', '')} | {summary['n_unique_patterns']} | "
            f"{summary['pattern_entropy_bits']:.3f} | {summary['active_bits']['mean']:.2f} | "
            f"{summary['pattern_label_normalized_mutual_information']:.3f} | "
            f"{prediction_accuracy(prediction):.3f} | "
            f"{prediction_macro_f1(prediction):.3f} | "
            f"{len(summary['dead_bits'])}/{len(summary['universal_bits'])} |"
        )

    best_prediction = predictions[best]
    lines.extend(
        [
            "",
            "### Scope of the predictive columns",
            "",
            (
                f"The predictive columns use one common seeded stratified split: "
                f"{best_prediction['train_rows']:,} documents (80%) fit a separate "
                f"class-balanced logistic regression for every embedding model and "
                f"{best_prediction['test_rows']:,} documents (20%) are evaluated once "
                f"(seed {best_prediction['seed']}). All models receive exactly the same "
                "document IDs. The regressions use only each run's hard `bit_XX` columns; "
                "the unsupervised SAE was fitted earlier on the complete corpus without "
                "labels. See the [predictive split assignment](predictive_split.csv)."
                if holdout_protocol
                else f"Legacy report: predictive columns use "
                f"{best_prediction['folds']}-fold stratified, class-balanced logistic "
                "regression on only each run's hard `bit_XX` columns."
            ),
        ]
    )

    if result.get("predictive_modeling"):
        lines.extend([""] + modeling_markdown(result["predictive_modeling"]))

    if result.get("information_retention"):
        lines.extend(
            [""]
            + information_retention_markdown(result["information_retention"])
        )

    lines.extend(
        [
            "",
            "## Most-associated category by bit and model",
            "",
            "For each model and hard bit, the table reports the category maximizing "
            "`P(bit=1 | category)`. Exact ties are separated by semicolons, and the same "
            "category may legitimately be strongest for multiple bits. Full activation "
            "rates and lift over each bit's marginal activation are stored in the JSON.",
            "",
            "**Bit indices are run-local.** A shared row number does not claim that, for "
            "example, Qwen `bit_00` corresponds to MiniLM `bit_00`; cross-model alignment "
            "is given only by the Hungarian tables below.",
            "",
        ]
    )
    lines.extend(strongest_category_matrix_markdown(result))

    lines.extend(
        [
            "",
            "## Bit-by-category activation matrices",
            "",
            "Each model has a bit × category heatmap. Cell color is the conditional "
            "hard-bit activation rate `P(bit=1 | category)` on a common 0–1 scale; white "
            "outlines mark the maximum category or exact ties for each bit. These matrices "
            "show complete category profiles rather than only the strongest category. "
            "Rows remain run-local and should be aligned across models with the Hungarian "
            "tables below.",
        ]
    )
    for name in result["run_order"]:
        heatmap = result["category_association_heatmaps"][name]
        lines.extend(
            [
                "",
                f"### {name}",
                "",
                f"![{name}: bit-by-category activation matrix]({heatmap['filename']})",
            ]
        )

    lines.extend(
        [
            "",
            "## How the comparison columns are calculated",
            "",
            "Semantic direction, strength-aware similarity, and same-pair activation NMI "
            "are evaluated on one canonical semantic Hungarian assignment. A second, "
            "activation-optimal assignment is retained as a diagnostic upper bound.",
            "",
            "- **Semantic direction.** For each bit, form a vector whose entries are the "
            "bit's mean hard activation within each class. Score every cross-run pair by "
            "the absolute Pearson correlation between these class profiles. An originally "
            "negative correlation is marked as a complemented match. The reported value is "
            "the mean score over the optimal one-to-one assignment.",
            "- **Strength-aware.** For a semantic pair, multiply its direction score by "
            "`min(NMI(bit A, label), NMI(bit B, label)) / max(...)`; the ratio is zero when "
            "both label-NMIs are zero. The reported value is the mean of these products over "
            "the semantic assignment. It prevents two similarly shaped but unequally "
            "informative class profiles from looking equivalent.",
            "- **Activation NMI on semantic pairs.** For every pair already selected by "
            "semantic matching, compare its two binary activation vectors on the same "
            "documents using normalized mutual information, `2 MI / (H(A) + H(B))`. It is "
            "invariant to swapping 0 and 1; constant bits are assigned zero. This reveals "
            "whether semantically aligned bits also identify the same individual documents.",
            "- **Activation-optimal NMI.** Run a separate full Hungarian assignment using "
            "document-level NMI as its objective. It measures the strongest document-level "
            "correspondence available across the codes, even when the partners differ from "
            "the semantic matches.",
            "- **Predictive verdict.** This is a run-level heuristic, not another matching score. "
            "On the common predictive 20% split it returns `not_equivalent` when held-out accuracy or "
            "pattern-label NMI differs by more than 0.05 (or when only one run is degenerate); "
            "otherwise it returns `compatible_information_content`. Compatibility is not a "
            "claim that individual bits are identical.",
            "",
            "**How to read the measures together.** High semantic direction and high "
            "same-pair activation NMI provide the strongest evidence that two factors carry "
            "the same meaning and select the same documents. High direction with low NMI "
            "means the factors have similar aggregate class behavior but are realized on "
            "different documents. Low direction with high NMI means they often select the "
            "same documents without exhibiting the same class profile. A large gap between "
            "same-pair and activation-optimal NMI means document-level correspondences exist, "
            "but they do not follow the semantic permutation.",
        ]
    )

    labels = [
        ("semantic_direction", "Mean semantic direction similarity"),
        ("strength_aware", "Mean strength-aware semantic similarity"),
        ("semantic_pair_activation_nmi", "Mean activation NMI on semantic pairs"),
        ("activation_nmi", "Mean activation-optimal NMI"),
    ]
    for key, title in labels:
        lines.extend(["", f"## {title}", ""])
        lines.extend(matrix_markdown(names, result["matrices"][key]))

    lines.extend(
        [
            "",
            "## Pairwise comparison heatmaps",
            "",
            "The four panels summarize semantic direction, strength-aware similarity, "
            "Activation NMI on the same semantic pairs, and the separately optimized "
            "activation NMI for every model pair.",
            "",
            "![Pairwise factor-comparison heatmaps](factor_collection_heatmaps.png)",
        ]
    )

    lines.extend(["", "## Pairwise verdicts", ""])
    lines.extend(["| Pair | Verdict | Reason |", "|---|---|---|"])
    for pair in result["pairs"].values():
        lines.append(
            f"| {pair['run_a']} vs {pair['run_b']} | `{pair['verdict']['status']}` | "
            f"{pair['verdict']['reason']} |"
        )

    lines.extend(["", "## Optimal bit correspondences", ""])
    lines.append(
        "The semantic table is the full one-to-one assignment based on direction, and its "
        "Activation NMI column is computed on those exact pairs. The activation-optimal "
        "table is a separate full one-to-one assignment based on document-level NMI."
    )
    for pair in result["pairs"].values():
        lines.extend(
            [
                "",
                f"### {pair['run_a']} vs {pair['run_b']}",
                "",
                "Each document contributes one total flow unit split equally across its "
                "active A×B bit pairs. Colored ribbons are semantic Hungarian pairs; gray "
                "ribbons are all other active/active connections. A dagger marks a "
                "complemented semantic match, but flows retain the raw 1/1 activations.",
                "",
                f"![{pair['run_a']} vs {pair['run_b']} document-weighted activation Sankey]"
                f"({pair['sankey']['filename']})",
                "",
                "**Semantic matching**",
                "",
                "| Bit A | Bit B | Direction | Strength-aware | Activation NMI (same pair) | Complemented |",
                "|---|---|---:|---:|---:|:---:|",
            ]
        )
        for row in pair["semantic"]["pairs"]:
            lines.append(
                f"| {row['bit_a']} | {row['bit_b']} | {row['similarity']:.3f} | "
                f"{row['strength_aware_similarity']:.3f} | "
                f"{row['activation_nmi']:.3f} | "
                f"{'yes' if row['complemented'] else 'no'} |"
            )
        lines.extend(
            [
                "",
                "**Activation-optimal document matching**",
                "",
                "| Bit A | Bit B | Activation NMI |",
                "|---|---|---:|",
            ]
        )
        for row in pair["activation"]["pairs"]:
            lines.append(
                f"| {row['bit_a']} | {row['bit_b']} | "
                f"{row['normalized_mutual_information']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Labels: `{result['labels']['path']}` (`{result['label_column']}`).",
            f"- Shared rows: {result['n_aligned_rows']}.",
            f"- Minimum class frequency: {result['label_filter']['minimum_class_count']}; "
            f"excluded rows: {result['label_filter']['excluded_rows']}.",
            f"- Checkpoint: `{result['checkpoint']}`.",
            "",
        ]
    )
    lines.extend(
        [
            "## Training convergence and loss balance",
            "",
            "Each compact diagnostic keeps only the two requested views: convergence of "
            "soft/hard active factors per document toward the target, and the weighted "
            "contribution of every adopted loss term. Step-level series use a rolling "
            "window of 12; reports state explicitly when older runs provide only epoch means.",
        ]
    )
    for name in names:
        diagnostic = result["runs"][name].get("training_diagnostics")
        lines.extend(["", f"### {name}", ""])
        if not diagnostic:
            lines.append("No compact training diagnostic could be reconstructed for this run.")
            continue
        lines.extend(
            [
                f"Activity granularity: **{diagnostic['activity_granularity']}**; "
                f"loss granularity: **{diagnostic['loss_granularity']}**.",
                "",
                f"![{name}: active-bit and weighted-loss convergence]"
                f"({diagnostic['plot_relative_path']})",
                "",
                "Numeric artifacts: "
                f"[active-bit series]({diagnostic['active_bits_metrics_relative_path']}), "
                f"[weighted-loss series]({diagnostic['weighted_loss_metrics_relative_path']}), "
                f"[raw per-batch metrics]({diagnostic['batch_metrics_relative_path']}), "
                f"[summary JSON]({diagnostic['summary_relative_path']}), and "
                f"[run metadata]({diagnostic['run_metadata_relative_path']}).",
            ]
        )
    if profile_matching_markdown:
        lines.extend(["", profile_matching_markdown.rstrip(), ""])
    return add_navigation_index("\n".join(lines))


def add_navigation_index(markdown: str) -> str:
    """Add stable, compact navigation links to a generated collection report.

    Generated reports can be long because they retain both the aggregate evidence and
    the per-pair / per-profile diagnostics.  Rather than relying on a renderer's
    heading-slug convention, insert explicit anchors before each top-level section.
    This keeps the contents links portable between GitHub, VS Code, and Overleaf's
    preview while deliberately leaving detailed third-level headings out of the
    index.
    """
    source_lines = markdown.splitlines()
    sections: list[tuple[str, str]] = []
    section_by_line: dict[int, str] = {}
    for index, line in enumerate(source_lines):
        if not line.startswith("## ") or line == "## Contents":
            continue
        anchor = f"report-section-{len(sections) + 1}"
        title = line.removeprefix("## ")
        sections.append((anchor, title))
        section_by_line[index] = anchor

    contents = ["## Contents", ""]
    contents.extend(f"- [{title}](#{anchor})" for anchor, title in sections)
    contents.append("")

    rendered: list[str] = []
    inserted_contents = False
    for index, line in enumerate(source_lines):
        if index == 1 and not inserted_contents:
            # The first line is the single report title.  Put the index immediately
            # beneath it, before the TL;DR, so readers can navigate before scrolling.
            rendered.extend([""] + contents)
            inserted_contents = True
        if index in section_by_line:
            rendered.extend([f'<a id="{section_by_line[index]}"></a>', ""])
        rendered.append(line)
    if not inserted_contents:
        rendered.extend([""] + contents)
    return "\n".join(rendered)


def nested_profile_matching_markdown(markdown: str) -> str:
    """Nest the standalone report and make every local link bundle-relative."""
    nested_lines = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            line = "#" + line
        nested_lines.append(line)
    nested = "\n".join(nested_lines)

    def prefix_local_target(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://", "#", "profile_matching/")):
            return match.group(0)
        return f"[{label}](profile_matching/{target})"

    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", prefix_local_target, nested)


def generate_profile_matching(
    result: dict[str, Any],
    output_dir: Path,
    null_draws: int = 1000,
    second_taxonomy: str | None = None,
    top_reference_profiles: int = 20,
    tail_reference_profiles: int = 10,
    additional_label_summary: bool = False,
    missing_label_values: list[str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Generate and embed the profile analysis for collections with one shared bit width."""
    n_bits = {run["code_summary"]["n_bits"] for run in result["runs"].values()}
    if len(n_bits) != 1:
        return None, None
    script = Path(__file__).resolve().parent / "profile_matching" / "profile_matching.py"
    if not script.is_file():
        raise FileNotFoundError(f"Profile-matching generator not found: {script}")
    profile_dir = output_dir / "profile_matching"
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script),
        "--run-dir",
        str(output_dir.resolve()),
        "--reference",
        result["run_order"][0],
        "--top-reference-profiles",
        str(top_reference_profiles),
        "--tail-reference-profiles",
        str(tail_reference_profiles),
        "--all-references",
        "--graphs",
        "--null-draws",
        str(null_draws),
        "--out",
        str(profile_dir.resolve()),
    ]
    if second_taxonomy:
        command.extend(["--second-taxonomy", second_taxonomy])
    if additional_label_summary:
        command.append("--additional-label-summary")
    for value in missing_label_values or []:
        command.extend(["--missing-label-value", value])
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/latent-profile-matplotlib")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    manifest_path = profile_dir / "manifest.json"
    standalone_path = profile_dir / "output.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["directory"] = "profile_matching"
    manifest["generator_stdout"] = completed.stdout.strip()
    return manifest, nested_profile_matching_markdown(
        standalone_path.read_text(encoding="utf-8")
    )


def plot_matrices(result: dict[str, Any], output_path: Path) -> None:
    names = result["run_order"]
    panels = [
        ("semantic_direction", "Semantic direction"),
        ("strength_aware", "Strength-aware"),
        ("semantic_pair_activation_nmi", "Activation NMI\non semantic pairs"),
        ("activation_nmi", "Activation-optimal NMI"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(4.8 * 4, 4.8), constrained_layout=True)
    image = None
    for axis, (key, title) in zip(axes, panels):
        matrix = np.array(
            [[1.0 if row == column else result["matrices"][key][row][column]
              for column in names] for row in names]
        )
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set_title(title)
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                color = "white" if matrix[i, j] < 0.5 else "black"
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color)
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.75)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_category_association_heatmaps(
    result: dict[str, Any], output_dir: Path
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    classes = [str(label) for label in result["classes"]]
    for name in result["run_order"]:
        association = result["runs"][name]["strongest_category_associations"]
        bits = association["bits"]
        matrix = np.array(
            [
                [row["activation_rate"] for row in bit["category_activation_rates"]]
                for bit in bits
            ],
            dtype=float,
        )
        width = min(32.0, max(8.0, 4.5 + 0.38 * len(classes)))
        figure, axis = plt.subplots(
            figsize=(width, 7.5), constrained_layout=True
        )
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_title(f"{name}: P(bit=1 | category)")
        axis.set_xlabel("Category")
        axis.set_ylabel("Run-local bit")
        axis.set_xticks(range(len(classes)), classes, rotation=60, ha="right")
        axis.set_yticks(range(len(bits)), [bit["bit"] for bit in bits])
        axis.tick_params(axis="x", labelsize=7 if len(classes) > 20 else 9)
        axis.tick_params(axis="y", labelsize=8)
        for bit_index, bit in enumerate(bits):
            selected = {row["category"] for row in bit["selected_categories"]}
            for class_index, label in enumerate(classes):
                if label in selected:
                    axis.scatter(
                        class_index,
                        bit_index,
                        marker="s",
                        s=105,
                        facecolors="none",
                        edgecolors="white",
                        linewidths=1.0,
                    )
        figure.colorbar(image, ax=axis, label="Conditional hard-bit activation rate")
        filename = f"factor_category_heatmap_{name}.png"
        output_path = output_dir / filename
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        artifacts[name] = {
            "filename": filename,
            "path": str(output_path.resolve()),
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "value": "P(bit=1 | category)",
            "scale": [0.0, 1.0],
        }
    return artifacts


def main() -> None:
    args = parse_args()
    if args.min_class_count < 2:
        raise ValueError("--min_class_count must be at least 2 for stratified holdout")
    specs = parse_runs(args.run)
    loaded, y, label_filter, aligned_ids, predictive_groups = load_runs(args, specs)
    if len(np.unique(y)) < 2:
        raise ValueError("At least two label classes must remain after frequency filtering")
    names = [name for name, _ in specs]
    classes = sorted(np.unique(y).tolist())
    semantic_matching_identifiable = len(classes) >= 3
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if predictive_groups is None:
        train_indices, test_indices = stratified_holdout_indices(
            y, args.predictive_test_size, args.seed
        )
    else:
        train_indices, test_indices = stratified_group_holdout_indices(
            y, predictive_groups, args.predictive_test_size, args.seed
        )
    split_assignment = np.full(len(y), "train", dtype=object)
    split_assignment[test_indices] = "test"
    split_path = output_dir / "predictive_split.csv"
    split_columns: dict[str, Any] = {
        "id": aligned_ids,
        "predictive_split": split_assignment,
    }
    if predictive_groups is not None:
        split_columns[args.predictive_group_col] = predictive_groups
    pd.DataFrame(split_columns).to_csv(split_path, index=False)
    predictive_split = {
        "method": (
            "seeded stratified group holdout"
            if predictive_groups is not None
            else "seeded stratified random holdout"
        ),
        "requested_test_fraction": float(args.predictive_test_size),
        "seed": int(args.seed),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "assignment": file_record(split_path),
    }
    if predictive_groups is not None:
        predictive_split.update(
            {
                "group_column": args.predictive_group_col,
                "train_groups": int(len(np.unique(predictive_groups[train_indices]))),
                "test_groups": int(len(np.unique(predictive_groups[test_indices]))),
                "group_overlap": 0,
            }
        )

    public_runs: dict[str, dict[str, Any]] = {}
    for name in names:
        run = loaded[name]
        summary = code_summary(run["matrix"], y, run["bit_names"])
        public_runs[name] = {
            key: value
            for key, value in run.items()
            if key
            not in {
                "matrix",
                "bit_names",
                "probability_matrix",
                "encoded_path",
            }
        }
        public_runs[name]["code_summary"] = summary
        public_runs[name]["strongest_category_associations"] = (
            strongest_category_associations(
                run["matrix"], y, run["bit_names"], classes
            )
        )
        public_runs[name]["predictive_holdout"] = predictive_holdout(
            run["matrix"], y, train_indices, test_indices, args.seed
        )
        run_dir = Path(run["run_dir"])
        diagnostic = bundle_compact_training_diagnostics(run_dir, output_dir, name)
        if diagnostic is not None:
            public_runs[name]["training_diagnostics"] = diagnostic

    predictive_modeling = None
    raw_embeddings_by_run: dict[str, np.ndarray] = {}
    representations: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        run = loaded[name]
        if not args.skip_extended_modeling:
            representations[name] = {
                "hard_bits": run["matrix"].astype(bool),
                "head_log_odds": probability_log_odds(run["probability_matrix"]),
            }
        raw_matrix, raw_metadata = load_raw_embeddings(run, aligned_ids, args)
        if raw_matrix is not None:
            raw_embeddings_by_run[name] = raw_matrix
            public_runs[name]["raw_embedding_cache"] = raw_metadata
            loaded[name]["raw_cache_key"] = raw_metadata.get("matrix_sha256", "")
            if not args.skip_extended_modeling:
                representations[name]["raw_embeddings"] = raw_matrix
    if not args.skip_extended_modeling:
        predictive_modeling = run_modeling_benchmark(
            representations=representations,
            y=y,
            ids=aligned_ids,
            train_indices=train_indices,
            test_indices=test_indices,
            seed=args.seed,
            output_dir=output_dir,
            split_record=predictive_split["assignment"],
            target_name=args.label_col,
        )

    matrices = {
        key: {row: {column: 1.0 for column in names} for row in names}
        for key in (
            "semantic_direction",
            "strength_aware",
            "semantic_pair_activation_nmi",
            "activation_nmi",
        )
    }
    pairs: dict[str, dict[str, Any]] = {}
    for name_a, name_b in combinations(names, 2):
        run_a, run_b = loaded[name_a], loaded[name_b]
        summary_a = public_runs[name_a]["code_summary"]
        summary_b = public_runs[name_b]["code_summary"]
        semantic = semantic_matching(
            class_profiles(run_a["matrix"], y, classes),
            class_profiles(run_b["matrix"], y, classes),
            run_a["bit_names"],
            run_b["bit_names"],
            np.array([row["label_normalized_mutual_information"] for row in summary_a["per_bit"]]),
            np.array([row["label_normalized_mutual_information"] for row in summary_b["per_bit"]]),
            run_a["matrix"],
            run_b["matrix"],
        )
        activation = activation_matching(
            run_a["matrix"], run_b["matrix"], run_a["bit_names"], run_b["bit_names"]
        )
        sankey = plot_activation_sankey(
            run_a["matrix"],
            run_b["matrix"],
            run_a["bit_names"],
            run_b["bit_names"],
            semantic,
            name_a,
            name_b,
            output_dir / f"factor_activation_sankey_{name_a}_vs_{name_b}.svg",
        )
        pair_verdict = verdict(
            summary_a,
            summary_b,
            public_runs[name_a]["predictive_holdout"],
            public_runs[name_b]["predictive_holdout"],
        )
        key = f"{name_a}__vs__{name_b}"
        pairs[key] = {
            "run_a": name_a,
            "run_b": name_b,
            "semantic": semantic,
            "activation": activation,
            "sankey": sankey,
            "verdict": pair_verdict,
        }
        values = {
            "semantic_direction": semantic["mean_similarity"],
            "strength_aware": semantic["mean_strength_aware_similarity"],
            "semantic_pair_activation_nmi": semantic[
                "mean_activation_nmi_on_semantic_assignment"
            ],
            "activation_nmi": activation["mean_normalized_mutual_information"],
        }
        for matrix_name, value in values.items():
            matrices[matrix_name][name_a][name_b] = value
            matrices[matrix_name][name_b][name_a] = value

    result = {
        "schema_version": 5,
        "comparison": "multi-run permutation-invariant binary-factor information",
        "label_column": args.label_col,
        "classes": classes,
        "semantic_matching_identifiable": semantic_matching_identifiable,
        "n_aligned_rows": int(len(y)),
        "label_filter": label_filter,
        "checkpoint": args.checkpoint,
        "labels": file_record(args.labels),
        "predictive_split": predictive_split,
        "run_order": names,
        "runs": public_runs,
        "matrices": matrices,
        "pairs": pairs,
    }
    if predictive_modeling is not None:
        result["predictive_modeling"] = predictive_modeling
    result["information_retention"] = compute_information_retention(
        runs=loaded,
        raw_embeddings=raw_embeddings_by_run,
        output_dir=output_dir,
        checkpoint=args.checkpoint,
        seed=args.seed,
        predictive_modeling=predictive_modeling,
        labels=y,
        ids=aligned_ids,
        split_assignment=split_assignment,
    )
    result["schema_version"] = max(7, int(result["schema_version"]))
    result["category_association_heatmaps"] = plot_category_association_heatmaps(
        result, output_dir
    )
    report_json_path = output_dir / "factor_collection_comparison.json"
    write_json(report_json_path, result)
    profile_manifest = None
    profile_markdown = None
    if not args.skip_profile_matching:
        profile_manifest, profile_markdown = generate_profile_matching(
            result,
            output_dir,
            null_draws=args.profile_null_draws,
            second_taxonomy=args.profile_second_taxonomy,
            top_reference_profiles=args.profile_top_reference_profiles,
            tail_reference_profiles=args.profile_tail_reference_profiles,
            additional_label_summary=args.profile_additional_label_summary,
            missing_label_values=args.profile_missing_label_value,
        )
        if profile_manifest is not None:
            result["profile_matching"] = profile_manifest
            write_json(report_json_path, result)
    (output_dir / "factor_collection_comparison.md").write_text(
        markdown_report(result, profile_markdown), encoding="utf-8"
    )
    plot_matrices(result, output_dir / "factor_collection_heatmaps.png")
    print(f"Wrote collection comparison for {len(names)} runs to {output_dir}")


if __name__ == "__main__":
    main()
