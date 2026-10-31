#!/usr/bin/env python3
"""Compare factor associations across label-column analysis directories."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_reporting import (
    discover_analysis_plots,
    fmt,
    json_safe,
    plot_markdown,
    read_json,
    technical_markdown,
    write_json,
)


EVIDENCE_FILES = (
    "marginal_associations.csv",
    "mutual_information.csv",
    "label_profiles.csv",
)
DISCOVERY_FILES = EVIDENCE_FILES + ("analysis_manifest.json", "factor_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate label-specific factor_summary.json files."
    )
    parser.add_argument(
        "--analysis_dir", required=True,
        help="Parent directory whose children are per-label analysis directories",
    )
    parser.add_argument("--output_dir", default=None,
                        help="Default: <analysis_dir>/cross_label")
    parser.add_argument(
        "--refresh_individual", dest="refresh_individual", action="store_true", default=True,
        help="Rebuild summaries for every existing label directory first (default: on)",
    )
    parser.add_argument(
        "--no_refresh_individual", dest="refresh_individual", action="store_false",
        help="Use existing factor_summary.json files without rebuilding them",
    )
    return parser.parse_args()


def is_cross_label_output(path: Path, root: Path) -> bool:
    """Return whether a path is inside a generated cross_label directory."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return "cross_label" in relative.parts


def directories_with_files(root: Path, filenames: tuple[str, ...]) -> list[Path]:
    """Find analysis directories recursively, excluding generated report directories."""
    directories: set[Path] = set()
    for filename in filenames:
        for path in root.rglob(filename):
            if path.is_file() and not is_cross_label_output(path, root):
                directories.add(path.parent.resolve())
    return sorted(directories, key=lambda path: str(path.relative_to(root.resolve())))


def label_analysis_directories(root: Path) -> list[Path]:
    """Return all nested directories with enough evidence for an individual summary."""
    return directories_with_files(root, EVIDENCE_FILES)


def refresh_individual_summaries(root: Path) -> None:
    script = Path(__file__).resolve().with_name("summarize_factors.py")
    directories = label_analysis_directories(root)
    if not directories:
        return
    print(
        f"Refreshing {len(directories)} individual label summary/summaries before aggregation...",
        flush=True,
    )
    for directory in directories:
        subprocess.run(
            [sys.executable, str(script), "--analysis_dir", str(directory)],
            check=True,
        )


def discover_summaries(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    summaries = []
    for path in sorted(root.rglob("factor_summary.json")):
        if is_cross_label_output(path, root):
            continue
        payload = read_json(path)
        if payload.get("factors"):
            summaries.append((path, payload))
    return summaries


def validate_unique_label_columns(
    summaries: list[tuple[Path, dict[str, Any]]]
) -> None:
    """Reject ambiguous duplicate summaries instead of combining them silently."""
    locations: dict[str, list[Path]] = {}
    for path, payload in summaries:
        label_column = str(payload.get("label_column") or path.parent.name)
        locations.setdefault(label_column, []).append(path.parent)
    duplicates = {label: paths for label, paths in locations.items() if len(paths) > 1}
    if not duplicates:
        return
    details = "; ".join(
        f"{label}: {', '.join(str(path) for path in paths)}"
        for label, paths in sorted(duplicates.items())
    )
    raise ValueError(
        "Multiple factor summaries claim the same label column. Remove or rename the "
        f"stale analysis directory before aggregation. {details}"
    )


def build_discovery_audit(
    root: Path, summaries: list[tuple[Path, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Record every detected label directory and whether it entered the comparison."""
    included = {path.parent.resolve(): payload for path, payload in summaries}
    records = []
    for directory in directories_with_files(root, DISCOVERY_FILES):
        summary = read_json(directory / "factor_summary.json")
        manifest = read_json(directory / "analysis_manifest.json")
        payload = included.get(directory)
        label_column = str(
            (payload or summary or manifest).get("label_column") or directory.name
        )
        evidence = [name for name in EVIDENCE_FILES if (directory / name).is_file()]
        if payload is not None:
            status = "included"
            reason = "non-empty factor_summary.json"
        elif summary and not summary.get("factors"):
            status = "excluded"
            reason = "factor_summary.json contains no factor records"
        elif evidence:
            status = "excluded"
            reason = "factor evidence exists but no usable individual summary was generated"
        else:
            status = "excluded"
            reason = "manifest exists, but no marginal, MI, or label-profile factor table exists"
        records.append(json_safe({
            "label_column": label_column,
            "analysis_directory": str(directory),
            "status": status,
            "reason": reason,
            "evidence_files": evidence,
        }))
    return records


def provenance_compatibility(summaries: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    issues = []
    if not summaries:
        return issues
    first = summaries[0][1].get("provenance", {})
    first_hash = first.get("inputs", {}).get("encoded", {}).get("sha256")
    first_training = first.get("training_parameters", {})
    for path, payload in summaries[1:]:
        provenance = payload.get("provenance", {})
        encoded_hash = provenance.get("inputs", {}).get("encoded", {}).get("sha256")
        if first_hash and encoded_hash and first_hash != encoded_hash:
            issues.append(f"{path.parent.name} uses a different encoded-file SHA-256")
        training = provenance.get("training_parameters", {})
        for key in (
            "model_name", "embedding_dim", "text_prefix",
            "k_factors", "objective", "pooling_mode", "seed",
        ):
            if key in first_training and key in training and first_training[key] != training[key]:
                issues.append(f"{path.parent.name} has a different training `{key}`")
    return issues


def build_rows(summaries: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for path, payload in summaries:
        label_column = str(payload.get("label_column") or path.parent.name)
        for factor in payload.get("factors", []):
            rows.append({
                "factor": factor.get("factor"),
                "factor_index": factor.get("factor_index"),
                "label_column": label_column,
                "dominant_category": factor.get("dominant_category"),
                "activation_rate": factor.get("activation_rate"),
                "marginal_role": factor.get("marginal_role"),
                "cramers_v": factor.get("cramers_v"),
                "mutual_information_bits": factor.get("mutual_information_bits"),
                "factor_explains_label": factor.get("factor_explains_label"),
                "label_explains_factor": factor.get("label_explains_factor"),
                "max_macro_f1_drop": factor.get("max_macro_f1_drop"),
                "label_specific_confidence": factor.get(
                    "interpretation_confidence_for_this_label"
                ),
                "source_summary": str(path),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    for column in (
        "cramers_v", "mutual_information_bits", "factor_explains_label",
        "label_explains_factor", "max_macro_f1_drop",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["association_rank_within_label"] = frame.groupby("label_column")[
        "label_explains_factor"
    ].rank(method="min", ascending=False)
    frame["association_percentile_within_label"] = frame.groupby("label_column")[
        "label_explains_factor"
    ].rank(method="average", pct=True, ascending=True)
    return [json_safe(row.to_dict()) for _, row in frame.iterrows()]


def factor_attribution(factor: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row["factor"] == factor]
    valid = [row for row in candidates if row.get("label_explains_factor") is not None]
    valid.sort(key=lambda row: row["label_explains_factor"], reverse=True)

    near_constant = [
        row for row in candidates
        if row.get("activation_rate") is not None
        and (row["activation_rate"] < 0.02 or row["activation_rate"] > 0.98)
    ]
    if candidates and len(near_constant) == len(candidates):
        association_type = "low_entropy_factor"
    elif not valid:
        association_type = "insufficient_evidence"
    else:
        top = valid[0]
        second = valid[1] if len(valid) > 1 else None
        top_u = top["label_explains_factor"]
        max_v = max(
            (row.get("cramers_v") or 0.0 for row in candidates), default=0.0
        )
        if top_u < 0.01 and max_v < 0.10:
            association_type = "weakly_annotated"
        elif second and second["label_explains_factor"] >= 0.8 * top_u:
            association_type = "shared_association"
        else:
            association_type = "primary_association"

    if not valid:
        primary = None
        secondary = []
        confidence = "low"
    else:
        primary = valid[0]
        secondary = valid[1:3]
        runner_u = valid[1]["label_explains_factor"] if len(valid) > 1 else 0.0
        top_u = primary["label_explains_factor"]
        top_v = primary.get("cramers_v") or 0.0
        if top_u >= 0.20 and top_v >= 0.30 and runner_u < 0.8 * top_u:
            confidence = "high"
        elif top_u >= 0.05 or top_v >= 0.10:
            confidence = "moderate"
        else:
            confidence = "low"

    return json_safe({
        "factor": factor,
        "association_type": association_type,
        "primary_label_column": primary.get("label_column") if primary else None,
        "primary_category": primary.get("dominant_category") if primary else None,
        "primary_label_explains_factor": (
            primary.get("label_explains_factor") if primary else None
        ),
        "primary_cramers_v": primary.get("cramers_v") if primary else None,
        "secondary_label_columns": [
            {
                "label_column": row.get("label_column"),
                "dominant_category": row.get("dominant_category"),
                "label_explains_factor": row.get("label_explains_factor"),
                "cramers_v": row.get("cramers_v"),
            }
            for row in secondary
        ],
        "cross_label_attribution_confidence": confidence,
        "all_label_evidence": valid,
    })


def flat_attribution(record: dict[str, Any]) -> dict[str, Any]:
    secondary = record.get("secondary_label_columns", [])
    return {
        "factor": record["factor"],
        "association_type": record["association_type"],
        "primary_label_column": record["primary_label_column"],
        "primary_category": record["primary_category"],
        "primary_label_explains_factor": record["primary_label_explains_factor"],
        "primary_cramers_v": record["primary_cramers_v"],
        "secondary_label_columns": "; ".join(
            f"{row['label_column']} ({fmt(row['label_explains_factor'])})"
            for row in secondary
        ),
        "cross_label_attribution_confidence": record["cross_label_attribution_confidence"],
    }


def numeric(value: Any) -> float | None:
    try:
        result = float(value)
        return result if pd.notna(result) else None
    except (TypeError, ValueError):
        return None


def build_cross_executive_summary(
    attributions: list[dict[str, Any]],
    summaries: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Consolidate the individual TL;DR evidence across label columns."""
    strongest = sorted(
        [
            record for record in attributions
            if numeric(record.get("primary_label_explains_factor")) is not None
        ],
        key=lambda record: numeric(record.get("primary_label_explains_factor")) or 0.0,
        reverse=True,
    )[:5]
    primary_counts: dict[str, int] = {}
    per_label: list[dict[str, Any]] = []
    exclusions: dict[tuple[str, str], float] = {}
    overlaps: dict[tuple[str, str], float] = {}
    coalitions: dict[tuple[str, ...], dict[str, Any]] = {}

    for record in attributions:
        label = record.get("primary_label_column")
        if label:
            primary_counts[str(label)] = primary_counts.get(str(label), 0) + 1

    for path, payload in summaries:
        label = str(payload.get("label_column") or path.parent.name)
        summary = payload.get("executive_summary", {})
        per_label.append({
            "label_column": label,
            "n_moderate_or_high": summary.get("n_moderate_or_high"),
            "n_factors": summary.get("n_factors"),
            "n_discriminative_factors": summary.get("n_discriminative_factors", 0),
            "n_positive_ablation_factors": summary.get(
                "n_positive_ablation_factors", 0
            ),
            "top_discriminative_factors": summary.get(
                "top_discriminative_factors", []
            ),
            "top_positive_ablation_factors": summary.get(
                "top_positive_ablation_factors", []
            ),
            "best_holdout_by_macro_f1": summary.get("best_holdout_by_macro_f1"),
            "best_holdout_by_accuracy": summary.get("best_holdout_by_accuracy"),
        })
        for relation in summary.get("strongest_mutual_exclusions", []):
            pair = tuple(sorted((str(relation["factor_a"]), str(relation["factor_b"]))))
            value = numeric(relation.get("value"))
            if value is not None and (pair not in exclusions or value < exclusions[pair]):
                exclusions[pair] = value
        for relation in summary.get("high_overlap_pairs", []):
            pair = tuple(sorted((str(relation["factor_a"]), str(relation["factor_b"]))))
            value = numeric(relation.get("value"))
            if value is not None and (pair not in overlaps or value > overlaps[pair]):
                overlaps[pair] = value
        for coalition in summary.get("positive_pmi_coalitions", []):
            key = tuple(sorted(str(factor) for factor in coalition.get("factors", [])))
            if key:
                coalitions.setdefault(key, coalition)

    strongest_exclusions = sorted(exclusions.items(), key=lambda item: item[1])[:5]
    strongest_overlaps = sorted(
        overlaps.items(), key=lambda item: item[1], reverse=True
    )[:5]
    return json_safe({
        "n_label_columns": len(summaries),
        "n_factors": len(attributions),
        "strongest_cross_label_attributions": [
            {
                "factor": record.get("factor"),
                "label_column": record.get("primary_label_column"),
                "category": record.get("primary_category"),
                "label_explains_factor": record.get("primary_label_explains_factor"),
                "cramers_v": record.get("primary_cramers_v"),
                "confidence": record.get("cross_label_attribution_confidence"),
            }
            for record in strongest
        ],
        "primary_label_counts": primary_counts,
        "per_label_highlights": per_label,
        "strongest_mutual_exclusions": [
            {"factor_a": pair[0], "factor_b": pair[1], "pmi": value}
            for pair, value in strongest_exclusions
        ],
        "positive_pmi_coalitions": list(coalitions.values())[:3],
        "high_overlap_pairs": [
            {"factor_a": pair[0], "factor_b": pair[1], "jaccard": value}
            for pair, value in strongest_overlaps
        ],
    })


def cross_executive_markdown(summary: dict[str, Any]) -> list[str]:
    strongest = summary["strongest_cross_label_attributions"]
    strongest_text = ", ".join(
        f"`{r['factor']}` → `{r['label_column']}`/{r.get('category')} "
        f"({(numeric(r.get('label_explains_factor')) or 0.0) * 100:.1f}% explained; "
        f"V={fmt(r.get('cramers_v'))}; {r.get('confidence')})"
        for r in strongest
    ) or "none available"
    counts_text = ", ".join(
        f"`{label}`: {count} bit(s)"
        for label, count in sorted(summary["primary_label_counts"].items())
    ) or "no primary attributions"

    discriminative_parts = []
    predictive_parts = []
    prediction_parts = []
    for item in summary["per_label_highlights"]:
        label = item["label_column"]
        discriminative = item.get("top_discriminative_factors", [])
        predictive = item.get("top_positive_ablation_factors", [])
        discriminative_parts.append(
            f"`{label}`: {item.get('n_discriminative_factors', 0)} total; strongest " + (
                ", ".join(f"`{r['factor']}`" for r in discriminative)
                if discriminative else "none"
            )
        )
        predictive_parts.append(
            f"`{label}`: {item.get('n_positive_ablation_factors', 0)} at/above "
            "0.5 points; strongest " + (
                ", ".join(
                    f"`{r['factor']}` "
                    f"({(numeric(r.get('max_macro_f1_drop')) or 0.0) * 100:.2f} points)"
                    for r in predictive[:3]
                ) if predictive else "none"
            )
        )
        best = item.get("best_holdout_by_macro_f1")
        if best:
            prediction_parts.append(
                f"`{label}`: {fmt(best.get('macro_f1'))} macro-F1 with "
                f"`{best.get('classifier')}`"
            )

    exclusions_text = ", ".join(
        f"`{r['factor_a']}`–`{r['factor_b']}` (PMI {r['pmi']:+.3f})"
        for r in summary["strongest_mutual_exclusions"]
    ) or "none available"
    coalition_descriptions = []
    for coalition in summary["positive_pmi_coalitions"]:
        members = coalition.get("factors", [])
        if len(members) <= 6:
            coalition_descriptions.append("`" + ", ".join(members) + "`")
            continue
        strongest_edges = ", ".join(
            f"{edge['factor_a']}–{edge['factor_b']} ({edge['pmi']:+.3f})"
            for edge in coalition.get("strongest_internal_edges", [])[:3]
        ) or "not available"
        coalition_descriptions.append(
            f"{len(members)} bits (strongest links: {strongest_edges})"
        )
    coalitions_text = "; ".join(coalition_descriptions) or (
        "none above the configured threshold"
    )
    overlap_text = ", ".join(
        f"`{r['factor_a']}`–`{r['factor_b']}` (Jaccard {r['jaccard']:.3f})"
        for r in summary["high_overlap_pairs"]
    ) or "none with Jaccard ≥ 0.70"

    return [
        "## TL;DR",
        "",
        f"- **Scope:** {summary['n_factors']} bits compared across "
        f"{summary['n_label_columns']} label column(s).",
        f"- **Strongest cross-label attributions:** {strongest_text}.",
        f"- **Primary label coverage:** {counts_text}.",
        f"- **Discriminative bits by label:** {'; '.join(discriminative_parts)}.",
        f"- **Held-out model-specific importance by label:** "
        f"{'; '.join(predictive_parts)}.",
        f"- **Strongest mutual exclusions:** {exclusions_text}.",
        f"- **Largest co-activating coalitions:** {coalitions_text}.",
        f"- **Potentially redundant high-overlap pairs:** {overlap_text}.",
        f"- **Best held-out macro-F1 by label:** "
        f"{'; '.join(prediction_parts) if prediction_parts else 'not available'}.",
        "- **Reading caution:** the primary label is the best among the annotations tested, "
        "not necessarily the semantic meaning of the bit; correlated labels can share the "
        "same signal.",
    ]


def generate_factor_label_heatmaps(
    matrix: pd.DataFrame, output_dir: Path
) -> list[dict[str, str]]:
    """Plot absolute associations and within-factor (row-rescaled) label profiles."""
    if matrix.empty:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(
            "matplotlib/numpy not available — skipping cross-label heatmaps",
            file=sys.stderr,
        )
        return []

    raw = matrix.to_numpy(dtype=float)
    row_relative = np.full_like(raw, np.nan, dtype=float)
    for row_index in range(raw.shape[0]):
        row = raw[row_index, :]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        maximum = float(np.max(row[finite]))
        row_relative[row_index, finite] = (
            row[finite] / maximum if maximum > 0 else 0.0
        )

    figure_width = max(8.0, 1.65 * len(matrix.columns) + 3.0)
    figure_height = max(6.0, 0.43 * len(matrix.index) + 2.5)
    finite_raw = raw[np.isfinite(raw)]
    observed_maximum = float(np.max(finite_raw)) if finite_raw.size else 1.0
    absolute_vmax = observed_maximum if observed_maximum > 0 else 1.0
    specifications = [
        {
            "values": raw,
            "vmax": absolute_vmax,
            "filename": "factor_label_absolute_association.png",
            "title": "Absolute factor–label association",
            "colour_bar": "Raw label_explains_factor score",
            "cell_format": ".3f",
            "description": (
                "Both colour and printed numbers show the raw `label_explains_factor` "
                "score. A common colour scale is used for every cell, so this view preserves "
                "absolute strength differences both across factors and across label columns. "
                "Darker cells indicate that a label explains more of a factor's on/off "
                "uncertainty."
            ),
            "normalization": "none; common scale from zero to the largest observed score",
            "cell_values": "raw label_explains_factor",
        },
        {
            "values": row_relative,
            "vmax": 1.0,
            "filename": "factor_label_row_rescaled_association.png",
            "title": "Within-factor label profile (row-rescaled)",
            "colour_bar": "Share of the row maximum",
            "cell_format": ".2f",
            "description": (
                "Each factor row is divided by its own largest raw score, which becomes 1. "
                "Read horizontally: darker cells identify the labels most characteristic of "
                "that bit relative to its other label associations. Row rescaling removes "
                "absolute magnitude, so a uniformly weak factor can still contain a dark "
                "cell; use the absolute heatmap above to judge whether the association is "
                "substantial at all."
            ),
            "normalization": "independent within each factor row; maximum = 1",
            "cell_values": "raw score divided by the maximum raw score in the factor row",
        },
    ]

    plots = []
    for specification in specifications:
        values = specification["values"]
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        colour_map = plt.colormaps["Blues"].copy()
        colour_map.set_bad("#eeeeee")
        image = ax.imshow(
            np.ma.masked_invalid(values),
            vmin=0,
            vmax=specification["vmax"],
            cmap=colour_map,
            aspect="auto",
        )
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(
            [str(column) for column in matrix.columns], rotation=35, ha="right"
        )
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([str(factor) for factor in matrix.index])
        ax.set_xlabel("Label column")
        ax.set_ylabel("Binary factor")
        ax.set_title(specification["title"])

        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if not np.isfinite(value):
                    label = "N/A"
                    text_colour = "#555555"
                    weight = "normal"
                else:
                    label = format(value, specification["cell_format"])
                    displayed_strength = value / specification["vmax"]
                    text_colour = "white" if displayed_strength >= 0.58 else "#202020"
                    row_maximum = np.nanmax(values[row_index, :])
                    weight = (
                        "bold"
                        if row_maximum > 0 and np.isclose(value, row_maximum)
                        else "normal"
                    )
                ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=text_colour,
                    fontweight=weight,
                )

        ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        colour_bar = fig.colorbar(image, ax=ax, pad=0.02)
        colour_bar.set_label(specification["colour_bar"])
        fig.tight_layout()

        filename = specification["filename"]
        plot_path = output_dir / filename
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved cross-label heatmap: {plot_path}")
        plots.append({
            "filename": filename,
            "title": specification["title"],
            "alt": specification["title"],
            "description": specification["description"],
            "relative_path": filename,
            "markdown_path": filename,
            "colour_normalization": specification["normalization"],
            "cell_values": specification["cell_values"],
        })
    return plots


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    if not analysis_dir.is_dir():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else analysis_dir / "cross_label"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_individual:
        refresh_individual_summaries(analysis_dir)

    summaries = discover_summaries(analysis_dir)
    if not summaries:
        raise ValueError(
            f"No usable factor_summary.json files found under {analysis_dir}. "
            "Run summarize_factors.py for each label directory first."
        )
    validate_unique_label_columns(summaries)
    discovery_audit = build_discovery_audit(analysis_dir, summaries)
    rows = build_rows(summaries)
    factors = sorted({row["factor"] for row in rows}, key=lambda value: int(value.split("_")[-1]))
    attributions = [factor_attribution(factor, rows) for factor in factors]
    issues = provenance_compatibility(summaries)
    association_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    for record in attributions:
        association_type = record["association_type"]
        confidence = record["cross_label_attribution_confidence"]
        association_counts[association_type] = (
            association_counts.get(association_type, 0) + 1
        )
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
    attributed = (
        association_counts.get("primary_association", 0)
        + association_counts.get("shared_association", 0)
    )
    executive_summary = build_cross_executive_summary(attributions, summaries)

    common_provenance = copy.deepcopy(summaries[0][1].get("provenance", {}))
    common_provenance["label_column"] = "cross-label comparison"
    common_provenance["analysis_parameters"] = {
        "label_columns_included": [
            payload.get("label_column", path.parent.name) for path, payload in summaries
        ],
        "source_summary_directories": [str(path.parent) for path, _ in summaries],
        "label_directories_detected": len(discovery_audit),
        "label_directories_included": sum(
            record["status"] == "included" for record in discovery_audit
        ),
        "cross_label_heatmaps": (
            "absolute raw scores on one common scale; second plot divides each factor "
            "row by its maximum label_explains_factor score"
        ),
    }
    common_provenance.setdefault("seeds", {})
    label_runs = [
        {
            "label_column": payload.get("label_column", path.parent.name),
            "summary_path": str(path),
            "analysis_parameters": payload.get("provenance", {}).get(
                "analysis_parameters", {}
            ),
            "seeds": payload.get("provenance", {}).get("seeds", {}),
            "plots": discover_analysis_plots(path.parent, output_dir),
        }
        for path, payload in summaries
    ]

    matrix = pd.DataFrame(rows).pivot(
        index="factor", columns="label_column", values="label_explains_factor"
    )
    matrix = matrix.reindex(factors)
    matrix.to_csv(output_dir / "label_explains_factor_matrix.csv")
    cross_label_plots = generate_factor_label_heatmaps(matrix, output_dir)

    payload = {
        "schema_version": 1,
        "provenance": json_safe(common_provenance),
        "compatibility_warnings": issues,
        "label_discovery": discovery_audit,
        "label_runs": json_safe(label_runs),
        "plots": cross_label_plots,
        "executive_summary": executive_summary,
        "metric_note": (
            "label_explains_factor = I(factor; label) / H(factor) is primary for "
            "cross-label attribution. factor_explains_label = I/H(label) answers the "
            "opposite predictive question. Cramer's V and within-label rank are supporting evidence."
        ),
        "factor_label_evidence": rows,
        "factor_attributions": attributions,
        "overall_conclusion": {
            "association_type_counts": association_counts,
            "confidence_counts": confidence_counts,
            "n_primary_or_shared": attributed,
        },
    }
    write_json(output_dir / "factor_cross_label_summary.json", payload)
    pd.DataFrame(rows).to_csv(output_dir / "factor_label_matrix.csv", index=False)
    pd.DataFrame([flat_attribution(record) for record in attributions]).to_csv(
        output_dir / "factor_cross_label_summary.csv", index=False
    )

    lines = [
        "# Cross-label binary factor summary",
        "",
        "This report compares which annotated label dimension best explains each factor's "
        "activation. It does not generate semantic factor names from document text.",
        "",
        *cross_executive_markdown(executive_summary),
        "",
        *technical_markdown(common_provenance),
        "",
        "### Per-label analysis seeds",
        "",
        "| Label column | Recorded seeds |",
        "|---|---|",
        *[
            "| " + str(run["label_column"]) + " | " + (
                ", ".join(
                    f"{name}={value}" for name, value in run["seeds"].items()
                    if value is not None
                ) or "none recorded"
            ) + " |"
            for run in label_runs
        ],
        "",
        "## How to read this comparison",
        "",
        "Each label column is a different way of describing the same documents—for example, "
        "topic, sentiment, or source. The report asks which of those descriptions best tracks "
        "each factor's on/off pattern.",
        "",
        "The main score, `label_explains_factor`, is a proportion from 0 to 1. A value of 0 "
        "means the label gives no information about whether the factor activates; larger values "
        "mean it explains more of the factor's uncertainty. The reverse score, "
        "`factor_explains_label`, asks how much the factor explains the label. Cramér's V is an "
        "additional 0-to-1 association check. Silent factors receive caution because they "
        "provide too few active examples. Factors called `universal` by this pipeline—active "
        "in more than 50% of documents—receive conservative treatment because they are broad; "
        "the term does not mean they activate in every document.",
        "",
        "Cross-label confidence is also a rule-based heuristic, not a probability. It is high "
        "when the leading label explains at least 20% of factor uncertainty, has Cramér's V of "
        "at least 0.30, and clearly exceeds the runner-up; moderate when the leading evidence "
        "passes a weaker threshold; otherwise it is low.",
        "",
        "Associations can reflect confounding between label columns; they are descriptive, "
        "not causal.",
        "",
        "## Label-directory discovery",
        "",
        "The aggregator searches recursively below the requested analysis directory. This "
        "table makes every detected processed-label directory explicit, including any "
        "directory that could not enter the comparison.",
        "",
        "| Label column | Status | Analysis directory | Reason |",
        "|---|---|---|---|",
        *[
            f"| {record['label_column']} | {record['status']} | "
            f"`{record['analysis_directory']}` | {record['reason']} |"
            for record in discovery_audit
        ],
    ]
    if issues:
        lines.extend(["", "## Compatibility warnings", ""])
        lines.extend(f"- {issue}" for issue in issues)

    matrix_columns = [str(column) for column in matrix.columns]
    lines.extend([
        "",
        "## Complete factor × label-column comparison",
        "",
        f"All {len(matrix_columns)} included label columns appear below. Each cell is "
        "`label_explains_factor`: the proportion of that factor's on/off uncertainty "
        "explained by the label column. The later `Primary label` field is only the largest "
        "value in each row; it does not mean that the other label columns were omitted.",
        "",
        "| Factor | " + " | ".join(matrix_columns) + " |",
        "|---|" + "---:|" * len(matrix_columns),
    ])
    for factor, values in matrix.iterrows():
        lines.append(
            f"| {factor} | "
            + " | ".join(fmt(values[column]) for column in matrix.columns)
            + " |"
        )
    if cross_label_plots:
        lines.extend([
            "",
            "Read the two heatmaps together: the absolute version answers whether an "
            "association is strong, while the row-rescaled version answers which label "
            "dimensions best describe each individual bit.",
            "",
            *plot_markdown(cross_label_plots, heading_level=3),
        ])

    runs_with_plots = [run for run in label_runs if run["plots"]]
    if runs_with_plots:
        lines.extend([
            "",
            "## Per-label analysis plots",
            "",
            "These are the figures already generated inside each label-specific analysis "
            "directory. Each caption explains the visual encoding; comparisons across label "
            "columns remain descriptive rather than causal.",
            "",
        ])
        for run in runs_with_plots:
            lines.extend([
                f"### {run['label_column']}",
                "",
                *plot_markdown(run["plots"], heading_level=4),
            ])

    lines.extend([
        "",
        "## Overview",
        "",
        "This compact table retains only the strongest (`Primary label`) association from "
        "each row of the complete matrix above.",
        "",
        "| Factor | Association type | Primary label | Primary category | "
        "Label explains factor | Cramér's V | Confidence |",
        "|---|---|---|---|---:|---:|---|",
    ])
    for record in attributions:
        lines.append(
            f"| {record['factor']} | {record['association_type']} | "
            f"{fmt(record['primary_label_column'])} | {fmt(record['primary_category'])} | "
            f"{fmt(record['primary_label_explains_factor'])} | "
            f"{fmt(record['primary_cramers_v'])} | "
            f"{record['cross_label_attribution_confidence']} |"
        )

    lines.extend(["", "## Per-factor cross-label evidence", ""])
    for record in attributions:
        lines.extend([
            f"### {record['factor']}",
            "",
            f"Classification: **{record['association_type']}**; attribution confidence: "
            f"**{record['cross_label_attribution_confidence']}**.",
        ])
        evidence = record.get("all_label_evidence", [])
        if evidence:
            lines.extend([
                "",
                "| Label column | Main category | Label explains factor | "
                "Factor explains label | Cramér's V | Within-label rank |",
                "|---|---|---:|---:|---:|---:|",
            ])
            for row in evidence:
                lines.append(
                    f"| {row.get('label_column')} | {row.get('dominant_category')} | "
                    f"{fmt(row.get('label_explains_factor'))} | "
                    f"{fmt(row.get('factor_explains_label'))} | "
                    f"{fmt(row.get('cramers_v'))} | "
                    f"{fmt(row.get('association_rank_within_label'), 0)} |"
                )
        else:
            lines.extend(["", "No comparable cross-label evidence is available."])
        lines.append("")

    lines.extend([
        "## Overall conclusion",
        "",
        f"{attributed} of {len(attributions)} factors have a primary or shared association "
        "with at least one available label column. Cross-label attribution confidence is "
        f"high for {confidence_counts.get('high', 0)}, moderate for "
        f"{confidence_counts.get('moderate', 0)}, and low for "
        f"{confidence_counts.get('low', 0)} factor(s).",
        "",
        "A weak or low-confidence label attribution does not mean that a factor is unreliable "
        "or unstructured. It means the available label columns do not explain its activation "
        "clearly enough. PMI, Jaccard, exact-pattern, FCA, and cluster evidence in the "
        "individual reports may still show stable co-activation or mutual-exclusion structure.",
        "",
    ])

    (output_dir / "factor_cross_label_summary.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )
    included_labels = common_provenance["analysis_parameters"]["label_columns_included"]
    excluded = [record for record in discovery_audit if record["status"] == "excluded"]
    print(f"Included {len(included_labels)} label column(s): {', '.join(included_labels)}")
    for record in excluded:
        print(
            f"Excluded label directory {record['analysis_directory']}: {record['reason']}",
            file=sys.stderr,
        )
    print(f"Saved cross-label reports to: {output_dir}")


if __name__ == "__main__":
    main()
