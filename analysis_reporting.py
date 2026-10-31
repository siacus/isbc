#!/usr/bin/env python3
"""Shared provenance and formatting helpers for analysis reports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from os.path import relpath
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


TRAINING_FIELDS = [
    "model_name",
    "model_name_short",
    "objective",
    "pooling_mode",
    "embedding_dim",
    "backbone_internal_hidden_size",
    "text_prefix",
    "k_factors",
    "target_active",
    "head_hidden_dim",
    "temperature",
    "threshold",
    "freeze_backbone",
    "max_length",
    "epochs",
    "batch_size",
    "num_workers",
    "lr",
    "weight_decay",
    "warmup_ratio",
    "grad_clip",
    "lambda_tweet_sparse",
    "lambda_balance",
    "lambda_decor",
    "lambda_recon",
    "lambda_entropy",
    "entropy_epsilon",
    "calibration_samples",
    "reconstruction_scale",
    "recon_scale_epsilon",
    "adaptive_interval",
    "adaptive_ema_beta",
    "adaptive_rate",
    "adaptive_tolerance",
    "coefficient_monitor_interval",
    "lambda_min",
    "lambda_max",
    "loss_spec",
    "trainer_variant",
    "backbone_dtype_resolved",
    "trainable_dtype",
    "lightweight_checkpoints",
    "bit_path_artifacts",
    "precision",
    "seed",
]

EXACT_MATCHING_ABLATION_NOTE = (
    "Not applicable: exact-pattern matching remains a full-model baseline, but removing a "
    "bit merges lookup cells and changes pattern coverage, confounding factor information "
    "with smoothing. Its per-factor ablation is therefore intentionally not computed."
)


ANALYSIS_PLOTS = (
    {
        "filename": "hierarchical_dendrogram.png",
        "title": "Hierarchical clustering of binary patterns",
        "alt": "Dendrogram of observed binary factor patterns",
        "description": (
            "Leaves are observed binary activation patterns, labelled by their dominant "
            "category abbreviation and document count. Branches that merge lower represent "
            "more similar patterns; compact label-homogeneous branches suggest recurring "
            "factor combinations associated with the selected label."
        ),
    },
    {
        "filename": "pmi_network.png",
        "title": "PMI factor interaction network",
        "alt": "Full pointwise mutual information network of binary factors",
        "description": (
            "Node size represents factor activation frequency and node colour its dominant "
            "label category. Green edges are positive PMI within the same dominant category, "
            "grey edges are positive cross-category PMI, and red edges indicate negative PMI; "
            "thicker edges have larger absolute PMI."
        ),
    },
    {
        "filename": "pmi_network_coalitions.png",
        "title": "Same-category factor coalitions",
        "alt": "Positive-PMI same-category factor coalition network",
        "description": (
            "This filtered PMI view retains only positively associated factor pairs that share "
            "the same dominant label category. Connected groups are candidate coalitions, but "
            "they describe co-activation rather than causal or necessarily unique effects."
        ),
    },
    {
        "filename": "jaccard_heatmap.png",
        "title": "Factor co-occurrence heatmap",
        "alt": "Clustered Jaccard similarity heatmap for binary factors",
        "description": (
            "Each cell is the Jaccard similarity between two factor activation sets. Factors "
            "are reordered by agglomerative clustering, so bright off-diagonal blocks identify "
            "groups that frequently activate together relative to their union."
        ),
    },
    {
        "filename": "label_profiles_heatmap.png",
        "title": "Label-conditional factor profiles",
        "alt": "Factor activation fingerprints and label-category similarity heatmaps",
        "description": (
            "The left panel shows mean factor activation for every label category; vertical "
            "contrast reveals discriminative factors. The right panel shows cosine similarity "
            "between complete category fingerprints, with high values indicating categories "
            "that use the factor code in similar proportions."
        ),
    },
)


def discover_analysis_plots(
    analysis_dir: Path | str, report_dir: Path | str
) -> list[dict[str, str]]:
    """Describe known pipeline plots that exist, with paths relative to a report."""
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    report_dir = Path(report_dir).expanduser().resolve()
    plots = []
    for spec in ANALYSIS_PLOTS:
        path = analysis_dir / spec["filename"]
        if not path.is_file():
            continue
        relative_path = Path(relpath(path, report_dir)).as_posix()
        plots.append(
            {
                **spec,
                "relative_path": relative_path,
                "markdown_path": quote(relative_path, safe="/._-"),
            }
        )
    return plots


def plot_markdown(plots: list[dict[str, str]], heading_level: int = 3) -> list[str]:
    """Render plot descriptions and portable relative image links."""
    lines: list[str] = []
    heading = "#" * heading_level
    for plot in plots:
        lines.extend([
            f"{heading} {plot['title']}",
            "",
            plot["description"],
            "",
            f"![{plot['alt']}]({plot['markdown_path']})",
            "",
        ])
    return lines


def retained_performance_pct(full_score: float, ablated_score: float) -> float | None:
    """Percentage of the full model's score retained after factor removal."""
    if full_score <= 0:
        return None
    return 100.0 * ablated_score / full_score


def ablation_feedback(
    classifier: str, factor: str, full_macro_f1: float, ablated_macro_f1: float
) -> str:
    """Explain the sign and scale of a leave-one-factor-out macro-F1 change."""
    completeness = retained_performance_pct(full_macro_f1, ablated_macro_f1)
    completeness_text = f"{completeness:.1f}%" if completeness is not None else "not available"
    drop_points = (full_macro_f1 - ablated_macro_f1) * 100
    if drop_points > 0:
        interpretation = (
            f"the ablated model lost {drop_points:.2f} percentage points, so the factor "
            "made a positive model-specific contribution"
        )
    elif drop_points < 0:
        interpretation = (
            f"the ablated model gained {abs(drop_points):.2f} percentage points; this negative "
            "drop means removal helped and suggests model-specific redundancy or overfitting"
        )
    else:
        interpretation = "macro-F1 was unchanged, so this model did not uniquely use the factor"
    return (
        f"{classifier} ablating {factor}: macro-F1 completeness {completeness_text}; "
        f"{interpretation} ({full_macro_f1:.4f} -> {ablated_macro_f1:.4f})."
    )


def read_json(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def file_sha256(path: Path | str) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path | str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def find_training_metadata(encoded_path: Path | str) -> tuple[dict[str, Any], str | None]:
    """Merge run metadata with checkpoint config, filling missing fields safely."""
    encoded_path = Path(encoded_path).expanduser().resolve()
    run_dir = encoded_path.parent
    inferred_checkpoint = next(
        (name for name in ("best", "last") if encoded_path.stem.endswith(f"_{name}")),
        None,
    )
    candidates = [run_dir / "metadata.json"]
    if inferred_checkpoint:
        candidates.append(run_dir / inferred_checkpoint / "run_config.json")
    candidates.extend(
        [run_dir / "best" / "run_config.json", run_dir / "last" / "run_config.json"]
    )
    candidates.extend(parent / "metadata.json" for parent in encoded_path.parents[1:3])
    seen: set[Path] = set()
    merged: dict[str, Any] = {}
    sources = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            source = read_json(candidate)
            for key, value in source.items():
                if key not in merged or merged[key] is None:
                    merged[key] = value
            sources.append(str(candidate))
    return merged, "; ".join(sources) if sources else None


def select_training_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    return {field: metadata[field] for field in TRAINING_FIELDS if field in metadata}


def git_state(repo_dir: Path | str) -> dict[str, Any]:
    repo_dir = Path(repo_dir).resolve()

    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=repo_dir, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            return proc.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty_worktree": bool(status) if status is not None else None,
    }


def software_versions(packages: Iterable[str] = ()) -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def build_provenance(
    *,
    encoded: Path | str,
    labels: Path | str,
    label_column: str,
    analysis_parameters: dict[str, Any] | None = None,
    analysis_seeds: dict[str, Any] | None = None,
    checkpoint: str | None = None,
    repo_dir: Path | str | None = None,
) -> dict[str, Any]:
    training_metadata, metadata_path = find_training_metadata(encoded)
    primary_metadata = read_json(Path(encoded).expanduser().resolve().parent / "metadata.json")
    if primary_metadata.get("seed") is not None:
        training_seed_status = "recorded in metadata.json"
    elif training_metadata.get("seed") is not None:
        training_seed_status = "recovered from checkpoint or parent metadata"
    else:
        training_seed_status = "not recorded"
    return {
        "schema_version": 1,
        "label_column": label_column,
        "checkpoint": checkpoint or training_metadata.get("checkpoint"),
        "inputs": {
            "encoded": file_record(encoded),
            "labels": file_record(labels),
            "training_metadata_path": metadata_path,
        },
        "training_parameters": select_training_fields(training_metadata),
        "analysis_parameters": analysis_parameters or {},
        "seeds": {
            "training": training_metadata.get("seed"),
            **(analysis_seeds or {}),
        },
        "seed_metadata": {
            "training_seed_status": training_seed_status,
            "training_metadata_sources": metadata_path,
        },
        "software": software_versions(
            ["numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers"]
        ),
        "repository": git_state(repo_dir or Path(__file__).resolve().parent),
    }


def json_safe(value: Any) -> Any:
    """Recursively turn pandas/numpy-like scalars and non-finite floats into JSON values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def fmt(value: Any, digits: int = 4, missing: str = "not available") -> str:
    if value is None:
        return missing
    try:
        if isinstance(value, float) and math.isnan(value):
            return missing
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def technical_markdown(provenance: dict[str, Any], heading_level: int = 2) -> list[str]:
    """Render a compact, reproducibility-oriented provenance section."""
    h = "#" * heading_level
    training = provenance.get("training_parameters", {})
    analysis = provenance.get("analysis_parameters", {})
    seeds = provenance.get("seeds", {})
    inputs = provenance.get("inputs", {})
    repo = provenance.get("repository", {})
    software = provenance.get("software", {})

    lines = [f"{h} Technical provenance", ""]
    if training:
        lines.extend([f"{h}# Training and encoding", ""])
        for key in TRAINING_FIELDS:
            if key in training:
                lines.append(f"- `{key}`: `{fmt(training[key])}`")
    else:
        lines.append("- Training metadata: not available (no discoverable `metadata.json`).")

    lines.extend(["", f"{h}# Analysis", ""])
    lines.append(f"- Label column: `{fmt(provenance.get('label_column'))}`")
    lines.append(f"- Checkpoint: `{fmt(provenance.get('checkpoint'))}`")
    for key, value in analysis.items():
        lines.append(f"- `{key}`: `{fmt(value)}`")

    lines.extend(["", f"{h}# Seeds", ""])
    if seeds:
        for key, value in seeds.items():
            lines.append(f"- `{key}`: `{value if value is not None else 'not recorded'}`")
    else:
        lines.append("- No stochastic seed was recorded for the available artifacts.")

    lines.extend(["", f"{h}# Inputs and environment", ""])
    for name in ("encoded", "labels"):
        record = inputs.get(name, {})
        if record:
            lines.append(
                f"- {name}: `{record.get('path')}`  "
                f"(SHA-256 `{record.get('sha256') or 'not available'}`)"
            )
    if repo:
        lines.append(
            f"- Repository commit: `{fmt(repo.get('commit'))}`; "
            f"dirty worktree at analysis time: `{fmt(repo.get('dirty_worktree'))}`"
        )
    if software:
        version_text = ", ".join(f"{k}={v}" for k, v in software.items())
        lines.append(f"- Software: `{version_text}`")
    return lines
