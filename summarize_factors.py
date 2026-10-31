#!/usr/bin/env python3
"""Aggregate one label-specific analysis directory into factor evidence cards."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_reporting import (
    EXACT_MATCHING_ABLATION_NOTE,
    ablation_feedback,
    discover_analysis_plots,
    fmt,
    json_safe,
    plot_markdown,
    read_json,
    retained_performance_pct,
    technical_markdown,
    write_json,
)


EXPECTED_ARTIFACTS = {
    "marginals": "marginal_associations.csv",
    "mutual information": "mutual_information.csv",
    "label profiles": "label_profiles.csv",
    "PMI edges": "pmi_pmi_edges.csv",
    "Jaccard edges": "jaccard_edges.csv",
    "decision-tree importances": "decision_tree_feature_importances.csv",
    "FCA concepts": "fca_concepts.csv",
    "hierarchical clusters": "hierarchical_clusters.csv",
    "patterns": "pattern_analysis.csv",
    "codebook statistics": "code_entropy_global.csv",
    "holdout metrics": "holdout_accuracy_metrics.csv",
    "holdout ablations": "holdout_accuracy_ablation.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSV, JSON, and Markdown evidence summaries for one label column."
    )
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument(
        "--output_prefix", default="factor_summary",
        help="Output filename prefix inside --analysis_dir (default: factor_summary)",
    )
    parser.add_argument("--top_relations", type=int, default=3)
    parser.add_argument("--top_combinations", type=int, default=3)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def record_for(df: pd.DataFrame, column: str, value: Any) -> dict[str, Any]:
    if df.empty or column not in df.columns:
        return {}
    rows = df[df[column].astype(str) == str(value)]
    return rows.iloc[0].to_dict() if len(rows) else {}


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def percentage(value: Any, digits: int = 1) -> str:
    numeric = number(value)
    return f"{numeric:.{digits}f}%" if numeric is not None else "not available"


def enrich_ablation_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Backfill completeness and feedback in ablation rows from older analyses."""
    row = json_safe(raw)
    if row.get("classifier") == "pattern_majority":
        return {
            "classifier": "pattern_majority",
            "omitted_factor": row.get("omitted_factor"),
            "ablation_applicable": False,
            "accuracy_drop": None,
            "macro_f1_drop": None,
            "macro_f1_completeness_pct": None,
            "feedback": EXACT_MATCHING_ABLATION_NOTE,
        }
    row["ablation_applicable"] = True
    full_accuracy = number(row.get("full_test_accuracy"))
    ablated_accuracy = number(row.get("ablated_test_accuracy"))
    full_macro_f1 = number(row.get("full_macro_f1"))
    ablated_macro_f1 = number(row.get("ablated_macro_f1"))

    if number(row.get("accuracy_completeness_pct")) is None:
        if full_accuracy is not None and ablated_accuracy is not None:
            row["accuracy_completeness_pct"] = retained_performance_pct(
                full_accuracy, ablated_accuracy
            )
    if number(row.get("macro_f1_completeness_pct")) is None:
        if full_macro_f1 is not None and ablated_macro_f1 is not None:
            row["macro_f1_completeness_pct"] = retained_performance_pct(
                full_macro_f1, ablated_macro_f1
            )

    classifier = str(row.get("classifier", "model"))
    factor = str(row.get("omitted_factor", "factor"))
    if full_macro_f1 is not None and ablated_macro_f1 is not None:
        row["feedback"] = ablation_feedback(
            classifier, factor, full_macro_f1, ablated_macro_f1
        )
    elif not row.get("feedback"):
        drop = number(row.get("macro_f1_drop"))
        if drop is None:
            row["feedback"] = "Insufficient score data to interpret this ablation."
        elif drop < 0:
            row["feedback"] = (
                f"{classifier} ablating {factor}: macro-F1 drop {drop:.4f}; the negative "
                "drop means removal improved performance, suggesting model-specific "
                "redundancy or overfitting."
            )
        elif drop > 0:
            row["feedback"] = (
                f"{classifier} ablating {factor}: macro-F1 drop {drop:.4f}; performance "
                "fell, indicating a positive model-specific contribution."
            )
        else:
            row["feedback"] = (
                f"{classifier} ablating {factor}: macro-F1 was unchanged."
            )
    return row


def bit_number(name: str) -> int:
    try:
        return int(str(name).split("_")[-1])
    except ValueError:
        return 10**9


def contains_factor(value: Any, factor: str) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    names = {item.strip() for item in str(value).split(",") if item.strip()}
    return factor in names


def contains_index(value: Any, index: int) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    values = {item.strip() for item in str(value).split(",") if item.strip()}
    return str(index) in values


def pair_relations(
    df: pd.DataFrame,
    factor: str,
    a_col: str,
    b_col: str,
    value_col: str,
    *,
    largest: bool,
    n: int,
) -> list[dict[str, Any]]:
    if df.empty or not {a_col, b_col, value_col}.issubset(df.columns):
        return []
    rows = df[(df[a_col] == factor) | (df[b_col] == factor)].copy()
    rows[value_col] = pd.to_numeric(rows[value_col], errors="coerce")
    rows = rows.dropna(subset=[value_col]).sort_values(value_col, ascending=not largest).head(n)
    result = []
    for _, row in rows.iterrows():
        partner = row[b_col] if row[a_col] == factor else row[a_col]
        result.append({"factor": str(partner), "value": float(row[value_col])})
    return result


def strength_label(value: float | None) -> str:
    if value is None:
        return "not available"
    if value >= 0.30:
        return "strong"
    if value >= 0.10:
        return "weak-to-moderate"
    return "negligible"


def role_explanation(role: str) -> str:
    explanations = {
        "silent": "active in less than 2% of documents",
        "universal": (
            "active in more than 50% of documents; this pipeline term means broad, "
            "not literally active everywhere"
        ),
        "discriminative": "not broad or silent, with Cramér's V above 0.30",
        "modifier": "not broad or silent, with Cramér's V at or below 0.30",
    }
    return explanations.get(str(role), "no standard role definition was available")


def ablation_assessment(rows: list[dict[str, Any]]) -> tuple[str, float | None]:
    drops = [
        number(row.get("macro_f1_drop"))
        for row in rows
        if row.get("ablation_applicable", True)
    ]
    drops = [value for value in drops if value is not None]
    if not drops:
        return "not available", None
    best = max(drops)
    if best >= 0.02:
        return "strong unique predictive contribution", best
    if best >= 0.005:
        return "moderate unique predictive contribution", best
    if best <= -0.005:
        return "potentially harmful or overfitting", best
    return "small or redundant predictive contribution", best


def relation_text(relations: list[dict[str, Any]], signed: bool = False) -> str:
    if not relations:
        return "not available"
    pattern = "+.3f" if signed else ".3f"
    return ", ".join(f"{r['factor']} ({r['value']:{pattern}})" for r in relations)


def combination_records(
    df: pd.DataFrame, factor: str, factor_idx: int, limit: int
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    if "shared_factor_names" in df.columns:
        rows = df[df["shared_factor_names"].apply(lambda v: contains_factor(v, factor))].copy()
        fields = [
            "shared_factor_names", "n_docs", "dominant", "dominant_pct", "entropy"
        ]
    elif "active_factors" in df.columns:
        rows = df[df["active_factors"].apply(lambda v: contains_index(v, factor_idx))].copy()
        fields = ["active_factors", "total", "dominant", "dominant_pct", "entropy"]
    else:
        return []
    if "dominant_pct" in rows.columns:
        rows["dominant_pct"] = pd.to_numeric(rows["dominant_pct"], errors="coerce")
    support = "n_docs" if "n_docs" in rows.columns else "total"
    if support in rows.columns:
        rows[support] = pd.to_numeric(rows[support], errors="coerce")
    rows = rows.sort_values(
        [c for c in ["dominant_pct", support] if c in rows.columns],
        ascending=False,
    ).head(limit)
    return [
        {key: json_safe(row.get(key)) for key in fields if key in rows.columns}
        for _, row in rows.iterrows()
    ]


def hierarchical_records(df: pd.DataFrame, factor_idx: int, limit: int) -> list[dict[str, Any]]:
    if df.empty or "core_bits" not in df.columns:
        return []
    rows = df[df["core_bits"].apply(lambda value: contains_index(value, factor_idx))].copy()
    rows = rows.sort_values(
        [c for c in ["dominant_pct", "n_docs"] if c in rows.columns], ascending=False
    ).head(limit)
    fields = ["cluster", "n_patterns", "n_docs", "dominant", "dominant_pct", "core_bits"]
    return [
        {key: json_safe(row.get(key)) for key in fields if key in rows.columns}
        for _, row in rows.iterrows()
    ]


def merge_holdout_provenance(provenance: dict[str, Any], analysis_dir: Path) -> None:
    metadata_path = analysis_dir / "holdout_accuracy_metadata.json"
    if not metadata_path.is_file():
        metadata_path = next(iter(sorted(analysis_dir.glob("*_metadata.json"))), metadata_path)
    holdout = read_json(metadata_path)
    if not holdout:
        return
    for key, value in holdout.get("seeds", {}).items():
        if value is not None:
            provenance.setdefault("seeds", {})[key] = value
    for key, value in holdout.get("analysis_parameters", {}).items():
        provenance.setdefault("analysis_parameters", {})[f"holdout.{key}"] = value
    if "split" in holdout:
        provenance["holdout_split"] = holdout["split"]


def load_artifacts(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    def artifact(exact: str, pattern: str | None = None) -> Path:
        path = analysis_dir / exact
        if path.is_file() or pattern is None:
            return path
        candidates = sorted(analysis_dir.glob(pattern))
        return candidates[0] if candidates else path

    return {
        "marginals": read_csv(artifact("marginal_associations.csv")),
        "mi": read_csv(artifact("mutual_information.csv")),
        "profiles": read_csv(artifact("label_profiles.csv")),
        "pmi": read_csv(artifact("pmi_pmi_edges.csv")),
        "jaccard": read_csv(artifact("jaccard_edges.csv")),
        "tree": read_csv(artifact("decision_tree_feature_importances.csv")),
        "fca": read_csv(artifact("fca_concepts.csv")),
        "hierarchical": read_csv(artifact("hierarchical_clusters.csv")),
        "patterns": read_csv(artifact("pattern_analysis.csv")),
        "code": read_csv(artifact("code_entropy_global.csv")),
        "holdout_metrics": read_csv(artifact("holdout_accuracy_metrics.csv", "*_metrics.csv")),
        "ablation": read_csv(artifact("holdout_accuracy_ablation.csv", "*_ablation.csv")),
    }


def build_factor_record(
    factor: str, data: dict[str, pd.DataFrame], top_relations: int, top_combinations: int
) -> dict[str, Any]:
    idx = bit_number(factor)
    marginal = record_for(data["marginals"], "bit", factor)
    mi = record_for(data["mi"], "bit", factor)
    profile = record_for(data["profiles"], "factor", factor)
    tree = record_for(data["tree"], "factor", factor)

    pmi_positive = pair_relations(
        data["pmi"], factor, "name_i", "name_j", "pmi",
        largest=True, n=top_relations,
    )
    pmi_negative = pair_relations(
        data["pmi"], factor, "name_i", "name_j", "pmi",
        largest=False, n=top_relations,
    )
    pmi_positive = [relation for relation in pmi_positive if relation["value"] > 0]
    pmi_negative = [relation for relation in pmi_negative if relation["value"] < 0]
    jaccard = pair_relations(
        data["jaccard"], factor, "factor_a", "factor_b", "jaccard",
        largest=True, n=top_relations,
    )

    ablation_rows = []
    if not data["ablation"].empty and "omitted_factor" in data["ablation"].columns:
        matches = data["ablation"][data["ablation"]["omitted_factor"] == factor]
        ablation_rows = [
            enrich_ablation_row(row.to_dict()) for _, row in matches.iterrows()
        ]
        if not any(row.get("classifier") == "pattern_majority" for row in ablation_rows):
            ablation_rows.append({
                "classifier": "pattern_majority",
                "omitted_factor": factor,
                "ablation_applicable": False,
                "accuracy_drop": None,
                "macro_f1_drop": None,
                "macro_f1_completeness_pct": None,
                "feedback": EXACT_MATCHING_ABLATION_NOTE,
            })
    predictive_assessment, max_macro_drop = ablation_assessment(ablation_rows)
    completeness_values = [
        number(row.get("macro_f1_completeness_pct"))
        for row in ablation_rows
        if row.get("ablation_applicable", True)
    ]
    completeness_values = [value for value in completeness_values if value is not None]
    minimum_completeness = min(completeness_values) if completeness_values else None

    v = number(marginal.get("cramers_v"))
    label_explains = number(mi.get("label_explains_factor"))
    activation = number(marginal.get("marginal", mi.get("marginal")))
    mi_bits = number(mi.get("MI_bits"))
    if label_explains is None and mi_bits is not None and activation is not None:
        probabilities = [activation, 1.0 - activation]
        h_factor = -sum(p * math.log2(p) for p in probabilities if p > 0)
        label_explains = mi_bits / h_factor if h_factor > 0 else 0.0
    dominant = marginal.get("dominant_cat", profile.get("dominant_class"))
    role = marginal.get("role", profile.get("role", "not available"))
    max_jaccard = jaccard[0]["value"] if jaccard else None
    category_specificity = []
    for column, value in mi.items():
        if str(column).startswith("PMI_") and number(value) is not None:
            category_specificity.append({
                "category": str(column)[4:], "pointwise_mi": number(value)
            })
    category_specificity.sort(key=lambda item: item["pointwise_mi"], reverse=True)

    flags = []
    nearly_constant = (
        activation is not None and (activation < 0.02 or activation > 0.98)
    )
    if nearly_constant:
        flags.append("near-constant factor; label-specific interpretation is unreliable")
    elif role == "universal":
        flags.append(
            "broad factor active in more than half of documents; interpret its category "
            "association rather than the role name alone"
        )
    if v is not None and v >= 0.30 and max_macro_drop is not None and max_macro_drop < 0.005:
        flags.append("strong marginal association but little unique predictive contribution")
    if max_jaccard is not None and max_jaccard >= 0.70:
        flags.append("high-overlap partner suggests redundancy")
    rf_drop = next(
        (number(row.get("macro_f1_drop")) for row in ablation_rows
         if row.get("classifier") == "random_forest"),
        None,
    )
    if v is not None and v < 0.10 and rf_drop is not None and rf_drop >= 0.01:
        flags.append("weak marginal signal but possible nonlinear interaction contribution")
    negative_ablation_models = [
        str(row.get("classifier"))
        for row in ablation_rows
        if row.get("ablation_applicable", True)
        and number(row.get("macro_f1_drop")) is not None
        and number(row.get("macro_f1_drop")) < 0
    ]
    if negative_ablation_models:
        flags.append(
            "negative ablation for " + ", ".join(negative_ablation_models)
            + ": removing the factor improved that model, suggesting model-specific "
            "redundancy or overfitting"
        )
    if not flags:
        flags.append("no major marginal/predictive contradiction detected")

    if nearly_constant:
        confidence = "low"
        if activation is not None and activation < 0.02:
            confidence_reason = (
                "the factor is active in less than 2% of documents, leaving too few active "
                "examples for a reliable label-specific interpretation"
            )
        else:
            confidence_reason = (
                "the factor is active in more than 98% of documents, leaving too few inactive "
                "examples for a reliable label-specific interpretation"
            )
    elif v is not None and v >= 0.30 and label_explains is not None and label_explains >= 0.10:
        confidence = "high"
        confidence_reason = (
            "both association checks pass the strong thresholds: Cramér's V >= 0.30 "
            "and the label explains at least 10% of factor entropy"
        )
    elif (v is not None and v >= 0.10) or (label_explains is not None and label_explains >= 0.05):
        confidence = "moderate"
        confidence_reason = (
            "at least one association check passes the moderate threshold: Cramér's V "
            ">= 0.10 or the label explains at least 5% of factor entropy"
        )
    else:
        confidence = "low"
        confidence_reason = (
            "neither association check passes the moderate threshold: Cramér's V >= 0.10 "
            "or the label explaining at least 5% of factor entropy"
        )

    return json_safe({
        "factor": factor,
        "factor_index": idx,
        "activation_rate": activation,
        "n_active": marginal.get("n_active"),
        "marginal_role": role,
        "dominant_category": dominant,
        "dominant_category_pct": number(marginal.get("dominant_pct")),
        "cramers_v": v,
        "association_strength": strength_label(v),
        "chi_squared_p_value": number(marginal.get("pval")),
        "mutual_information_bits": mi_bits,
        "factor_explains_label": number(mi.get("NMI")),
        "label_explains_factor": label_explains,
        "factor_entropy_bits": number(mi.get("H_factor_bits")),
        "profile_role": profile.get("role"),
        "profile_dominant_class": profile.get("dominant_class"),
        "profile_gap_to_second": number(profile.get("gap_to_second")),
        "profile_std_across_classes": number(profile.get("std_across_classes")),
        "decision_tree_importance": number(tree.get("importance")),
        "predictive_assessment": predictive_assessment,
        "max_macro_f1_drop": max_macro_drop,
        "minimum_macro_f1_completeness_pct": minimum_completeness,
        "holdout_ablation": ablation_rows,
        "top_positive_pmi_partners": pmi_positive,
        "top_negative_pmi_partners": pmi_negative,
        "top_jaccard_partners": jaccard,
        "category_specificity_by_pointwise_mi": category_specificity,
        "top_fca_combinations": combination_records(
            data["fca"], factor, idx, top_combinations
        ),
        "top_exact_patterns": combination_records(
            data["patterns"], factor, idx, top_combinations
        ),
        "hierarchical_clusters_where_core": hierarchical_records(
            data["hierarchical"], idx, top_combinations
        ),
        "evidence_flags": flags,
        "interpretation_confidence_for_this_label": confidence,
        "interpretation_confidence_reason": confidence_reason,
    })


def flatten_factor(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "factor": record["factor"],
        "factor_index": record["factor_index"],
        "activation_rate": record["activation_rate"],
        "marginal_role": record["marginal_role"],
        "dominant_category": record["dominant_category"],
        "dominant_category_pct": record["dominant_category_pct"],
        "cramers_v": record["cramers_v"],
        "association_strength": record["association_strength"],
        "mutual_information_bits": record["mutual_information_bits"],
        "factor_explains_label": record["factor_explains_label"],
        "label_explains_factor": record["label_explains_factor"],
        "decision_tree_importance": record["decision_tree_importance"],
        "max_macro_f1_drop": record["max_macro_f1_drop"],
        "minimum_macro_f1_completeness_pct": record[
            "minimum_macro_f1_completeness_pct"
        ],
        "predictive_assessment": record["predictive_assessment"],
        "top_positive_pmi_partners": relation_text(record["top_positive_pmi_partners"], True),
        "top_negative_pmi_partners": relation_text(record["top_negative_pmi_partners"], True),
        "top_jaccard_partners": relation_text(record["top_jaccard_partners"]),
        "evidence_flags": "; ".join(record["evidence_flags"]),
        "ablation_feedback": " ".join(
            str(row.get("feedback")) for row in record["holdout_ablation"]
            if row.get("feedback")
        ),
        "interpretation_confidence_for_this_label": record[
            "interpretation_confidence_for_this_label"
        ],
        "interpretation_confidence_reason": record["interpretation_confidence_reason"],
    }


def build_overall_conclusion(
    records: list[dict[str, Any]], label_column: str, coalitions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize label-relative evidence without calling low-confidence factors useless."""
    counts = {level: 0 for level in ("high", "moderate", "low")}
    for record in records:
        level = record.get("interpretation_confidence_for_this_label", "low")
        counts[level] = counts.get(level, 0) + 1

    associated = [
        record for record in records
        if record.get("interpretation_confidence_for_this_label") in {"high", "moderate"}
    ]
    association_ranked = sorted(
        [record for record in records if number(record.get("label_explains_factor")) is not None],
        key=lambda record: number(record.get("label_explains_factor")) or 0.0,
        reverse=True,
    )[:3]
    predictive_ranked = sorted(
        [
            record for record in records
            if number(record.get("max_macro_f1_drop")) is not None
            and number(record.get("max_macro_f1_drop")) > 0
        ],
        key=lambda record: number(record.get("max_macro_f1_drop")) or 0.0,
        reverse=True,
    )[:3]
    broad = [
        record["factor"]
        for record in records
        if record.get("marginal_role") == "universal"
    ]
    internally_structured = [
        record["factor"]
        for record in records
        if record.get("top_positive_pmi_partners")
        or record.get("top_negative_pmi_partners")
        or record.get("top_jaccard_partners")
        or record.get("top_fca_combinations")
        or record.get("top_exact_patterns")
        or record.get("hierarchical_clusters_where_core")
    ]
    negative_pairs: dict[tuple[str, str], float] = {}
    for record in records:
        factor = record["factor"]
        for relation in record.get("top_negative_pmi_partners", []):
            partner = str(relation["factor"])
            pair = tuple(sorted((factor, partner)))
            value = number(relation.get("value"))
            if value is not None and (
                pair not in negative_pairs or value < negative_pairs[pair]
            ):
                negative_pairs[pair] = value
    strongest_exclusions = sorted(negative_pairs.items(), key=lambda item: item[1])[:5]

    sentences = [
        f"{len(associated)} of {len(records)} factors have moderate or high evidence of a "
        f"stable association with `{label_column}` ({counts.get('high', 0)} high, "
        f"{counts.get('moderate', 0)} moderate, {counts.get('low', 0)} low)."
    ]
    if associated:
        sentences.append(
            "The report therefore does not indicate that every factor is unreliable. "
            "A low rating only means that this particular label column does not provide "
            "enough evidence under the stated rules; the factor may encode another, "
            "unmeasured distinction."
        )
    else:
        sentences.append(
            "None crosses the current label-specific thresholds. This does not show that the "
            "factors are intrinsically unreliable; it shows that this label column does not "
            "explain them strongly enough for a confident interpretation."
        )
    if internally_structured:
        sentences.append(
            f"Independently of the selected label, {len(internally_structured)} of "
            f"{len(records)} factors participate in observed co-activation, mutual-exclusion, "
            "overlap, pattern, or cluster structure. This is evidence that the code is "
            "internally organized even when a factor's label-specific confidence is low."
        )
    if strongest_exclusions:
        pairs = ", ".join(
            f"{left}–{right} (PMI {value:+.3f})"
            for (left, right), value in strongest_exclusions
        )
        sentences.append(
            "The strongest observed negative-PMI relationships are " + pairs
            + ". Negative PMI means the pair activates together less often than chance "
            "(statistical mutual exclusion), not that either factor is unreliable."
        )
    if coalitions:
        sentences.append(
            f"The positive-PMI network also contains {len(coalitions)} multi-factor "
            "coalition(s), showing groups that activate together more often than chance."
        )
    if association_ranked:
        strongest = ", ".join(
            f"{record['factor']} ({record.get('dominant_category')}, label explains "
            f"{(number(record.get('label_explains_factor')) or 0.0) * 100:.1f}% of factor "
            "uncertainty)"
            for record in association_ranked
        )
        sentences.append(f"The clearest label associations are {strongest}.")
    if predictive_ranked:
        strongest = ", ".join(
            f"{record['factor']} ({(number(record.get('max_macro_f1_drop')) or 0.0) * 100:.2f} "
            "macro-F1 percentage points)"
            for record in predictive_ranked
        )
        sentences.append(
            "The largest positive held-out ablation effects among applicable classifiers are "
            f"{strongest}. These effects are model-specific, not semantic definitions."
        )
    if broad:
        sentences.append(
            f"{len(broad)} factor(s) are labelled `universal` only because they activate in "
            "more than half of documents. Broad activation alone no longer lowers confidence; "
            "only near-constant activation below 2% or above 98% does."
        )
    return json_safe({
        "confidence_counts": counts,
        "n_moderate_or_high": len(associated),
        "top_label_associations": [record["factor"] for record in association_ranked],
        "top_positive_ablation_factors": [record["factor"] for record in predictive_ranked],
        "n_internally_structured_factors": len(internally_structured),
        "strongest_mutual_exclusions": [
            {"factor_a": pair[0], "factor_b": pair[1], "pmi": value}
            for pair, value in strongest_exclusions
        ],
        "n_positive_pmi_coalitions": len(coalitions),
        "summary_sentences": sentences,
    })


def unique_relation_pairs(
    records: list[dict[str, Any]], relation_key: str, *, limit: int,
    minimum: float | None = None, maximum: float | None = None,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """Collect one copy of each pairwise relation from per-factor records."""
    pairs: dict[tuple[str, str], float] = {}
    for record in records:
        factor = str(record["factor"])
        for relation in record.get(relation_key, []):
            partner = str(relation.get("factor"))
            value = number(relation.get("value"))
            if value is None:
                continue
            if minimum is not None and value < minimum:
                continue
            if maximum is not None and value > maximum:
                continue
            pair = tuple(sorted((factor, partner), key=bit_number))
            if pair not in pairs or (
                ascending and value < pairs[pair]
            ) or (
                not ascending and value > pairs[pair]
            ):
                pairs[pair] = value
    ordered = sorted(pairs.items(), key=lambda item: item[1], reverse=not ascending)
    return [
        {"factor_a": pair[0], "factor_b": pair[1], "value": value}
        for pair, value in ordered[:limit]
    ]


def build_executive_summary(
    records: list[dict[str, Any]], label_column: str,
    coalitions: list[dict[str, Any]], holdout_metrics: list[dict[str, Any]],
    conclusion: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact, evidence-typed summary for the top of the report."""
    label_associated = sorted(
        [r for r in records if number(r.get("label_explains_factor")) is not None],
        key=lambda r: number(r.get("label_explains_factor")) or 0.0,
        reverse=True,
    )[:5]
    all_discriminative = sorted(
        [r for r in records if r.get("marginal_role") == "discriminative"],
        key=lambda r: number(r.get("cramers_v")) or 0.0,
        reverse=True,
    )
    discriminative = all_discriminative[:5]
    all_predictive = sorted(
        [
            r for r in records
            if (number(r.get("max_macro_f1_drop")) or 0.0) >= 0.005
        ],
        key=lambda r: number(r.get("max_macro_f1_drop")) or 0.0,
        reverse=True,
    )
    predictive = all_predictive[:5]
    exclusions = unique_relation_pairs(
        records, "top_negative_pmi_partners", limit=5, maximum=-1e-12,
        ascending=True,
    )
    redundancy = unique_relation_pairs(
        records, "top_jaccard_partners", limit=5, minimum=0.70,
    )
    valid_holdout = [
        row for row in holdout_metrics
        if number(row.get("macro_f1")) is not None
    ]
    best_macro = max(
        valid_holdout, key=lambda row: number(row.get("macro_f1")) or -1.0,
        default=None,
    )
    best_accuracy = max(
        [row for row in holdout_metrics if number(row.get("test_accuracy")) is not None],
        key=lambda row: number(row.get("test_accuracy")) or -1.0,
        default=None,
    )
    return json_safe({
        "label_column": label_column,
        "confidence_counts": conclusion.get("confidence_counts", {}),
        "n_moderate_or_high": conclusion.get("n_moderate_or_high", 0),
        "n_factors": len(records),
        "n_discriminative_factors": len(all_discriminative),
        "n_positive_ablation_factors": len(all_predictive),
        "top_label_associated_factors": [
            {
                "factor": r["factor"],
                "dominant_category": r.get("dominant_category"),
                "label_explains_factor": r.get("label_explains_factor"),
                "cramers_v": r.get("cramers_v"),
                "confidence": r.get("interpretation_confidence_for_this_label"),
            }
            for r in label_associated
        ],
        "top_discriminative_factors": [
            {
                "factor": r["factor"],
                "dominant_category": r.get("dominant_category"),
                "cramers_v": r.get("cramers_v"),
            }
            for r in discriminative
        ],
        "top_positive_ablation_factors": [
            {
                "factor": r["factor"],
                "max_macro_f1_drop": r.get("max_macro_f1_drop"),
                "predictive_assessment": r.get("predictive_assessment"),
            }
            for r in predictive
        ],
        "strongest_mutual_exclusions": exclusions,
        "positive_pmi_coalitions": coalitions[:3],
        "high_overlap_pairs": redundancy,
        "best_holdout_by_macro_f1": best_macro,
        "best_holdout_by_accuracy": best_accuracy,
    })


def executive_summary_markdown(summary: dict[str, Any]) -> list[str]:
    """Render the compact summary while keeping evidence types separate."""
    associated = summary["top_label_associated_factors"]
    discriminative = summary["top_discriminative_factors"]
    predictive = summary["top_positive_ablation_factors"]
    exclusions = summary["strongest_mutual_exclusions"]
    coalitions = summary["positive_pmi_coalitions"]
    redundant = summary["high_overlap_pairs"]
    counts = summary["confidence_counts"]

    association_text = ", ".join(
        f"`{r['factor']}` → {r.get('dominant_category')} "
        f"({(number(r.get('label_explains_factor')) or 0.0) * 100:.1f}% explained; "
        f"V={fmt(r.get('cramers_v'))}; {r.get('confidence')})"
        for r in associated
    ) or "none available"
    discriminative_text = ", ".join(
        f"`{r['factor']}` → {r.get('dominant_category')} (V={fmt(r.get('cramers_v'))})"
        for r in discriminative
    ) or "none under the report's V > 0.30 role rule"
    predictive_text = ", ".join(
        f"`{r['factor']}` ({(number(r.get('max_macro_f1_drop')) or 0.0) * 100:.2f} "
        "macro-F1 points)"
        for r in predictive
    ) or "none reached the 0.5 macro-F1-point reporting threshold"
    exclusion_text = ", ".join(
        f"`{r['factor_a']}`–`{r['factor_b']}` (PMI {r['value']:+.3f})"
        for r in exclusions
    ) or "none available"
    coalition_descriptions = []
    for coalition in coalitions:
        members = coalition["factors"]
        if len(members) <= 6:
            coalition_descriptions.append("`" + ", ".join(members) + "`")
            continue
        strongest = ", ".join(
            f"{edge['factor_a']}–{edge['factor_b']} ({edge['pmi']:+.3f})"
            for edge in coalition.get("strongest_internal_edges", [])[:3]
        ) or "not available"
        coalition_descriptions.append(
            f"{len(members)} bits (strongest links: {strongest})"
        )
    coalition_text = "; ".join(coalition_descriptions) or (
        "none above the configured positive-PMI threshold"
    )
    redundancy_text = ", ".join(
        f"`{r['factor_a']}`–`{r['factor_b']}` (Jaccard {r['value']:.3f})"
        for r in redundant
    ) or "none with Jaccard ≥ 0.70"

    best_macro = summary.get("best_holdout_by_macro_f1")
    best_accuracy = summary.get("best_holdout_by_accuracy")
    if best_macro:
        prediction_text = (
            f"best macro-F1 is {fmt(best_macro.get('macro_f1'))} with "
            f"`{best_macro.get('classifier')}`"
        )
        if best_accuracy:
            prediction_text += (
                f"; best accuracy is {fmt(best_accuracy.get('test_accuracy'))} with "
                f"`{best_accuracy.get('classifier')}` (majority baseline "
                f"{fmt(best_accuracy.get('baseline_accuracy'))})"
            )
    else:
        prediction_text = "held-out metrics are not available"

    return [
        "## TL;DR",
        "",
        f"- **Overall label evidence:** {summary['n_moderate_or_high']} of "
        f"{summary['n_factors']} bits have moderate/high evidence for "
        f"`{summary['label_column']}` ({counts.get('high', 0)} high, "
        f"{counts.get('moderate', 0)} moderate, {counts.get('low', 0)} low).",
        f"- **Most label-associated bits:** {association_text}.",
        f"- **Discriminative bits:** {summary['n_discriminative_factors']} total; "
        f"strongest: {discriminative_text}.",
        f"- **Held-out model-specific importance:** "
        f"{summary['n_positive_ablation_factors']} bits reach at least 0.5 macro-F1 points "
        f"in one applicable model; strongest: {predictive_text}.",
        f"- **Strongest mutual exclusions:** {exclusion_text}.",
        f"- **Largest co-activating coalitions:** {coalition_text}.",
        f"- **Potentially redundant high-overlap pairs:** {redundancy_text}.",
        f"- **Held-out prediction:** {prediction_text}.",
        "- **Reading caution:** association, ablation importance, co-activation, and "
        "redundancy are different evidence types; none alone supplies a semantic name or "
        "a causal interpretation for a bit.",
    ]


def build_coalitions(
    factors: list[str], records: list[dict[str, Any]], pmi_edges: pd.DataFrame, threshold: float
) -> list[dict[str, Any]]:
    if pmi_edges.empty or not {"name_i", "name_j", "pmi"}.issubset(pmi_edges.columns):
        return []
    parent = {factor: factor for factor in factors}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        parent[find(left)] = find(right)

    edges = pmi_edges.copy()
    edges["pmi"] = pd.to_numeric(edges["pmi"], errors="coerce")
    strong = edges[edges["pmi"] >= threshold]
    for _, row in strong.iterrows():
        left, right = str(row["name_i"]), str(row["name_j"])
        if left in parent and right in parent:
            union(left, right)

    groups: dict[str, list[str]] = {}
    for factor in factors:
        groups.setdefault(find(factor), []).append(factor)
    record_by_factor = {record["factor"]: record for record in records}
    result = []
    for members in sorted(groups.values(), key=lambda group: (-len(group), bit_number(group[0]))):
        if len(members) < 2:
            continue
        member_set = set(members)
        internal = strong[
            strong["name_i"].isin(member_set) & strong["name_j"].isin(member_set)
        ].sort_values("pmi", ascending=False)
        category_counts: dict[str, int] = {}
        for member in members:
            category = record_by_factor.get(member, {}).get("dominant_category")
            if category is not None:
                category_counts[str(category)] = category_counts.get(str(category), 0) + 1
        result.append({
            "factors": sorted(members, key=bit_number),
            "dominant_category_counts": category_counts,
            "strongest_internal_edges": [
                {
                    "factor_a": str(row["name_i"]),
                    "factor_b": str(row["name_j"]),
                    "pmi": float(row["pmi"]),
                }
                for _, row in internal.head(5).iterrows()
            ],
        })
    return result


def factor_markdown(record: dict[str, Any]) -> list[str]:
    activation = record.get("activation_rate")
    activation_text = f"{activation * 100:.1f}%" if activation is not None else "not available"
    label_explains = record.get("label_explains_factor")
    factor_explains = record.get("factor_explains_label")
    max_drop = record.get("max_macro_f1_drop")
    minimum_completeness = record.get("minimum_macro_f1_completeness_pct")
    lines = [f"### {record['factor']}", ""]
    lines.append(
        f"Active in **{activation_text}** of documents; marginal role: "
        f"**{record.get('marginal_role', 'not available')}** "
        f"({role_explanation(record.get('marginal_role', 'not available'))})."
    )
    lines.append(
        f"Its main observed category is **{record.get('dominant_category', 'not available')}** "
        f"({fmt(record.get('dominant_category_pct'), 1)}% of active documents). "
        f"Marginal association is **{record.get('association_strength')}** "
        f"(Cramér's V={fmt(record.get('cramers_v'))}, where 0 means no association and "
        f"1 means a perfect association; MI={fmt(record.get('mutual_information_bits'), 6)} "
        "bits, where 0 means statistical independence and larger values mean more shared "
        "information)."
    )
    lines.append(
        f"The factor explains {fmt(factor_explains, 4)} of label uncertainty; the label "
        f"explains {fmt(label_explains, 4)} of factor uncertainty. These are proportions "
        "from 0 to 1: larger values mean that knowing one variable removes more uncertainty "
        "about the other."
    )
    lines.append(
        f"Predictive evidence: **{record.get('predictive_assessment')}** "
        f"(largest macro-F1 drop after removal={fmt(max_drop, 4)}). "
        f"Minimum ablation completeness={percentage(minimum_completeness)}. "
        f"Decision-tree importance={fmt(record.get('decision_tree_importance'), 4)}."
    )
    lines.append(
        "Positive-PMI partners (factors that activate together more often than chance): "
        f"{relation_text(record['top_positive_pmi_partners'], True)}.  "
        "Negative-PMI partners (factors that activate together less often than chance; "
        "statistical mutual exclusion): "
        f"{relation_text(record['top_negative_pmi_partners'], True)}.  "
        "Highest-overlap partners (largest share of jointly active documents among documents "
        "where either factor is active): "
        f"{relation_text(record['top_jaccard_partners'])}."
    )
    lines.append("Evidence assessment: " + "; ".join(record["evidence_flags"]) + ".")
    lines.append(
        "Rule-based interpretation confidence relative to this label column: "
        f"**{record['interpretation_confidence_for_this_label']}** "
        f"({record['interpretation_confidence_reason']}). This is a transparent heuristic, "
        "not a probability or a statistical significance test."
    )

    if record["holdout_ablation"]:
        lines.extend([
            "",
            "Ablation removes this factor, retrains the classifier, and compares it with the "
            "full model on the same held-out documents. Drop is `full - ablated`: positive "
            "means removal hurt performance; negative means removal improved performance. "
            "Completeness is the percentage of the full model's macro-F1 retained.",
            "",
            "| Classifier | Accuracy drop | Macro-F1 drop | Completeness | Feedback |",
            "|---|---:|---:|---:|---|",
        ])
        for row in record["holdout_ablation"]:
            if not row.get("ablation_applicable", True):
                lines.append(
                    f"| {row.get('classifier')} | not applicable | not applicable | "
                    f"not applicable | {row.get('feedback')} |"
                )
            else:
                lines.append(
                    f"| {row.get('classifier')} | {fmt(row.get('accuracy_drop'), 4)} | "
                    f"{fmt(row.get('macro_f1_drop'), 4)} | "
                    f"{percentage(row.get('macro_f1_completeness_pct'))} | "
                    f"{row.get('feedback', 'not available')} |"
                )
    if record["top_fca_combinations"]:
        lines.extend(["", "Most label-pure FCA combinations containing this factor:", ""])
        for item in record["top_fca_combinations"]:
            lines.append(
                f"- `{item.get('shared_factor_names')}`: n={item.get('n_docs')}, "
                f"dominant `{item.get('dominant')}` ({fmt(item.get('dominant_pct'), 1)}%)."
            )
    if record["top_exact_patterns"]:
        lines.extend(["", "Most label-pure exact patterns containing this factor:", ""])
        for item in record["top_exact_patterns"]:
            lines.append(
                f"- Active indices `{item.get('active_factors')}`: n={item.get('total')}, "
                f"dominant `{item.get('dominant')}` ({fmt(item.get('dominant_pct'), 1)}%)."
            )
    if record["hierarchical_clusters_where_core"]:
        lines.extend(["", "Hierarchical clusters where this factor is a core bit:", ""])
        for item in record["hierarchical_clusters_where_core"]:
            lines.append(
                f"- Cluster {item.get('cluster')}: n={item.get('n_docs')}, dominant "
                f"`{item.get('dominant')}` ({fmt(item.get('dominant_pct'), 1)}%)."
            )
    return lines


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    if not analysis_dir.is_dir():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    data = load_artifacts(analysis_dir)
    provenance = read_json(analysis_dir / "analysis_manifest.json")
    if not provenance:
        provenance = {
            "schema_version": 1,
            "label_column": analysis_dir.name,
            "training_parameters": {},
            "analysis_parameters": {},
            "seeds": {
                "training": None,
                "decision_tree": None,
                "pmi_spring_layout": None,
                "holdout_split_and_classifiers": None,
            },
            "seed_metadata": {
                "status": "No analysis manifest was available; seed values were not inferred."
            },
            "inputs": {},
            "repository": {},
            "software": {},
        }
    merge_holdout_provenance(provenance, analysis_dir)

    factor_sources = [
        (data["marginals"], "bit"),
        (data["mi"], "bit"),
        (data["profiles"], "factor"),
        (data["tree"], "factor"),
    ]
    factors: set[str] = set()
    for frame, column in factor_sources:
        if not frame.empty and column in frame.columns:
            factors.update(frame[column].dropna().astype(str))
    if not factors:
        raise ValueError(
            f"No factor tables found in {analysis_dir}; expected at least "
            "marginal_associations.csv or mutual_information.csv."
        )
    ordered_factors = sorted(factors, key=bit_number)
    records = [
        build_factor_record(factor, data, args.top_relations, args.top_combinations)
        for factor in ordered_factors
    ]

    threshold = number(provenance.get("analysis_parameters", {}).get("pmi_threshold"))
    if threshold is None:
        threshold = 0.5
    coalitions = build_coalitions(ordered_factors, records, data["pmi"], threshold)
    label_column = str(provenance.get("label_column", analysis_dir.name))
    conclusion = build_overall_conclusion(records, label_column, coalitions)

    availability = {
        label: (analysis_dir / filename).is_file()
        for label, filename in EXPECTED_ARTIFACTS.items()
    }
    availability["holdout metrics"] = not data["holdout_metrics"].empty
    availability["holdout ablations"] = not data["ablation"].empty
    codebook = (
        json_safe(data["code"].iloc[0].to_dict()) if not data["code"].empty else None
    )
    holdout_metrics_frame = data["holdout_metrics"]
    holdout_metrics = [
        json_safe(row.to_dict()) for _, row in holdout_metrics_frame.iterrows()
    ]
    executive_summary = build_executive_summary(
        records, label_column, coalitions, holdout_metrics, conclusion
    )
    prefix = analysis_dir / args.output_prefix
    plots = discover_analysis_plots(analysis_dir, prefix.parent)
    payload = {
        "schema_version": 1,
        "label_column": label_column,
        "provenance": json_safe(provenance),
        "artifact_availability": availability,
        "codebook": codebook,
        "holdout_metrics": holdout_metrics,
        "excluded_ablations": {
            "pattern_majority": EXACT_MATCHING_ABLATION_NOTE,
        },
        "plots": plots,
        "executive_summary": executive_summary,
        "factors": records,
        "factor_coalitions": coalitions,
        "overall_conclusion": conclusion,
        "interpretation_scope": (
            "Statistical evidence relative to the selected label column only. "
            "No semantic names were generated from document text."
        ),
    }

    write_json(prefix.with_suffix(".json"), payload)
    pd.DataFrame([flatten_factor(record) for record in records]).to_csv(
        prefix.with_suffix(".csv"), index=False
    )

    lines = [
        f"# Binary factor evidence summary: {payload['label_column']}",
        "",
        "A factor is a learned yes/no feature: it is either active or inactive for each "
        "document. This report combines several checks to show what each factor may represent "
        "relative to the selected label. Association does not prove causation, and the report "
        "does not assign free-form semantic names from document text.",
        "",
        *executive_summary_markdown(executive_summary),
        "",
        *technical_markdown(provenance),
        "",
        "## Plain-language reading guide",
        "",
        "- **Active percentage and role**: how often the factor switches on. In this pipeline, "
        "`silent` means active in less than 2% of documents and `universal` means active in "
        "more than 50%. `Universal` means broad, not literally active in every document.",
        "- **Main category**: the most common label among documents where the factor is on. "
        "It is a description, not proof that the factor means that category.",
        "- **Cramér's V**: strength of association between factor activation and the label, "
        "from 0 (none) to 1 (perfect).",
        "- **Mutual information (MI)**: how much knowing the factor reduces uncertainty about "
        "the label, and vice versa. Zero means statistical independence; larger is stronger.",
        "- **PMI partners**: positive pairs switch on together more often than chance; negative "
        "pairs switch on together less often than chance (statistical mutual exclusion).",
        "- **Jaccard overlap**: among documents where either factor is on, the share where both "
        "are on. It ranges from 0 (no overlap) to 1 (identical activation sets).",
        "- **Ablation**: remove one factor, retrain a model, and measure the change on unseen "
        "documents. Positive drop means removal hurt; negative drop means removal helped.",
        "- **Interpretation confidence**: a rule-based summary of activation variability and "
        "association strength. It is not a probability and not a significance test; every "
        "factor card states the exact reason for its rating.",
        "",
        "## Evidence coverage",
        "",
    ]
    for label, available in availability.items():
        lines.append(f"- {label}: {'available' if available else 'not available'}")

    if plots:
        lines.extend([
            "",
            "## Analysis plots",
            "",
            "The following figures were generated by the analysis pipeline. Their descriptions "
            "state what is encoded visually and what kind of evidence the plot supports.",
            "",
            *plot_markdown(plots),
        ])

    if codebook:
        lines.extend([
            "",
            "## Codebook overview",
            "",
            f"- Documents: {fmt(codebook.get('n_documents'))}",
            f"- Observed patterns: {fmt(codebook.get('n_observed_patterns'))} of "
            f"{fmt(codebook.get('n_possible_patterns'))} possible",
            f"- Pattern efficiency: {fmt(codebook.get('efficiency_pct'), 1)}%",
            f"- Mean active factors per document: "
            f"{fmt(codebook.get('mean_active_factors_per_document'), 2)}",
        ])

    if holdout_metrics:
        lines.extend([
            "",
            "## Held-out prediction overview",
            "",
            "Ablation completeness is `100 × ablated macro-F1 / full macro-F1`: "
            "100% means removal retained all macro-F1, lower values mean the factor "
            "was uniquely useful, and values above 100% mean removal improved the model.",
            "",
            "| Classifier | Test accuracy | Macro-F1 | Baseline accuracy |",
            "|---|---:|---:|---:|",
        ])
        for row in holdout_metrics:
            lines.append(
                f"| {row.get('classifier')} | {fmt(row.get('test_accuracy'))} | "
                f"{fmt(row.get('macro_f1'))} | {fmt(row.get('baseline_accuracy'))} |"
            )

    if coalitions:
        lines.extend([
            "",
            "## Factor coalitions",
            "",
            f"Connected components formed by positive PMI edges at or above {threshold:.3f}.",
            "",
        ])
        for index, coalition in enumerate(coalitions, 1):
            categories = ", ".join(
                f"{category}: {count} factor(s)"
                for category, count in coalition["dominant_category_counts"].items()
            ) or "not available"
            strongest = ", ".join(
                f"{edge['factor_a']}–{edge['factor_b']} ({edge['pmi']:+.3f})"
                for edge in coalition["strongest_internal_edges"]
            )
            lines.append(
                f"- Coalition {index}: `{', '.join(coalition['factors'])}`. "
                f"Label evidence: {categories}. Strongest links: {strongest}."
            )

    lines.extend([
        "",
        "## Factor overview",
        "",
        "| Factor | Active | Main category | Cramér's V | Label explains factor | "
        "Max macro-F1 drop | Min completeness | Confidence |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ])
    for record in records:
        active = record.get("activation_rate")
        lines.append(
            f"| {record['factor']} | "
            f"{active * 100:.1f}%" if active is not None else f"| {record['factor']} | n/a"
        )
        # Replace the intentionally partial row above with the remaining cells.
        lines[-1] += (
            f" | {record.get('dominant_category')} | {fmt(record.get('cramers_v'))} | "
            f"{fmt(record.get('label_explains_factor'))} | "
            f"{fmt(record.get('max_macro_f1_drop'))} | "
            f"{percentage(record.get('minimum_macro_f1_completeness_pct'))} | "
            f"{record.get('interpretation_confidence_for_this_label')} |"
        )

    lines.extend(["", "## Factor evidence cards", ""])
    for record in records:
        lines.extend(factor_markdown(record))
        lines.append("")

    lines.extend(["## Overall conclusion", ""])
    for sentence in conclusion["summary_sentences"]:
        lines.extend([sentence, ""])

    prefix.with_suffix(".md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved factor summary CSV : {prefix.with_suffix('.csv')}")
    print(f"Saved factor summary JSON: {prefix.with_suffix('.json')}")
    print(f"Saved factor summary Markdown: {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
