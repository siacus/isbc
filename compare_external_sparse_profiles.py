#!/usr/bin/env python3
"""Run symmetric-reference profile matching for the AG News sparse-coder bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("ours", "hypothesaes", "binarysentemb")
DISPLAY = {
    "ours": "Consolidated SAE",
    "hypothesaes": "HypotheSAEs",
    "binarysentemb": "BinarySentEmb",
}
COLORS = {
    "ours": "#1F77B4",
    "hypothesaes": "#D62728",
    "binarysentemb": "#2CA02C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path(
            "runs/agnews_embedding_comparison/external_sparse_encoders_qwen_k16"
        ),
    )
    parser.add_argument("--max_profiles", type=int, default=10)
    parser.add_argument("--null_draws", type=int, default=1000)
    parser.add_argument("--min_docs", type=int, default=30)
    return parser.parse_args()


def run_reference(
    run_dir: Path,
    reference: str,
    max_profiles: int,
    null_draws: int,
    min_docs: int,
) -> Path:
    output_dir = run_dir / "profile_matching" / f"from_{reference}"
    # A rerun may request fewer reference profiles than an earlier invocation.
    # Remove only numbered, generated profile figures so the artifact directory
    # represents the current selection instead of retaining stale tail profiles.
    graph_dir = output_dir / "graphs"
    if graph_dir.exists():
        for stale_graph in graph_dir.glob("graph_*.png"):
            stale_graph.unlink()
    command = [
        sys.executable,
        "profile_matching/profile_matching.py",
        "--run-dir",
        str(run_dir),
        "--reference",
        reference,
        "--min-share",
        "0",
        "--min-top-share",
        "0",
        "--top-reference-profiles",
        str(max_profiles),
        "--tail-reference-profiles",
        "0",
        "--min-docs",
        str(min_docs),
        "--null-draws",
        str(null_draws),
        "--graphs",
        "--out",
        str(output_dir),
    ]
    subprocess.run(command, check=True)
    return output_dir


def summarize_reference(reference: str, output_dir: Path, n_documents: int) -> dict:
    matches = pd.read_csv(output_dir / "matches.csv")
    invariance = pd.read_csv(output_dir / "invariance.csv")
    if matches.empty:
        raise ValueError(f"No profile matches generated for {reference}")
    if invariance.empty:
        raise ValueError(f"No invariance statistics generated for {reference}")
    same_category = (
        matches["start_top_category"].astype(str)
        == matches["final_top_category"].astype(str)
    )
    start_docs = int(matches["start_docs"].sum())
    final_docs = int(matches["final_docs"].sum())
    per_profile_retention = (
        matches["final_docs"].to_numpy(dtype=float)
        / matches["start_docs"].to_numpy(dtype=float)
    )
    weights = invariance["start_share"].to_numpy(dtype=float)
    weights = weights / weights.sum()

    def weighted(column: str) -> tuple[float, int]:
        values = invariance[column].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if not valid.any():
            return float("nan"), 0
        selected_weights = weights[valid]
        return float(np.average(values[valid], weights=selected_weights / selected_weights.sum())), int(valid.sum())

    coherence, coherence_n = weighted("coherence")
    swap_mean, swap_n = weighted("swap_mean")
    chain_mean, chain_n = weighted("chain_mean")
    information, information_n = weighted("information")

    return {
        "reference": reference,
        "reference_display": DISPLAY[reference],
        "n_profiles": int(len(matches)),
        "start_docs": start_docs,
        "start_coverage": start_docs / n_documents,
        "final_docs": final_docs,
        "final_coverage": final_docs / n_documents,
        "pooled_retention": final_docs / start_docs,
        "median_profile_retention": float(np.median(per_profile_retention)),
        "weak_matches": int(matches["weak"].sum()),
        "category_agreement": float(
            np.average(same_category, weights=matches["start_docs"])
        ),
        "weighted_coherence": coherence,
        "weighted_coherence_profiles": coherence_n,
        "weighted_swap_null": swap_mean,
        "weighted_swap_profiles": swap_n,
        "weighted_chain_null": chain_mean,
        "weighted_chain_profiles": chain_n,
        "weighted_information_bits": information,
        "weighted_information_profiles": information_n,
        "significant_swap_profiles": int((invariance["swap_p"] <= 0.05).sum()),
        "significant_chain_profiles": int((invariance["chain_p"] <= 0.05).sum()),
        "output": str(output_dir.resolve()),
    }


def plot_summary(summary: pd.DataFrame, output_path: Path, n_documents: int) -> None:
    positions = np.arange(len(summary))
    width = 0.34
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    axes[0].bar(
        positions - width / 2,
        summary["start_coverage"],
        width,
        color=[COLORS[value] for value in summary["reference"]],
        alpha=0.45,
        label="top-10 reference profiles",
    )
    axes[0].bar(
        positions + width / 2,
        summary["final_coverage"],
        width,
        color=[COLORS[value] for value in summary["reference"]],
        label="final exact intersections",
    )
    axes[0].set_xticks(positions, summary["reference_display"], rotation=12)
    axes[0].set_ylabel(f"Share of all {n_documents:,} documents")
    axes[0].set_title("Coverage before and after profile-chain matching")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        positions,
        summary["pooled_retention"],
        color=[COLORS[value] for value in summary["reference"]],
    )
    axes[1].set_xticks(positions, summary["reference_display"], rotation=12)
    axes[1].set_ylabel("Final documents / reference-profile documents")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Pooled retention within the matched profile chains")
    axes[1].grid(axis="y", alpha=0.25)
    for position, value in zip(positions, summary["pooled_retention"]):
        axes[1].text(position, value + 0.025, f"{value:.1%}", ha="center")

    axes[2].plot(
        positions,
        summary["weighted_coherence"],
        "o-",
        color="#111111",
        linewidth=2,
        label="observed category coherence",
    )
    axes[2].plot(
        positions,
        summary["weighted_swap_null"],
        "o--",
        color="#9467BD",
        label="profile-swap null",
    )
    axes[2].plot(
        positions,
        summary["weighted_chain_null"],
        "o--",
        color="#8C564B",
        label="chain-permutation null",
    )
    axes[2].set_xticks(positions, summary["reference_display"], rotation=12)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Weighted category coherence")
    axes[2].set_title("Semantic coherence versus null baselines")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def matching_conclusion(summary: pd.DataFrame, min_docs: int) -> str:
    """State only the empirical full-corpus result, never a stale prior conclusion."""
    all_weak = bool((summary["weak_matches"] == summary["n_profiles"]).all())
    coherence = summary["weighted_coherence"].dropna()
    null_values = pd.concat(
        [summary["weighted_swap_null"], summary["weighted_chain_null"]], ignore_index=True
    ).dropna()
    weak_text = (
        f"Every selected chain remains below the predeclared {min_docs}-document threshold."
        if all_weak
        else f"Some selected chains exceed the predeclared {min_docs}-document threshold; inspect the direction-specific tables."
    )
    null_text = (
        f" versus {null_values.min():.3f}--{null_values.max():.3f} under the available null schemes"
        if not null_values.empty else " (no null scheme was identifiable for these profiles)"
    )
    return (
        f"{weak_text} Weighted category coherence ranges from {coherence.min():.3f} to "
        f"{coherence.max():.3f}{null_text}. "
        "Exact-profile agreement and category-composition coherence describe different, complementary levels of alignment."
    )


def write_overview(run_dir: Path, summary: pd.DataFrame, max_profiles: int, min_docs: int) -> None:
    profile_root = run_dir / "profile_matching"
    lines = [
        "# Cross-coder hard-profile matching",
        "",
        "## TL;DR",
        "",
        (
            "The comparison is deliberately repeated three times, taking each sparse coder "
            "in turn as the reference. This matters because the matching is conditional and "
            "asymmetric: a broad reference profile can split into several fine profiles in another "
            "coder, while starting from the fine coder asks the reverse question."
        ),
        "",
        (
            f"For comparability, each direction begins with up to {max_profiles} most "
            "frequent exact 16-bit profiles (fewer only when that coder has fewer observed profiles). "
            "The algorithm then restricts to the documents in one "
            "reference profile, selects the most prevalent exact profile in either remaining model, "
            "restricts again, and repeats for the last model. No label is used to select a match."
        ),
        "",
        matching_conclusion(summary, min_docs),
        "",
        "![Reference-direction summary](profile_matching_summary.png)",
        "",
        "| Starting model | Top profiles | Initial corpus coverage | Final exact-intersection coverage | Pooled retention | Median per-profile retention | Weak matches (<30 docs) | Category agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['reference_display']} | {int(row['n_profiles'])} | "
            f"{row['start_coverage']:.1%} | {row['final_coverage']:.1%} | "
            f"{row['pooled_retention']:.1%} | {row['median_profile_retention']:.1%} | "
            f"{int(row['weak_matches'])} | {row['category_agreement']:.1%} |"
        )
    lines.extend(
        [
            "",
            "`Category agreement` is the reference-document-weighted share of matched chains whose "
            "prevalent AG News category is unchanged from the initial subgroup to the final exact "
            "intersection. It is descriptive, not part of the matching objective.",
            "",
            "`Pooled retention` must be interpreted together with initial coverage. A model with many "
            "small profiles can have low top-10 corpus coverage even when its chains are locally "
            "coherent. Conversely, high retention from a broad reference profile does not imply a "
            "one-to-one global equivalence between partitions.",
            "",
            "### Category-composition coherence",
            "",
            "For every starting profile, coherence measures how similar the AG News category "
            "distribution remains across the matched model profiles. Values closer to 1 indicate "
            "more similar category mixtures. The two null columns show the coherence expected after "
            "breaking the observed profile correspondence by swapping profiles or permuting the "
            "matching chain. Higher observed coherence than both nulls supports semantic alignment; "
            "it does not imply exact code equality.",
            "The profile-swap and chain-permutation nulls each use 1,000 fixed-seed replications "
            "(seeds 0 and 1 respectively). Parenthetical `n` values are the number of selected "
            "reference profiles for which each weighted statistic is identifiable; a swap null is "
            "reported as unavailable when a matched model has no alternative observed profile.",
            "",
            "| Starting model | Observed coherence (n) | Profile-swap null (n) | Chain-permutation null (n) | Information (bits; n) | Significant profiles: swap | Significant profiles: chain |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['reference_display']} | {row['weighted_coherence']:.3f} ({int(row['weighted_coherence_profiles'])}) | "
            f"{row['weighted_swap_null']:.3f} ({int(row['weighted_swap_profiles'])}) | "
            f"{row['weighted_chain_null']:.3f} ({int(row['weighted_chain_profiles'])}) | "
            f"{row['weighted_information_bits']:.3f} ({int(row['weighted_information_profiles'])}) | "
            f"{int(row['significant_swap_profiles'])}/{int(row['weighted_swap_profiles'])} | "
            f"{int(row['significant_chain_profiles'])}/{int(row['weighted_chain_profiles'])} |"
        )
    lines.extend(["", "## Direction-specific reports", ""])
    for method in METHODS:
        lines.extend(
            [
                f"### Starting from {DISPLAY[method]}",
                "",
                f"- [Detailed report](from_{method}/output.md)",
                f"- [Complete match table](from_{method}/matches.csv)",
                f"- [All profile/category graphs](from_{method}/graphs/contact_sheet.png)",
                f"- [Invariance statistics](from_{method}/invariance.csv)",
                "",
                f"![{DISPLAY[method]} contact sheet](from_{method}/graphs/contact_sheet.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## What the matching does and does not establish",
            "",
            "A retained document is one for which all three coders simultaneously assign the exact "
            "profiles selected by the greedy chain. The threshold flag in the table is a practical "
            "reliability diagnostic rather than a global equivalence test. Coherence above the null "
            "baselines instead supports a coarser semantic or topological alignment despite arbitrary "
            "bit numbering. These diagnostics do not prove "
            "that individual bits are identical, that the complete partitions are interchangeable, "
            "or that one coder is globally superior. The direction-specific views expose refinement "
            "and many-to-one relationships that a single reference would hide.",
        ]
    )
    (profile_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_handoff(run_dir: Path, summary: pd.DataFrame, min_docs: int) -> None:
    """Write paper-ready prose only from the current full-corpus artifacts."""
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    n_documents = int(metadata["n_documents"])
    metrics = pd.read_csv(run_dir / "predictive_metrics.csv")
    codes = pd.read_csv(run_dir / "code_summary.csv").set_index("method")
    ols = pd.read_csv(run_dir / "hard_code_affine_ols_reconstruction.csv").set_index("method")
    logistic = metrics[metrics["classifier"].eq("logistic_regression")].copy()
    metric_lookup = logistic.set_index(["run", "representation"])

    def metric(method: str, representation: str, column: str = "balanced_accuracy") -> float:
        return float(metric_lookup.loc[(method, representation), column])

    def matching_value(value: float, latex: bool = False) -> str:
        """Avoid presenting an unavailable null comparison as a numerical result."""
        if pd.isna(value):
            return "--" if latex else "not identifiable"
        return f"{float(value):.3f}"

    lines = [
        "# Paper-writing handoff: full-corpus AG News sparse-coder comparison",
        "",
        "This supersedes the earlier 15,600-document comparison. All figures and numbers below refer "
        f"to the full **{n_documents:,}-document** corpus.",
        "",
        "## Protocol",
        "",
        f"- Frozen backbone: Qwen/Qwen3-Embedding-0.6B; cached, unnormalised {metadata['embedding_dim']}-dimensional embeddings.",
        f"- Unsupervised fitting: all {n_documents:,} documents; labels are excluded from encoder fitting.",
        "- Evaluation split: fixed IDs, 102,080 training and 25,520 test documents.",
        "- Width: 16 units; proposed SAE target activity six; HypotheSAEs M=16/K=6; BinarySentEmb has no imposed activity target.",
        "- Prediction: all representations were refit with the same logistic, elastic-net, and random-forest protocols. Logistic is the primary probe. Its 95% BA intervals use shared class-stratified bootstrap resamples.",
        "- Continuous values: proposed SAE = bit-probability log-odds; HypotheSAEs = official nonnegative sparse activations; BinarySentEmb = pre-threshold encoder logits.",
        "- BinarySentEmb remains a transfer experiment: its official objective is applied to seeded random AG News Qwen-vector pairs, not a reproduction of its InferSent/NLI experiment.",
        "",
        "## Core results",
        "",
        "| Method | Logistic hard BA | Logistic continuous BA | Hard/Raw BA ratio | Mean active bits | Effective profiles | Hard-code OLS R² |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            f"| {DISPLAY[method]} | {metric(method, 'hard_bits'):.3f} | "
            f"{metric(method, 'continuous_latent'):.3f} | "
            f"{metric(method, 'hard_bits', 'ba_binary_over_raw_embedding'):.3f} | "
            f"{codes.loc[method, 'mean_active_bits']:.3f} | "
            f"{codes.loc[method, 'effective_profiles']:.1f} | "
            f"{ols.loc[method, 'posthoc_linear_decoder_r2']:.3f} |"
        )
    raw_ba = metric("qwen_baseline", "raw_embeddings")
    lines.extend([
        f"| Raw Qwen | -- | {raw_ba:.3f} | 1.000 | -- | -- | -- |",
        "",
        f"On the common logistic probe, raw Qwen reaches BA {raw_ba:.3f}. The three hard-code values "
        "above are directly comparable because the documents, split, classifier, and bootstrap resamples are fixed. "
        "Hard-code OLS R² is in-sample reconstruction from a newly fitted affine decoder with the code fixed; it is not the original decoder's score and is not a predictive metric.",
        "",
        "## LaTeX table: predictive and code summary",
        "",
        "```latex",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Hard BA & Continuous BA & Hard/raw & Active bits & $2^H$ & Hard OLS $R^2$ \\\\",
        "\\midrule",
    ])
    for method in METHODS:
        lines.append(
            f"{DISPLAY[method]} & {metric(method, 'hard_bits'):.3f} & "
            f"{metric(method, 'continuous_latent'):.3f} & "
            f"{metric(method, 'hard_bits', 'ba_binary_over_raw_embedding'):.3f} & "
            f"{codes.loc[method, 'mean_active_bits']:.2f} & "
            f"{codes.loc[method, 'effective_profiles']:.1f} & "
            f"{ols.loc[method, 'posthoc_linear_decoder_r2']:.3f} \\\\")
    lines.extend([
        f"Raw Qwen & -- & {raw_ba:.3f} & 1.000 & -- & -- & -- \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "```",
        "",
        "## Profile matching",
        "",
        "Profile matching is document-based and label-free during selection. Each method becomes the reference in turn; up to its ten most frequent complete hard profiles initiate the chain (BinarySentEmb can contribute fewer if it has fewer observed profiles). Category composition is a post-hoc diagnostic only.",
        "",
        "| Starting model | Top-10 coverage | Final exact coverage | Pooled retention | Coherence | Swap null | Chain null |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['reference_display']} | {row['start_coverage']:.1%} | {row['final_coverage']:.1%} | "
            f"{row['pooled_retention']:.1%} | {row['weighted_coherence']:.3f} | "
            f"{matching_value(row['weighted_swap_null'])} | "
            f"{matching_value(row['weighted_chain_null'])} |"
        )
    lines.extend([
        "",
        matching_conclusion(summary, min_docs),
        "",
        "## LaTeX table: directional profile matching",
        "",
        "```latex",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Reference & Top-10 coverage & Exact coverage & Retention & Coherence & Swap/chain null \\\\",
        "\\midrule",
    ])
    for _, row in summary.iterrows():
        lines.append(
            f"{row['reference_display']} & {row['start_coverage']:.1%} & {row['final_coverage']:.1%} & "
            f"{row['pooled_retention']:.1%} & {row['weighted_coherence']:.3f} & "
            f"{matching_value(row['weighted_swap_null'], latex=True)}/"
            f"{matching_value(row['weighted_chain_null'], latex=True)} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "```",
        "",
        "## Claims and cautions",
        "",
        "- Do not call the BinarySentEmb result a reproduction of the original paper.",
        "- Do not compare binary-support and continuous-activation results as if they used identical information.",
        "- Do not treat profile entropy, observed profiles, or OLS reconstruction as a scalar measure of semantic quality.",
        "- Do not treat greedy directional profile retention as a global partition-alignment optimum.",
        "",
        "## Source and artifacts",
        "",
        "- HypotheSAEs paper: https://proceedings.mlr.press/v267/movva25a.html",
        "- BinarySentEmb paper: https://aclanthology.org/P19-1011/",
        "- [Full comparison](comparison.md); [profile matching](profile_matching/README.md).",
        "- Exact values: `predictive_metrics.csv`, `code_summary.csv`, `reconstruction_summary.csv`, `hard_code_affine_ols_reconstruction.csv`, and `profile_matching/profile_matching_summary.csv`.",
    ])
    (run_dir / "paper_handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def embed_profile_section(run_dir: Path, summary: pd.DataFrame, min_docs: int) -> None:
    report_path = run_dir / "comparison.md"
    text = report_path.read_text(encoding="utf-8")
    marker = "## Cross-coder profile matching"
    if marker in text:
        start = text.index(marker)
        artifacts = text.index("## Artifacts", start)
        text = text[:start] + text[artifacts:]
    section = "\n".join(
        [
            "## Cross-coder profile matching",
            "",
            "Exact hard-code profiles are also compared document by document, taking each of the "
            "three coders in turn as the reference. The directional design exposes refinement and "
            "many-to-one relations that a single starting model would hide. Each direction uses the "
            "ten most frequent reference profiles; labels describe the resulting subsets but never "
            "select the matches.",
            "",
            matching_conclusion(summary, min_docs),
            "",
            "![Directional profile-matching summary](profile_matching/profile_matching_summary.png)",
            "",
            "See the [complete profile-matching overview](profile_matching/README.md), its three "
            "direction-specific reports, graphs, match tables, and null-based invariance diagnostics.",
            "",
        ]
    )
    text = text.replace("## Artifacts", section + "\n## Artifacts", 1)
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_profiles <= 0 or args.null_draws < 0 or args.min_docs <= 0:
        raise ValueError("Require positive profile/min-doc counts and nonnegative null draws")
    run_dir = args.run_dir.resolve()
    manifest = run_dir / "factor_collection_comparison.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing {manifest}; regenerate the external benchmark bundle first"
        )
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    n_documents = int(metadata["n_documents"])
    summaries = []
    for reference in METHODS:
        print(f"profile matching from {reference}")
        output_dir = run_reference(
            run_dir,
            reference,
            args.max_profiles,
            args.null_draws,
            args.min_docs,
        )
        summaries.append(summarize_reference(reference, output_dir, n_documents))
    summary = pd.DataFrame(summaries)
    profile_root = run_dir / "profile_matching"
    summary.to_csv(profile_root / "profile_matching_summary.csv", index=False)
    plot_summary(summary, profile_root / "profile_matching_summary.png", n_documents)
    write_overview(run_dir, summary, args.max_profiles, args.min_docs)
    write_paper_handoff(run_dir, summary, args.min_docs)
    embed_profile_section(run_dir, summary, args.min_docs)
    print(f"Profile matching complete: {profile_root / 'README.md'}")


if __name__ == "__main__":
    main()
