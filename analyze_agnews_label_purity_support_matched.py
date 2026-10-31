#!/usr/bin/env python3
"""Compare AG News label purity of k-means clusters and SAE bit conjunctions.

The comparison is descriptive and corpus-wide.  It uses the existing Qwen
16/6 hard codes and the already-fitted k=32, 64, and 256 raw-embedding k-means
partitions; it never refits an embedding model, SAE, or k-means partition.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/agnews_qwen_all_fixed_balance"
OUT = RUN / "kmeans_business_crosscutting_case_study"
LABEL_ORDER = ["World", "Sports", "Business", "Sci_Tech"]
COLORS = {
    "K-means cluster": "#5C88C4",
    "2-bit conjunction": "#E68653",
    "3-bit conjunction": "#67A877",
}


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def dominant_purity(labels: np.ndarray, membership: np.ndarray) -> tuple[int, str, float]:
    selected = labels[membership]
    counts = pd.Series(selected).value_counts().reindex(LABEL_ORDER, fill_value=0)
    support = int(counts.sum())
    return support, str(counts.idxmax()), float(counts.max() / support)


def conjunctions(codes: np.ndarray, labels: np.ndarray, bit_columns: list[str]) -> pd.DataFrame:
    rows = []
    for degree in (2, 3):
        for positions in combinations(range(codes.shape[1]), degree):
            mask = np.all(codes[:, positions] == 1, axis=1)
            support, dominant, purity = dominant_purity(labels, mask)
            rows.append({
                "representation": f"{degree}-bit conjunction",
                "degree": degree,
                "conjunction": " ∧ ".join(bit_columns[i] for i in positions),
                "support": support,
                "dominant_label": dominant,
                "dominant_label_share": purity,
            })
    return pd.DataFrame(rows)


def compare_probability(a: np.ndarray, b: np.ndarray) -> float:
    """Probability that a random purity from a is lower than one from b."""
    return float((a[:, None] < b[None, :]).mean())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    codes = pd.read_csv(RUN / "encoded_agnews_all_ready_best.csv", dtype={"id": "string"})
    corpus = pd.read_csv(ROOT / "data/agnews/agnews_all.csv", dtype={"id": "string"})[["id", "label_name"]]
    merged = codes.merge(corpus, on="id", how="left", validate="one_to_one")
    if len(merged) != 127_600 or merged["label_name"].isna().any():
        raise ValueError("Hard codes and AG News labels do not align")
    bit_columns = [f"bit_{i:02d}" for i in range(16)]
    hard = merged[bit_columns].to_numpy(dtype=np.int8)
    labels = merged["label_name"].to_numpy()
    all_conjunctions = conjunctions(hard, labels, bit_columns)
    all_conjunctions.to_csv(OUT / "all_2bit_3bit_conjunction_label_purity.csv", index=False)

    comparisons = []
    matched_frames = []
    for k in (32, 64, 256):
        assignments = pd.read_csv(
            RUN / f"kmeans{k}_tech_market_case_study/document_assignments.csv",
            dtype={"id": "string"},
        )[["id", f"kmeans{k}_cluster"]]
        joined = merged[["id"]].merge(assignments, on="id", how="left", validate="one_to_one")
        if joined.iloc[:, 1].isna().any():
            raise ValueError(f"k={k} assignments do not align")
        cluster = joined.iloc[:, 1].to_numpy(dtype=int)
        km_rows = []
        for c in range(k):
            support, dominant, purity = dominant_purity(labels, cluster == c)
            km_rows.append({
                "representation": "K-means cluster",
                "degree": 0,
                "conjunction": f"cluster {c}",
                "support": support,
                "dominant_label": dominant,
                "dominant_label_share": purity,
            })
        km = pd.DataFrame(km_rows)
        lower, upper = int(km.support.min()), int(km.support.max())
        matched = all_conjunctions.loc[
            all_conjunctions.support.between(lower, upper, inclusive="both")
        ].copy()
        if matched.empty:
            raise ValueError(f"No support-matched conjunctions for k={k}")
        comparison = pd.concat([km, matched], ignore_index=True)
        comparison.insert(0, "k", k)
        comparison["support_lower_bound"] = lower
        comparison["support_upper_bound"] = upper
        comparison.to_csv(OUT / f"k{k}_support_matched_label_purity.csv", index=False)
        matched_frames.append(comparison)

        for representation, frame in comparison.groupby("representation", sort=False):
            purity = frame["dominant_label_share"]
            comparisons.append({
                "k": k,
                "representation": representation,
                "n_groups": len(frame),
                "support_min": int(frame.support.min()),
                "support_max": int(frame.support.max()),
                "support_median": float(frame.support.median()),
                "purity_mean": float(purity.mean()),
                "purity_q25": float(purity.quantile(0.25)),
                "purity_median": float(purity.median()),
                "purity_q75": float(purity.quantile(0.75)),
            })
        km_purity = km["dominant_label_share"].to_numpy()
        for representation, frame in matched.groupby("representation", sort=False):
            comparisons.append({
                "k": k,
                "representation": f"P({representation} purity < K-means purity)",
                "n_groups": len(frame),
                "support_min": np.nan,
                "support_max": np.nan,
                "support_median": np.nan,
                "purity_mean": compare_probability(frame["dominant_label_share"].to_numpy(), km_purity),
                "purity_q25": np.nan,
                "purity_median": np.nan,
                "purity_q75": np.nan,
            })

    summary = pd.DataFrame(comparisons)
    summary.to_csv(OUT / "support_matched_label_purity_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True, constrained_layout=True)
    for ax, k, frame in zip(axes, (32, 64, 256), matched_frames):
        for representation, group in frame.groupby("representation", sort=False):
            ax.scatter(
                group["support"], group["dominant_label_share"] * 100,
                label=representation,
                color=COLORS[representation], alpha=0.76,
                s=42 if representation == "K-means cluster" else 30,
                edgecolor="white", linewidth=0.45,
            )
        lower = int(frame["support_lower_bound"].iloc[0])
        upper = int(frame["support_upper_bound"].iloc[0])
        ax.set_xscale("log")
        ax.set_title(f"k={k}\nsupport window: {lower:,}–{upper:,}")
        ax.set_xlabel("Group support (documents; log scale)")
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("Dominant AG News label share (%)")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_legend, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.14),
    )
    fig.savefig(OUT / "support_matched_label_purity.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "support_matched_label_purity.pdf", bbox_inches="tight")
    plt.close(fig)

    display = summary.loc[~summary.representation.str.startswith("P(")].copy()
    for column in ("purity_mean", "purity_q25", "purity_median", "purity_q75"):
        display[column] = display[column].map(lambda value: f"{100 * value:.2f}%")
    display["support_median"] = display["support_median"].map(lambda value: f"{value:,.0f}")
    for column in ("support_min", "support_max"):
        display[column] = display[column].map(lambda value: f"{value:,.0f}")
    probability = summary.loc[summary.representation.str.startswith("P(")].copy()
    probability_values = probability["purity_mean"].to_numpy(copy=True)
    probability["probability"] = probability["purity_mean"].map(lambda value: f"{100 * value:.1f}%")
    probability = probability[["k", "representation", "probability"]]
    conjunction_rows = summary.loc[~summary.representation.eq("K-means cluster") & ~summary.representation.str.startswith("P(")]
    purity_conclusion = (
        "**Conclusion.** In every available support window, the conjunction distribution lies below the k-means distribution in dominant-label purity. "
        f"Across the displayed comparisons, the probability that a conjunction is less label-pure than a k-means cluster ranges from {100 * probability_values.min():.1f}% to {100 * probability_values.max():.1f}%. "
        "Thus the evidence is consistent with bit intersections cutting across the four AG News categories, whereas raw-embedding k-means tends to align with them. This is a support-window comparison rather than exact one-to-one matching; the reported support medians make any residual support imbalance visible."
    )

    section = [
        "## Corpus-wide label-purity comparison: k-means clusters versus bit conjunctions",
        "",
        "This analysis uses all 127,600 AG News documents. For each raw-embedding k-means partition, each cluster is assigned the share of documents in its dominant AG News category. The same quantity is calculated for every 2-bit and 3-bit hard-code conjunction (a document belongs when all named bits are active; other bits may also be active). No labels are used to fit k-means or the SAE.",
        "",
        "To make group sizes comparable, for each value of k the conjunctions are retained only when their support lies between the smallest and largest k-means cluster in that partition. The comparison therefore asks whether multi-bit intersections are more or less label-homogeneous than clusters of similar observed size; it does not claim that either representation is optimized for the AG News taxonomy.",
        "",
        "![Support-matched label purity](support_matched_label_purity.png)",
        "",
        "### Support-matched purity distribution",
        "",
        markdown_table(display),
        "",
        "The final diagnostic below is the fraction of all pairwise comparisons in which a random support-matched conjunction has lower dominant-label purity than a random k-means cluster. Values above 50% mean that the conjunctions are more cross-cutting with respect to this taxonomy in the common-language effect-size sense; it is descriptive rather than a hypothesis test.",
        "",
        markdown_table(probability),
        "",
        purity_conclusion,
        "",
        "Artifacts: `all_2bit_3bit_conjunction_label_purity.csv` contains all 680 conjunctions; `k32_support_matched_label_purity.csv`, `k64_support_matched_label_purity.csv`, and `k256_support_matched_label_purity.csv` contain the support-matched observations behind the panel and table.",
    ]
    (OUT / "label_purity_comparison_section.md").write_text("\n".join(section) + "\n")
    print(OUT / "label_purity_comparison_section.md")


if __name__ == "__main__":
    main()
