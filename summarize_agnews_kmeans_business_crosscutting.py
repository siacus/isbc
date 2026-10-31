#!/usr/bin/env python3
"""Write one comparative report for the AG News Qwen k-means case study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/agnews_qwen_all_fixed_balance"
TARGETS = [
    ("tech_market", "Business ∩ Sci/Tech (technology-market)", 147),
    ("business_world", "Business ∩ World", 22),
]


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def main() -> None:
    destination = RUN / "kmeans_business_crosscutting_case_study"
    destination.mkdir(exist_ok=True)
    rows = []
    purity_tables: dict[int, pd.DataFrame] = {}
    assignment_frames: dict[tuple[int, str], pd.DataFrame] = {}
    detailed_links = []

    for k in (32, 64, 256):
        for slug, name, expected_n in TARGETS:
            directory = RUN / f"kmeans{k}_{slug}_case_study"
            overlap = pd.read_csv(directory / "cluster_target_overlap.csv")
            assignments = pd.read_csv(directory / "document_assignments.csv")
            cluster_column = f"kmeans{k}_cluster"
            target_column = f"is_sae_{slug}_business_target"
            target = assignments[target_column].astype(bool).to_numpy()
            clusters = assignments[cluster_column].to_numpy()
            if target.sum() != expected_n:
                raise ValueError(f"Unexpected target size in {directory}")
            assignment_frames[(k, slug)] = assignments
            best = overlap.iloc[0]
            target_clusters = int(np.unique(clusters[target]).size)
            rows.append({
                "k": k,
                "SAE-defined subset": name,
                "target documents": int(target.sum()),
                "clusters containing target documents": target_clusters,
                "best cluster": int(best["cluster"]),
                "target documents in best cluster": int(best["tech_market_documents"]),
                "precision": best["precision_target_within_cluster"],
                "recall": best["recall_target_covered"],
                "F1": best["f1_single_cluster"],
                "enrichment": best["enrichment_vs_corpus_rate"],
            })
            detailed_links.append(
                f"- k={k}, {name}: [`report.md`](../kmeans{k}_{slug}_case_study/report.md)"
            )
            if slug == "tech_market":
                purity_tables[k] = pd.read_csv(directory / "cluster_label_purity.csv")

    summary = pd.DataFrame(rows)
    display = summary.copy()
    for col in ("precision", "recall", "F1"):
        display[col] = display[col].map(lambda x: f"{100 * x:.2f}%")
    display["enrichment"] = display["enrichment"].map(lambda x: f"{x:.2f}×")
    summary.to_csv(destination / "recovery_summary.csv", index=False)
    tech_dispersion = summary.loc[summary["SAE-defined subset"].str.contains("Sci/Tech")]
    world_dispersion = summary.loc[summary["SAE-defined subset"].eq("Business ∩ World")]
    recovery_conclusion = (
        "**Conclusion.** Neither SAE-defined subset is recovered as one k-means cluster at any resolution. "
        f"The 147-document technology-market subset is split across {', '.join(str(int(x)) for x in tech_dispersion['clusters containing target documents'])} clusters at k=32, 64, and 256; "
        f"the 22-document Business–World subset is split across {', '.join(str(int(x)) for x in world_dispersion['clusters containing target documents'])}. "
        "Increasing k can raise the purity of a small best-matching fragment, but it simultaneously fragments the full conjunction; it does not yield a one-cluster recovery."
    )

    report = [
        "# AG News Qwen: k-means recovery of two SAE-defined cross-cutting subsets",
        "",
        "## Design",
        "",
        "Full Lloyd k-means is fitted to all 127,600 cached raw, unnormalised Qwen embeddings (1,024 dimensions), without AG News labels or SAE codes. The two Business-only targets are defined after fitting from the Qwen 16/6 SAE community analysis: the 147-document Sci/Tech-dominant technology-market conjunction (community 0) and the 22-document World-dominant Business-World conjunction (community 2). Each k-means partition is fit once and evaluated against both targets.",
        "",
        "A target appearing in one cluster would have `clusters containing target documents = 1` and a best-cluster recall of 100%. Precision indicates how much of that cluster belongs to the target; F1 requires both precision and recall. The targets are SAE-defined, so this is a post-hoc recovery test rather than an independent supervised benchmark.",
        "",
        "## Recovery summary",
        "",
        markdown_table(display),
        "",
        recovery_conclusion,
        "",
        "## Detailed target reports",
        "",
        *detailed_links,
    ]
    seed_stability_section = destination / "profile_partition_seed_stability_section.md"
    if seed_stability_section.exists():
        report.extend(["", seed_stability_section.read_text().strip()])
    bit_residual_section = destination / "bit_residual_information_section.md"
    if bit_residual_section.exists():
        report.extend(["", bit_residual_section.read_text().strip()])
    robustness_section = RUN / "normalized_kmeans_robustness" / "robustness_section.md"
    if robustness_section.exists():
        report.extend(["", robustness_section.read_text().strip()])
    corpus_wide_section = destination / "label_purity_comparison_section.md"
    if corpus_wide_section.exists():
        report.extend(["", corpus_wide_section.read_text().strip()])
    for k in (32, 64, 256):
        purity = purity_tables[k].copy()
        tech = assignment_frames[(k, "tech_market")]
        world = assignment_frames[(k, "business_world")]
        cluster_column = f"kmeans{k}_cluster"
        if not np.array_equal(tech[cluster_column].to_numpy(), world[cluster_column].to_numpy()):
            raise ValueError(f"Target evaluations do not share the same k={k} partition")
        purity["Business ∩ Sci/Tech target documents (n=147)"] = (
            tech.loc[tech["is_sae_tech_market_business_target"], cluster_column]
            .value_counts().reindex(purity["cluster"], fill_value=0).to_numpy()
        )
        purity["Business ∩ World target documents (n=22)"] = (
            world.loc[world["is_sae_business_world_business_target"], cluster_column]
            .value_counts().reindex(purity["cluster"], fill_value=0).to_numpy()
        )
        purity.to_csv(destination / f"k{k}_cluster_label_purity.csv", index=False)

        total_row = {column: np.nan for column in purity.columns}
        total_row.update({
            "cluster": "Total",
            "cluster_documents": int(purity["cluster_documents"].sum()),
            "dominant_label": "—",
            "Business ∩ Sci/Tech target documents (n=147)": int(purity["Business ∩ Sci/Tech target documents (n=147)"].sum()),
            "Business ∩ World target documents (n=22)": int(purity["Business ∩ World target documents (n=22)"].sum()),
        })
        for column in [c for c in purity if c.startswith("share_")]:
            total_row[column] = (
                purity[column].mul(purity["cluster_documents"]).sum()
                / total_row["cluster_documents"]
            )
        display_purity = pd.concat([purity, pd.DataFrame([total_row])], ignore_index=True)
        for column in [c for c in display_purity if c.endswith("_share") or c.startswith("share_")]:
            display_purity[column] = display_purity[column].map(
                lambda x: "—" if pd.isna(x) else f"{100 * x:.2f}%"
            )
        median_purity = purity["dominant_label_share"].median()
        high_purity = int((purity["dominant_label_share"] >= 0.8).sum())
        composition_conclusion = (
            f"**Conclusion.** At k={k}, the median cluster's dominant AG News category accounts for {median_purity:.1%} of its documents, "
            f"and {high_purity} of {k} clusters have at least 80% of one category. The raw-embedding partition is therefore substantially aligned with this four-way taxonomy; the target-count columns show that this label alignment does not imply recovery of the cross-cutting SAE conjunctions."
        )
        report.extend([
            "",
            f"## AG News category composition of every k={k} cluster",
            "",
            "The categories below were not used in fitting. `dominant_label_share` is the cluster's maximum category share; the four following columns give the complete AG News composition. The two target-count columns show where the 147-document and 22-document SAE-defined subsets fall in this same partition. The final `Total` row verifies the full corpus and both target counts. Thus purity is descriptive, not a classification accuracy.",
            "",
            f"**Target dispersion:** the 147-document Business ∩ Sci/Tech subset appears in **{int((purity['Business ∩ Sci/Tech target documents (n=147)'] > 0).sum())} of {k}** clusters; the 22-document Business ∩ World subset appears in **{int((purity['Business ∩ World target documents (n=22)'] > 0).sum())} of {k}** clusters.",
            "",
            markdown_table(display_purity),
            "",
            composition_conclusion,
        ])
    (destination / "report.md").write_text("\n".join(report) + "\n")
    print(destination / "report.md")


if __name__ == "__main__":
    main()
