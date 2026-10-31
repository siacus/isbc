#!/usr/bin/env python3
"""Create a reproducible paper handoff for AG News cross-cutting communities.

The analysis deliberately uses external AG News labels only after fitting the
label-free PMI/Leiden graph.  It isolates Business-labelled documents that
jointly activate every bit in either the Sci/Tech-dominant community 0 or the
World-dominant community 2 of the Qwen 16-bit/6-active-bit run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


COMMUNITIES = {
    0: ("Sci/Tech-dominant", (0, 5, 6, 10, 12)),
    2: ("World-dominant", (3, 7, 8, 11, 15)),
}

# Illustrative texts are deliberately selected to cover distinct mechanisms,
# not sampled for inference.  The complete selected population is exported too.
ILLUSTRATIVE_IDS = {
    0: [2079, 2822, 30040, 85857],
    2: [40589, 62621, 104216, 125602],
}

SPORTS_WORLD_COMMUNITY = 2


def distinctive_terms(text: pd.Series, in_group: pd.Series) -> list[tuple[str, float, float]]:
    """Return visible, descriptive term-presence contrasts within Business."""
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        binary=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b",
    )
    matrix = vectorizer.fit_transform(text)
    names = vectorizer.get_feature_names_out()
    group_rate = matrix[in_group.to_numpy()].mean(axis=0).A1
    other_rate = matrix[~in_group.to_numpy()].mean(axis=0).A1
    group_count = matrix[in_group.to_numpy()].sum(axis=0).A1
    score = (group_rate + 0.012) / (other_rate + 0.012)
    order = score.argsort()[::-1]
    selected = [i for i in order if group_count[i] >= 3][:10]
    return [(str(names[i]), float(group_rate[i]), float(other_rate[i])) for i in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    codes = pd.read_csv(args.run_dir / "encoded_agnews_all_ready_best.csv")
    labels = pd.read_csv(args.labels_csv, usecols=["id", "label_name"])
    data = codes.merge(labels, on="id", how="inner", validate="one_to_one")
    if len(data) != len(codes):
        raise ValueError("Label join dropped encoded documents.")
    business = data.loc[data["label_name"].eq("Business")].copy()

    all_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    terms_by_community: dict[int, list[tuple[str, float, float]]] = {}
    within_community_terms: dict[int, list[tuple[str, float, float]]] = {}
    within_community_counts: dict[int, tuple[int, int]] = {}
    term_rows: list[dict[str, object]] = []
    for community, (dominant_context, bit_numbers) in COMMUNITIES.items():
        bit_columns = [f"bit_{bit:02d}" for bit in bit_numbers]
        joint_all = (data[bit_columns] == 1).all(axis=1)
        joint = joint_all.loc[business.index]
        subset = business.loc[joint].copy()
        subset.insert(0, "community", community)
        subset.insert(1, "context_dominant_label", dominant_context)
        subset.insert(2, "required_bits", "; ".join(bit_columns))
        all_rows.append(subset[["community", "context_dominant_label", "required_bits", "id", "text"]])
        terms_by_community[community] = distinctive_terms(business["text"], joint)
        for term, focal_rate, comparison_rate in terms_by_community[community]:
            term_rows.append(
                {
                    "community": community,
                    "comparison": "Business joint members vs all other Business documents",
                    "focal_group": "Business joint members",
                    "term": term,
                    "focal_document_presence": focal_rate,
                    "comparison_document_presence": comparison_rate,
                }
            )

        joint_members = data.loc[joint_all].copy()
        non_business = joint_members["label_name"].ne("Business")
        within_community_counts[community] = (int(non_business.sum()), int((~non_business).sum()))
        within_community_terms[community] = distinctive_terms(joint_members["text"], non_business)
        for term, focal_rate, comparison_rate in within_community_terms[community]:
            term_rows.append(
                {
                    "community": community,
                    "comparison": "Non-Business joint members vs Business joint members within the same community",
                    "focal_group": "Non-Business joint members",
                    "term": term,
                    "focal_document_presence": focal_rate,
                    "comparison_document_presence": comparison_rate,
                }
            )
        summaries.append(
            {
                "community": community,
                "context_dominant_label": dominant_context,
                "required_bits": "; ".join(bit_columns),
                "business_documents": len(subset),
                "share_of_all_business_documents": len(subset) / len(business),
            }
        )

    population = pd.concat(all_rows, ignore_index=True)
    population.to_csv(args.output_dir / "business_crosscutting_communities_all_documents.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "business_crosscutting_communities_summary.csv", index=False)
    pd.DataFrame(term_rows).to_csv(
        args.output_dir / "business_crosscutting_communities_term_contrasts.csv", index=False
    )
    examples = population.loc[population["id"].isin(sum(ILLUSTRATIVE_IDS.values(), []))].copy()
    examples["illustrative_selection"] = "purposive coverage; not a statistical sample"
    examples.to_csv(args.output_dir / "business_crosscutting_communities_illustrative_examples.csv", index=False)

    # There are only four Sports-labelled documents in the World-dominant
    # community. Export the complete set and treat it as a qualitative case,
    # rather than presenting unstable term-frequency estimates.
    world_bits = [f"bit_{bit:02d}" for bit in COMMUNITIES[SPORTS_WORLD_COMMUNITY][1]]
    sports_world = data.loc[
        (data[world_bits] == 1).all(axis=1) & data["label_name"].eq("Sports")
    ].copy()
    sports_world.insert(0, "community", SPORTS_WORLD_COMMUNITY)
    sports_world.insert(1, "context_dominant_label", "World-dominant")
    sports_world.insert(2, "required_bits", "; ".join(world_bits))
    sports_world = sports_world[["community", "context_dominant_label", "required_bits", "id", "text"]]
    sports_world.to_csv(args.output_dir / "sports_world_crosscutting_all_documents.csv", index=False)

    lines = [
        "# AG News: Business documents in cross-cutting bit communities",
        "",
        "## Scope and reproducibility",
        "",
        "This is a post-hoc qualitative case study for the Qwen 16-bit, target-six-active-bit SAE. "
        "The community graph and Leiden partition were fit without AG News labels. Only afterwards, "
        "the original four-way `label_name` was used to locate Business-labelled documents within two "
        "communities. A document belongs to a community here only when **every** bit listed for that "
        "community is active; it is not enough that any one community bit is active.",
        "",
        f"The encoded corpus has {len(data):,} documents and the original Business class has {len(business):,} documents.",
        "",
        "## Quantitative description",
        "",
        "| Community | Predominant post-hoc label among all joint members | Required active bits | Business documents | Share of all Business documents |",
        "|---:|---|---|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['community']} | {summary['context_dominant_label']} | "
            f"{summary['required_bits']} | {summary['business_documents']:,} | "
            f"{100 * summary['share_of_all_business_documents']:.3f}% |"
        )
    lines += [
        "",
        "The two sets contain 169 Business documents in total. The joint-community overlap table records "
        "no document that satisfies both full conjunctions; the two contextualizations are therefore disjoint "
        "in this fitted representation.",
        "",
        "## What differs descriptively",
        "",
        "Within the original Business class, community 0 is a **technology-market** context: corporate "
        "technology, devices, software, product launches, technology-company investment, and digital goods. "
        "Community 2 is an **international political-economy** context: government and multilateral institutions, "
        "macroeconomics, conflict, trade, security, and country-level financial consequences. These descriptions "
        "are interpretations of the documents, not labels assigned to individual bits.",
        "",
        "The following terms compare **only Business-labelled documents**. For each community, the focal set is "
        "the Business documents satisfying its full bit conjunction; the comparison set is every other Business "
        "document, including those outside that community and those in other communities. The contrasts are "
        "147 focal documents versus 31,753 other Business documents for community 0, and 22 versus 31,878 for "
        "community 2. Non-Business documents do not enter these term ratios. The values are smoothed "
        "document-presence ratios (minimum three occurrences in the focal set), and are descriptive diagnostics "
        "rather than hypothesis tests:",
        "",
    ]
    for community in COMMUNITIES:
        formatted = "; ".join(
            f"`{term}` ({100 * group:.0f}% vs {100 * other:.0f}%)"
            for term, group, other in terms_by_community[community]
        )
        lines += [f"- Community {community}: {formatted}.", ""]
    lines += [
        "## Complement: remaining members of the same community",
        "",
        "The preceding analysis asks what distinguishes the Business minority from other Business documents. "
        "The complementary contrast below asks what characterizes the **non-Business members of the same bit "
        "community**. This is the appropriate comparison for the contextual majority, not a replacement for the "
        "within-Business contrast above. The percentages read as ‘non-Business joint members versus Business joint "
        "members’, again using smoothed document presence and a minimum of three focal occurrences.",
        "",
    ]
    for community in COMMUNITIES:
        non_business_count, business_count = within_community_counts[community]
        formatted = "; ".join(
            f"`{term}` ({100 * focal:.0f}% vs {100 * comparison:.0f}%)"
            for term, focal, comparison in within_community_terms[community]
        )
        lines += [
            f"- Community {community} ({non_business_count:,} non-Business versus {business_count:,} Business joint members): {formatted}.",
            "",
        ]
    lines += [
        "## Illustrative excerpts",
        "",
        "The following are deliberately diverse, short, identifiable examples from the complete exported population. "
        "They illustrate the qualitative claim but are not used to estimate any quantity.",
        "",
    ]
    for community, (_, _) in COMMUNITIES.items():
        context = COMMUNITIES[community][0]
        lines += [f"### Community {community}: Business in a {context} context", ""]
        for document_id in ILLUSTRATIVE_IDS[community]:
            row = examples.loc[examples["id"].eq(document_id)].iloc[0]
            excerpt = str(row["text"]).replace("\n", " ")
            lines += [f"- **ID {document_id}:** {excerpt}", ""]
    lines += [
        "## Paper-ready interpretation",
        "",
        "The original AG News Business label pools at least two contextually distinct semantic configurations. "
        "Business documents jointly activating community 0 concern commercial activity internal to technological "
        "markets—firms, products, platforms, intellectual property, and investment. Business documents jointly "
        "activating community 2 concern economic consequences of international political settings—states, "
        "multilateral organizations, conflict, trade, and macroeconomic governance. Thus the sparse code does not "
        "replace the editorial label; it supplies a compositional contextual dimension that cuts across it. This is "
        "a qualitative illustration of representation structure, not evidence that a bit or community has a fixed "
        "semantic meaning in every corpus.",
        "",
        "## Artifacts",
        "",
        "- `business_crosscutting_communities_all_documents.csv`: all 169 documents supporting the case study.",
        "- `business_crosscutting_communities_summary.csv`: counts and conjunction definitions.",
        "- `business_crosscutting_communities_term_contrasts.csv`: both term-comparison designs with exact rates.",
        "- `business_crosscutting_communities_illustrative_examples.csv`: the eight excerpts reproduced above.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python analyze_agnews_crosscutting_communities.py \\",
        "  --run-dir runs/agnews_qwen_all_fixed_balance \\",
        "  --labels-csv data/agnews/agnews_all.csv \\",
        "  --output-dir runs/agnews_qwen_all_fixed_balance/bit_structure",
        "```",
    ]
    (args.output_dir / "business_crosscutting_communities_paper_handoff.md").write_text("\n".join(lines) + "\n")

    sports_lines = [
        "# AG News: Sports documents in the World-dominant bit community",
        "",
        "## Scope",
        "",
        "This post-hoc qualitative case study uses the same Qwen 16-bit, target-six-active-bit SAE and "
        "the label-free Leiden community partition as the main community analysis. It selects the four "
        "originally Sports-labelled documents for which **all** bits of community 2 (`bit_03`, `bit_07`, "
        "`bit_08`, `bit_11`, and `bit_15`) are active. Community 2 is World-dominant: 1,536 of its 1,568 "
        "joint members (97.96%) have the original World label. The AG News label is used only after fitting "
        "the code and graph.",
        "",
        "## Complete case set",
        "",
        f"The four documents are {len(sports_world) / len(data):.3%} of the {len(data):,}-document corpus and "
        f"{len(sports_world) / int((data['label_name'] == 'Sports').sum()):.3%} of the "
        f"{int((data['label_name'] == 'Sports').sum()):,} Sports-labelled documents. Because the complete "
        "case set has only four documents, no term-frequency ranking or inferential claim is reported.",
        "",
    ]
    for row in sports_world.itertuples(index=False):
        sports_lines += [f"- **ID {row.id}:** {str(row.text).replace(chr(10), ' ')}", ""]
    sports_lines += [
        "## Qualitative reading",
        "",
        "These documents are not primarily about sporting performance, scores, or athlete competition. Three "
        "place sport within public order, security, government, or crowd-control concerns: security for the Indian "
        "cricket team, police management of a Manchester United–Arsenal match, and a police warning to players "
        "before the same rivalry. The remaining item concerns a proposed takeover of Manchester United by a U.S. "
        "businessman. Taken together, the shared contextual layer is the institutional and political-economic "
        "environment around sport—security, policing, collective conflict, and ownership—rather than sport alone.",
        "",
        "This reading is illustrative and post-hoc. It does not establish a stable semantic meaning for individual "
        "bits, nor does it show that every Sports item with these themes will activate the same conjunction.",
        "",
        "## Paper-ready interpretation",
        "",
        "The World-dominant community contains a small set of editorially Sports-labelled items whose content is "
        "structured around governance rather than sporting performance. This supports a compositional reading of "
        "the sparse code: a document can retain its editorial topic while jointly expressing a cross-cutting context, "
        "here public order, institutional authority, or transnational ownership. Given four cases, this is qualitative "
        "evidence for interpretation and not a prevalence estimate.",
        "",
        "## Artifact",
        "",
        "- `sports_world_crosscutting_all_documents.csv`: complete four-document case set.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python analyze_agnews_crosscutting_communities.py \\",
        "  --run-dir runs/agnews_qwen_all_fixed_balance \\",
        "  --labels-csv data/agnews/agnews_all.csv \\",
        "  --output-dir runs/agnews_qwen_all_fixed_balance/bit_structure",
        "```",
    ]
    (args.output_dir / "sports_world_crosscutting_paper_handoff.md").write_text(
        "\n".join(sports_lines) + "\n"
    )


if __name__ == "__main__":
    main()
