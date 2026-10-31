#!/usr/bin/env python3
"""Test whether a Qwen SAE-defined tech-market subset emerges under k-means.

This is a narrow AG News case study only.  K-means is fit without labels on
cached raw Qwen embeddings; labels and the SAE community definition enter only
after fitting to evaluate overlap with the 147 Business documents in community
0 (the technology-market conjunction).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# The analysis is batch-only: never initialise the macOS interactive backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs/agnews_qwen_all_fixed_balance"
TARGETS = {
    0: {
        "slug": "tech_market",
        "title": "technology-market",
        "description": "Sci/Tech-dominant technology-market conjunction",
        "expected_n": 147,
    },
    2: {
        "slug": "business_world",
        "title": "Business-World",
        "description": "World-dominant Business-World conjunction",
        "expected_n": 22,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--n-clusters", type=int, default=32)
    parser.add_argument(
        "--community", type=int, default=0, choices=sorted(TARGETS),
        help="SAE community defining the post-hoc Business target.",
    )
    parser.add_argument(
        "--assignments-csv", type=Path,
        help="Reuse an existing id/cluster assignment file instead of refitting k-means.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def main() -> None:
    args = parse_args()
    if args.n_clusters < 2:
        raise ValueError("--n-clusters must be at least 2")
    k = args.n_clusters
    target_spec = TARGETS[args.community]
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / f"kmeans{k}_{target_spec['slug']}_case_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = run_dir / "raw_embeddings_agnews_all_ready_best.npy"
    embedding_ids_path = run_dir / "raw_embeddings_agnews_all_ready_best_ids.csv"
    community_path = run_dir / "bit_structure/business_crosscutting_communities_all_documents.csv"
    agnews_path = ROOT / "data/agnews/agnews_all.csv"
    for path in (embedding_path, embedding_ids_path, community_path, agnews_path):
        if not path.exists():
            raise FileNotFoundError(path)

    # The source metadata documents raw, unnormalised embeddings.  We preserve
    # that geometry: no labels, SAE codes, normalisation, or dimensionality
    # reduction is supplied to k-means.
    embedding_metadata = json.loads((run_dir / "raw_embeddings_agnews_all_ready_best_metadata.json").read_text())
    if embedding_metadata.get("l2_normalized") is not False:
        raise ValueError("This case study is defined for cached raw, unnormalised embeddings")
    embeddings = np.load(embedding_path, mmap_mode="r")
    ids = pd.read_csv(embedding_ids_path, dtype={"id": "string"})
    if len(ids) != len(embeddings) or ids["id"].duplicated().any():
        raise ValueError("Embedding row/ID alignment is invalid")

    corpus = pd.read_csv(agnews_path, dtype={"id": "string"})
    if corpus["id"].duplicated().any() or "label_name" not in corpus:
        raise ValueError("AG News labels are unavailable or non-unique")
    target_rows = pd.read_csv(community_path, dtype={"id": "string"})
    target_ids = set(target_rows.loc[
        target_rows["community"].eq(args.community), "id"
    ].astype("string"))
    if len(target_ids) != target_spec["expected_n"]:
        raise ValueError(
            f"Expected {target_spec['expected_n']} {target_spec['title']} target documents, found {len(target_ids)}"
        )
    lookup = corpus.set_index("id")["label_name"]
    if not all(lookup.reindex(list(target_ids)).eq("Business")):
        raise ValueError("The target must consist solely of Business-labelled documents")
    target = ids["id"].isin(target_ids).to_numpy(dtype=bool)
    if target.sum() != target_spec["expected_n"]:
        raise ValueError("Some target IDs are missing from the embedding cache")
    corpus_labels = lookup.reindex(ids["id"])
    if corpus_labels.isna().any():
        raise ValueError("Some embedding IDs are absent from the AG News corpus")

    if args.assignments_csv:
        previous = pd.read_csv(args.assignments_csv, dtype={"id": "string"})
        cluster_columns = [c for c in previous if c.startswith("kmeans") and c.endswith("_cluster")]
        if len(cluster_columns) != 1 or len(previous) != len(ids):
            raise ValueError("Existing assignments must have one kmeans*_cluster column and aligned rows")
        previous = previous.set_index("id").reindex(ids["id"])
        if previous.index.hasnans or previous[cluster_columns[0]].isna().any():
            raise ValueError("Existing assignment IDs do not align with embedding IDs")
        assignments = previous[cluster_columns[0]].to_numpy(dtype=int)
        if assignments.min() != 0 or assignments.max() != k - 1:
            raise ValueError(f"Existing assignments do not represent exactly k={k} clusters")
        fit_source = f"reused assignments from {args.assignments_csv.resolve()}"
    else:
        # Full Lloyd k-means. It runs on CPU because sklearn's implementation
        # does not use Apple MPS; no embedding computation occurs.
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=args.n_init,
            max_iter=args.max_iter,
            algorithm="lloyd",
            random_state=args.seed,
            verbose=0,
        )
        assignments = model.fit_predict(embeddings)
        fit_source = "new full Lloyd k-means fit"

    base_rate = target.mean()
    rows: list[dict[str, float | int]] = []
    for cluster in range(k):
        member = assignments == cluster
        n_cluster = int(member.sum())
        n_target = int((member & target).sum())
        precision = n_target / n_cluster
        recall = n_target / target.sum()
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "cluster": cluster,
                "cluster_documents": n_cluster,
                "tech_market_documents": n_target,
                "precision_target_within_cluster": precision,
                "recall_target_covered": recall,
                "f1_single_cluster": f1,
                "enrichment_vs_corpus_rate": precision / base_rate if base_rate else np.nan,
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["f1_single_cluster", "tech_market_documents", "precision_target_within_cluster"],
        ascending=False,
        ignore_index=True,
    )
    summary.to_csv(out_dir / "cluster_target_overlap.csv", index=False)

    # `Sci_Tech` is the canonical value in data/agnews/agnews_all.csv.
    label_order = ["World", "Sports", "Business", "Sci_Tech"]
    label_counts = pd.crosstab(
        pd.Series(assignments, name="cluster"),
        # `corpus_labels` retains document IDs as its index.  Build a
        # positional series to align it with the one-row-per-embedding vector.
        pd.Series(corpus_labels.to_numpy(), name="label_name"),
    ).reindex(index=range(k), columns=label_order, fill_value=0)
    purity = pd.DataFrame({
        "cluster": label_counts.index,
        "cluster_documents": label_counts.sum(axis=1).to_numpy(),
        "dominant_label": label_counts.idxmax(axis=1).to_numpy(),
        "dominant_label_share": (label_counts.max(axis=1) / label_counts.sum(axis=1)).to_numpy(),
    })
    for label in label_order:
        purity[f"share_{label.lower().replace('/', '_')}"] = (
            label_counts[label] / label_counts.sum(axis=1)
        ).to_numpy()
    purity.to_csv(out_dir / "cluster_label_purity.csv", index=False)

    output = ids.copy()
    output[f"kmeans{k}_cluster"] = assignments
    output[f"is_sae_{target_spec['slug']}_business_target"] = target
    output.to_csv(out_dir / "document_assignments.csv", index=False)

    best = summary.iloc[0]
    target_binary = target.astype(int)
    nmi = normalized_mutual_info_score(target_binary, assignments, average_method="arithmetic")
    ami = adjusted_mutual_info_score(target_binary, assignments, average_method="arithmetic")

    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    point_size = 22 + 0.05 * summary["cluster_documents"]
    scatter = ax.scatter(
        summary["recall_target_covered"] * 100,
        summary["precision_target_within_cluster"] * 100,
        s=point_size,
        c=summary["enrichment_vs_corpus_rate"],
        cmap="viridis",
        edgecolor="white",
        linewidth=0.7,
    )
    for _, row in summary.head(5).iterrows():
        ax.annotate(
            f"C{int(row.cluster)}", (row.recall_target_covered * 100, row.precision_target_within_cluster * 100),
            xytext=(4, 4), textcoords="offset points", fontsize=9,
        )
    ax.axhline(base_rate * 100, color="black", linestyle="--", linewidth=1, label="Corpus target rate")
    ax.set(
        xlabel="Recall of SAE technology-market target (%)",
        ylabel="Precision: target share within k-means cluster (%)",
        title=f"k={k} raw-embedding k-means: recovery of SAE {target_spec['title']} Business subset",
    )
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Enrichment versus corpus target rate")
    fig.savefig(out_dir / "cluster_precision_recall.png", dpi=220)
    fig.savefig(out_dir / "cluster_precision_recall.pdf")
    plt.close(fig)

    display = summary.head(10).copy()
    for column in ("precision_target_within_cluster", "recall_target_covered", "f1_single_cluster"):
        display[column] = (100 * display[column]).map(lambda value: f"{value:.2f}%")
    display["enrichment_vs_corpus_rate"] = display["enrichment_vs_corpus_rate"].map(lambda value: f"{value:.2f}×")
    purity_display = purity.copy()
    for column in [c for c in purity_display if c.endswith("_share") or c.startswith("share_")]:
        purity_display[column] = purity_display[column].map(lambda value: f"{100 * value:.2f}%")
    report = [
        f"# k={k} k-means test: AG News {target_spec['title']} case",
        "",
        "## Question",
        "",
        f"This case study tests whether the {target.sum():,} Business-labelled documents in the SAE's {target_spec['description']} emerge as a single cluster under k-means with {k} centres.",
        "",
        "## Design",
        "",
        f"K-means is fitted once, without labels or SAE codes, to all 127,600 cached raw unnormalised Qwen embeddings (1,024 dimensions). The target is defined only after fitting: the {target.sum():,} Business documents in SAE community {args.community}. This therefore tests post-hoc recovery of the SAE-defined subset; it does not establish an independent ground truth category.",
        "",
        "A single cluster is judged by precision (target documents within the cluster), recall (target documents captured), F1, and enrichment over the target's corpus prevalence. A high precision alone is not sufficient if recall is negligible; a high recall alone is not sufficient if the cluster is mostly unrelated documents.",
        "",
        "## Result",
        "",
        f"The corpus target rate is **{base_rate:.4%}** ({target.sum():,} / {len(target):,}). The best single k-means cluster by F1 is **cluster {int(best.cluster)}**: {int(best.tech_market_documents):,} target documents in {int(best.cluster_documents):,} documents, precision **{best.precision_target_within_cluster:.2%}**, recall **{best.recall_target_covered:.2%}**, F1 **{best.f1_single_cluster:.3f}**, and enrichment **{best.enrichment_vs_corpus_rate:.2f}×**. Binary target-versus-cluster agreement is NMI **{nmi:.4f}** and adjusted MI **{ami:.4f}**.",
        "",
        "![Cluster recovery](cluster_precision_recall.png)",
        "",
        "## Ten clusters with highest target F1",
        "",
        markdown_table(display),
        "",
        "## AG News label purity of every k-means cluster",
        "",
        "Each row is one k-means cluster. `dominant_label_share` is the share of its documents belonging to its most frequent AG News category; the remaining columns give the full four-category composition. These labels are **not** used while fitting k-means, and purity should not be read as a test-set accuracy.",
        "",
        markdown_table(purity_display),
        "",
        "## Interpretation",
        "",
        f"This is a focused falsification-style comparison, not a general comparison of k-means and sparse codes. If no cluster jointly attains substantial precision and recall, the claim that this conjunction appears as one k={k} cluster is not supported by this geometry. Conversely, a strong single-cluster match would show that this particular configuration is available to a coarse partition, but would not by itself reproduce the compositional or cross-cutting use of the bits.",
        "",
        "## Reproducibility",
        "",
        f"- seed: {args.seed}",
        f"- k: {k}; n_init: {args.n_init}; max_iter: {args.max_iter}; algorithm: Lloyd k-means",
        f"- assignment source: {fit_source}",
        "- embeddings: cached raw unnormalised Qwen Qwen3-Embedding-0.6B last-token embeddings; no encoder or embedding rerun",
        f"- `cluster_target_overlap.csv`: all {k} cluster-level overlap statistics",
        "- `cluster_label_purity.csv`: cluster size, dominant category, and the full AG News category distribution",
        "- `document_assignments.csv`: the 127,600 assignments and target indicator",
    ]
    (out_dir / "report.md").write_text("\n".join(report) + "\n")
    metadata = {
        "seed": args.seed,
        "k": k,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "algorithm": "lloyd",
        "n_documents": int(len(target)),
        "target_documents": int(target.sum()),
        "target_definition": f"Business documents in Qwen SAE community {args.community} ({target_spec['description']})",
        "assignment_source": fit_source,
        "embedding_metadata": embedding_metadata,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(out_dir)


if __name__ == "__main__":
    main()
