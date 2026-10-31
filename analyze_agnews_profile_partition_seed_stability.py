#!/usr/bin/env python3
"""Compare complete 16-bit AG News profile partitions across SAE seeds.

This is deliberately different from aligned bit-by-bit stability: each document
is labelled by its *entire* hard-code profile and ARI/AMI compare the resulting
document partitions.  Both statistics are invariant to a permutation of bit
names, so no Hungarian alignment is needed or appropriate here.
"""

from __future__ import annotations

import json
import platform
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUT = RUNS / "agnews_qwen_all_fixed_balance" / "kmeans_business_crosscutting_case_study"
SEEDS = (1234, 2025, 31415)
BIT_COLUMNS = [f"bit_{index:02d}" for index in range(16)]


def load_profiles(seed: int, reference_ids: pd.Series | None = None) -> tuple[pd.Series, np.ndarray]:
    path = RUNS / f"agnews_qwen_loss_ablation_full_seed{seed}" / "encoded_agnews_all_ready_best.csv"
    frame = pd.read_csv(path, usecols=["id", *BIT_COLUMNS], dtype={"id": "string"})
    if frame["id"].duplicated().any():
        raise ValueError(f"Duplicate document IDs in {path}")
    if reference_ids is not None:
        frame = frame.set_index("id").reindex(reference_ids)
        if frame[BIT_COLUMNS].isna().any().any():
            raise ValueError(f"Document IDs do not agree with reference seed in {path}")
        frame = frame.reset_index()
    bits = frame[BIT_COLUMNS].to_numpy(dtype=np.uint16)
    if not np.isin(bits, [0, 1]).all():
        raise ValueError(f"Non-binary hard code found in {path}")
    # An integer is just a compact label for the complete 16-bit profile.
    profiles = bits @ (1 << np.arange(16, dtype=np.uint16))
    return frame["id"], profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="rebuild Markdown/metadata from an already validated CSV without recomputing ARI/AMI",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / "profile_partition_seed_stability.csv"
    if args.reuse_existing:
        results = pd.read_csv(results_path)
        if len(results) != 3 or not results.n_documents.eq(127_600).all():
            raise ValueError("Existing stability CSV is incomplete; rerun without --reuse-existing")
        n_documents = 127_600
    else:
        ids, first = load_profiles(SEEDS[0])
        profiles = {SEEDS[0]: first}
        for seed in SEEDS[1:]:
            _, profiles[seed] = load_profiles(seed, ids)
        n_documents = len(ids)
        if n_documents != 127_600:
            raise ValueError(f"Expected full AG News corpus of 127,600 documents; found {n_documents:,}")

        rows = []
        for left, right in combinations(SEEDS, 2):
            left_profiles, right_profiles = profiles[left], profiles[right]
            rows.append({
                "left_seed": left,
                "right_seed": right,
                "n_documents": n_documents,
                "left_observed_profiles": int(np.unique(left_profiles).size),
                "right_observed_profiles": int(np.unique(right_profiles).size),
                "profile_adjusted_rand_index": adjusted_rand_score(left_profiles, right_profiles),
                "profile_adjusted_mutual_information": adjusted_mutual_info_score(
                    left_profiles, right_profiles, average_method="arithmetic"
                ),
            })
        results = pd.DataFrame(rows)
        results.to_csv(results_path, index=False)

    display = results.copy()
    for column in ("profile_adjusted_rand_index", "profile_adjusted_mutual_information"):
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    for column in ("left_observed_profiles", "right_observed_profiles", "n_documents"):
        display[column] = display[column].map(lambda value: f"{value:,}")

    kmeans_metrics_path = OUT / "profile_cluster_information_decomposition.csv"
    kmeans_context = ""
    if kmeans_metrics_path.exists():
        metrics = pd.read_csv(kmeans_metrics_path)
        view = metrics[[
            "k",
            "profile_adjusted_mutual_information_with_cluster",
            "profile_adjusted_rand_index_with_cluster",
        ]].copy()
        view.columns = ["k", "profile AMI with k-means", "profile ARI with k-means"]
        for column in ("profile AMI with k-means", "profile ARI with k-means"):
            view[column] = view[column].map(lambda value: f"{value:.3f}")
        kmeans_context = "\n".join([
            "",
            "For scale, the same seed-1234 profile partition has the following chance-corrected agreement with raw-embedding k-means:",
            "",
            view.to_markdown(index=False),
            "",
            "These are deliberately different comparisons: the table above asks whether independently fitted SAEs induce the same profile partition, while this table asks whether one SAE profile partition matches a geometric k-means partition.",
            "",
            f"On the stringent co-assignment scale, seed-to-seed ARI ({results.profile_adjusted_rand_index.min():.3f}–{results.profile_adjusted_rand_index.max():.3f}) exceeds profile–k-means ARI ({metrics.profile_adjusted_rand_index_with_cluster.min():.3f}–{metrics.profile_adjusted_rand_index_with_cluster.max():.3f}). AMI orders these comparisons differently (seed-to-seed {results.profile_adjusted_mutual_information.min():.3f}–{results.profile_adjusted_mutual_information.max():.3f}; profile–k-means {metrics.profile_adjusted_mutual_information_with_cluster.min():.3f}–{metrics.profile_adjusted_mutual_information_with_cluster.max():.3f}). This is expected to depend on the markedly different numbers and size distributions of groups (thousands of observed profiles versus 32–256 clusters), so AMI and ARI should be reported together rather than treated as interchangeable absolute scales.",
        ])

    section = [
        "## Complete-profile stability across SAE seeds",
        "",
        "The existing seed check aligns **individual bits** with a Hungarian assignment and measures their document-level activation NMI. That establishes stability of the factor inventory, but it does not answer whether the **complete 16-bit profile partition** is stable: small differences in several bits can compound into a different exact code. Here every document is assigned its unsigned 16-bit hard-code profile, and the three full-objective Qwen 16/6 fits (seeds 1234, 2025, and 31415) are compared on all 127,600 common documents.",
        "",
        "ARI and AMI both compare partitions rather than bit names, so they are invariant to any permutation of bit indices; Hungarian alignment is neither required nor used. In both measures, 1 denotes identical document partitions and values near 0 denote chance-corrected agreement. ARI is pairwise and is especially stringent with many rare profiles; AMI measures the information shared by the two categorical profile assignments after chance correction.",
        "",
        display.to_markdown(index=False),
        "",
        f"**Conclusion.** The three fits reproduce a non-trivial but imperfect complete-profile partition (profile AMI {results.profile_adjusted_mutual_information.min():.3f}–{results.profile_adjusted_mutual_information.max():.3f}; profile ARI {results.profile_adjusted_rand_index.min():.3f}–{results.profile_adjusted_rand_index.max():.3f}). This is necessarily lower than aligned one-bit stability because the metric requires consistency of the whole code. It prevents interpreting any profile–k-means agreement in isolation: the appropriate benchmark is the seed-to-seed stability of the profile partition, alongside the already documented bit-level Hungarian/NMI stability. Conversely, imperfect profile ARI/AMI does not imply that stable individual bits are noise; it quantifies the additional composition sensitivity of a 16-bit exact profile.",
        kmeans_context,
        "",
        "Artifact: `profile_partition_seed_stability.csv`. The IDs are validated before comparison; each seed contributes one row per document, and no encoder or embedding is recomputed.",
    ]
    (OUT / "profile_partition_seed_stability_section.md").write_text("\n".join(section) + "\n")
    metadata = {
        "analysis": "complete hard-profile partition stability across full-objective Qwen SAE seeds",
        "seeds": list(SEEDS),
        "n_documents": n_documents,
        "code_width": 16,
        "metrics": {
            "adjusted_rand_index": "sklearn.metrics.adjusted_rand_score",
            "adjusted_mutual_information": "sklearn.metrics.adjusted_mutual_info_score, average_method=arithmetic",
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "alignment": "IDs validated and aligned to seed 1234; no bit Hungarian matching because partition metrics are permutation invariant",
        "refresh": "Use --reuse-existing only to rebuild prose from the validated cached CSV; default recomputes all metrics.",
    }
    (OUT / "profile_partition_seed_stability_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(OUT / "profile_partition_seed_stability_section.md")


if __name__ == "__main__":
    main()
