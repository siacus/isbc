#!/usr/bin/env python3
"""Compare angular and Euclidean geometry on the saved Qwen/AG News codes.

Uses exactly the document/pair sampling of information_retention.geometry_metrics.
No labels, backbone inference, or encoder training are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

from analyze_qwen_compression_extensions import (
    DEFAULT_CONFIGURATIONS, load_raw, read_codes, sha256_file,
)


def main() -> None:
    output = Path("runs/agnews_embedding_comparison/qwen_compression_extensions")
    rows, sources = [], []
    reference_ids = reference_raw = sample_indices = pair_i = pair_j = None
    seed = 1234
    for config in DEFAULT_CONFIGURATIONS:
        ids, hard, _, encoded_path = read_codes(config)
        raw, _, raw_path = load_raw(config, ids)
        raw_hash = sha256_file(raw_path)
        if reference_ids is None:
            reference_ids = ids
            reference_raw = raw_hash
            rng = np.random.default_rng(seed)
            # Collection reports align documents in lexicographic string-ID order.
            ordered_positions = np.argsort(ids, kind="stable")
            sample_indices = ordered_positions[np.sort(rng.choice(len(ids), min(2000, len(ids)), replace=False))]
            pair_i, pair_j = np.triu_indices(len(sample_indices), k=1)
            if len(pair_i) > 100_000:
                selected = rng.choice(len(pair_i), 100_000, replace=False)
                pair_i, pair_j = pair_i[selected], pair_j[selected]
            x = np.asarray(raw[sample_indices], dtype=np.float32)
            if np.any(np.linalg.norm(x, axis=1) == 0):
                raise ValueError("Zero embedding: cosine is undefined")
            raw_cosine = pairwise_distances(x, metric="cosine", n_jobs=-1)[pair_i, pair_j]
            raw_euclidean = pairwise_distances(x.astype(float), metric="euclidean")[pair_i, pair_j]
            pd.DataFrame({"sample_index": sample_indices, "id": ids[sample_indices]}).to_csv(
                output / "geometry_sample_documents.csv", index=False)
            np.savez_compressed(output / "geometry_sample_pairs.npz", i=pair_i, j=pair_j)
        elif not np.array_equal(ids, reference_ids) or raw_hash != reference_raw:
            raise ValueError(f"Inputs differ across configurations: {config.name}")

        b = hard[sample_indices]
        if not np.isin(b, [0, 1]).all():
            raise ValueError("Nonbinary code")
        zero_count = int((b.sum(axis=1) == 0).sum())
        if zero_count:
            raise ValueError(f"{config.name}: {zero_count} zero codes; angular comparison needs explicit handling")
        hard_cosine = pairwise_distances(b.astype(float), metric="cosine")[pair_i, pair_j]
        hard_hamming = pairwise_distances(b.astype(bool), metric="hamming")[pair_i, pair_j]
        hard_jaccard = pairwise_distances(b.astype(bool), metric="jaccard")[pair_i, pair_j]
        hard_euclidean = np.sqrt(config.k * hard_hamming)
        # Independent check of the binary Euclidean/Hamming identity.
        explicit = np.linalg.norm(b[pair_i].astype(float) - b[pair_j], axis=1)
        np.testing.assert_allclose(hard_euclidean, explicit)
        rho = lambda left, right: float(spearmanr(left, right).statistic)
        row = {
            "configuration": config.name, "K": config.k, "target_active": config.target,
            "documents": len(sample_indices), "pairs": len(pair_i), "zero_codes": zero_count,
            "cosine_cosine": rho(raw_cosine, hard_cosine),
            "euclidean_hamming": rho(raw_euclidean, hard_hamming),
            "euclidean_jaccard": rho(raw_euclidean, hard_jaccard),
            "cosine_jaccard": rho(raw_cosine, hard_jaccard),
            "cosine_hamming": rho(raw_cosine, hard_hamming),
        }
        np.testing.assert_allclose(row["euclidean_hamming"], rho(raw_euclidean, hard_euclidean), atol=1e-14)
        if config.comparison_dir is not None:
            old = pd.read_csv(config.comparison_dir / "information_retention_geometry.csv")
            old = old.loc[(old.run == "qwen") & (old.metric == "pairwise_distance_spearman")]
            for representation, key in [("hard_jaccard", "cosine_jaccard"), ("hard_hamming", "cosine_hamming")]:
                np.testing.assert_allclose(row[key], old.loc[old.representation == representation, "value"].item(), atol=1e-12, rtol=0)
        rows.append(row)
        sources.append({"configuration": config.name, "encoded": str(encoded_path.resolve()),
                        "encoded_sha256": sha256_file(encoded_path), "raw": str(raw_path.resolve()),
                        "raw_sha256": raw_hash})
        print(row, flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "geometry_distance_sensitivity.csv", index=False)
    metadata = {
        "seed": seed, "sample_documents": len(sample_indices), "sample_pairs": len(pair_i),
        "sampling": "Lexicographic string-ID alignment, sorted uniform document sample without replacement, then uniform unordered pairs without replacement; identical across configurations and to original geometry analysis",
        "raw_embedding_preprocessing": "None; cached unnormalised embeddings",
        "statistic": "Spearman correlation, average ranks for ties",
        "hamming_definition": "Fraction of unequal bit coordinates; sqrt(K * Hamming) equals binary Euclidean distance",
        "cosine_zero_policy": "Fail on zero embedding or sampled zero code; none observed",
        "checks": ["IDs and raw hashes identical across all configurations", "Legacy cosine correlations reproduced within 1e-12", "Binary Euclidean/Hamming equivalence verified"],
        "raw_cosine_vs_raw_euclidean_spearman": rho(raw_cosine, raw_euclidean),
        "sources": sources,
    }
    (output / "geometry_distance_sensitivity.json").write_text(json.dumps(metadata, indent=2) + "\n")
    columns = ["K", "target_active", "cosine_cosine", "euclidean_hamming", "euclidean_jaccard", "cosine_jaccard"]
    table = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    for row in rows:
        table += "| " + " | ".join(str(row[c]) if c in columns[:2] else f"{row[c]:.4f}" for c in columns) + " |\n"
    (output / "geometry_distance_sensitivity.md").write_text(
        "# Qwen AG News: sensitivity to distance definitions\n\n"
        "All entries are Spearman correlations on the same 100,000 pairs from the same 2,000 documents (seed 1234). "
        "Distances are calculated in the original cached embedding space without normalisation and in the hard binary code space. "
        "No category labels are used.\n\n" + table + "\n"
        "Cosine/cosine measures preservation of angular ordering. Euclidean/Hamming measures preservation of Euclidean distance ordering: "
        "for binary vectors, squared Euclidean distance equals the number of unequal bits. Normalised Hamming divides this count by K, "
        "which does not affect ranks within a configuration. Jaccard divides unequal active bits by the union of active bits, providing "
        "a complementary overlap-based measure. Neither correlation is a fraction of information or variance retained.\n\n"
        "The legacy cosine/Jaccard and cosine/Hamming values were reproduced for all five previously evaluated configurations. "
        "The 8/6 configuration is included here as well. No sampled embedding or code has zero norm. "
        "The pairs are dependent because documents recur; no independent-pair confidence intervals are claimed.\n\n"
        "Reproduce from the repository root: `python analyze_qwen_geometry_sensitivity.py`.\n")


if __name__ == "__main__":
    main()
