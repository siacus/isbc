#!/usr/bin/env python3
"""Qwen/AG News compression extensions: collisions, centroids, paired probes.

All unsupervised analyses in this program use only hard codes, cached raw
embeddings, and label-free Leiden communities.  AG News labels are loaded only
for the already-defined held-out predictive probe and its stratified bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from information_retention import _decoder_parameters
from predictive_modeling import run_modeling_benchmark


SEED = 1234
BOOTSTRAPS = 2000
COLLISION_BOOTSTRAPS = 1000
MAX_COLLISION_REFERENCES = 5000


@dataclass(frozen=True)
class Configuration:
    name: str
    k: int
    target: int
    run_dir: Path
    comparison_dir: Path | None


DEFAULT_CONFIGURATIONS = (
    Configuration(
        "k8_target4", 8, 4,
        Path("runs/agnews_qwen_all_k8_target4_fixed_balance"),
        Path("runs/agnews_embedding_comparison/all_k8_target4_label_name"),
    ),
    Configuration(
        "k8_target6", 8, 6,
        Path("runs/agnews_qwen_all_k8_target6_fixed_balance"), None,
    ),
    Configuration(
        "k16_target6", 16, 6,
        Path("runs/agnews_qwen_all_fixed_balance"),
        Path("runs/agnews_embedding_comparison/all_fixed_balance_label_name"),
    ),
    Configuration(
        "k32_target6", 32, 6,
        Path("runs/agnews_qwen_all_k32_target6_fixed_balance"),
        Path("runs/agnews_embedding_comparison/all_k32_target6_label_name"),
    ),
    Configuration(
        "k32_target8", 32, 8,
        Path("runs/agnews_qwen_all_k32_target8_fixed_balance"),
        Path("runs/agnews_embedding_comparison/all_k32_target8_label_name"),
    ),
    Configuration(
        "k32_target16", 32, 16,
        Path("runs/agnews_qwen_all_k32_target16_fixed_balance"),
        Path("runs/agnews_embedding_comparison/all_k32_target16_label_name"),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))


def profile_keys(bits: np.ndarray) -> np.ndarray:
    """A compact sortable categorical key for every complete hard profile."""
    packed = np.packbits(np.asarray(bits, dtype=np.uint8), axis=1, bitorder="little")
    packed = np.ascontiguousarray(packed)
    return packed.view(np.dtype((np.void, packed.shape[1]))).ravel()


def read_codes(config: Configuration) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    path = config.run_dir / "encoded_agnews_all_ready_best.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    bit_columns = [f"bit_{index:02d}" for index in range(config.k)]
    probability_columns = [f"prob_{index:02d}" for index in range(config.k)]
    if not set(bit_columns + probability_columns + ["id"]).issubset(header):
        raise ValueError(f"{path} does not contain the expected {config.k} bits")
    frame = pd.read_csv(path, usecols=["id", *bit_columns, *probability_columns], dtype={"id": str})
    return (
        frame["id"].to_numpy(dtype=str),
        frame[bit_columns].to_numpy(dtype=np.uint8),
        frame[probability_columns].to_numpy(dtype=np.float32),
        path,
    )


def raw_cache_paths(config: Configuration) -> tuple[Path, Path, Path]:
    stem = "agnews_all_ready"
    return (
        config.run_dir / f"raw_embeddings_{stem}_best.npy",
        config.run_dir / f"raw_embeddings_{stem}_best_ids.csv",
        config.run_dir / f"raw_embeddings_{stem}_best_metadata.json",
    )


def load_raw(config: Configuration, expected_ids: np.ndarray) -> tuple[np.memmap, dict[str, Any], Path]:
    matrix_path, ids_path, metadata_path = raw_cache_paths(config)
    if not all(path.is_file() for path in (matrix_path, ids_path, metadata_path)):
        raise FileNotFoundError(f"Missing raw cache for {config.name}")
    ids = pd.read_csv(ids_path, dtype={"id": str})["id"].to_numpy(dtype=str)
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"Raw-cache IDs do not align for {config.name}")
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.shape[0] != len(expected_ids) or matrix.ndim != 2:
        raise ValueError(f"Malformed raw cache for {config.name}: {matrix.shape}")
    return matrix, json.loads(metadata_path.read_text()), matrix_path


def load_communities(config: Configuration) -> dict[int, np.ndarray]:
    path = config.run_dir / "bit_structure" / "bit_community_nodes.csv"
    metadata_path = config.run_dir / "bit_structure" / "bit_community_metadata.json"
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Label-free Leiden community artifacts are required for {config.name}: {path}"
        )
    metadata = json.loads(metadata_path.read_text())
    if float(metadata.get("principal_resolution", -1)) != 1.0:
        raise ValueError(f"Unexpected principal Leiden resolution for {config.name}")
    nodes = pd.read_csv(path)
    column = "community_resolution_1"
    if column not in nodes:
        raise ValueError(f"Missing principal-community column for {config.name}")
    nodes["bit_index"] = nodes["bit"].str.extract(r"(\d+)$").astype(int)
    communities: dict[int, np.ndarray] = {}
    for community, group in nodes.groupby(column, sort=True):
        communities[int(community)] = np.sort(group["bit_index"].to_numpy(dtype=int))
    if sum(len(value) for value in communities.values()) != config.k:
        raise ValueError(f"Community nodes do not cover exactly K bits for {config.name}")
    return communities


def cosine_distances(raw: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = np.asarray(raw[first], dtype=np.float32)
    right = np.asarray(raw[second], dtype=np.float32)
    numerator = np.einsum("ij,ij->i", left, right, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    result = np.full(len(first), np.nan, dtype=float)
    valid = denominator > 0
    result[valid] = 1.0 - numerator[valid] / denominator[valid]
    return result


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    draw_indices = rng.integers(0, len(values), size=(draws, len(values)), endpoint=False)
    draws_mean = values[draw_indices].mean(axis=1)
    low, high = percentile_interval(draws_mean)
    return float(values.mean()), low, high


def _profile_groups(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(np.r_[0, counts[:-1]])
    return unique, inverse, counts, np.column_stack([starts, starts + counts]), order


def sample_same_profile_partners(
    reference: np.ndarray,
    inverse: np.ndarray,
    bounds: np.ndarray,
    sorted_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.empty(len(reference), dtype=np.int64)
    for position, index in enumerate(reference):
        start, stop = bounds[inverse[index]]
        count = stop - start
        offset = int(rng.integers(count - 1))
        candidate = sorted_indices[start + offset]
        if candidate == index:
            candidate = sorted_indices[stop - 1]
        result[position] = candidate
    return result


def sample_community_partners(
    reference: np.ndarray,
    community_indices: np.ndarray,
    inverse: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.empty(len(reference), dtype=np.int64)
    for position, index in enumerate(reference):
        own = inverse[index]
        # Rejection sampling is uniform over community members conditional on a
        # different full profile. Fallback is exact when a profile dominates.
        for _ in range(1000):
            candidate = community_indices[int(rng.integers(len(community_indices)))]
            if inverse[candidate] != own:
                result[position] = candidate
                break
        else:
            possible = community_indices[inverse[community_indices] != own]
            if len(possible) == 0:
                raise RuntimeError("Community partner requested without an eligible profile")
            result[position] = possible[int(rng.integers(len(possible)))]
    return result


def sample_random_partners(reference: np.ndarray, n_documents: int, rng: np.random.Generator) -> np.ndarray:
    candidates = rng.integers(0, n_documents - 1, size=len(reference), endpoint=False)
    return candidates + (candidates >= reference)


def summarize_collision_scope(
    *,
    config: Configuration,
    scope: str,
    community: int | None,
    required_bits: np.ndarray | None,
    raw: np.ndarray,
    reference: np.ndarray,
    inverse: np.ndarray,
    bounds: np.ndarray,
    sorted_indices: np.ndarray,
    community_indices: np.ndarray | None,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    same_partner = sample_same_profile_partners(reference, inverse, bounds, sorted_indices, rng)
    random_partner = sample_random_partners(reference, len(raw), rng)
    values: dict[str, np.ndarray] = {
        "same_full_profile": cosine_distances(raw, reference, same_partner),
        "random_corpus_document": cosine_distances(raw, reference, random_partner),
    }
    if community_indices is not None:
        community_partner = sample_community_partners(reference, community_indices, inverse, rng)
        values["different_profile_same_community"] = cosine_distances(
            raw, reference, community_partner
        )
    rows: list[dict[str, Any]] = []
    for partner_type, distances in values.items():
        mean, low, high = bootstrap_mean(distances, rng, COLLISION_BOOTSTRAPS)
        rows.append(
            {
                "configuration": config.name,
                "k": config.k,
                "target_active": config.target,
                "scope": scope,
                "community": community,
                "required_bits": ";".join(f"bit_{bit:02d}" for bit in required_bits)
                if required_bits is not None else "",
                "partner_type": partner_type,
                "reference_documents_sampled": int(len(reference)),
                "mean_cosine_distance": mean,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "q05": float(np.quantile(distances, 0.05)),
                "q25": float(np.quantile(distances, 0.25)),
                "median": float(np.median(distances)),
                "q75": float(np.quantile(distances, 0.75)),
                "q95": float(np.quantile(distances, 0.95)),
            }
        )
    return rows


def collision_analysis(
    config: Configuration,
    bits: np.ndarray,
    raw: np.ndarray,
    communities: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = profile_keys(bits)
    _, inverse, counts, bounds, order = _profile_groups(keys)
    repeated = counts[inverse] > 1
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    profile_eligible = np.flatnonzero(repeated)
    profile_reference = rng.choice(
        profile_eligible, size=min(MAX_COLLISION_REFERENCES, len(profile_eligible)), replace=False
    )
    rows.extend(summarize_collision_scope(
        config=config, scope="full_profile", community=None, required_bits=None,
        raw=raw, reference=np.sort(profile_reference), inverse=inverse, bounds=bounds,
        sorted_indices=order, community_indices=None, rng=rng,
    ))
    coverage_rows.append({
        "configuration": config.name, "scope": "full_profile", "community": np.nan,
        "documents_total": len(bits), "documents_eligible": len(profile_eligible),
        "eligible_share": len(profile_eligible) / len(bits),
        "singleton_documents": int((counts[inverse] == 1).sum()),
        "singleton_document_share": float((counts[inverse] == 1).mean()),
        "reference_documents_sampled": len(profile_reference),
        "exclusion_reason": "singleton full profile",
    })

    for community, bit_indices in communities.items():
        member = np.flatnonzero(bits[:, bit_indices].all(axis=1))
        if len(member) == 0:
            coverage_rows.append({
                "configuration": config.name, "scope": "community", "community": community,
                "documents_total": len(bits), "documents_eligible": 0, "eligible_share": 0.0,
                "singleton_documents": np.nan, "singleton_document_share": np.nan,
                "reference_documents_sampled": 0, "exclusion_reason": "no joint community members",
            })
            continue
        by_profile = np.bincount(inverse[member], minlength=len(counts))
        eligible = member[(counts[inverse[member]] > 1) & (len(member) - by_profile[inverse[member]] > 0)]
        coverage_rows.append({
            "configuration": config.name, "scope": "community", "community": community,
            "documents_total": len(bits), "documents_eligible": len(eligible),
            "eligible_share": len(eligible) / len(bits),
            "singleton_documents": int((counts[inverse[member]] == 1).sum()),
            "singleton_document_share": float((counts[inverse[member]] == 1).mean()),
            "reference_documents_sampled": min(MAX_COLLISION_REFERENCES, len(eligible)),
            "exclusion_reason": "requires another same-profile document and a different-profile joint community member",
        })
        if len(eligible) == 0:
            continue
        reference = np.sort(rng.choice(eligible, size=min(MAX_COLLISION_REFERENCES, len(eligible)), replace=False))
        rows.extend(summarize_collision_scope(
            config=config, scope="community", community=community, required_bits=bit_indices,
            raw=raw, reference=reference, inverse=inverse, bounds=bounds, sorted_indices=order,
            community_indices=member, rng=rng,
        ))
    return pd.DataFrame(rows), pd.DataFrame(coverage_rows)


def _mse_summary(sse: float, n_rows: int, dim: int, baseline_sse: float) -> dict[str, float]:
    return {
        "sse": float(sse),
        "mse": float(sse / (n_rows * dim)),
        "r2": float(1.0 - sse / baseline_sse) if baseline_sse > 0 else float("nan"),
    }


def _centroids_from_indices(raw: np.ndarray, keys: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_keys = keys[indices]
    unique, inverse, counts = np.unique(selected_keys, return_inverse=True, return_counts=True)
    dim = raw.shape[1]
    sums = np.zeros((len(unique), dim), dtype=np.float32)
    for start in range(0, len(indices), 2048):
        stop = min(len(indices), start + 2048)
        np.add.at(sums, inverse[start:stop], np.asarray(raw[indices[start:stop]], dtype=np.float32))
    return unique, sums / counts[:, None].astype(np.float32), counts


def _lookup_positions(keys: np.ndarray, unique: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(unique, keys)
    seen = positions < len(unique)
    seen[seen] &= unique[positions[seen]] == keys[seen]
    return positions, seen


def _sse_against_profiles(raw: np.ndarray, indices: np.ndarray, positions: np.ndarray, seen: np.ndarray,
                          centroids: np.ndarray, fallback: np.ndarray) -> tuple[float, np.ndarray]:
    sse = 0.0
    per_row = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), 2048):
        stop = min(len(indices), start + 2048)
        prediction = np.repeat(fallback[None, :], stop - start, axis=0).astype(np.float32)
        local_seen = seen[start:stop]
        if local_seen.any():
            prediction[local_seen] = centroids[positions[start:stop][local_seen]]
        residual = np.asarray(raw[indices[start:stop]], dtype=np.float32) - prediction
        squared = np.square(residual, dtype=np.float64).sum(axis=1)
        per_row[start:stop] = squared
        sse += float(squared.sum())
    return sse, per_row


def _fit_affine_decoder(bits: np.ndarray, raw: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(train_indices), dtype=np.float64), bits[train_indices].astype(np.float64)])
    xtx = x.T @ x
    cross = np.zeros((x.shape[1], raw.shape[1]), dtype=np.float64)
    for start in range(0, len(train_indices), 2048):
        stop = min(len(train_indices), start + 2048)
        cross += x[start:stop].T @ np.asarray(raw[train_indices[start:stop]], dtype=np.float64)
    return np.linalg.lstsq(xtx, cross, rcond=None)[0]


def _sse_affine(bits: np.ndarray, raw: np.ndarray, indices: np.ndarray, coefficients: np.ndarray) -> float:
    sse = 0.0
    for start in range(0, len(indices), 2048):
        stop = min(len(indices), start + 2048)
        x = np.column_stack([np.ones(stop - start), bits[indices[start:stop]].astype(np.float64)])
        residual = np.asarray(raw[indices[start:stop]], dtype=np.float64) - x @ coefficients
        sse += float(np.square(residual).sum())
    return sse


def _sse_saved_decoder(config: Configuration, bits: np.ndarray, raw: np.ndarray, indices: np.ndarray) -> float:
    decoder = _decoder_parameters(config.run_dir, "best")
    if decoder is None:
        return float("nan")
    weight, bias = decoder
    sse = 0.0
    for start in range(0, len(indices), 2048):
        stop = min(len(indices), start + 2048)
        reconstruction = bits[indices[start:stop]].astype(np.float64) @ weight.T.astype(np.float64) + bias
        residual = np.asarray(raw[indices[start:stop]], dtype=np.float64) - reconstruction
        sse += float(np.square(residual).sum())
    return sse


def centroid_reconstruction(
    config: Configuration,
    bits: np.ndarray,
    raw: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keys = profile_keys(bits)
    all_indices = np.arange(len(bits), dtype=np.int64)
    dim = raw.shape[1]
    overall_mean = np.asarray(raw.mean(axis=0, dtype=np.float64), dtype=np.float64)
    total_sse = 0.0
    for start in range(0, len(raw), 2048):
        observed = np.asarray(raw[start:start + 2048], dtype=np.float64)
        total_sse += float(np.square(observed - overall_mean).sum())
    unique_all, centroids_all, counts_all = _centroids_from_indices(raw, keys, all_indices)
    positions_all, seen_all = _lookup_positions(keys, unique_all)
    centroid_sse, per_row_sse = _sse_against_profiles(
        raw, all_indices, positions_all, seen_all, centroids_all, overall_mean
    )
    ols_all = _fit_affine_decoder(bits, raw, all_indices)
    ols_sse = _sse_affine(bits, raw, all_indices, ols_all)
    saved_sse = _sse_saved_decoder(config, bits, raw, all_indices)
    tolerance = max(1e-6 * total_sse, 1e-2)
    if not (centroid_sse <= ols_sse + tolerance and ols_sse <= saved_sse + tolerance):
        raise RuntimeError(
            f"In-sample decoder ordering failed for {config.name}: centroid={centroid_sse}, "
            f"ols={ols_sse}, saved={saved_sse}"
        )
    full_rows = []
    for method, sse in (("profile_centroid", centroid_sse), ("affine_ols", ols_sse), ("saved_decoder", saved_sse)):
        full_rows.append({
            "configuration": config.name, "k": config.k, "target_active": config.target,
            "split": "full_corpus_in_sample", "method": method,
            "documents": len(bits), "embedding_dimensions": dim,
            **_mse_summary(sse, len(bits), dim, total_sse),
        })

    bands = [(1, 1, "singleton"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 49, "10-49"), (50, np.inf, "50+")]
    band_rows = []
    document_counts = counts_all[positions_all]
    for lower, upper, name in bands:
        mask = (document_counts >= lower) & (document_counts <= upper)
        band_rows.append({
            "configuration": config.name, "profile_size_band": name,
            "documents": int(mask.sum()), "document_share": float(mask.mean()),
            "profiles": int(((counts_all >= lower) & (counts_all <= upper)).sum()),
            "centroid_mse": float(per_row_sse[mask].mean() / dim) if mask.any() else float("nan"),
        })

    train_mean = np.asarray(raw[train_indices].mean(axis=0, dtype=np.float64), dtype=np.float64)
    test_mean = np.asarray(raw[test_indices].mean(axis=0, dtype=np.float64), dtype=np.float64)
    test_baseline_sse = 0.0
    for start in range(0, len(test_indices), 2048):
        observed = np.asarray(raw[test_indices[start:start + 2048]], dtype=np.float64)
        test_baseline_sse += float(np.square(observed - test_mean).sum())
    unique_train, centroids_train, _ = _centroids_from_indices(raw, keys, train_indices)
    positions_test, seen_test = _lookup_positions(keys[test_indices], unique_train)
    test_centroid_sse, test_centroid_per_row = _sse_against_profiles(
        raw, test_indices, positions_test, seen_test, centroids_train, train_mean
    )
    ols_train = _fit_affine_decoder(bits, raw, train_indices)
    test_ols_sse = _sse_affine(bits, raw, test_indices, ols_train)
    test_saved_sse = _sse_saved_decoder(config, bits, raw, test_indices)
    test_rows = []
    for method, sse in (("profile_centroid", test_centroid_sse), ("affine_ols_train", test_ols_sse), ("saved_decoder_transductive_code", test_saved_sse)):
        test_rows.append({
            "configuration": config.name, "k": config.k, "target_active": config.target,
            "split": "held_out_test", "method": method, "documents": len(test_indices),
            "embedding_dimensions": dim, **_mse_summary(sse, len(test_indices), dim, test_baseline_sse),
        })
    for subset, mask in (("seen_profile", seen_test), ("unseen_profile_fallback_to_train_mean", ~seen_test)):
        if mask.any():
            sse = float(test_centroid_per_row[mask].sum())
            subset_baseline = float(np.square(
                np.asarray(raw[test_indices[mask]], dtype=np.float64) - test_mean
            ).sum())
            test_rows.append({
                "configuration": config.name, "k": config.k, "target_active": config.target,
                "split": subset, "method": "profile_centroid", "documents": int(mask.sum()),
                "embedding_dimensions": dim,
                **_mse_summary(sse, int(mask.sum()), dim, subset_baseline),
            })
    return full_rows, test_rows, band_rows


def validate_existing_prediction_source(config: Configuration, expected_ids: np.ndarray) -> pd.DataFrame:
    if config.comparison_dir is None:
        raise ValueError("New configuration must be modeled separately")
    path = config.comparison_dir / "predictive_modeling_predictions.csv"
    frame = pd.read_csv(path, dtype={"id": str})
    result = frame.loc[
        (frame["run"] == "qwen")
        & (frame["representation"] == "hard_bits")
        & (frame["classifier"] == "logistic_regression")
    ].copy()
    if result["id"].duplicated().any() or len(result) == 0:
        raise ValueError(f"Bad Qwen hard-bit predictions for {config.name}")
    return result


def model_k8_target6(
    *, ids: np.ndarray, bits: np.ndarray, labels: np.ndarray,
    train_indices: np.ndarray, test_indices: np.ndarray, split_record: dict[str, Any], output_dir: Path,
) -> pd.DataFrame:
    modeling = run_modeling_benchmark(
        representations={"qwen": {"hard_bits": bits.astype(bool)}},
        y=labels, ids=ids, train_indices=train_indices, test_indices=test_indices,
        seed=SEED, output_dir=output_dir, split_record=split_record, target_name="label_name",
    )
    predictions = pd.read_csv(output_dir / modeling["artifacts"]["predictions"], dtype={"id": str})
    return predictions.loc[
        (predictions["run"] == "qwen")
        & (predictions["representation"] == "hard_bits")
        & (predictions["classifier"] == "logistic_regression")
    ].copy()


def stratified_bootstrap_metrics(
    truth: np.ndarray, probabilities: np.ndarray, predicted: np.ndarray,
    *, seed: int, draws: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    classes = np.unique(truth)
    strata = [np.flatnonzero(truth == value) for value in classes]
    samples = np.empty((draws, len(truth)), dtype=np.int32)
    offset = 0
    for stratum in strata:
        samples[:, offset:offset + len(stratum)] = stratum[
            rng.integers(0, len(stratum), size=(draws, len(stratum)), endpoint=False)
        ]
        offset += len(stratum)
    sample_hash = hashlib.sha256(samples[:10].tobytes()).hexdigest()
    correct = (predicted == truth)
    ba = np.empty(draws, dtype=float)
    for draw in range(draws):
        positions = samples[draw]
        ba[draw] = balanced_accuracy_score(truth[positions], predicted[positions])
    ce = -np.log2(np.clip(probabilities[samples], 1e-15, 1.0)).mean(axis=1)
    return ba, ce, sample_hash


def paired_predictive_analysis(predictions: dict[str, pd.DataFrame], configuration_order: list[str], output_dir: Path) -> pd.DataFrame:
    ordered: dict[str, pd.DataFrame] = {}
    reference_ids: np.ndarray | None = None
    reference_truth: np.ndarray | None = None
    for name in configuration_order:
        frame = predictions[name].sort_values("id", kind="stable").reset_index(drop=True)
        ids = frame["id"].to_numpy(dtype=str)
        truth = frame["true_label"].to_numpy(dtype=str)
        if reference_ids is None:
            reference_ids, reference_truth = ids, truth
        elif not np.array_equal(ids, reference_ids) or not np.array_equal(truth, reference_truth):
            raise ValueError(f"Held-out predictions do not align for {name}")
        ordered[name] = frame
    assert reference_truth is not None
    bootstrap: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    hashes: set[str] = set()
    point_metrics: dict[str, tuple[float, float]] = {}
    for name, frame in ordered.items():
        probability = frame["true_class_probability"].to_numpy(dtype=float)
        predicted = frame["predicted_label"].to_numpy(dtype=str)
        point_metrics[name] = (
            float(balanced_accuracy_score(reference_truth, predicted)),
            float((-np.log2(np.clip(probability, 1e-15, 1.0))).mean()),
        )
        ba, ce, sample_hash = stratified_bootstrap_metrics(
            reference_truth, probability, predicted, seed=SEED, draws=BOOTSTRAPS
        )
        bootstrap[name] = (ba, ce)
        hashes.add(sample_hash)
    if len(hashes) != 1:
        raise RuntimeError("Bootstrap samples unexpectedly differ across configurations")
    rows = []
    for left, right in combinations(configuration_order, 2):
        left_ba, left_ce = point_metrics[left]
        right_ba, right_ce = point_metrics[right]
        ba_delta = (bootstrap[left][0] - bootstrap[right][0]) * 100.0
        ce_delta = bootstrap[left][1] - bootstrap[right][1]
        ba_low, ba_high = percentile_interval(ba_delta)
        ce_low, ce_high = percentile_interval(ce_delta)
        rows.append({
            "configuration_a": left, "configuration_b": right,
            "balanced_accuracy_a": left_ba, "balanced_accuracy_b": right_ba,
            "balanced_accuracy_difference_pp_a_minus_b": (left_ba - right_ba) * 100.0,
            "balanced_accuracy_ci_low_pp": ba_low, "balanced_accuracy_ci_high_pp": ba_high,
            "cross_entropy_bits_a": left_ce, "cross_entropy_bits_b": right_ce,
            "cross_entropy_difference_bits_a_minus_b": left_ce - right_ce,
            "cross_entropy_ci_low_bits": ce_low, "cross_entropy_ci_high_bits": ce_high,
            "bootstrap_draws": BOOTSTRAPS, "bootstrap_seed": SEED,
            "bootstrap_first_draws_sha256": next(iter(hashes)),
        })
    return pd.DataFrame(rows)


def plot_collisions(frame: pd.DataFrame, output: Path) -> None:
    configs = list(dict.fromkeys(frame["configuration"]))
    figure, axes = plt.subplots(3, 2, figsize=(13, 14), sharey=True)
    for axis, config in zip(axes.flat, configs):
        subset = frame.loc[(frame["configuration"] == config) & (frame["scope"] == "community")]
        labels = []
        for community in sorted(subset["community"].dropna().unique()):
            block = subset.loc[subset["community"] == community]
            for partner in ("same_full_profile", "different_profile_same_community", "random_corpus_document"):
                row = block.loc[block["partner_type"] == partner]
                if row.empty:
                    continue
                row = row.iloc[0]
                x = len(labels)
                labels.append(f"C{int(community)}\n{partner.replace('_', ' ')}")
                axis.errorbar(x, row["mean_cosine_distance"],
                              yerr=[[row["mean_cosine_distance"] - row["bootstrap_ci_low"]],
                                    [row["bootstrap_ci_high"] - row["mean_cosine_distance"]]],
                              fmt="o", color={"same_full_profile":"#2166ac", "different_profile_same_community":"#f4a582", "random_corpus_document":"#b2182b"}[partner])
        axis.set_title(config.replace("_", " "))
        axis.set_xticks(range(len(labels)), labels, rotation=50, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=.25)
    figure.suptitle("Cosine distances: full-profile, community, and random partners")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_reconstruction(frame: pd.DataFrame, output: Path) -> None:
    data = frame.loc[frame["split"].eq("held_out_test") & frame["method"].isin(
        ["profile_centroid", "affine_ols_train", "saved_decoder_transductive_code"]
    )].copy()
    methods = ["profile_centroid", "affine_ols_train", "saved_decoder_transductive_code"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, color in zip(methods, ["#2166ac", "#4daf4a", "#b2182b"]):
        block = data.loc[data["method"] == method]
        axes[0].plot(block["configuration"], block["mse"], marker="o", label=method, color=color)
        axes[1].plot(block["configuration"], block["r2"], marker="o", label=method, color=color)
    for axis, title in zip(axes, ["Held-out reconstruction MSE", "Held-out reconstruction R²"]):
        axis.set_title(title); axis.tick_params(axis="x", rotation=35); axis.grid(alpha=.25); axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(output.with_suffix(".png"), dpi=180); figure.savefig(output.with_suffix(".pdf")); plt.close(figure)


def plot_predictive(frame: pd.DataFrame, output: Path) -> None:
    labels = [f"{a} − {b}" for a, b in zip(frame["configuration_a"], frame["configuration_b"])]
    y = np.arange(len(frame))
    figure, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    for axis, point, low, high, title in (
        (axes[0], "balanced_accuracy_difference_pp_a_minus_b", "balanced_accuracy_ci_low_pp", "balanced_accuracy_ci_high_pp", "Balanced-accuracy difference (pp)"),
        (axes[1], "cross_entropy_difference_bits_a_minus_b", "cross_entropy_ci_low_bits", "cross_entropy_ci_high_bits", "Cross-entropy difference (bits/document)"),
    ):
        values = frame[point].to_numpy(float)
        axis.errorbar(values, y, xerr=np.vstack([values-frame[low], frame[high]-values]), fmt="o", color="#2166ac")
        axis.axvline(0, color="black", lw=.8); axis.set_title(title); axis.grid(axis="x", alpha=.25)
    axes[0].set_yticks(y, labels, fontsize=8); axes[0].invert_yaxis()
    figure.tight_layout(); figure.savefig(output.with_suffix(".png"), dpi=180); figure.savefig(output.with_suffix(".pdf")); plt.close(figure)


def paper_handoff(output_dir: Path, collision: pd.DataFrame, reconstruction: pd.DataFrame,
                  predictive: pd.DataFrame, configs: list[Configuration]) -> None:
    lines = [
        "# Qwen AG News compression extensions: paper handoff", "",
        "## Design", "",
        "Six Qwen sparse autoencoders are compared on the same 127,600 AG News documents: "
        "8/4, 8/6, 16/6, 32/6, 32/8, and 32/16, where the first number is code width and the second is the target number of active bits. "
        "The 8/6 encoder was trained for three epochs on MPS with seed 1234 and a validated hard-linked cache of raw, unnormalised frozen Qwen embeddings. No frozen-backbone embeddings were recomputed.", "",
        "## Collision analysis", "",
        "For each configuration, a reference document is paired with one distinct document with the same full profile, one distinct document with a different profile but all bits of the same label-free Leiden community active, and one distinct random document. "
        "Communities use the positive-PMI weighted Leiden procedure at resolution 1. The reported intervals resample reference documents. AG News categories do not enter community construction or sampling.", "",
        "See `collision_summary.csv` and `collision_coverage.csv`. Community rows can overlap at document level and must not be added as disjoint counts.", "",
        "## Profile-centroid reconstruction", "",
        "A profile centroid is the mean raw embedding for documents with exactly the same complete hard code. It is evaluated in sample and with centroids fitted on the shared predictive training split and applied to test documents. Unseen test profiles fall back to the training embedding mean. The affine OLS decoder is refit on training documents for held-out evaluation. The saved decoder remains transductive because the unsupervised encoder and codes were fit on all documents.", "",
        "The in-sample check `SSE_centroid <= SSE_affine_OLS <= SSE_saved_decoder` passed for every configuration; see `reconstruction_full.csv`, `reconstruction_test.csv`, and `reconstruction_profile_size_bands.csv`.", "",
        "## Paired prediction", "",
        "All 15 pairs compare the hard-bit logistic probe on the same seeded, stratified 80/20 test IDs. Balanced-accuracy differences are A minus B in percentage points, so positive values favor A. Cross-entropy differences are A minus B in bits per document, so negative values favor A. The 95% intervals are individual percentile intervals from 2,000 common stratified bootstrap draws and are conditional on these fitted encoders and this split.", "",
        "See `paired_predictive_contrasts.csv` and `paired_predictive_contrasts.png`.", "",
        "## Artifacts", "",
        "- `manifest.json`: source paths, hashes, seeds, definitions, and validation checks.",
        "- `collision_*.csv` and `collision_distances.png`: label-free within-code and within-community distance analysis.",
        "- `reconstruction_*.csv` and `centroid_reconstruction.png`: centroid, OLS, and saved-decoder comparisons.",
        "- `paired_predictive_contrasts.csv` and `paired_predictive_contrasts.png`: all paired held-out prediction comparisons.",
    ]
    (output_dir / "paper_handoff.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/agnews_embedding_comparison/qwen_compression_extensions"))
    parser.add_argument("--labels", type=Path, default=Path("data/agnews/agnews_all.csv"))
    parser.add_argument("--split", type=Path, default=Path("runs/agnews_embedding_comparison/all_fixed_balance_label_name/predictive_split.csv"))
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = list(DEFAULT_CONFIGURATIONS)
    loaded: dict[str, dict[str, Any]] = {}
    reference_ids: np.ndarray | None = None
    raw_hashes: set[str] = set()
    for config in configurations:
        ids, bits, probabilities, encoded_path = read_codes(config)
        if reference_ids is None:
            reference_ids = ids
        elif not np.array_equal(ids, reference_ids):
            raise ValueError(f"Encoded IDs do not align for {config.name}")
        raw, raw_metadata, raw_path = load_raw(config, ids)
        raw_hashes.add(raw_metadata.get("matrix_sha256", sha256_file(raw_path)))
        loaded[config.name] = {"config": config, "ids": ids, "bits": bits, "probabilities": probabilities,
                               "encoded_path": encoded_path, "raw": raw, "raw_metadata": raw_metadata,
                               "raw_path": raw_path, "communities": load_communities(config)}
    assert reference_ids is not None
    if len(raw_hashes) != 1:
        raise ValueError("Qwen configurations do not share an identical raw embedding cache")

    labels_frame = pd.read_csv(args.labels, dtype={"id": str}, usecols=["id", "label_name"])
    labels_by_id = labels_frame.set_index("id")["label_name"]
    labels = labels_by_id.reindex(reference_ids)
    if labels.isna().any():
        raise ValueError("Labels do not align to encoded IDs")
    labels_array = labels.to_numpy(dtype=str)
    split_frame = pd.read_csv(args.split, dtype={"id": str})
    split_by_id = split_frame.set_index("id")["predictive_split"].reindex(reference_ids)
    if split_by_id.isna().any() or not set(split_by_id.unique()).issubset({"train", "test"}):
        raise ValueError("Predictive split does not align to encoded IDs")
    train_indices = np.flatnonzero(split_by_id.to_numpy() == "train")
    test_indices = np.flatnonzero(split_by_id.to_numpy() == "test")
    split_record = {"path": str(args.split.resolve()), "sha256": sha256_file(args.split),
                    "train_rows": int(len(train_indices)), "test_rows": int(len(test_indices))}

    collision_rows, coverage_rows = [], []
    reconstruction_full, reconstruction_test, reconstruction_bands = [], [], []
    for index, config in enumerate(configurations):
        item = loaded[config.name]
        rng = np.random.default_rng(SEED + 1009 * index)
        collision, coverage = collision_analysis(config, item["bits"], item["raw"], item["communities"], rng)
        collision_rows.append(collision); coverage_rows.append(coverage)
        full, test, bands = centroid_reconstruction(config, item["bits"], item["raw"], train_indices, test_indices)
        reconstruction_full.extend(full); reconstruction_test.extend(test); reconstruction_bands.extend(bands)
        print(f"completed unsupervised extensions: {config.name}", flush=True)
    collision_frame = pd.concat(collision_rows, ignore_index=True)
    coverage_frame = pd.concat(coverage_rows, ignore_index=True)
    full_frame = pd.DataFrame(reconstruction_full)
    test_frame = pd.DataFrame(reconstruction_test)
    bands_frame = pd.DataFrame(reconstruction_bands)
    collision_frame.to_csv(output_dir / "collision_summary.csv", index=False)
    coverage_frame.to_csv(output_dir / "collision_coverage.csv", index=False)
    full_frame.to_csv(output_dir / "reconstruction_full.csv", index=False)
    test_frame.to_csv(output_dir / "reconstruction_test.csv", index=False)
    bands_frame.to_csv(output_dir / "reconstruction_profile_size_bands.csv", index=False)
    plot_collisions(collision_frame, output_dir / "collision_distances")
    plot_reconstruction(test_frame, output_dir / "centroid_reconstruction")

    predictions: dict[str, pd.DataFrame] = {}
    for config in configurations:
        if config.name == "k8_target6":
            predictions[config.name] = model_k8_target6(
                ids=reference_ids, bits=loaded[config.name]["bits"], labels=labels_array,
                train_indices=train_indices, test_indices=test_indices, split_record=split_record,
                output_dir=output_dir / "k8_target6_predictive_modeling",
            )
        else:
            predictions[config.name] = validate_existing_prediction_source(config, reference_ids)
    predictive_frame = paired_predictive_analysis(predictions, [config.name for config in configurations], output_dir)
    predictive_frame.to_csv(output_dir / "paired_predictive_contrasts.csv", index=False)
    plot_predictive(predictive_frame, output_dir / "paired_predictive_contrasts")

    manifest = {
        "schema_version": 1, "purpose": "Qwen AG News compression extensions", "seed": SEED,
        "collision_bootstrap_draws": COLLISION_BOOTSTRAPS, "predictive_bootstrap_draws": BOOTSTRAPS,
        "max_collision_references_per_scope": MAX_COLLISION_REFERENCES,
        "raw_embedding_definition": "cached unnormalised frozen Qwen pooled embeddings",
        "collision_definition": "cosine distance; category labels excluded from community construction and sampling",
        "community_definition": "positive finite PMI graph, PMI weights, Leiden weighted modularity, resolution 1",
        "centroid_definition": "mean raw embedding conditional on complete hard profile",
        "test_centroid_fallback": "training embedding mean for a profile unseen in training",
        "predictive_difference_direction": {"balanced_accuracy": "A minus B, positive favors A", "cross_entropy": "A minus B, negative favors A"},
        "split": split_record,
        "labels": {"path": str(args.labels.resolve()), "sha256": sha256_file(args.labels)},
        "configurations": [
            {"name": config.name, "k": config.k, "target_active": config.target,
             "run_dir": str(config.run_dir.resolve()), "encoded": str(loaded[config.name]["encoded_path"].resolve()),
             "raw_cache": str(loaded[config.name]["raw_path"].resolve()),
             "raw_cache_sha256": loaded[config.name]["raw_metadata"].get("matrix_sha256"),
             "community_nodes": str((config.run_dir / "bit_structure" / "bit_community_nodes.csv").resolve())}
            for config in configurations
        ],
        "checks": {"encoded_ids_aligned": True, "raw_cache_hashes_identical": True,
                   "in_sample_reconstruction_ordering": "passed"},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    paper_handoff(output_dir, collision_frame, test_frame, predictive_frame, configurations)
    print(f"Wrote Qwen compression extensions to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
