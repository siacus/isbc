#!/usr/bin/env python3
"""Extended references and uncertainty diagnostics for information retention.

All calculations consume saved raw embeddings and learned codes.  They never load an
embedding backbone or relearn the sparse autoencoder.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mutual_info_score,
)
from sklearn.metrics.cluster import contingency_matrix, expected_mutual_information


EPSILON = 1e-12


def binary_entropy_bits(probability: float) -> float:
    p = float(probability)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def categorical_entropy_bits(values: np.ndarray) -> float:
    _, counts = np.unique(np.asarray(values).astype(str), return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def profile_strings(bits: np.ndarray) -> np.ndarray:
    return np.asarray(["".join(map(str, row.astype(np.uint8))) for row in bits])


def profile_label_association(
    hard: np.ndarray, labels: np.ndarray
) -> dict[str, float | int]:
    """Return plug-in and exact chance-adjusted profile/label information."""
    profiles = profile_strings(hard)
    y = np.asarray(labels).astype(str)
    observed_nats = float(mutual_info_score(y, profiles))
    table = contingency_matrix(y, profiles, sparse=False)
    expected_nats = float(expected_mutual_information(table, len(y)))
    adjusted = float(
        adjusted_mutual_info_score(y, profiles, average_method="arithmetic")
    )
    label_entropy = categorical_entropy_bits(y)
    observed_bits = observed_nats / np.log(2.0)
    expected_bits = expected_nats / np.log(2.0)
    corrected_bits = observed_bits - expected_bits
    return {
        "n_documents": int(len(y)),
        "n_profiles": int(len(np.unique(profiles))),
        "n_label_classes": int(len(np.unique(y))),
        "label_entropy_bits_per_document": label_entropy,
        "plugin_mutual_information_bits_per_document": observed_bits,
        "expected_mutual_information_under_fixed_margins_bits_per_document": expected_bits,
        "chance_corrected_mutual_information_bits_per_document": corrected_bits,
        "plugin_fraction_label_entropy": (
            observed_bits / label_entropy if label_entropy > 0 else float("nan")
        ),
        "chance_corrected_fraction_label_entropy": (
            corrected_bits / label_entropy if label_entropy > 0 else float("nan")
        ),
        "adjusted_mutual_information": adjusted,
    }


def _chunked_projection(
    raw: np.ndarray,
    mean: np.ndarray,
    directions: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    result = np.empty((len(raw), directions.shape[0]), dtype=np.float32)
    directions64 = np.asarray(directions, dtype=np.float64)
    mean64 = np.asarray(mean, dtype=np.float64)
    for start in range(0, len(raw), chunk_size):
        stop = min(len(raw), start + chunk_size)
        centered = np.asarray(raw[start:stop], dtype=np.float64) - mean64
        result[start:stop] = (centered @ directions64.T).astype(np.float32)
    return result


def _itq(scores: np.ndarray, seed: int, iterations: int = 50) -> np.ndarray:
    """Unsupervised ITQ following alternating sign and Procrustes updates."""
    values = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.standard_normal((values.shape[1], values.shape[1])))
    for _ in range(iterations):
        binary = np.where(values @ rotation >= 0.0, 1.0, -1.0)
        left, _, right_t = np.linalg.svd(values.T @ binary, full_matrices=False)
        rotation = left @ right_t
    return (values @ rotation >= 0.0).astype(np.uint8)


def _top_s_codes(values: np.ndarray, active: int) -> np.ndarray:
    active = max(0, min(int(active), values.shape[1]))
    result = np.zeros(values.shape, dtype=np.uint8)
    if active == 0:
        return result
    indices = np.argpartition(values, values.shape[1] - active, axis=1)[
        :, -active:
    ]
    rows = np.arange(len(values))[:, None]
    result[rows, indices] = 1
    return result


def _cache_key(
    ids: np.ndarray,
    raw_shape: tuple[int, ...],
    n_bits: int,
    seed: int,
    raw_cache_key: str,
) -> str:
    digest = hashlib.sha256()
    digest.update("\0".join(np.asarray(ids).astype(str)).encode("utf-8"))
    digest.update(str(tuple(raw_shape)).encode("ascii"))
    digest.update(f"|{n_bits}|{seed}|{raw_cache_key}".encode("utf-8"))
    return digest.hexdigest()[:20]


def pca_and_quantizer_codes(
    raw: np.ndarray,
    ids: np.ndarray,
    *,
    n_bits: int,
    target_active: int,
    seed: int,
    cache_dir: Path,
    raw_cache_key: str = "",
    itq_iterations: int = 50,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Compute or load PCA ceiling inputs and nonlearned binary controls."""
    key = _cache_key(ids, tuple(raw.shape), n_bits, seed, raw_cache_key)
    cache_path = cache_dir / f"information_reference_codes_{key}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as saved:
            metadata = {
                "cache": str(cache_path.resolve()),
                "cache_key": key,
                "reused": True,
                "pca_solver": str(saved["pca_solver"].item()),
                "itq_iterations": int(saved["itq_iterations"].item()),
            }
            pca_metrics = {
                "pca_components": int(saved["pca_components"].item()),
                "pca_reconstruction_r2_ceiling": float(saved["pca_r2"].item()),
                "pca_normalized_mse_floor": float(saved["pca_nmse"].item()),
            }
            codes = {
                name: saved[name].astype(np.uint8)
                for name in (
                    "pca_sign",
                    "itq",
                    "random_hyperplane",
                    "pca_top_s",
                    "random_top_s",
                )
            }
        return pca_metrics, codes, metadata

    cache_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # A d x d covariance eigendecomposition is prohibitively expensive for
        # the largest embedding collections. Randomized SVD scales with K and
        # is deterministic under the recorded seed. It is a numerical estimate
        # of the theoretical PCA-K optimum, not a certified upper bound.
        pca = PCA(
            n_components=n_bits,
            svd_solver="randomized",
            random_state=seed,
            n_oversamples=max(32, 2 * n_bits),
            iterated_power=7,
        )
        pca.fit(raw)
    scores = _chunked_projection(raw, pca.mean_, pca.components_)
    pca_r2 = float(np.sum(pca.explained_variance_ratio_))

    rng = np.random.default_rng(seed)
    random_directions, _ = np.linalg.qr(
        rng.standard_normal((raw.shape[1], n_bits))
    )
    random_scores = _chunked_projection(raw, pca.mean_, random_directions.T)
    codes = {
        "pca_sign": (scores >= 0.0).astype(np.uint8),
        "itq": _itq(scores, seed=seed + 17, iterations=itq_iterations),
        "random_hyperplane": (random_scores >= 0.0).astype(np.uint8),
        "pca_top_s": _top_s_codes(scores, target_active),
        "random_top_s": _top_s_codes(random_scores, target_active),
    }
    np.savez_compressed(
        cache_path,
        pca_components=np.asarray(n_bits, dtype=np.int64),
        pca_r2=np.asarray(pca_r2, dtype=np.float64),
        pca_nmse=np.asarray(1.0 - pca_r2, dtype=np.float64),
        pca_solver=np.asarray("randomized"),
        itq_iterations=np.asarray(itq_iterations, dtype=np.int64),
        **codes,
    )
    return (
        {
            "pca_components": int(n_bits),
            "pca_reconstruction_r2_ceiling": pca_r2,
            "pca_normalized_mse_floor": float(1.0 - pca_r2),
        },
        codes,
        {
            "cache": str(cache_path.resolve()),
            "cache_key": key,
            "reused": False,
            "pca_solver": "randomized",
            "pca_reference_is_numerical_estimate": True,
            "itq_iterations": int(itq_iterations),
        },
    )


def posthoc_linear_decoder_metrics_many(
    raw: np.ndarray,
    codes: dict[str, np.ndarray],
    *,
    chunk_size: int = 2048,
) -> dict[str, dict[str, float | int]]:
    """Fit optimal affine least-squares decoders using sufficient statistics."""
    mean_raw = np.asarray(raw.mean(axis=0, dtype=np.float64), dtype=np.float64)
    means = {
        name: np.asarray(code.mean(axis=0, dtype=np.float64), dtype=np.float64)
        for name, code in codes.items()
    }
    xtx = {
        name: np.zeros((code.shape[1], code.shape[1]), dtype=np.float64)
        for name, code in codes.items()
    }
    xty = {
        name: np.zeros((code.shape[1], raw.shape[1]), dtype=np.float64)
        for name, code in codes.items()
    }
    baseline_sse = 0.0
    for start in range(0, len(raw), chunk_size):
        stop = min(len(raw), start + chunk_size)
        centered_raw = np.asarray(raw[start:stop], dtype=np.float64) - mean_raw
        baseline_sse += float(np.square(centered_raw).sum())
        for name, code in codes.items():
            centered_code = np.asarray(code[start:stop], dtype=np.float64) - means[name]
            xtx[name] += centered_code.T @ centered_code
            xty[name] += centered_code.T @ centered_raw

    results: dict[str, dict[str, float | int]] = {}
    for name in codes:
        inverse = np.linalg.pinv(xtx[name], rcond=1e-10)
        coefficient = inverse @ xty[name]
        explained_sse = float(np.sum(xty[name] * coefficient))
        r2 = explained_sse / baseline_sse if baseline_sse > 0 else float("nan")
        r2 = min(1.0, max(0.0, r2)) if np.isfinite(r2) else r2
        condition = float(np.linalg.cond(xtx[name]))
        results[name] = {
            "posthoc_linear_decoder_r2": float(r2),
            "posthoc_linear_decoder_normalized_mse": float(1.0 - r2),
            "code_rank": int(np.linalg.matrix_rank(xtx[name])),
            "code_condition_number": condition if np.isfinite(condition) else np.nan,
        }
    return results


def simple_code_rate(bits: np.ndarray) -> dict[str, float | int]:
    profiles = profile_strings(bits)
    _, counts = np.unique(profiles, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return {
        "mean_active_bits": float(np.asarray(bits).sum(axis=1).mean()),
        "observed_profile_count": int(len(counts)),
        "joint_profile_entropy_bits_per_document": entropy,
        "effective_profile_count": float(2.0**entropy),
        "collision_probability": float(np.square(probabilities).sum()),
    }


def fano_metrics(
    accuracy: float, label_entropy_bits: float, n_classes: int
) -> dict[str, float]:
    error = min(1.0, max(0.0, 1.0 - float(accuracy)))
    penalty = binary_entropy_bits(error)
    if n_classes > 1 and error > 0:
        penalty += error * np.log2(n_classes - 1)
    lower = max(0.0, label_entropy_bits - penalty)
    return {
        "classification_error": error,
        "fano_mutual_information_lower_bound_bits_per_document": float(lower),
        "fano_fraction_label_entropy_resolved": (
            float(lower / label_entropy_bits)
            if label_entropy_bits > 0
            else float("nan")
        ),
    }


def paired_cross_entropy_contrasts(
    predictions: pd.DataFrame,
    *,
    seed: int,
    bootstrap_draws: int = 2_000,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Paired class-stratified percentile intervals for CE(code)-CE(raw)."""
    frame = predictions[
        predictions["classifier"].astype(str) == "logistic_regression"
    ].copy()
    frame["id"] = frame["id"].astype(str)
    frame["true_label"] = frame["true_label"].astype(str)
    frame["true_class_probability"] = pd.to_numeric(
        frame["true_class_probability"], errors="coerce"
    ).clip(1e-15, 1.0)
    rows: list[dict[str, Any]] = []
    for run in dict.fromkeys(frame["run"].astype(str)):
        selected = frame[frame["run"].astype(str) == run]
        raw = selected[selected["representation"] == "raw_embeddings"].set_index("id")
        if raw.empty:
            continue
        for representation in dict.fromkeys(selected["representation"].astype(str)):
            current = selected[selected["representation"].astype(str) == representation].set_index("id")
            shared = raw.index.intersection(current.index, sort=False)
            if len(shared) != len(raw) or len(shared) != len(current):
                raise ValueError(f"Unpaired cross-entropy predictions for {run}/{representation}")
            raw_aligned = raw.loc[shared]
            current_aligned = current.loc[shared]
            if not np.array_equal(
                raw_aligned["true_label"].to_numpy(),
                current_aligned["true_label"].to_numpy(),
            ):
                raise ValueError(f"True labels differ for {run}/{representation}")
            truth = raw_aligned["true_label"].to_numpy()
            difference = (
                -np.log(current_aligned["true_class_probability"].to_numpy())
                + np.log(raw_aligned["true_class_probability"].to_numpy())
            ) / np.log(2.0)
            digest = hashlib.sha256(
                f"{seed}|{run}|{representation}|paired-ce".encode("utf-8")
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            bootstrap = np.zeros(bootstrap_draws, dtype=np.float64)
            for label in sorted(set(truth)):
                values = difference[truth == label]
                weight = len(values) / len(difference)
                for start in range(0, bootstrap_draws, batch_size):
                    stop = min(bootstrap_draws, start + batch_size)
                    sampled = rng.integers(
                        0, len(values), size=(stop - start, len(values))
                    )
                    bootstrap[start:stop] += weight * values[sampled].mean(axis=1)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            rows.append(
                {
                    "run": run,
                    "representation": representation,
                    "classifier": "logistic_regression",
                    "cross_entropy_loss_vs_raw_bits_per_document": float(difference.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_draws": int(bootstrap_draws),
                    "seed": int(seed),
                    "test_documents": int(len(difference)),
                }
            )
    return pd.DataFrame(rows)
