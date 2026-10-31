from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from information_retention_extended import (
    categorical_entropy_bits,
    fano_metrics,
    paired_cross_entropy_contrasts,
    pca_and_quantizer_codes,
    posthoc_linear_decoder_metrics_many,
    profile_label_association,
)


class ExtendedInformationRetentionTests(unittest.TestCase):
    def test_profile_label_adjustment_detects_identical_partitions(self) -> None:
        hard = np.asarray(
            [[0, 0], [0, 0], [1, 0], [1, 0], [0, 1], [0, 1]],
            dtype=np.uint8,
        )
        labels = np.asarray(["a", "a", "b", "b", "c", "c"])
        result = profile_label_association(hard, labels)
        self.assertAlmostEqual(result["adjusted_mutual_information"], 1.0)
        self.assertGreater(
            result["chance_corrected_mutual_information_bits_per_document"], 0.0
        )
        self.assertGreater(
            result["plugin_mutual_information_bits_per_document"],
            result[
                "chance_corrected_mutual_information_bits_per_document"
            ],
        )

    def test_fano_uses_ordinary_error_and_label_entropy(self) -> None:
        entropy = categorical_entropy_bits(np.asarray(["a", "b"] * 20))
        perfect = fano_metrics(1.0, entropy, 2)
        chance = fano_metrics(0.5, entropy, 2)
        self.assertAlmostEqual(
            perfect["fano_mutual_information_lower_bound_bits_per_document"],
            1.0,
        )
        self.assertAlmostEqual(
            chance["fano_mutual_information_lower_bound_bits_per_document"],
            0.0,
        )

    def test_paired_cross_entropy_contrast_has_expected_sign(self) -> None:
        rows = []
        for index, label in enumerate(["a", "a", "b", "b"]):
            for representation, probability in (
                ("raw_embeddings", 0.8),
                ("hard_bits", 0.4),
            ):
                rows.append(
                    {
                        "id": str(index),
                        "run": "model",
                        "representation": representation,
                        "classifier": "logistic_regression",
                        "true_label": label,
                        "true_class_probability": probability,
                    }
                )
        result = paired_cross_entropy_contrasts(
            pd.DataFrame(rows), seed=123, bootstrap_draws=100
        ).set_index("representation")
        self.assertAlmostEqual(
            result.loc[
                "raw_embeddings", "cross_entropy_loss_vs_raw_bits_per_document"
            ],
            0.0,
        )
        self.assertAlmostEqual(
            result.loc["hard_bits", "cross_entropy_loss_vs_raw_bits_per_document"],
            1.0,
        )
        self.assertGreater(result.loc["hard_bits", "ci_low"], 0.0)

    def test_pca_reference_and_sparse_controls_are_reproducible(self) -> None:
        rng = np.random.default_rng(9)
        latent = rng.normal(size=(180, 2))
        loading = rng.normal(size=(2, 6))
        raw = (latent @ loading + 0.02 * rng.normal(size=(180, 6))).astype(
            np.float32
        )
        ids = np.asarray([f"d{index}" for index in range(len(raw))])
        with tempfile.TemporaryDirectory() as directory:
            metrics, codes, protocol = pca_and_quantizer_codes(
                raw,
                ids,
                n_bits=2,
                target_active=1,
                seed=41,
                cache_dir=Path(directory),
            )
            decoded = posthoc_linear_decoder_metrics_many(raw, codes)
            self.assertGreater(metrics["pca_reconstruction_r2_ceiling"], 0.99)
            self.assertTrue(np.all(codes["pca_top_s"].sum(axis=1) == 1))
            self.assertTrue(np.all(codes["random_top_s"].sum(axis=1) == 1))
            for result in decoded.values():
                self.assertLessEqual(
                    result["posthoc_linear_decoder_r2"],
                    metrics["pca_reconstruction_r2_ceiling"] + 1e-6,
                )
            _, cached_codes, cached_protocol = pca_and_quantizer_codes(
                raw,
                ids,
                n_bits=2,
                target_active=1,
                seed=41,
                cache_dir=Path(directory),
            )
            self.assertFalse(protocol["reused"])
            self.assertTrue(cached_protocol["reused"])
            for name in codes:
                np.testing.assert_array_equal(codes[name], cached_codes[name])


if __name__ == "__main__":
    unittest.main()
