import unittest

import numpy as np

from information_retention import (
    _ranks_from_distances,
    _trustworthiness_continuity_from_ranks,
    binary_entropy,
    code_rate_metrics,
    information_retention_markdown,
)


class InformationRetentionTests(unittest.TestCase):
    def test_binary_entropy_endpoints_and_fair_coin(self):
        observed = binary_entropy(np.asarray([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(observed, [0.0, 1.0, 0.0], atol=1e-12)

    def test_total_correlation_detects_duplicated_bit(self):
        base = np.asarray([[0], [1], [0], [1]], dtype=np.uint8)
        hard = np.column_stack([base, base])
        summary, per_bit = code_rate_metrics(hard, hard.astype(float))
        self.assertEqual(len(per_bit), 2)
        self.assertAlmostEqual(summary["joint_profile_entropy_bits_per_document"], 1.0)
        self.assertAlmostEqual(summary["total_correlation_bits_per_document"], 1.0)
        self.assertAlmostEqual(summary["normalized_redundancy"], 0.5)

    def test_identical_geometry_has_perfect_local_scores(self):
        distances = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 0.0, 1.5, 2.5, 3.5, 4.5],
                [2.0, 1.5, 0.0, 1.0, 2.0, 3.0],
                [3.0, 2.5, 1.0, 0.0, 1.5, 2.5],
                [4.0, 3.5, 2.0, 1.5, 0.0, 1.0],
                [5.0, 4.5, 3.0, 2.5, 1.0, 0.0],
            ]
        )
        order, ranks = _ranks_from_distances(distances)
        overlap, trustworthiness, continuity = (
            _trustworthiness_continuity_from_ranks(
                order, ranks, order, ranks, k=1
            )
        )
        self.assertAlmostEqual(overlap, 1.0)
        self.assertAlmostEqual(trustworthiness, 1.0)
        self.assertAlmostEqual(continuity, 1.0)

    def test_report_explains_metric_directions_and_both_plots(self):
        manifest = {
            "summary_rows": [],
            "geometry_rows": [],
            "utility_rows": [],
            "artifacts": {
                "summary": {"path": "information_retention_summary.csv"},
                "per_bit": {"path": "information_retention_per_bit.csv"},
                "geometry": {"path": "information_retention_geometry.csv"},
                "utility": {"path": "information_retention_utility.csv"},
                "metadata": {"path": "information_retention_metadata.json"},
                "overview_plot": {"path": "information_retention_overview.png"},
                "rate_distortion_plot": {"path": "information_retention_rate_distortion.png"},
            },
        }
        report = "\n".join(information_retention_markdown(manifest))
        self.assertIn("How to read the code-rate statistics", report)
        self.assertIn("How to read the reconstruction statistics", report)
        self.assertIn("Reading the rate--distortion plot", report)
        self.assertIn("How to read the geometry statistics", report)
        self.assertIn("Reading the three-panel overview", report)
        self.assertIn("How to read the predictive-utility statistics", report)
        self.assertIn("negative means the code outperforms raw embeddings", report)


if __name__ == "__main__":
    unittest.main()
