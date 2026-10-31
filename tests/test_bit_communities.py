"""Focused checks for the reusable positive-PMI Leiden community analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from report_bit_structure import (
    build_community_analysis,
    build_posthoc_profile_label_distribution,
    leiden_membership,
)


class BitCommunityAnalysisTests(unittest.TestCase):
    def test_edgeless_graph_retains_each_isolated_bit(self) -> None:
        membership, modularity = leiden_membership(
            n_bits=4, edges=[], weights=[], resolution=1.0, seed=1234
        )
        self.assertEqual(membership, [0, 1, 2, 3])
        self.assertEqual(modularity, 0.0)

    def test_pmi_is_the_only_graph_weight_and_fit_is_reproducible(self) -> None:
        bits = [f"bit_{index:02d}" for index in range(4)]
        pair_table = pd.DataFrame(
            [
                ("bit_00", "bit_01", 20, 0.20, 0.12, 0.50, 0.21),
                ("bit_00", "bit_02", 4, 0.04, 0.08, -1.00, -0.11),
                ("bit_00", "bit_03", 0, 0.00, 0.03, float("-inf"), -0.20),
                ("bit_01", "bit_02", 18, 0.18, 0.13, 0.47, 0.17),
                ("bit_01", "bit_03", 0, 0.00, 0.04, float("-inf"), -0.19),
                ("bit_02", "bit_03", 0, 0.00, 0.05, float("-inf"), -0.23),
            ],
            columns=[
                "bit_a", "bit_b", "joint_activation_count", "joint_activation_share",
                "expected_if_independent", "pmi_bits", "phi_correlation",
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            pair_table.to_csv(output / "bit_coactivation.csv", index=False)
            result = build_community_analysis(
                bits=bits,
                prevalence=np.asarray([0.4, 0.5, 0.3, 0.1]),
                pair_table=pair_table,
                output_dir=output,
                run_dir=Path("/synthetic/run"),
                encoded_path=Path("/synthetic/encoded.csv"),
            )
            edges = pd.read_csv(output / "bit_community_edges.csv")
            nodes = pd.read_csv(output / "bit_community_nodes.csv")
            metadata = json.loads((output / "bit_community_metadata.json").read_text())
            self.assertEqual(len(edges), 2)
            self.assertTrue(np.allclose(edges["weight_pmi_bits"], edges["pmi_bits"]))
            self.assertEqual(metadata["edge_weight_column"], "pmi_bits")
            self.assertTrue(all(metadata["self_checks"].values()))
            isolate = nodes.loc[nodes["bit"] == "bit_03"].iloc[0]
            self.assertTrue(isolate["is_isolated"])
            self.assertEqual(result["isolated_nodes"], 1)
            self.assertTrue((output / "bit_community_graph.png").is_file())
            self.assertTrue((output / "bit_community_graph.pdf").is_file())

    def test_profile_label_overlay_requires_all_community_bits_jointly(self) -> None:
        bits = ["bit_00", "bit_01", "bit_02"]
        encoded = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "text": ["joint A", "not joint B", "joint A again"],
                "bit_00": [1, 1, 1],
                "bit_01": [1, 0, 1],
                "bit_02": [0, 1, 1],
            }
        )
        nodes = pd.DataFrame(
            {
                "bit": bits,
                "community_resolution_1": [0, 0, 1],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            labels_path = output / "labels.csv"
            pd.DataFrame({"id": ["a", "b", "c"], "label": ["A", "B", "A"]}).to_csv(
                labels_path, index=False
            )
            result = build_posthoc_profile_label_distribution(
                encoded,
                bits,
                nodes,
                {
                    "label_column": "label",
                    "label_path": str(labels_path),
                    "categories": ["A", "B"],
                },
                output,
            )
            first = result["summary"].loc[result["summary"]["community"] == 0].iloc[0]
            # Row b has bit_00 but not bit_01, so it must be excluded.
            self.assertEqual(first["documents_with_all_community_bits"], 2)
            self.assertEqual(first["dominant_label"], "A")
            self.assertEqual(first["dominant_label_share"], 1.0)
            profiles = result["joint_profiles"]
            selected = profiles.loc[profiles["community"] == 0, "profile"].tolist()
            self.assertEqual(set(selected), {"110", "111"})
            self.assertNotIn("101", selected)
            samples = result["text_samples"]
            self.assertEqual(
                samples.loc[(samples["community"] == 0) & (samples["label"] == "A")].shape[0],
                2,
            )


if __name__ == "__main__":
    unittest.main()
