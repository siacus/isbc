from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import binary_factors_train as train


class _TextDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"text": str(index)}


class _Backbone:
    def eval(self) -> "_Backbone":
        return self


class _Model:
    hidden_size = 3

    def __init__(self, interrupt_on_call: int | None = None) -> None:
        self.backbone = _Backbone()
        self.interrupt_on_call = interrupt_on_call
        self.calls = 0

    def eval(self) -> "_Model":
        return self

    def encode(self, *, texts: list[str]) -> torch.Tensor:
        self.calls += 1
        if self.interrupt_on_call == self.calls:
            raise KeyboardInterrupt
        values = torch.tensor([float(text) for text in texts], dtype=torch.float32)
        return torch.stack((values, values + 0.25, -values), dim=1)


class ResumableEmbeddingCacheTest(unittest.TestCase):
    @staticmethod
    def _inputs(tmp_path: Path, rows: int = 40):
        source = tmp_path / "ready.csv"
        frame = pd.DataFrame(
            {
                "id": [str(index) for index in range(rows)],
                "text": [str(index) for index in range(rows)],
            }
        )
        frame.to_csv(source, index=False)
        output_dir = tmp_path / "run"
        output_dir.mkdir()
        args = SimpleNamespace(
            input=str(source),
            output_dir=str(output_dir),
            model_name="test/model",
            pooling_mode="mean",
            text_prefix="",
            objective="sparse_ae",
            max_length=8,
            backbone_dtype_resolved="float32",
            batch_size=2,
            num_workers=0,
        )
        return frame, output_dir, args

    def test_training_embedding_cache_resumes_after_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            rows = 40
            frame, output_dir, args = self._inputs(tmp_path, rows)

            with patch.object(
                train,
                "tokenize_texts",
                lambda tokenizer, texts, max_length, device: {"texts": texts},
            ):
                interrupted = _Model(interrupt_on_call=18)
                with self.assertRaises(KeyboardInterrupt):
                    train.ensure_training_embedding_cache(
                        interrupted,
                        tokenizer=None,
                        df=frame,
                        dataset=_TextDataset(rows),
                        args=args,
                        device=torch.device("cpu"),
                    )

                partial, progress = train.training_embedding_partial_paths(
                    output_dir, "ready"
                )
                self.assertTrue(partial.is_file())
                self.assertTrue(progress.is_file())
                self.assertEqual(json.loads(progress.read_text())["rows_written"], 32)

                resumed = _Model()
                matrix_path, metadata = train.ensure_training_embedding_cache(
                    resumed,
                    tokenizer=None,
                    df=frame,
                    dataset=_TextDataset(rows),
                    args=args,
                    device=torch.device("cpu"),
                )

            self.assertEqual(resumed.calls, 4)  # Eight rows at batch size two.
            expected_values = np.arange(rows, dtype=np.float32)
            expected = np.column_stack(
                (expected_values, expected_values + 0.25, -expected_values)
            )
            np.testing.assert_array_equal(np.load(matrix_path), expected)
            self.assertEqual(metadata["rows"], rows)
            self.assertFalse(partial.exists())
            self.assertFalse(progress.exists())

    def test_mismatched_partial_cache_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            rows = 40
            frame, output_dir, args = self._inputs(tmp_path, rows)

            with patch.object(
                train,
                "tokenize_texts",
                lambda tokenizer, texts, max_length, device: {"texts": texts},
            ):
                with self.assertRaises(KeyboardInterrupt):
                    train.ensure_training_embedding_cache(
                        _Model(interrupt_on_call=18),
                        tokenizer=None,
                        df=frame,
                        dataset=_TextDataset(rows),
                        args=args,
                        device=torch.device("cpu"),
                    )
                _, progress = train.training_embedding_partial_paths(output_dir, "ready")
                payload = json.loads(progress.read_text())
                payload["model_name"] = "incompatible/model"
                progress.write_text(json.dumps(payload))

                restarted = _Model()
                matrix_path, _ = train.ensure_training_embedding_cache(
                    restarted,
                    tokenizer=None,
                    df=frame,
                    dataset=_TextDataset(rows),
                    args=args,
                    device=torch.device("cpu"),
                )

            self.assertEqual(restarted.calls, 20)
            self.assertEqual(np.load(matrix_path).shape, (rows, 3))


if __name__ == "__main__":
    unittest.main()
