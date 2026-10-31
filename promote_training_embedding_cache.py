#!/usr/bin/env python3
"""Validate and hard-link a frozen training-embedding cache into a new SAE run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from binary_factors_train import (
    read_input_table,
    training_embedding_cache_is_valid,
    training_embedding_cache_paths,
    write_training_embedding_metadata,
)


def link_or_copy(source: Path, target: Path) -> str:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache artifact: {target}")
    try:
        os.link(source, target)
        return "hard_link"
    except OSError:
        shutil.copyfile(source, target)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--target-run-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--pooling-mode", required=True)
    parser.add_argument("--text-prefix", default="")
    parser.add_argument("--objective", default="sparse_ae")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--backbone-precision", required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    args = parser.parse_args()

    input_path = args.input.resolve()
    ids = read_input_table(input_path)["id"].astype(str).tolist()
    stem = input_path.stem
    validation = dict(
        input_stem=stem,
        source_input=input_path,
        ids=ids,
        model_name=args.model_name,
        pooling_mode=args.pooling_mode,
        text_prefix=args.text_prefix,
        objective=args.objective,
        max_length=args.max_length,
        backbone_precision=args.backbone_precision,
        embedding_dim=args.embedding_dim,
    )
    if not training_embedding_cache_is_valid(output_dir=args.source_run_dir, **validation):
        raise SystemExit("Source training cache does not validate for the requested frozen representation.")

    target_matrix, target_ids, target_metadata = training_embedding_cache_paths(args.target_run_dir, stem)
    if any(path.exists() for path in (target_matrix, target_ids, target_metadata)):
        if training_embedding_cache_is_valid(output_dir=args.target_run_dir, **validation):
            print("Target cache is already valid; no action taken.")
            return
        raise SystemExit("Target run has an incomplete or incompatible cache; refusing to overwrite it.")

    args.target_run_dir.mkdir(parents=True, exist_ok=True)
    source_matrix, source_ids, _ = training_embedding_cache_paths(args.source_run_dir, stem)
    matrix_method = link_or_copy(source_matrix, target_matrix)
    ids_method = link_or_copy(source_ids, target_ids)
    metadata = write_training_embedding_metadata(output_dir=args.target_run_dir, **validation)
    provenance = {
        "schema_version": 1,
        "operation": "validated training-cache promotion across SAE runs",
        "source_run_dir": str(args.source_run_dir.resolve()),
        "target_run_dir": str(args.target_run_dir.resolve()),
        "matrix_link_method": matrix_method,
        "ids_link_method": ids_method,
        "cache_metadata": metadata,
    }
    (args.target_run_dir / "training_embedding_cache_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validated cache promoted from {args.source_run_dir} to {args.target_run_dir} ({matrix_method}).")


if __name__ == "__main__":
    main()
