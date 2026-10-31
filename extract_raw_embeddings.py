#!/usr/bin/env python3
"""Export the exact unnormalised pooled backbone vectors used by one SAE run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from binary_factors_train import (
    basic_clean,
    choose_device,
    load_model_for_inference,
    raw_embedding_cache_is_valid,
    raw_embedding_cache_paths,
    tokenize_texts,
    write_raw_embedding_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--encoded", required=True)
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    encoded_path = Path(args.encoded).resolve()
    config_path = run_dir / args.checkpoint / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(encoded_path, dtype={"id": str}, usecols=["id", "text"])
    if frame["id"].duplicated().any():
        raise ValueError("Encoded input contains duplicate ids")

    stem = encoded_path.stem
    prefix = stem.removeprefix("encoded_").removesuffix(f"_{args.checkpoint}")
    matrix_path, _, metadata_path = raw_embedding_cache_paths(
        run_dir, prefix, args.checkpoint
    )
    configured_input = Path(config["input"]) if config.get("input") else None
    if configured_input is not None and not configured_input.is_absolute():
        configured_input = (Path.cwd() / configured_input).resolve()
    if configured_input is not None and not configured_input.is_file():
        configured_input = None
    if raw_embedding_cache_is_valid(
        run_dir,
        prefix,
        args.checkpoint,
        encoded_path,
        source_input=configured_input,
        encoding_threshold=config.get("threshold"),
    ):
        print(f"Reusing validated raw embeddings: {matrix_path}")
        return
    batch_size = args.batch_size or int(config.get("batch_size", 16))
    max_length = int(config.get("max_length", 128))

    inference_args = SimpleNamespace(
        output_dir=str(run_dir),
        precision=config.get("precision", "bf16"),
    )
    device = choose_device()
    model, tokenizer = load_model_for_inference(
        inference_args, device=device, which=args.checkpoint
    )
    temporary_path = matrix_path.with_suffix(".npy.partial")
    embeddings = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(frame), int(model.hidden_size)),
    )
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        texts = [basic_clean(text) for text in frame.iloc[start:stop]["text"].tolist()]
        tokens = tokenize_texts(tokenizer, texts, max_length, device)
        embeddings[start:stop] = model.encode(**tokens).detach().cpu().float().numpy()
        if start == 0 or stop == len(frame) or stop % (batch_size * 100) == 0:
            print(f"raw embeddings: {stop:,}/{len(frame):,}", flush=True)
    embeddings.flush()
    del embeddings
    temporary_path.replace(matrix_path)
    metadata = write_raw_embedding_metadata(
        output_dir=run_dir,
        input_stem=prefix,
        checkpoint=args.checkpoint,
        encoded_path=encoded_path,
        ids=frame["id"].astype(str).tolist(),
        embedding_dim=int(model.hidden_size),
        config=config,
        created_during_encoding=False,
        source_input=configured_input,
        encoding_threshold=config.get("threshold"),
    )
    print(f"Saved {matrix_path} and {metadata_path}")


if __name__ == "__main__":
    main()
