#!/usr/bin/env python3
"""
Self-supervised sparse binary factor discovery on text using Hugging Face Transformers.

Design choices
--------------
- Backbone : any HuggingFace AutoModel (frozen by default)
- Binary head : MLP -> K Bernoulli probabilities (straight-through estimator)
- Data : CSV, JSON, or JSONL with at least columns/fields: `id`, `text`

Pooling modes (--pooling_mode)
------------------------------
  mean         : mean pool over all non-padding tokens [default]
  last_token   : last non-padding token — correct for embedding-finetuned models
                 (Qwen3-Embedding, GTE-Qwen2, E5-Mistral, LLaMA-based embedders)
  weighted_mean: position-weighted mean (more weight to later tokens)

Model / pooling_mode pairings
-----------------------------
  google/embeddinggemma-2             --pooling_mode mean
  Qwen/Qwen3.5-0.8B-Base              --pooling_mode mean
  Qwen/Qwen3-Embedding-0.6B           --pooling_mode last_token
  Qwen/Qwen3-Embedding-4B             --pooling_mode last_token
  Qwen/Qwen3-Embedding-8B             --pooling_mode last_token
  Alibaba-NLP/gte-Qwen2-1.5B-instruct --pooling_mode last_token
  meta-llama/Llama-3.2-1B             --pooling_mode last_token

Objectives (--objective)
------------------------
  sparse_ae     : consolidated sparse binary autoencoder — variance-scaled
                  reconstruction plus adaptive document sparsity, fixed bit balance,
                  decorrelation, and Bernoulli entropy.  [default, recommended]
  consistency   : contrastive loss across two perturbed views of the same text
                  + sparsity + balance + decorrelation  [original]
  last_token    : [DEPRECATED — use --pooling_mode last_token instead]
  token_sample  : same losses but each view pools a random 50% subset of tokens
  mid_layer     : same losses but uses the middle transformer layer

Typical usage
-------------
# Qwen3-Embedding + sparse_ae (recommended):
python binary_factors_train.py \\
  --input data/agnews/agnews_all_ready.csv \\
  --output_dir runs/agnews_sae \\
  --model_name Qwen/Qwen3-Embedding-0.6B \\
  --pooling_mode last_token \\
  --objective sparse_ae \\
  --k_factors 16 --target_active 6 --epochs 3

# Base model + sparse_ae:
python binary_factors_train.py \\
  --input data/agnews/agnews_all_ready.csv \\
  --output_dir runs/agnews_sae \\
  --model_name Qwen/Qwen3.5-0.8B-Base \\
  --pooling_mode mean \\
  --objective sparse_ae \\
  --k_factors 16 --target_active 6 --epochs 3

Then encode:
python binary_factors_train.py \\
  --input data/agnews/agnews_all_ready.csv \\
  --output_dir runs/agnews_sae \\
  --mode encode

If training is interrupted, resume from the last completed epoch by re-running
the exact same command with --resume added (resume_state.pt in --output_dir
holds the model/optimizer/scheduler state, saved after every epoch):
python binary_factors_train.py \\
  --input data/agnews/agnews_all_ready.csv \\
  --output_dir runs/agnews_sae \\
  --model_name Qwen/Qwen3-Embedding-0.6B \\
  --pooling_mode last_token \\
  --objective sparse_ae \\
  --k_factors 16 --target_active 6 --epochs 3 \\
  --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from training_diagnostic_plots import write_compact_training_diagnostics


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    """Resolve a device without silently weakening an explicit requirement."""
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was explicitly requested but is unavailable. Run outside the "
                "sandbox with the local latent environment; CPU fallback is disabled."
            )
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "auto":
        raise ValueError(f"Unsupported device selection: {requested!r}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def choose_dtype(precision: str = "auto") -> Optional[torch.dtype]:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    # precision == "auto" — pick the best default for the available device
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:  # Ampere or newer — native bf16 tensor cores
            return torch.bfloat16
        return torch.float16  # Volta/Turing — native fp16 tensor cores
    if torch.backends.mps.is_available():
        return torch.bfloat16  # Apple Silicon — same 2-byte width as fp16 (halves memory
        # bandwidth) but fp32's exponent range, so it doesn't overflow to inf/NaN the way
        # fp16 can on losses like reconstruction MSE
    return None

def set_gpu_optimizations() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def short_model_name(model_name: str) -> str:
    """HuggingFace repo id ("Qwen/Qwen3-Embedding-0.6B") -> short tag ("Qwen3-Embedding-0.6B")."""
    return model_name.rstrip("/").split("/")[-1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_embedding_cache_paths(
    output_dir: str | Path, input_stem: str, checkpoint: str
) -> tuple[Path, Path, Path]:
    run_dir = Path(output_dir).resolve()
    prefix = f"raw_embeddings_{input_stem}_{checkpoint}"
    return (
        run_dir / f"{prefix}.npy",
        run_dir / f"{prefix}_ids.csv",
        run_dir / f"{prefix}_metadata.json",
    )


def training_embedding_cache_paths(
    output_dir: str | Path, input_stem: str
) -> tuple[Path, Path, Path]:
    """Canonical pooled-embedding cache used by frozen sparse-AE training."""
    run_dir = Path(output_dir).resolve()
    prefix = f"training_embeddings_{input_stem}"
    return (
        run_dir / f"{prefix}.npy",
        run_dir / f"{prefix}_ids.csv",
        run_dir / f"{prefix}_metadata.json",
    )


def training_embedding_partial_paths(
    output_dir: str | Path, input_stem: str
) -> tuple[Path, Path]:
    """Interrupted-cache matrix and its atomically updated progress record."""
    matrix_path, _, _ = training_embedding_cache_paths(output_dir, input_stem)
    partial_path = matrix_path.with_suffix(".npy.partial")
    return partial_path, partial_path.with_suffix(".partial.json")


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def synchronize_device(device: torch.device) -> None:
    """Make elapsed-time measurements include queued accelerator work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def record_bit_generation_timing(
    output_dir: str | Path, stage: str, payload: Dict[str, object]
) -> None:
    """Persist timing without conflating cached embeddings with bit generation."""
    output_path = Path(output_dir)
    timing_path = output_path / "bit_generation_timing.json"
    if timing_path.is_file():
        current = json.loads(timing_path.read_text(encoding="utf-8"))
    else:
        current = {
            "schema_version": 1,
            "definition": (
                "Wall-clock timing of SAE fitting and code materialisation. "
                "Cached frozen-backbone embedding construction is excluded when "
                "the recorded stage says embedding_generation_included=false."
            ),
        }
    current[stage] = payload
    write_json_atomic(timing_path, current)

    metadata_path = output_path / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["bit_generation_timing"] = {
            "schema_version": 1,
            "path": str(timing_path.resolve()),
            "definition": current["definition"],
        }
        write_json_atomic(metadata_path, metadata)


def training_embedding_cache_is_valid(
    *,
    output_dir: str | Path,
    input_stem: str,
    source_input: Path,
    ids: list[str],
    model_name: str,
    pooling_mode: str,
    text_prefix: str,
    objective: str,
    max_length: int,
    backbone_precision: str,
    embedding_dim: int,
) -> bool:
    """Validate that a training cache is the exact requested frozen representation."""
    matrix_path, ids_path, metadata_path = training_embedding_cache_paths(
        output_dir, input_stem
    )
    if not all(path.is_file() for path in (matrix_path, ids_path, metadata_path)):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "model_name": model_name,
            "pooling_mode": pooling_mode,
            "text_prefix": text_prefix,
            "objective": objective,
            "max_length": int(max_length),
            "backbone_precision": backbone_precision,
            "embedding_dim": int(embedding_dim),
            "rows": len(ids),
            "source_input_sha256": sha256_file(source_input),
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        cached_ids = pd.read_csv(ids_path, dtype={"id": str})["id"].astype(str).tolist()
        if cached_ids != ids:
            return False
        matrix = np.load(matrix_path, mmap_mode="r")
        return (
            matrix.dtype == np.float32
            and matrix.ndim == 2
            and matrix.shape == (len(ids), int(embedding_dim))
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def write_training_embedding_metadata(
    *,
    output_dir: str | Path,
    input_stem: str,
    source_input: Path,
    ids: list[str],
    model_name: str,
    pooling_mode: str,
    text_prefix: str,
    objective: str,
    max_length: int,
    backbone_precision: str,
    embedding_dim: int,
) -> dict:
    """Atomically finish metadata for a reusable frozen-backbone training cache."""
    matrix_path, ids_path, metadata_path = training_embedding_cache_paths(
        output_dir, input_stem
    )
    ids_temporary = ids_path.with_suffix(".csv.partial")
    pd.DataFrame({"id": ids}).to_csv(ids_temporary, index=False)
    ids_temporary.replace(ids_path)
    metadata = {
        "schema_version": 1,
        "definition": "raw unnormalised pooled backbone output before the binary head",
        "purpose": "single-pass frozen-backbone cache reused by SAE training and encoding",
        "l2_normalized": False,
        "storage_dtype": "float32",
        "backbone_eval_mode": True,
        "source_input": str(source_input.resolve()),
        "source_input_sha256": sha256_file(source_input),
        "model_name": model_name,
        "pooling_mode": pooling_mode,
        "text_prefix": text_prefix,
        "objective": objective,
        "max_length": int(max_length),
        "backbone_precision": backbone_precision,
        "rows": len(ids),
        "embedding_dim": int(embedding_dim),
        "matrix": str(matrix_path),
        "ids": str(ids_path),
        "matrix_sha256": sha256_file(matrix_path),
        "ids_sha256": sha256_file(ids_path),
    }
    temporary_metadata = metadata_path.with_suffix(".json.partial")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    return metadata


def promote_raw_cache_for_training(
    *,
    output_dir: str | Path,
    input_stem: str,
    source_input: Path,
    ids: list[str],
    model_name: str,
    pooling_mode: str,
    text_prefix: str,
    objective: str,
    max_length: int,
    backbone_precision: str,
    embedding_dim: int,
) -> bool:
    """Reuse an existing pooled-embedding cache for a fresh SAE fit.

    Raw pooled embeddings depend on the frozen backbone configuration and input
    corpus, but not on the learned binary head or decoder.  This promotes a
    validated encode-time cache to the canonical training-cache name using a
    hard link when possible, so corrected SAE runs never repeat backbone
    inference merely because their output checkpoint changes.
    """
    target_matrix, target_ids, target_metadata = training_embedding_cache_paths(
        output_dir, input_stem
    )
    if any(path.exists() for path in (target_matrix, target_ids, target_metadata)):
        return False

    precision_aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    expected_precision = precision_aliases.get(backbone_precision, backbone_precision)
    source_hash = sha256_file(source_input)
    for checkpoint in ("best", "last"):
        matrix_path, ids_path, metadata_path = raw_embedding_cache_paths(
            output_dir, input_stem, checkpoint
        )
        if not all(path.is_file() for path in (matrix_path, ids_path, metadata_path)):
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached_precision = precision_aliases.get(
                str(metadata.get("backbone_precision", "")),
                str(metadata.get("backbone_precision", "")),
            )
            expected = {
                "model_name": model_name,
                "pooling_mode": pooling_mode,
                "text_prefix": text_prefix,
                "max_length": int(max_length),
                "rows": len(ids),
                "embedding_dim": int(embedding_dim),
                "source_input_sha256": source_hash,
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                continue
            if cached_precision != expected_precision:
                continue
            cached_ids = (
                pd.read_csv(ids_path, dtype={"id": str})["id"].astype(str).tolist()
            )
            if cached_ids != ids:
                continue
            matrix = np.load(matrix_path, mmap_mode="r")
            if (
                matrix.dtype != np.float32
                or matrix.ndim != 2
                or matrix.shape != (len(ids), int(embedding_dim))
            ):
                continue

            temporary = target_matrix.with_suffix(".npy.partial")
            if temporary.exists():
                temporary.unlink()
            try:
                os.link(matrix_path, temporary)
            except OSError:
                shutil.copyfile(matrix_path, temporary)
            temporary.replace(target_matrix)
            write_training_embedding_metadata(
                output_dir=output_dir,
                input_stem=input_stem,
                source_input=source_input,
                ids=ids,
                model_name=model_name,
                pooling_mode=pooling_mode,
                text_prefix=text_prefix,
                objective=objective,
                max_length=max_length,
                backbone_precision=backbone_precision,
                embedding_dim=embedding_dim,
            )
            print(
                "Promoted validated raw embeddings to the training cache: "
                f"{matrix_path} -> {target_matrix}"
            )
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return False


def raw_embedding_cache_is_valid(
    output_dir: str | Path,
    input_stem: str,
    checkpoint: str,
    encoded_path: Path,
    source_input: Path | None = None,
    encoding_threshold: float | None = None,
) -> bool:
    matrix_path, ids_path, metadata_path = raw_embedding_cache_paths(
        output_dir, input_stem, checkpoint
    )
    if not all(path.is_file() for path in (matrix_path, ids_path, metadata_path)):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config_path = Path(output_dir) / checkpoint / "run_config.json"
        if not config_path.is_file():
            return False
        current_config = json.loads(config_path.read_text(encoding="utf-8"))
        if not all(
            metadata.get(metadata_key) == current_config.get(config_key, default)
            for metadata_key, config_key, default in (
                ("model_name", "model_name", None),
                ("pooling_mode", "pooling_mode", "mean"),
                ("text_prefix", "text_prefix", ""),
                ("max_length", "max_length", 128),
                ("backbone_precision", "precision", "bf16"),
            )
        ):
            return False
        if metadata.get("source_encoded_sha256") != sha256_file(encoded_path):
            return False
        if source_input is not None:
            if metadata.get("source_input_sha256") != sha256_file(source_input):
                return False
        if encoding_threshold is not None:
            cached_threshold = metadata.get("encoding_threshold")
            if cached_threshold is None or not math.isclose(
                float(cached_threshold), float(encoding_threshold), rel_tol=0, abs_tol=1e-12
            ):
                return False
        matrix = np.load(matrix_path, mmap_mode="r")
        ids = pd.read_csv(ids_path, dtype={"id": str})["id"]
        return (
            matrix.ndim == 2
            and len(matrix) == len(ids)
            and int(metadata.get("rows", -1)) == len(ids)
            and int(metadata.get("embedding_dim", -1)) == matrix.shape[1]
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def write_raw_embedding_metadata(
    *,
    output_dir: str | Path,
    input_stem: str,
    checkpoint: str,
    encoded_path: Path,
    ids: list[str],
    embedding_dim: int,
    config: dict,
    created_during_encoding: bool,
    source_input: Path | None = None,
    encoding_threshold: float | None = None,
) -> dict:
    """Finish and describe the persistent raw pooled-embedding cache."""
    matrix_path, ids_path, metadata_path = raw_embedding_cache_paths(
        output_dir, input_stem, checkpoint
    )
    pd.DataFrame({"id": ids}).to_csv(ids_path, index=False)
    metadata = {
        "schema_version": 2,
        "definition": "raw unnormalised pooled backbone output before the binary head",
        "l2_normalized": False,
        "storage_dtype": "float32",
        "created_during_encoding": bool(created_during_encoding),
        "run_dir": str(Path(output_dir).resolve()),
        "checkpoint": checkpoint,
        "source_encoded": str(encoded_path.resolve()),
        "source_encoded_sha256": sha256_file(encoded_path),
        "source_input": str(source_input.resolve()) if source_input is not None else None,
        "source_input_sha256": (
            sha256_file(source_input) if source_input is not None else None
        ),
        "encoding_threshold": (
            float(encoding_threshold) if encoding_threshold is not None else None
        ),
        "model_name": config["model_name"],
        "pooling_mode": config.get("pooling_mode", "mean"),
        "text_prefix": config.get("text_prefix", ""),
        "max_length": int(config.get("max_length", 128)),
        "backbone_precision": config.get("precision", "bf16"),
        "rows": len(ids),
        "embedding_dim": int(embedding_dim),
        "matrix": str(matrix_path),
        "ids": str(ids_path),
        "matrix_sha256": sha256_file(matrix_path),
        "ids_sha256": sha256_file(ids_path),
    }
    temporary_metadata = metadata_path.with_suffix(".json.partial")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(metadata_path)

    run_metadata_path = Path(output_dir) / "metadata.json"
    if run_metadata_path.is_file():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        run_metadata["raw_embedding_cache"] = metadata
        temporary_run_metadata = run_metadata_path.with_suffix(".json.partial")
        temporary_run_metadata.write_text(
            json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
        )
        temporary_run_metadata.replace(run_metadata_path)
    return metadata


def write_model_metadata(
    output_dir: str, args: argparse.Namespace, model: Optional[nn.Module] = None
) -> None:
    """Write launch parameters plus dimensions resolved from the selected backbone."""
    meta = vars(args).copy()
    meta["model_name_short"] = short_model_name(args.model_name)
    if model is not None:
        meta["embedding_dim"] = int(model.hidden_size)
        cfg = model.backbone.config
        text_cfg = getattr(cfg, "text_config", None)
        internal_dim = getattr(text_cfg, "hidden_size", getattr(cfg, "hidden_size", None))
        if internal_dim is not None:
            meta["backbone_internal_hidden_size"] = int(internal_dim)
    with open(Path(output_dir) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def backfill_resolved_dimensions(output_dir: str, ckpt_dir: Path, model: nn.Module) -> None:
    """Add dynamically resolved dimensions to checkpoints created by older versions."""
    cfg = model.backbone.config
    text_cfg = getattr(cfg, "text_config", None)
    internal_dim = getattr(text_cfg, "hidden_size", getattr(cfg, "hidden_size", None))
    dimensions = {"embedding_dim": int(model.hidden_size)}
    if internal_dim is not None:
        dimensions["backbone_internal_hidden_size"] = int(internal_dim)

    for path in (ckpt_dir / "run_config.json", Path(output_dir) / "metadata.json"):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            current = json.load(handle)
        if all(current.get(key) == value for key, value in dimensions.items()):
            continue
        current.update(dimensions)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(current, handle, indent=2)
            handle.write("\n")


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
MULTISPACE_RE = re.compile(r"\s+")
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1+")
HASHTAG_RE = re.compile(r"#(\w+)")


def basic_clean(text: str) -> str:
    text = str(text).strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def perturb_text(text: str, rng: random.Random, token_drop_prob: float = 0.10) -> str:
    """
    Light perturbation meant to preserve broad latent identity while removing brittle surface details.
    """
    text = basic_clean(text)

    # stochastic URL removal
    if rng.random() < 0.70:
        text = URL_RE.sub(" ", text)

    # stochastic mention removal
    if rng.random() < 0.70:
        text = MENTION_RE.sub(" ", text)

    # stochastic hashtag masking (keep marker of hashtag-ness sometimes)
    if rng.random() < 0.50:
        text = HASHTAG_RE.sub("[HASHTAG]", text)

    # punctuation normalization
    if rng.random() < 0.50:
        text = REPEAT_PUNCT_RE.sub(r"\1", text)

    tokens = text.split()
    if len(tokens) > 2:
        kept = []
        for tok in tokens:
            if rng.random() > token_drop_prob:
                kept.append(tok)
        if len(kept) >= 2:
            tokens = kept
    text = " ".join(tokens)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text if text else "[EMPTY]"


# -----------------------------
# Data
# -----------------------------


def read_input_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path, columns=["id", "text"])
    elif suffix == ".json":
        # try records first, then plain JSON list/object
        try:
            df = pd.read_json(path)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            df = pd.DataFrame(obj)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(
            f"Unsupported input format: {suffix}. Use .csv, .parquet, .pq, .json, or .jsonl"
        )

    missing = {"id", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"Input must contain columns/fields: id, text. Missing: {sorted(missing)}")

    df = df[["id", "text"]].copy()
    df["text"] = df["text"].astype(str).map(basic_clean)
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df


class TweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seed: int = 1234):
        self.df = df.reset_index(drop=True)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.df.iloc[idx]
        return {"id": str(row["id"]), "text": str(row["text"]), "idx": idx}


class CachedEmbeddingDataset(Dataset):
    """Memory-mapped FP32 pooled embeddings with row order fixed by the ID sidecar."""

    def __init__(self, matrix_path: Path):
        self.matrix = np.load(matrix_path, mmap_mode="r")
        if self.matrix.ndim != 2 or self.matrix.dtype != np.float32:
            raise ValueError(
                f"Expected a 2D float32 embedding cache, got "
                f"shape={self.matrix.shape}, dtype={self.matrix.dtype}"
            )

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Copy one row so PyTorch receives writable, contiguous memory while the
        # complete matrix remains memory-mapped instead of occupying RAM.
        return torch.from_numpy(np.array(self.matrix[idx], dtype=np.float32, copy=True))


@dataclass
class BatchViews:
    ids: List[str]
    text_a: List[str]
    text_b: List[str]


class ContrastiveCollator:
    def __init__(self, seed: int = 1234):
        self.seed = seed

    def __call__(self, batch: List[Dict[str, str]]) -> BatchViews:
        ids = []
        text_a = []
        text_b = []
        for item in batch:
            ids.append(item["id"])
            text = item["text"]
            rng_a = random.Random(self.seed + hash((item["id"], "a")) % (10**9))
            rng_b = random.Random(self.seed + hash((item["id"], "b")) % (10**9))
            text_a.append(perturb_text(text, rng_a))
            text_b.append(perturb_text(text, rng_b))
        return BatchViews(ids=ids, text_a=text_a, text_b=text_b)


# -----------------------------
# Model
# -----------------------------

class MeanPooler(nn.Module):
    """Mean pool over non-padding tokens (default)."""
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked = hidden_states * mask
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return masked.sum(dim=1) / denom


class LastTokenPooler(nn.Module):
    """Pool using the last non-padding token (option 1: last_token)."""
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lens = attention_mask.sum(dim=1) - 1  # index of last real token
        seq_lens = seq_lens.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, seq_lens]


class TokenSamplePooler(nn.Module):
    """Pool over a random 50% subset of non-padding tokens (option 2: token_sample).
    Each call samples a different subset, creating genuine view variability."""
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L, D = hidden_states.shape
        pooled = []
        for b in range(B):
            valid = attention_mask[b].bool()
            valid_hidden = hidden_states[b][valid]          # (n_valid, D)
            n = max(1, valid_hidden.size(0) // 2)
            idx = torch.randperm(valid_hidden.size(0), device=hidden_states.device)[:n]
            pooled.append(valid_hidden[idx].mean(dim=0))
        return torch.stack(pooled, dim=0)


class WeightedMeanPooler(nn.Module):
    """Position-weighted mean pool — later tokens get more weight.
    Useful for causal LMs where later positions have seen more context."""
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        L = hidden_states.size(1)
        # weights: 1, 2, 3, ... L  (only for non-padding positions)
        pos_weights = torch.arange(1, L + 1, device=hidden_states.device, dtype=hidden_states.dtype)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)       # (B, L, 1)
        weights = pos_weights.unsqueeze(0).unsqueeze(-1) * mask            # (B, L, 1)
        denom = weights.sum(dim=1).clamp_min(1e-6)                         # (B, 1)
        return (hidden_states * weights).sum(dim=1) / denom


class BernoulliHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, k_factors: int,
                 target_rho: float, temperature: float = 0.50):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, k_factors)
        self.temperature = temperature
        self.k_factors = k_factors

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)

        # Initialize biases so sigmoid(logit / temperature) starts at target rho.
        target_rho = float(np.clip(target_rho, 1e-4, 1 - 1e-4))
        init_bias = temperature * math.log(target_rho / (1.0 - target_rho))
        nn.init.constant_(self.fc2.bias, init_bias)

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.norm(pooled)
        x = F.gelu(self.fc1(x))
        logits = self.fc2(x)
        probs = torch.sigmoid(logits / self.temperature)

        # Straight-through hard sample — keep everything in head dtype
        u = torch.rand_like(probs)
        hard = (probs > u).to(probs.dtype)   # bool→head dtype, not float32
        z_st = hard + probs - probs.detach()
        return {"logits": logits, "probs": probs, "z_st": z_st}


class LinearDecoder(nn.Module):
    """Simple linear decoder for sparse_ae objective: binary code -> embedding."""
    def __init__(self, k_factors: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(k_factors, output_dim, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class BinaryFactorsModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        k_factors: int,
        target_active: float,
        head_hidden_dim: int = 512,
        temperature: float = 0.50,
        freeze_backbone: bool = True,
        trust_remote_code: bool = True,
        objective: str = "sparse_ae",
        pooling_mode: str = "mean",
        backbone_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.objective = objective
        self.pooling_mode = pooling_mode

        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=backbone_dtype,
            trust_remote_code=trust_remote_code,
        )

        # Resolve the dimension emitted by the backbone. Some embedding models
        # project token states to a dedicated embedding dimension (for example,
        # EmbeddingGemma 2 maps 512d internal states to 768d outputs).
        cfg = self.backbone.config
        text_cfg = getattr(cfg, "text_config", None)
        if objective != "mid_layer" and text_cfg is not None and hasattr(text_cfg, "embedding_dim"):
            hidden_size = text_cfg.embedding_dim
        elif objective != "mid_layer" and hasattr(cfg, "embedding_dim"):
            hidden_size = cfg.embedding_dim
        elif hasattr(cfg, "hidden_size"):
            hidden_size = cfg.hidden_size
        elif text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            hidden_size = text_cfg.hidden_size
        else:
            raise ValueError(
                f"Cannot determine hidden_size from config: {type(cfg)}. "
                f"Available keys: {list(cfg.__dict__.keys())}"
            )

        # Choose pooler — fully independent of objective
        if pooling_mode == "last_token":
            self.pooler = LastTokenPooler()
        elif pooling_mode == "weighted_mean":
            self.pooler = WeightedMeanPooler()
        elif pooling_mode == "token_sample":
            self.pooler = TokenSamplePooler()
        else:  # mean (default)
            self.pooler = MeanPooler()

        rho = target_active / k_factors
        self.head = BernoulliHead(
            input_dim=hidden_size,
            hidden_dim=head_hidden_dim,
            k_factors=k_factors,
            target_rho=rho,
            temperature=temperature,
        )

        # Decoder only used for sparse_ae objective
        self.decoder = LinearDecoder(k_factors, hidden_size) if objective == "sparse_ae" else None

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.k_factors = k_factors
        self.target_active = target_active
        self.target_rho = rho
        self.hidden_size = hidden_size

    def _run_backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      use_mid_layer: bool = False, **backbone_inputs) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **backbone_inputs,
        )
        if use_mid_layer:
            layer_idx = len(out.hidden_states) // 2
            hidden = out.hidden_states[layer_idx]
        else:
            # Use the model's public final output rather than hidden_states[-1].
            # They are usually identical, but embedding-specific projection heads
            # (including EmbeddingGemma 2's 512->768 projection) are reflected only
            # in last_hidden_state.
            hidden = out.last_hidden_state
        pooled = self.pooler(hidden, attention_mask)
        if pooled.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"Backbone emitted {pooled.shape[-1]} pooled dimensions, but the model "
                f"configuration resolved embedding_dim={self.hidden_size}."
            )
        return pooled

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               **backbone_inputs) -> torch.Tensor:
        """Return pooled embedding cast to head dtype."""
        use_mid = (self.objective == "mid_layer")
        pooled = self._run_backbone(
            input_ids, attention_mask, use_mid_layer=use_mid, **backbone_inputs
        )
        return pooled.to(self.head.norm.weight.dtype)

    def forward_from_pooled(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run only the trainable SAE on an already-pooled raw embedding."""
        pooled = pooled.to(self.head.norm.weight.dtype)
        head_out = self.head(pooled)
        head_out["pooled"] = pooled
        if self.decoder is not None:
            z = head_out["z_st"].to(self.decoder.linear.weight.dtype)
            head_out["reconstructed"] = self.decoder(z)
        return head_out

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                **backbone_inputs) -> Dict[str, torch.Tensor]:
        pooled = self.encode(input_ids, attention_mask, **backbone_inputs)
        return self.forward_from_pooled(pooled)

# -----------------------------
# Losses
# -----------------------------

def cosine_consistency_loss(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(p1, p2, dim=-1)).mean()


def tweet_sparsity_loss(probs: torch.Tensor, target_rho: float) -> torch.Tensor:
    # Encourage each tweet to have average activation near target rho.
    per_tweet_mean = probs.mean(dim=1)
    return ((per_tweet_mean - target_rho) ** 2).mean()


def bit_balance_loss(probs: torch.Tensor, target_rho: float) -> torch.Tensor:
    # Encourage each bit to be used at the target rate across the batch.
    per_bit_mean = probs.mean(dim=0)
    return ((per_bit_mean - target_rho) ** 2).mean()


def decorrelation_loss(probs: torch.Tensor) -> torch.Tensor:
    # Penalize off-diagonal covariance among bits.
    x = probs - probs.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(1, probs.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).mean()


def reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between decoded binary code and original pooled embedding (sparse_ae only)."""
    return F.mse_loss(reconstructed.float(), target.detach().float())


@torch.no_grad()
def initialize_decoder_intercept_from_embedding_mean(
    model: "BinaryFactorsModel", coordinate_mean: torch.Tensor
) -> None:
    """Match the decoder's initial expected output to the embedding mean.

    The Bernoulli head is initialized at ``target_rho`` for every bit.  With the
    decoder weights held at their seeded initialization, the affine intercept
    that makes ``E[decoder(z)]`` equal the calibration mean is therefore
    ``mean - W @ E[z]``.  This removes a backbone-specific location offset from
    the initial reconstruction problem without centering or normalizing the raw
    embeddings themselves.
    """
    if model.decoder is None:
        raise ValueError("decoder-intercept initialization requires sparse_ae")
    bias = model.decoder.linear.bias
    weight = model.decoder.linear.weight
    mean = coordinate_mean.to(device=bias.device, dtype=bias.dtype)
    if mean.ndim != 1 or mean.numel() != bias.numel():
        raise ValueError(
            "embedding mean shape does not match decoder output: "
            f"mean={tuple(mean.shape)}, decoder_bias={tuple(bias.shape)}"
        )
    expected_code = torch.full(
        (model.k_factors,),
        float(model.target_rho),
        device=weight.device,
        dtype=weight.dtype,
    )
    bias.copy_(mean - weight @ expected_code)


def bernoulli_entropy_loss(
    probabilities: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """Mean conditional Bernoulli entropy; minimizing it sharpens soft bits."""
    clipped = probabilities.clamp(epsilon, 1.0 - epsilon)
    return -(
        clipped * clipped.log()
        + (1.0 - clipped) * (1.0 - clipped).log()
    ).mean()


@torch.no_grad()
def calibrate_embedding_scale(
    model: "BinaryFactorsModel",
    tokenizer,
    dataset: "TweetDataset",
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """Estimate one fixed variance scale without normalizing the embeddings."""
    n_requested = min(len(dataset), args.calibration_samples)
    subset = torch.utils.data.Subset(dataset, list(range(n_requested)))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    model.backbone.eval()
    sum_vector: Optional[torch.Tensor] = None
    sumsq_vector: Optional[torch.Tensor] = None
    n_seen = 0
    for batch in tqdm(loader, desc="Calibrating embedding scale", unit="batch"):
        texts = [basic_clean(text) for text in batch["text"]]
        tokenized = tokenize_texts(tokenizer, texts, args.max_length, device)
        pooled = model.encode(**tokenized).detach().float().cpu().double()
        if sum_vector is None:
            sum_vector = torch.zeros(pooled.shape[1], dtype=torch.float64)
            sumsq_vector = torch.zeros_like(sum_vector)
        sum_vector += pooled.sum(dim=0)
        sumsq_vector += pooled.pow(2).sum(dim=0)
        n_seen += pooled.shape[0]
    if n_seen < 2 or sum_vector is None or sumsq_vector is None:
        raise RuntimeError("At least two calibration examples are required.")
    coordinate_mean = sum_vector / n_seen
    coordinate_variance = (sumsq_vector / n_seen) - coordinate_mean.pow(2)
    scalar_variance = float(coordinate_variance.clamp_min(0).mean().item())
    if not math.isfinite(scalar_variance) or scalar_variance <= args.recon_scale_epsilon:
        raise RuntimeError(f"Invalid reconstruction calibration variance: {scalar_variance!r}.")
    return {
        "samples": int(n_seen),
        "embedding_dim": int(sum_vector.numel()),
        "coordinate_variance_mean": scalar_variance,
        "coordinate_mean_squared_mean": float(coordinate_mean.pow(2).mean().item()),
        "coordinate_second_moment_mean": float((sumsq_vector / n_seen).mean().item()),
        "_coordinate_mean": coordinate_mean.float(),
    }


@torch.no_grad()
def ensure_training_embedding_cache(
    model: "BinaryFactorsModel",
    tokenizer,
    df: pd.DataFrame,
    dataset: "TweetDataset",
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Path, dict]:
    """Compute the frozen pooled representation once, then reuse it everywhere."""
    source_input = Path(args.input)
    input_stem = source_input.stem
    ids = df["id"].astype(str).tolist()
    matrix_path, _, metadata_path = training_embedding_cache_paths(
        args.output_dir, input_stem
    )
    validation = {
        "output_dir": args.output_dir,
        "input_stem": input_stem,
        "source_input": source_input,
        "ids": ids,
        "model_name": args.model_name,
        "pooling_mode": args.pooling_mode,
        "text_prefix": args.text_prefix,
        "objective": args.objective,
        "max_length": args.max_length,
        "backbone_precision": args.backbone_dtype_resolved,
        "embedding_dim": int(model.hidden_size),
    }
    if training_embedding_cache_is_valid(**validation):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(
            "Reusing frozen training embeddings: "
            f"{matrix_path} ({metadata['rows']:,} x {metadata['embedding_dim']:,})"
        )
        return matrix_path, metadata

    partial_path, progress_path = training_embedding_partial_paths(
        args.output_dir, input_stem
    )
    progress_signature = {
        "schema_version": 1,
        "definition": "resumable frozen pooled-embedding cache",
        "source_input_sha256": sha256_file(source_input),
        "model_name": args.model_name,
        "pooling_mode": args.pooling_mode,
        "text_prefix": args.text_prefix,
        "objective": args.objective,
        "max_length": int(args.max_length),
        "backbone_precision": args.backbone_dtype_resolved,
        "rows": len(ids),
        "embedding_dim": int(model.hidden_size),
        "storage_dtype": "float32",
    }

    matrix = None
    offset = 0
    if partial_path.is_file() and progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if any(progress.get(key) != value for key, value in progress_signature.items()):
                raise ValueError("partial-cache signature mismatch")
            offset = int(progress["rows_written"])
            if not 0 <= offset <= len(df):
                raise ValueError("partial-cache row offset is out of bounds")
            matrix = np.load(partial_path, mmap_mode="r+")
            if (
                matrix.dtype != np.float32
                or matrix.ndim != 2
                or matrix.shape != (len(df), int(model.hidden_size))
            ):
                raise ValueError("partial-cache matrix shape or dtype mismatch")
            print(
                "Resuming frozen training-embedding cache: "
                f"{partial_path} from row {offset:,}/{len(df):,}"
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            if matrix is not None:
                del matrix
                matrix = None
            partial_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
            offset = 0
    elif partial_path.exists() or progress_path.exists():
        partial_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    if matrix is None:
        matrix = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(df), int(model.hidden_size)),
        )
        write_json_atomic(
            progress_path,
            {**progress_signature, "rows_written": 0},
        )

    remaining_dataset = Subset(dataset, range(offset, len(dataset)))
    loader = DataLoader(
        remaining_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    model.backbone.eval()
    checkpoint_every_batches = 16
    batches_since_checkpoint = 0
    try:
        for batch in tqdm(loader, desc="Caching frozen embeddings", unit="batch"):
            texts = [basic_clean(text) for text in batch["text"]]
            tokenized = tokenize_texts(tokenizer, texts, args.max_length, device)
            pooled = model.encode(**tokenized).detach().cpu().float().numpy()
            next_offset = offset + pooled.shape[0]
            matrix[offset:next_offset] = pooled
            offset = next_offset
            batches_since_checkpoint += 1
            if (
                batches_since_checkpoint >= checkpoint_every_batches
                or offset == len(df)
            ):
                matrix.flush()
                write_json_atomic(
                    progress_path,
                    {**progress_signature, "rows_written": offset},
                )
                batches_since_checkpoint = 0
        if offset != len(df):
            raise RuntimeError(
                f"Training embedding row mismatch: wrote {offset}, expected {len(df)}"
            )
        matrix.flush()
        del matrix
        matrix = None
        partial_path.replace(matrix_path)
        progress_path.unlink(missing_ok=True)
    except BaseException:
        if matrix is not None:
            matrix.flush()
            del matrix
        print(
            "Preserved resumable frozen-embedding cache at "
            f"{partial_path}; the next identical run will continue from the last "
            "committed batch."
        )
        raise

    metadata = write_training_embedding_metadata(**validation)
    print(
        "Saved frozen training embeddings: "
        f"{matrix_path} ({metadata['rows']:,} x {metadata['embedding_dim']:,})"
    )
    return matrix_path, metadata


def calibrate_cached_embedding_scale(
    matrix_path: Path, args: argparse.Namespace
) -> Dict[str, object]:
    """Apply the established variance calibration to the first cached rows."""
    matrix = np.load(matrix_path, mmap_mode="r")
    n_requested = min(len(matrix), args.calibration_samples)
    sum_vector = torch.zeros(matrix.shape[1], dtype=torch.float64)
    sumsq_vector = torch.zeros_like(sum_vector)
    n_seen = 0
    for start in range(0, n_requested, args.batch_size):
        stop = min(start + args.batch_size, n_requested)
        pooled = torch.from_numpy(
            np.array(matrix[start:stop], dtype=np.float32, copy=True)
        ).double()
        sum_vector += pooled.sum(dim=0)
        sumsq_vector += pooled.pow(2).sum(dim=0)
        n_seen += pooled.shape[0]
    if n_seen < 2:
        raise RuntimeError("At least two calibration examples are required.")
    coordinate_mean = sum_vector / n_seen
    coordinate_variance = (sumsq_vector / n_seen) - coordinate_mean.pow(2)
    scalar_variance = float(coordinate_variance.clamp_min(0).mean().item())
    if not math.isfinite(scalar_variance) or scalar_variance <= args.recon_scale_epsilon:
        raise RuntimeError(f"Invalid reconstruction calibration variance: {scalar_variance!r}.")
    return {
        "samples": int(n_seen),
        "embedding_dim": int(matrix.shape[1]),
        "coordinate_variance_mean": scalar_variance,
        "coordinate_mean_squared_mean": float(coordinate_mean.pow(2).mean().item()),
        "coordinate_second_moment_mean": float((sumsq_vector / n_seen).mean().item()),
        "source": "persistent_training_embedding_cache",
        "_coordinate_mean": coordinate_mean.float(),
    }


def release_frozen_backbone_from_accelerator(
    model: "BinaryFactorsModel", device: torch.device
) -> None:
    """Free accelerator memory once a deterministic frozen cache is available."""
    model.backbone.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def update_sparsity_multiplier(
    value: float, violation_ema: float, args: argparse.Namespace
) -> float:
    """Multiplicative constraint update used by the consolidated SAE."""
    log_step = args.adaptive_rate * (violation_ema - args.adaptive_tolerance)
    updated = value * math.exp(max(-2.0, min(2.0, log_step)))
    return float(min(args.lambda_max, max(args.lambda_min, updated)))


def adaptive_sparsity_is_enabled(args: argparse.Namespace) -> bool:
    """Return whether the adaptive activity penalty is part of this objective."""
    return bool(args.lambda_tweet_sparse > 0)


def consolidated_loss_spec(args: argparse.Namespace) -> Dict[str, object]:
    """Machine-readable record of the adopted sparse-AE objective."""
    return {
        "name": "fixed_balance_adaptive_sparsity_mean_intercept_v2",
        "formula": (
            "lambda_recon * (reconstruction_mse / embedding_variance) + "
            "lambda_sparse(t) * document_sparsity + lambda_balance * bit_balance + "
            "lambda_decor * decorrelation + lambda_entropy * bernoulli_entropy"
        ),
        "raw_embeddings_l2_normalized": False,
        "frozen_embedding_source": (
            "persistent_single_pass_cache"
            if getattr(args, "training_embedding_cache_used", False)
            else "live_backbone_forward"
        ),
        "reconstruction_scale": "fixed mean coordinate variance from calibration sample",
        "weights": {
            "lambda_recon": float(args.lambda_recon),
            "lambda_tweet_sparse_initial": float(args.lambda_tweet_sparse),
            "lambda_balance_fixed": float(args.lambda_balance),
            "lambda_decor": float(args.lambda_decor),
            "lambda_entropy": float(args.lambda_entropy),
        },
        "adaptive_sparsity": {
            "enabled": adaptive_sparsity_is_enabled(args),
            "interval_steps": int(args.adaptive_interval),
            "ema_beta": float(args.adaptive_ema_beta),
            "rate": float(args.adaptive_rate),
            "relative_tolerance": float(args.adaptive_tolerance),
            "lambda_min": float(args.lambda_min),
            "lambda_max": float(args.lambda_max),
        },
    }


def capture_bit_path_rows(
    model: "BinaryFactorsModel",
    epoch: int,
    batch: int,
    global_step: int,
    lambda_sparse: float,
    probabilities: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> List[Dict[str, object]]:
    """Capture lasso-style parameter, activation, and pre-clip gradient paths."""
    decoder_norm = model.decoder.linear.weight.detach().float().norm(dim=0).cpu().numpy()
    head_norm = model.head.fc2.weight.detach().float().norm(dim=1).cpu().numpy()
    head_bias = model.head.fc2.bias.detach().float().cpu().numpy()
    soft = np.full(model.k_factors, np.nan)
    hard = np.full(model.k_factors, np.nan)
    if probabilities is not None:
        soft = probabilities.detach().float().mean(dim=0).cpu().numpy()
        hard = (
            probabilities.detach().float().gt(threshold).float().mean(dim=0).cpu().numpy()
        )

    decoder_grad = np.full(model.k_factors, np.nan)
    head_grad = np.full(model.k_factors, np.nan)
    bias_grad = np.full(model.k_factors, np.nan)
    if model.decoder.linear.weight.grad is not None:
        decoder_grad = (
            model.decoder.linear.weight.grad.detach().float().norm(dim=0).cpu().numpy()
        )
    if model.head.fc2.weight.grad is not None:
        head_grad = model.head.fc2.weight.grad.detach().float().norm(dim=1).cpu().numpy()
    if model.head.fc2.bias.grad is not None:
        bias_grad = model.head.fc2.bias.grad.detach().float().abs().cpu().numpy()

    rows: List[Dict[str, object]] = []
    for bit_index in range(model.k_factors):
        rows.append(
            {
                "epoch": epoch,
                "batch": batch,
                "global_step": global_step,
                "bit": f"bit_{bit_index:02d}",
                "lambda_tweet_sparse": float(lambda_sparse),
                "decoder_column_norm": float(decoder_norm[bit_index]),
                "head_output_row_norm": float(head_norm[bit_index]),
                "head_output_bias": float(head_bias[bit_index]),
                "soft_activation_batch": float(soft[bit_index]),
                "hard_activation_batch": float(hard[bit_index]),
                "grad_decoder_column_norm_preclip": float(decoder_grad[bit_index]),
                "grad_head_output_row_norm_preclip": float(head_grad[bit_index]),
                "grad_head_output_bias_abs_preclip": float(bias_grad[bit_index]),
            }
        )
    return rows


def write_bit_path_plots(output_dir: Path, metrics: pd.DataFrame) -> None:
    """Render per-bit coefficient, activation, and gradient paths."""
    if metrics.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"Warning: bit-path plots unavailable: {exc}")
        return

    bits = list(dict.fromkeys(metrics["bit"].tolist()))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(bits))))
    color_by_bit = dict(zip(bits, colors))

    def draw_paths(axes, panels, frame: pd.DataFrame) -> None:
        for axis, (column, title) in zip(axes, panels):
            for bit in bits:
                group = frame[frame["bit"] == bit]
                axis.plot(
                    group["global_step"], group[column],
                    color=color_by_bit[bit], linewidth=1.35, label=bit,
                )
            axis.set_title(title)
            axis.set_xlabel("optimizer step")
            axis.set_ylabel(column)

    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    draw_paths(
        axes.flat,
        [
            ("decoder_column_norm", "Decoder column norm"),
            ("head_output_row_norm", "Head output-row norm"),
            ("head_output_bias", "Head output bias"),
            ("soft_activation_batch", "Per-bit soft activation (monitored batch)"),
        ],
        metrics,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    figure.suptitle("Per-bit coefficient paths")
    figure.savefig(output_dir / "bit_coefficient_paths.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    activity = metrics.dropna(subset=["soft_activation_batch"])
    if not activity.empty:
        figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        draw_paths(
            axes,
            [
                ("soft_activation_batch", "Soft activation rate"),
                ("hard_activation_batch", "Hard activation rate at configured threshold"),
            ],
            activity,
        )
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        figure.suptitle("Per-bit activation paths on monitored batches")
        figure.savefig(output_dir / "bit_activation_paths.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    gradients = metrics.dropna(subset=["grad_decoder_column_norm_preclip"])
    if not gradients.empty:
        figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
        panels = [
            ("grad_decoder_column_norm_preclip", "Decoder-column gradient"),
            ("grad_head_output_row_norm_preclip", "Head-row gradient"),
            ("grad_head_output_bias_abs_preclip", "Head-bias |gradient|"),
        ]
        draw_paths(axes, panels, gradients)
        for axis, (column, _) in zip(axes, panels):
            if (gradients[column] > 0).any():
                axis.set_yscale("log")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        figure.suptitle("Per-bit total-loss gradients before clipping")
        figure.savefig(output_dir / "bit_gradient_paths.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def write_training_history_plot(output_dir: Path, history: pd.DataFrame) -> None:
    """Plot every adopted loss contribution and adaptive sparsity state by epoch."""
    if history.empty or "recon_normalized" not in history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"Warning: training-history plot unavailable: {exc}")
        return

    epochs = history["epoch"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for column in (
        "recon_normalized", "tweet_sparse", "balance", "decor", "entropy"
    ):
        axes[0, 0].plot(epochs, history[column], marker="o", label=column)
    axes[0, 0].set_title("Raw loss terms")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)

    for column in (
        "weighted_recon", "weighted_tweet_sparse", "weighted_balance",
        "weighted_decor", "weighted_entropy",
    ):
        axes[0, 1].plot(epochs, history[column], marker="o", label=column)
    axes[0, 1].set_title("Weighted contributions to total loss")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(epochs, history["total"], marker="o", label="training total")
    axes[1, 0].plot(
        epochs,
        history["selection_loss_fixed_weights"],
        marker="o",
        label="fixed-weight checkpoint score",
    )
    axes[1, 0].set_title("Total and checkpoint-selection objective")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        epochs, history["lambda_tweet_sparse_start"], marker="o", label="lambda start"
    )
    axes[1, 1].plot(
        epochs, history["lambda_tweet_sparse_end"], marker="o", label="lambda end"
    )
    axes[1, 1].plot(
        epochs,
        history["sparse_violation_ema_end"],
        marker="o",
        label="relative-violation EMA",
    )
    axes[1, 1].set_title("Adaptive document-sparsity state")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.2)
    figure.suptitle("Consolidated SAE training dynamics")
    figure.savefig(output_dir / "training_loss_paths.png", dpi=180)
    plt.close(figure)


# -----------------------------
# Tokenization helpers
# -----------------------------

def build_tokenizer(model_name: str, trust_remote_code: bool = True, text_prefix: str = ""):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Runtime-only attribute: the value itself is persisted in run_config.json.
    tokenizer._latent_text_prefix = text_prefix
    return tokenizer


def tokenize_texts(tokenizer, texts: List[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    text_prefix = getattr(tokenizer, "_latent_text_prefix", "")
    if text_prefix:
        texts = [text_prefix + text for text in texts]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


# -----------------------------
# Training / encoding
# -----------------------------

def train(args: argparse.Namespace) -> None:
    set_gpu_optimizations()
    set_seed(args.seed)
    device = choose_device(args.device)
    dtype  = choose_dtype(args.precision)
    print(f"Device: {device} | dtype: {dtype or torch.float32}")
    print(f"Model:    {args.model_name}")
    print(f"Pooling:  {args.pooling_mode}")
    print(f"Text prefix: {args.text_prefix!r}")
    print(f"Objective: {args.objective}")
    os.makedirs(args.output_dir, exist_ok=True)

    df = read_input_table(args.input)
    dataset = TweetDataset(df, seed=args.seed)

    tokenizer = build_tokenizer(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        text_prefix=args.text_prefix,
    )
    model = BinaryFactorsModel(
        model_name=args.model_name,
        k_factors=args.k_factors,
        target_active=args.target_active,
        head_hidden_dim=args.head_hidden_dim,
        temperature=args.temperature,
        freeze_backbone=args.freeze_backbone,
        trust_remote_code=args.trust_remote_code,
        objective=args.objective,
        pooling_mode=args.pooling_mode,
        backbone_dtype=dtype,
    ).to(device)

    # The frozen feature extractor may use 16-bit weights; optimized parameters
    # remain FP32 to keep the small SAE numerically stable.
    model.head = model.head.to(device=device, dtype=torch.float32)
    if model.decoder is not None:
        model.decoder = model.decoder.to(device=device, dtype=torch.float32)

    print(f"Backbone device: {next(model.backbone.parameters()).device}")
    print(f"Head device:     {next(model.head.parameters()).device}")
    args.trainable_dtype = "fp32"
    args.backbone_dtype_resolved = str(dtype or torch.float32).replace("torch.", "")

    use_training_embedding_cache = bool(
        args.cache_training_embeddings
        and args.objective == "sparse_ae"
        and args.freeze_backbone
    )
    training_matrix_path: Optional[Path] = None
    if use_training_embedding_cache:
        input_stem = Path(args.input).stem
        ids = df["id"].astype(str).tolist()
        cache_is_ready = training_embedding_cache_is_valid(
            output_dir=args.output_dir,
            input_stem=input_stem,
            source_input=Path(args.input),
            ids=ids,
            model_name=args.model_name,
            pooling_mode=args.pooling_mode,
            text_prefix=args.text_prefix,
            objective=args.objective,
            max_length=args.max_length,
            backbone_precision=args.backbone_dtype_resolved,
            embedding_dim=int(model.hidden_size),
        )
        if not cache_is_ready:
            promote_raw_cache_for_training(
                output_dir=args.output_dir,
                input_stem=input_stem,
                source_input=Path(args.input),
                ids=ids,
                model_name=args.model_name,
                pooling_mode=args.pooling_mode,
                text_prefix=args.text_prefix,
                objective=args.objective,
                max_length=args.max_length,
                backbone_precision=args.backbone_dtype_resolved,
                embedding_dim=int(model.hidden_size),
            )
        if args.require_cached_embeddings and not training_embedding_cache_is_valid(
            output_dir=args.output_dir,
            input_stem=input_stem,
            source_input=Path(args.input),
            ids=ids,
            model_name=args.model_name,
            pooling_mode=args.pooling_mode,
            text_prefix=args.text_prefix,
            objective=args.objective,
            max_length=args.max_length,
            backbone_precision=args.backbone_dtype_resolved,
            embedding_dim=int(model.hidden_size),
        ):
            raise RuntimeError(
                "A validated cached embedding matrix is required, but neither the "
                "training cache nor a compatible raw-embedding cache is available. "
                "Backbone inference was not started."
            )
        training_matrix_path, cache_metadata = ensure_training_embedding_cache(
            model, tokenizer, df, dataset, args, device
        )
        args.training_embedding_cache = cache_metadata
        args.training_embedding_cache_used = True
        loader = DataLoader(
            CachedEmbeddingDataset(training_matrix_path),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        release_frozen_backbone_from_accelerator(model, device)
        print("Training source: persistent frozen pooled-embedding cache")
    elif args.objective == "sparse_ae":
        args.training_embedding_cache_used = False
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print("Training source: live frozen-backbone forward passes")
    else:
        args.training_embedding_cache_used = False
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=ContrastiveCollator(seed=args.seed),
        )

    args.trainer_variant = (
        "consolidated_fixed_balance_sae_mean_intercept_v2"
        if args.objective == "sparse_ae"
        else "legacy_contrastive_v1"
    )
    if args.objective == "sparse_ae":
        args.bit_path_artifacts = {
            "losses": "training_loss_paths.png",
            "batch_diagnostic_metrics": "training_diagnostic_metrics.csv",
            "compact_diagnostics": "training_convergence_diagnostics.png",
            "active_bit_diagnostics": "training_active_bits_diagnostics.csv",
            "weighted_loss_diagnostics": "training_weighted_loss_diagnostics.csv",
            "diagnostic_summary": "training_convergence_diagnostics.json",
            "metrics": "training_coefficient_metrics.csv",
            "coefficients": "bit_coefficient_paths.png",
            "activations": "bit_activation_paths.png",
            "gradients": "bit_gradient_paths.png",
        }
        args.loss_spec = consolidated_loss_spec(args)
    else:
        args.loss_spec = {
            "name": "legacy_contrastive_v1",
            "formula": (
                "consistency + lambda_sparse * document_sparsity + "
                "lambda_balance * bit_balance + lambda_decor * decorrelation"
            ),
        }

    reconstruction_scale = 1.0
    if args.objective == "sparse_ae":
        if training_matrix_path is not None:
            calibration = calibrate_cached_embedding_scale(training_matrix_path, args)
        else:
            calibration = calibrate_embedding_scale(model, tokenizer, dataset, args, device)
        coordinate_mean = calibration.pop("_coordinate_mean")
        reconstruction_scale = calibration["coordinate_variance_mean"]
        args.reconstruction_scale = reconstruction_scale
        if args.decoder_intercept_init == "embedding_mean":
            initialize_decoder_intercept_from_embedding_mean(model, coordinate_mean)
            calibration["decoder_intercept_initialization"] = (
                "embedding_mean_minus_initial_decoder_weight_times_target_rho"
            )
        else:
            calibration["decoder_intercept_initialization"] = "pytorch_default"
        args.decoder_intercept_initialization = calibration[
            "decoder_intercept_initialization"
        ]
        args.loss_spec["decoder_intercept_initialization"] = (
            args.decoder_intercept_initialization
        )
        calibration_path = Path(args.output_dir) / "reconstruction_calibration.json"
        with calibration_path.open("w", encoding="utf-8") as handle:
            json.dump(calibration, handle, indent=2)
            handle.write("\n")
        print(f"Fixed reconstruction variance: {reconstruction_scale:.8g}")

    # Cache construction and calibration may instantiate different iterator paths,
    # which consume framework RNG state even though the frozen embeddings themselves
    # are deterministic. Re-anchor optimization so a newly built cache, a reused
    # cache, and legacy live forwards produce the same seeded SAE trajectory.
    set_seed(args.seed)
    args.optimization_rng_seed = int(args.seed)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    # A frozen feature extractor must remain deterministic. Calling train() on the
    # full wrapper also enables dropout inside BERT-like backbones, despite their
    # parameters not receiving gradients (and some MPS attention kernels do not
    # support that dropout path). Keep only the trainable head/decoder in train mode.
    if args.freeze_backbone:
        model.backbone.eval()
    start_epoch = 1
    best_loss = float("inf")
    history = []
    lambda_sparse = float(args.lambda_tweet_sparse)
    sparse_violation_ema: Optional[float] = None
    global_step = 0

    if args.resume:
        resumed = load_resume_state(args.output_dir, args)
        if resumed is not None:
            load_checkpoint_state(model, resumed["model_state"], resumed["args"])
            optimizer.load_state_dict(resumed["optimizer_state"])
            scheduler.load_state_dict(resumed["scheduler_state"])
            best_loss = resumed["best_loss"]
            history = resumed["history"]
            start_epoch = resumed["epoch"] + 1
            training_state = resumed.get("training_state", {})
            lambda_sparse = float(training_state.get("lambda_tweet_sparse", lambda_sparse))
            sparse_violation_ema = training_state.get("sparse_violation_ema")
            global_step = int(training_state.get("global_step", 0))

    coefficient_path = Path(args.output_dir) / "training_coefficient_metrics.csv"
    if args.resume and coefficient_path.is_file():
        bit_path_rows: List[Dict[str, object]] = pd.read_csv(
            coefficient_path
        ).to_dict(orient="records")
    elif args.objective == "sparse_ae":
        bit_path_rows = capture_bit_path_rows(
            model,
            epoch=0,
            batch=0,
            global_step=global_step,
            lambda_sparse=lambda_sparse,
            threshold=args.threshold,
        )
    else:
        bit_path_rows = []

    diagnostic_path = Path(args.output_dir) / "training_diagnostic_metrics.csv"
    if args.resume and diagnostic_path.is_file():
        diagnostic_rows: List[Dict[str, object]] = pd.read_csv(
            diagnostic_path
        ).to_dict(orient="records")
    else:
        diagnostic_rows = []

    # Loss keys depend on objective
    is_contrastive = args.objective != "sparse_ae"
    if is_contrastive:
        loss_keys = ["total", "cons", "tweet_sparse", "balance", "decor"]
    else:
        loss_keys = [
            "total", "recon_raw", "recon_normalized", "tweet_sparse",
            "balance", "decor", "entropy", "weighted_recon",
            "weighted_tweet_sparse", "weighted_balance", "weighted_decor",
            "weighted_entropy",
        ]

    # This is the operational bit-generation time: queued accelerator work is
    # synchronized, and cache creation, model loading, calibration, checkpoint
    # serialization, encoding, and every downstream analysis are excluded.
    timing_global_step_start = global_step
    timing_epochs_executed = max(0, args.epochs - start_epoch + 1)
    synchronize_device(device)
    optimization_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_losses = {k: 0.0 for k in loss_keys}
        epoch_losses["epoch"] = epoch
        epoch_losses["lambda_tweet_sparse_start"] = lambda_sparse
        n_batches = 0

        for batch_index, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch"), start=1
        ):

            if args.objective == "sparse_ae":
                global_step += 1
                if use_training_embedding_cache:
                    pooled = batch.to(device=device, dtype=torch.float32)
                    out = model.forward_from_pooled(pooled)
                else:
                    # Single view — no perturbation needed
                    texts = [basic_clean(t) for t in batch["text"]]
                    tok = tokenize_texts(tokenizer, texts, args.max_length, device)
                    out = model(**tok)
                probs = out["probs"].float()  # fp32 for stable loss computation

                loss_recon_raw = reconstruction_loss(
                    out["reconstructed"].float(), out["pooled"].float()
                )
                loss_recon = loss_recon_raw / (
                    reconstruction_scale + args.recon_scale_epsilon
                )
                loss_tweet_sparse = tweet_sparsity_loss(probs, model.target_rho)
                loss_balance = bit_balance_loss(probs, model.target_rho)
                loss_decor = decorrelation_loss(probs)
                loss_entropy = bernoulli_entropy_loss(probs, args.entropy_epsilon)

                weighted_recon = args.lambda_recon * loss_recon
                weighted_sparse = lambda_sparse * loss_tweet_sparse
                weighted_balance = args.lambda_balance * loss_balance
                weighted_decor = args.lambda_decor * loss_decor
                weighted_entropy = args.lambda_entropy * loss_entropy
                loss = (
                    weighted_recon
                    + weighted_sparse
                    + weighted_balance
                    + weighted_decor
                    + weighted_entropy
                )

                epoch_losses["recon_raw"]    += loss_recon_raw.item()
                epoch_losses["recon_normalized"] += loss_recon.item()
                epoch_losses["tweet_sparse"] += loss_tweet_sparse.item()
                epoch_losses["balance"]      += loss_balance.item()
                epoch_losses["decor"]        += loss_decor.item()
                epoch_losses["entropy"]      += loss_entropy.item()
                epoch_losses["weighted_recon"] += weighted_recon.item()
                epoch_losses["weighted_tweet_sparse"] += weighted_sparse.item()
                epoch_losses["weighted_balance"] += weighted_balance.item()
                epoch_losses["weighted_decor"] += weighted_decor.item()
                epoch_losses["weighted_entropy"] += weighted_entropy.item()

            else:
                # Contrastive objectives: consistency / last_token / token_sample / mid_layer
                tok_a = tokenize_texts(tokenizer, batch.text_a, args.max_length, device)
                tok_b = tokenize_texts(tokenizer, batch.text_b, args.max_length, device)

                out_a = model(**tok_a)
                out_b = model(**tok_b)

                p_a = out_a["probs"].float()  # fp32 for stable loss computation
                p_b = out_b["probs"].float()
                p_mean = 0.5 * (p_a + p_b)

                loss_cons        = cosine_consistency_loss(p_a, p_b)
                loss_tweet_sparse = tweet_sparsity_loss(p_mean, model.target_rho)
                loss_balance     = bit_balance_loss(p_mean, model.target_rho)
                loss_decor       = decorrelation_loss(p_mean)

                loss = (
                    loss_cons
                    + args.lambda_tweet_sparse * loss_tweet_sparse
                    + args.lambda_balance * loss_balance
                    + args.lambda_decor * loss_decor
                )

                epoch_losses["cons"]         += loss_cons.item()
                epoch_losses["tweet_sparse"] += loss_tweet_sparse.item()
                epoch_losses["balance"]      += loss_balance.item()
                epoch_losses["decor"]        += loss_decor.item()

            epoch_losses["total"] += loss.item()
            n_batches += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            monitor_bit_paths = (
                global_step == 1
                or global_step % args.coefficient_monitor_interval == 0
                or batch_index == len(loader)
            )
            if args.objective == "sparse_ae" and monitor_bit_paths:
                bit_path_rows.extend(
                    capture_bit_path_rows(
                        model,
                        epoch=epoch,
                        batch=batch_index,
                        global_step=global_step,
                        lambda_sparse=lambda_sparse,
                        probabilities=probs,
                        threshold=args.threshold,
                    )
                )
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()

            if args.objective == "sparse_ae":
                with torch.no_grad():
                    soft_active = float(probs.sum(dim=1).mean().item())
                    hard_active = float(
                        probs.gt(args.threshold).float().sum(dim=1).mean().item()
                    )
                    relative_violation = abs(
                        soft_active - args.target_active
                    ) / max(args.target_active, 1e-8)
                    if sparse_violation_ema is None:
                        sparse_violation_ema = relative_violation
                    else:
                        beta = args.adaptive_ema_beta
                        sparse_violation_ema = (
                            beta * sparse_violation_ema
                            + (1.0 - beta) * relative_violation
                        )
                diagnostic_rows.append(
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "global_step": global_step,
                        "target_active": float(args.target_active),
                        "lambda_tweet_sparse": float(lambda_sparse),
                        "loss_total": float(loss.item()),
                        "loss_recon_raw": float(loss_recon_raw.item()),
                        "loss_recon_normalized": float(loss_recon.item()),
                        "loss_tweet_sparse": float(loss_tweet_sparse.item()),
                        "loss_balance": float(loss_balance.item()),
                        "loss_decor": float(loss_decor.item()),
                        "loss_entropy": float(loss_entropy.item()),
                        "weighted_recon": float(weighted_recon.item()),
                        "weighted_tweet_sparse": float(weighted_sparse.item()),
                        "weighted_balance": float(weighted_balance.item()),
                        "weighted_decor": float(weighted_decor.item()),
                        "weighted_entropy": float(weighted_entropy.item()),
                        "soft_active_mean": soft_active,
                        "hard_active_mean": hard_active,
                    }
                )
                # A zero initial weight is the explicit leave-one-loss-term-out
                # setting.  Do not let the adaptive controller silently revive
                # the omitted activity penalty at lambda_min.
                if (
                    adaptive_sparsity_is_enabled(args)
                    and global_step % args.adaptive_interval == 0
                ):
                    lambda_sparse = update_sparsity_multiplier(
                        lambda_sparse, sparse_violation_ema, args
                    )

        for k in loss_keys:
            if k != "epoch":
                epoch_losses[k] /= max(1, n_batches)

        if args.objective == "sparse_ae":
            epoch_losses["lambda_tweet_sparse_end"] = lambda_sparse
            epoch_losses["sparse_violation_ema_end"] = float(
                sparse_violation_ema or 0.0
            )
            selection_loss = (
                args.lambda_recon * epoch_losses["recon_normalized"]
                + args.lambda_tweet_sparse * epoch_losses["tweet_sparse"]
                + args.lambda_balance * epoch_losses["balance"]
                + args.lambda_decor * epoch_losses["decor"]
                + args.lambda_entropy * epoch_losses["entropy"]
            )
            epoch_losses["selection_loss_fixed_weights"] = selection_loss
            print(
                f"Epoch {epoch}/{args.epochs} | total={epoch_losses['total']:.4f} "
                f"recon_raw={epoch_losses['recon_raw']:.4f} "
                f"recon_norm={epoch_losses['recon_normalized']:.4f} "
                f"sparse={epoch_losses['tweet_sparse']:.4f} "
                f"balance={epoch_losses['balance']:.4f} "
                f"entropy={epoch_losses['entropy']:.4f} "
                f"decor={epoch_losses['decor']:.4f} lambda_s={lambda_sparse:.4g}"
            )
        else:
            selection_loss = epoch_losses["total"]
            print(
                f"Epoch {epoch}/{args.epochs} | total={epoch_losses['total']:.4f} "
                f"cons={epoch_losses['cons']:.4f} "
                f"sparse={epoch_losses['tweet_sparse']:.4f} "
                f"balance={epoch_losses['balance']:.4f} "
                f"decor={epoch_losses['decor']:.4f}"
            )

        history.append(epoch_losses)
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(Path(args.output_dir) / "train_history.csv", index=False)
        write_training_history_plot(Path(args.output_dir), history_frame)
        if args.objective == "sparse_ae":
            bit_path_frame = pd.DataFrame(bit_path_rows)
            bit_path_frame.to_csv(coefficient_path, index=False)
            write_bit_path_plots(Path(args.output_dir), bit_path_frame)
            pd.DataFrame(diagnostic_rows).to_csv(diagnostic_path, index=False)
            write_compact_training_diagnostics(Path(args.output_dir))

        if selection_loss < best_loss:
            best_loss = selection_loss
            save_checkpoint(args.output_dir, model, tokenizer, args, best=True)

        save_resume_state(args.output_dir, model, optimizer, scheduler,
                           epoch, best_loss, history, args,
                           training_state={
                               "lambda_tweet_sparse": lambda_sparse,
                               "sparse_violation_ema": sparse_violation_ema,
                               "global_step": global_step,
                           })

    synchronize_device(device)
    optimization_seconds = time.perf_counter() - optimization_started
    optimizer_steps_executed = global_step - timing_global_step_start
    document_updates = len(dataset) * timing_epochs_executed
    save_checkpoint(args.output_dir, model, tokenizer, args, best=False)
    pd.DataFrame(history).to_csv(Path(args.output_dir) / "train_history.csv", index=False)
    if args.objective == "sparse_ae":
        pd.DataFrame(diagnostic_rows).to_csv(diagnostic_path, index=False)
        write_compact_training_diagnostics(Path(args.output_dir))

    resume_path = Path(args.output_dir) / RESUME_FILENAME
    if resume_path.exists():
        resume_path.unlink()  # training finished — no longer resumable, nothing to clean up next run

    # Write only after the final checkpoint has rewritten metadata, so the
    # metadata pointer survives; elapsed time itself stopped before checkpoint
    # serialization and output materialisation.
    record_bit_generation_timing(
        args.output_dir,
        "optimization",
        {
            "status": "completed",
            "definition": (
                "Wall-clock time from the first optimizer step to the completed "
                "last optimizer step, synchronized on the selected accelerator."
            ),
            "elapsed_seconds": optimization_seconds,
            "embedding_generation_included": False,
            "model_loading_included": False,
            "calibration_included": False,
            "checkpoint_and_encoding_included": False,
            "cached_embeddings_used": bool(args.training_embedding_cache_used),
            "device": str(device),
            "n_documents": int(len(dataset)),
            "embedding_dim": int(model.hidden_size),
            "k_factors": int(args.k_factors),
            "target_active": float(args.target_active),
            "epochs_executed_this_invocation": int(timing_epochs_executed),
            "optimizer_steps_executed": int(optimizer_steps_executed),
            "document_updates": int(document_updates),
            "optimizer_steps_per_second": (
                optimizer_steps_executed / optimization_seconds
                if optimization_seconds > 0 else None
            ),
            "document_updates_per_second": (
                document_updates / optimization_seconds
                if optimization_seconds > 0 else None
            ),
            "resumed": bool(args.resume),
        },
    )

    print(f"Saved outputs to: {args.output_dir}")


def checkpoint_model_state(model: nn.Module, lightweight: bool) -> Dict[str, torch.Tensor]:
    """Return all weights, or only trainable modules when the backbone is frozen."""
    state = model.state_dict()
    if not lightweight:
        return state
    return {name: value for name, value in state.items() if not name.startswith("backbone.")}


def load_checkpoint_state(
    model: nn.Module, state: Dict[str, torch.Tensor], config: Dict[str, object]
) -> None:
    """Strictly load either a full or a lightweight head/decoder checkpoint."""
    if not config.get("lightweight_checkpoints", False):
        model.load_state_dict(state, strict=True)
        return
    expected = {name for name in model.state_dict() if not name.startswith("backbone.")}
    actual = set(state)
    if actual != expected:
        raise RuntimeError(
            "Lightweight checkpoint does not contain exactly the head/decoder state "
            f"(missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)})."
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith("backbone.") for name in missing):
        raise RuntimeError(
            f"Invalid lightweight checkpoint load: missing={missing}, unexpected={unexpected}."
        )


def save_checkpoint(output_dir: str, model: nn.Module, tokenizer, args: argparse.Namespace, best: bool = True) -> None:
    suffix = "best" if best else "last"
    ckpt_dir = Path(output_dir) / suffix
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        checkpoint_model_state(model, args.lightweight_checkpoints),
        ckpt_dir / "model.pt",
    )
    tokenizer.save_pretrained(ckpt_dir)

    config = vars(args).copy()
    config["embedding_dim"] = int(model.hidden_size)
    cfg = model.backbone.config
    text_cfg = getattr(cfg, "text_config", None)
    internal_dim = getattr(text_cfg, "hidden_size", getattr(cfg, "hidden_size", None))
    if internal_dim is not None:
        config["backbone_internal_hidden_size"] = int(internal_dim)
    with open(ckpt_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    write_model_metadata(output_dir, args, model=model)


RESUME_FILENAME = "resume_state.pt"

# Architecture-defining args — if any of these differ from the interrupted run,
# the saved model/optimizer state_dicts won't line up with a freshly built model.
RESUME_CRITICAL_ARGS = [
    "model_name", "k_factors", "target_active", "head_hidden_dim", "temperature",
    "objective", "pooling_mode", "text_prefix", "freeze_backbone", "lambda_recon",
    "lambda_tweet_sparse", "lambda_balance", "lambda_decor", "lambda_entropy",
    "adaptive_interval", "adaptive_ema_beta", "adaptive_rate",
    "adaptive_tolerance", "coefficient_monitor_interval", "lambda_min", "lambda_max",
    "decoder_intercept_init",
]


def save_resume_state(output_dir: str, model: nn.Module, optimizer, scheduler,
                       epoch: int, best_loss: float, history: list,
                       args: argparse.Namespace,
                       training_state: Optional[Dict[str, object]] = None) -> None:
    """Full training state for --resume: model + optimizer + scheduler + progress.

    Separate from the best/last checkpoints in save_checkpoint(), which only ever
    hold inference-ready model weights + tokenizer, not optimizer/scheduler state.
    """
    state = {
        "epoch": epoch,
        "best_loss": best_loss,
        "history": history,
        "model_state": checkpoint_model_state(model, args.lightweight_checkpoints),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": vars(args),
        "training_state": training_state or {},
    }
    torch.save(state, Path(output_dir) / RESUME_FILENAME)


def load_resume_state(output_dir: str, args: argparse.Namespace) -> Optional[dict]:
    path = Path(output_dir) / RESUME_FILENAME
    if not path.exists():
        print(f"--resume requested but no {RESUME_FILENAME} found in {output_dir} — starting from scratch.")
        return None
    state = torch.load(path, map_location="cpu")
    saved_args = state["args"]
    mismatches = [
        f"  --{k}: saved={saved_args.get(k)!r} vs now={getattr(args, k)!r}"
        for k in RESUME_CRITICAL_ARGS
        if saved_args.get(k) != getattr(args, k)
    ]
    if mismatches:
        raise ValueError(
            "Cannot resume: these arguments differ from the interrupted run:\n"
            + "\n".join(mismatches)
            + "\nRe-run with the original arguments, or start a fresh --output_dir."
        )
    if state["epoch"] >= args.epochs:
        print(f"Checkpoint already completed {state['epoch']}/{args.epochs} epochs — nothing to resume.")
    else:
        print(f"Resuming from epoch {state['epoch'] + 1}/{args.epochs} "
              f"(best_loss so far: {state['best_loss']:.4f})")
    return state


def load_model_for_inference(args: argparse.Namespace, device: torch.device, which: str = "best") -> Tuple[BinaryFactorsModel, any]:
    set_gpu_optimizations()
    ckpt_dir = Path(args.output_dir) / which
    with open(ckpt_dir / "run_config.json", "r", encoding="utf-8") as f:
        saved = json.load(f)
    dtype = choose_dtype(saved.get("precision", args.precision))
    print(f"Device: {device} | dtype: {dtype or torch.float32}")

    tokenizer = build_tokenizer(
        saved["model_name"],
        trust_remote_code=saved.get("trust_remote_code", True),
        text_prefix=saved.get("text_prefix", ""),
    )
    model = BinaryFactorsModel(
        model_name=saved["model_name"],
        k_factors=saved["k_factors"],
        target_active=saved["target_active"],
        head_hidden_dim=saved["head_hidden_dim"],
        temperature=saved["temperature"],
        freeze_backbone=True,
        trust_remote_code=saved.get("trust_remote_code", True),
        objective=saved.get("objective", "consistency"),
        pooling_mode=saved.get("pooling_mode", "mean"),
        backbone_dtype=dtype,
    ).to(device)
    state = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    load_checkpoint_state(model, state, saved)
    model.eval()
    model.head = model.head.to(device=device, dtype=torch.float32)
    if model.decoder is not None:
        model.decoder = model.decoder.to(device=device, dtype=torch.float32)
    backfill_resolved_dimensions(args.output_dir, ckpt_dir, model)
    print(f"Backbone device: {next(model.backbone.parameters()).device}")
    print(f"Head device:     {next(model.head.parameters()).device}")
    return model, tokenizer


def materialize_raw_cache_from_training_cache(
    training_matrix_path: Path, raw_matrix_path: Path
) -> None:
    """Expose one matrix under the established encode-cache name without duplication."""
    temporary = raw_matrix_path.with_suffix(".npy.partial")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(training_matrix_path, temporary)
    except OSError:
        # Cross-filesystem output directories cannot hard-link. copyfile may still
        # use the platform's copy-on-write primitive; correctness does not depend on it.
        shutil.copyfile(training_matrix_path, temporary)
    temporary.replace(raw_matrix_path)


@torch.no_grad()
def encode(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    df = read_input_table(args.input)
    dataset = TweetDataset(df, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_stem = Path(args.input).stem
    out_path = Path(args.output_dir) / f"encoded_{input_stem}_{args.checkpoint}.csv"
    if (
        args.save_raw_embeddings
        and not args.force_encode
        and out_path.is_file()
        and raw_embedding_cache_is_valid(
            args.output_dir,
            input_stem,
            args.checkpoint,
            out_path,
            source_input=Path(args.input),
            encoding_threshold=args.threshold,
        )
    ):
        print(
            f"Reusing {out_path} and its validated raw-embedding cache — "
            "skipping model load and forward pass."
        )
        return

    model, tokenizer = load_model_for_inference(args, device=device, which=args.checkpoint)

    config_path = Path(args.output_dir) / args.checkpoint / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_matrix_path, _, _ = training_embedding_cache_paths(
        args.output_dir, input_stem
    )
    ids = df["id"].astype(str).tolist()
    can_reuse_training_cache = training_embedding_cache_is_valid(
        output_dir=args.output_dir,
        input_stem=input_stem,
        source_input=Path(args.input),
        ids=ids,
        model_name=config["model_name"],
        pooling_mode=config.get("pooling_mode", "mean"),
        text_prefix=config.get("text_prefix", ""),
        objective=config.get("objective", "consistency"),
        max_length=int(config.get("max_length", args.max_length)),
        backbone_precision=config.get(
            "backbone_dtype_resolved", config.get("precision", "bf16")
        ),
        embedding_dim=int(model.hidden_size),
    )
    if can_reuse_training_cache:
        release_frozen_backbone_from_accelerator(model, device)
        matrix = np.load(training_matrix_path, mmap_mode="r")
        rows = []
        for start in tqdm(
            range(0, len(df), args.batch_size),
            desc="Encoding cached embeddings",
            unit="batch",
        ):
            stop = min(start + args.batch_size, len(df))
            pooled = torch.from_numpy(
                np.array(matrix[start:stop], dtype=np.float32, copy=True)
            ).to(device=device, dtype=torch.float32)
            out = model.forward_from_pooled(pooled)
            probs = out["probs"].detach().cpu().float().numpy()
            bits = (probs > args.threshold).astype(np.int64)
            for local_index, row_index in enumerate(range(start, stop)):
                row = {
                    "id": ids[row_index],
                    "text": str(df.iloc[row_index]["text"]),
                    "n_active": int(bits[local_index].sum()),
                }
                for factor_index in range(bits.shape[1]):
                    row[f"bit_{factor_index:02d}"] = int(
                        bits[local_index, factor_index]
                    )
                    row[f"prob_{factor_index:02d}"] = float(
                        probs[local_index, factor_index]
                    )
                rows.append(row)
        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_path, index=False)
        print(f"Saved encoded factors from the training cache to: {out_path}")
        if args.save_raw_embeddings:
            raw_matrix_path, _, _ = raw_embedding_cache_paths(
                args.output_dir, input_stem, args.checkpoint
            )
            materialize_raw_cache_from_training_cache(
                training_matrix_path, raw_matrix_path
            )
            metadata = write_raw_embedding_metadata(
                output_dir=args.output_dir,
                input_stem=input_stem,
                checkpoint=args.checkpoint,
                encoded_path=out_path,
                ids=ids,
                embedding_dim=int(model.hidden_size),
                config=config,
                created_during_encoding=False,
                source_input=Path(args.input),
                encoding_threshold=args.threshold,
            )
            print(
                "Reused frozen training embeddings as the predictive cache: "
                f"{raw_matrix_path} ({metadata['rows']:,} x "
                f"{metadata['embedding_dim']:,})"
            )
        return

    raw_matrix_path, _, _ = raw_embedding_cache_paths(
        args.output_dir, input_stem, args.checkpoint
    )
    raw_temporary_path = raw_matrix_path.with_suffix(".npy.partial")
    raw_embeddings = None
    if args.save_raw_embeddings:
        raw_embeddings = np.lib.format.open_memmap(
            raw_temporary_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(df), int(model.hidden_size)),
        )

    rows = []
    raw_offset = 0
    for batch in tqdm(loader, desc="Inspecting", unit="batch"):
#    for batch in loader:
        texts = [basic_clean(t) for t in batch["text"]]
        ids = [str(x) for x in batch["id"]]
        tok = tokenize_texts(tokenizer, texts, args.max_length, device)
        out = model(**tok)
        probs = out["probs"].detach().cpu().float().numpy()
        bits = (probs > args.threshold).astype(np.int64)
        if raw_embeddings is not None:
            next_offset = raw_offset + len(ids)
            raw_embeddings[raw_offset:next_offset] = (
                out["pooled"].detach().cpu().float().numpy()
            )
            raw_offset = next_offset

        for i, tweet_id in enumerate(ids):
            row = {"id": tweet_id, "text": texts[i]}
            active = int(bits[i].sum())
            row["n_active"] = active
            for k in range(bits.shape[1]):
                row[f"bit_{k:02d}"] = int(bits[i, k])
                row[f"prob_{k:02d}"] = float(probs[i, k])
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Saved encoded factors to: {out_path}")
    if raw_embeddings is not None:
        if raw_offset != len(df):
            raise RuntimeError(
                f"Raw embedding row mismatch: wrote {raw_offset}, expected {len(df)}"
            )
        raw_embeddings.flush()
        del raw_embeddings
        raw_temporary_path.replace(raw_matrix_path)
        metadata = write_raw_embedding_metadata(
            output_dir=args.output_dir,
            input_stem=input_stem,
            checkpoint=args.checkpoint,
            encoded_path=out_path,
            ids=out_df["id"].astype(str).tolist(),
            embedding_dim=int(model.hidden_size),
            config=config,
            created_during_encoding=True,
            source_input=Path(args.input),
            encoding_threshold=args.threshold,
        )
        print(
            "Saved reusable raw embeddings to: "
            f"{raw_matrix_path} ({metadata['rows']:,} x {metadata['embedding_dim']:,})"
        )


def _load_probs_from_encoded(encoded_path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """Read id/text/prob_XX columns from an existing encoded_*.csv, so inspect_bits
    doesn't have to reload the model and re-run the forward pass over --input."""
    enc = pd.read_csv(encoded_path, dtype={"id": str}, float_precision="round_trip")
    prob_cols = sorted(c for c in enc.columns if c.startswith("prob_"))
    ids = enc["id"].astype(str).tolist()
    texts = enc["text"].astype(str).tolist()
    probs = enc[prob_cols].to_numpy(dtype=float)
    return ids, texts, probs


@torch.no_grad()
def inspect_bits(args: argparse.Namespace) -> None:
    input_stem = Path(args.input).stem
    encoded_path = Path(args.output_dir) / f"encoded_{input_stem}_{args.checkpoint}.csv"

    if encoded_path.exists():
        print(f"Reusing {encoded_path} — skipping model load and forward pass.")
        all_ids, all_texts, P = _load_probs_from_encoded(encoded_path)
    else:
        device = choose_device(args.device)
        df = read_input_table(args.input)
        dataset = TweetDataset(df, seed=args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model, tokenizer = load_model_for_inference(args, device=device, which=args.checkpoint)

        all_ids: List[str] = []
        all_texts: List[str] = []
        all_probs: List[np.ndarray] = []

        for batch in tqdm(loader, desc="Inspecting", unit="batch", leave=True, dynamic_ncols=True):
            texts = [basic_clean(t) for t in batch["text"]]
            ids = [str(x) for x in batch["id"]]
            tok = tokenize_texts(tokenizer, texts, args.max_length, device)
            out = model(**tok)
            probs = out["probs"].detach().cpu().float().numpy()
            all_ids.extend(ids)
            all_texts.extend(texts)
            all_probs.append(probs)

        P = np.vstack(all_probs)

    k_factors = P.shape[1]
    top_n = args.top_n

    report = []
    for k in range(k_factors):
        idx = np.argsort(-P[:, k])[:top_n]
        report.append({
            "bit": k,
            "mean_prob": float(P[:, k].mean()),
            "top_examples": [
                {"id": all_ids[i], "prob": float(P[i, k]), "text": all_texts[i]}
                for i in idx
            ],
        })

    print(f"\n{'='*80}")
    print(f"BIT REPORT — top {top_n} examples per bit (K={k_factors})")
    print(f"{'='*80}")
    for row in report:
        print(f"\nBit {row['bit']:02d}  (mean_prob={row['mean_prob']:.3f})")
        for ex in row["top_examples"]:
            preview = ex["text"][:100].replace("\n", " ")
            print(f"  {ex['prob']:.3f}  id={ex['id']:<10}  {preview}")

    out_path = Path(args.output_dir) / f"bit_report_{input_stem}_{args.checkpoint}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved bit report to: {out_path}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse binary factor discovery with any Hugging Face Transformers backbone")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "encode", "inspect"])
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default=os.environ.get("LATENT_DEVICE", "auto"),
        help=(
            "Execution device. An explicit accelerator requirement fails rather "
            "than silently falling back to CPU. LATENT_DEVICE supplies the default."
        ),
    )
    parser.add_argument("--precision", type=str, default="bf16", choices=["auto", "fp32", "fp16", "bf16"],
                        help=(
                            "Frozen-backbone dtype (default: bf16). The trainable head and decoder always "
                            "remain fp32. Use 'auto' for device-dependent selection."
                        ))
    parser.add_argument("--objective", type=str, default="sparse_ae",
                        choices=["consistency", "last_token", "token_sample", "mid_layer", "sparse_ae"],
                        help=(
                            "Training objective. "
                            "'sparse_ae' (default): consolidated scale-aware sparse binary autoencoder. "
                            "Recommended for frozen backbones. "
                            "'consistency': contrastive loss across two perturbed text views (original default). "
                            "'last_token': same but pools the last non-padding token per view. "
                            "'token_sample': same but each view pools a random 50%% subset of tokens. "
                            "'mid_layer': same but uses the middle transformer layer instead of last."
                        ))
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="CSV/Parquet/JSON/JSONL with columns: id, text",
    )
    parser.add_argument("--output_dir", type=str, required=True)

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="Any HuggingFace AutoModel. "
                             "Use --pooling_mode last_token for embedding-finetuned models "
                             "(Qwen3-Embedding, GTE-Qwen2, LLaMA-based embedders). "
                             "Use --pooling_mode mean for mean-pooled encoders such as "
                             "EmbeddingGemma 2 and for base causal LMs when appropriate.")
    parser.add_argument("--pooling_mode", type=str, default="last_token",
                        choices=["mean", "last_token", "weighted_mean", "token_sample"],
                        help="How to pool token hidden states into a single vector. "
                             "last_token: correct for Qwen3-Embedding and similar. "
                             "mean: correct for base causal LMs (Qwen3.5-Base, LLaMA base). "
                             "weighted_mean: position-weighted mean. "
                             "token_sample: random 50%% subset (contrastive view diversity).")
    parser.add_argument(
        "--text_prefix",
        type=str,
        default="",
        help=(
            "Optional prefix prepended to every text before tokenization. Use the model's "
            "document/task instruction when required; it is saved in run metadata and reused "
            "during encode/inspect."
        ),
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze_backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--max_length", type=int, default=128)

    # Binary factors
    parser.add_argument("--k_factors", type=int, default=16)
    parser.add_argument("--target_active", type=float, default=6.0, help="Target average number of active factors per tweet")
    parser.add_argument("--head_hidden_dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.50)
    parser.add_argument("--threshold", type=float, default=0.50)

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from resume_state.pt in --output_dir "
                             "(saved after every epoch). Re-run with the exact same "
                             "arguments as the interrupted run, plus --resume.")
    parser.add_argument(
        "--cache_training_embeddings",
        action="store_true",
        default=True,
        help=(
            "For a frozen sparse-AE backbone, compute pooled raw embeddings once, "
            "persist them, and reuse them across every training epoch and encoding "
            "pass (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no_cache_training_embeddings",
        dest="cache_training_embeddings",
        action="store_false",
        help="Use legacy live backbone forwards during every sparse-AE training epoch.",
    )
    parser.add_argument(
        "--require_cached_embeddings",
        action="store_true",
        help=(
            "Refuse training unless a validated training cache or compatible raw "
            "embedding cache already exists; never fall back to backbone inference."
        ),
    )

    # Loss weights
    parser.add_argument("--lambda_tweet_sparse", type=float, default=1.0,
                        help="Initial adaptive weight on per-document sparsity")
    parser.add_argument("--lambda_balance", type=float, default=1.0,
                        help="Fixed weight on per-bit marginal balance")
    parser.add_argument("--lambda_decor", type=float, default=0.05)
    parser.add_argument("--lambda_recon", type=float, default=1.0,
                        help="Weight on reconstruction loss (sparse_ae only)")
    parser.add_argument("--lambda_entropy", type=float, default=0.1,
                        help="Weight on conditional Bernoulli entropy (sparse_ae only)")
    parser.add_argument("--entropy_epsilon", type=float, default=1e-6)
    parser.add_argument("--calibration_samples", type=int, default=1024,
                        help="Rows used once to estimate the fixed raw-embedding variance")
    parser.add_argument("--recon_scale_epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--decoder_intercept_init",
        choices=("default", "embedding_mean"),
        default="embedding_mean",
        help=(
            "Initialize the sparse-AE decoder bias either with the PyTorch default "
            "or so the initial expected reconstruction matches the calibration-set "
            "embedding mean. The latter changes no embedding normalization."
        ),
    )
    parser.add_argument("--adaptive_interval", type=int, default=50)
    parser.add_argument("--adaptive_ema_beta", type=float, default=0.95)
    parser.add_argument("--adaptive_rate", type=float, default=0.25)
    parser.add_argument("--adaptive_tolerance", type=float, default=0.05)
    parser.add_argument("--coefficient_monitor_interval", type=int, default=100,
                        help="Optimizer-step interval for per-bit coefficient/activation paths")
    parser.add_argument("--lambda_min", type=float, default=0.1)
    parser.add_argument("--lambda_max", type=float, default=1000.0)

    # Misc
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    parser.add_argument(
        "--save_raw_embeddings",
        action="store_true",
        default=True,
        help=(
            "During --mode encode, persist the exact unnormalised pooled backbone vectors "
            "already computed for the binary head (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no_save_raw_embeddings",
        dest="save_raw_embeddings",
        action="store_false",
        help="Do not persist pooled backbone vectors during --mode encode.",
    )
    parser.add_argument(
        "--force_encode",
        action="store_true",
        help="Regenerate probabilities and bits even when a validated encoded file exists.",
    )
    parser.add_argument(
        "--lightweight_checkpoints",
        action="store_true",
        help=(
            "Save only head/decoder weights and reload the frozen backbone from the "
            "local Hugging Face cache."
        ),
    )
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_active <= 0:
        raise ValueError("--target_active must be > 0")
    if args.target_active >= args.k_factors:
        raise ValueError("--target_active must be smaller than --k_factors for sparse codes")
    if args.calibration_samples < 2:
        raise ValueError("--calibration_samples must be at least 2")
    if args.adaptive_interval < 1:
        raise ValueError("--adaptive_interval must be positive")
    if args.coefficient_monitor_interval < 1:
        raise ValueError("--coefficient_monitor_interval must be positive")
    if not 0 <= args.adaptive_ema_beta < 1:
        raise ValueError("--adaptive_ema_beta must be in [0, 1)")
    if not 0 < args.lambda_min <= args.lambda_max:
        raise ValueError("invalid adaptive lambda bounds")
    if args.lightweight_checkpoints and not args.freeze_backbone:
        raise ValueError("--lightweight_checkpoints requires --freeze_backbone")
    if args.require_cached_embeddings and not args.cache_training_embeddings:
        raise ValueError(
            "--require_cached_embeddings cannot be combined with "
            "--no_cache_training_embeddings"
        )
    if min(
        args.lambda_recon,
        args.lambda_tweet_sparse,
        args.lambda_balance,
        args.lambda_decor,
        args.lambda_entropy,
    ) < 0:
        raise ValueError("loss weights cannot be negative")

    if args.mode == "train":
        train(args)
    elif args.mode == "encode":
        encode(args)
    elif args.mode == "inspect":
        inspect_bits(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
