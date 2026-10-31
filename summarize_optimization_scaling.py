#!/usr/bin/env python3
"""Create a provenance-aware SAE optimizer-scaling inventory.

Optimizer time excludes frozen-backbone embedding creation, model loading,
calibration, checkpoint writing, code CSV materialisation, and downstream work.
Legacy runs without a directly measured artifact remain unavailable rather than
being reconstructed from file timestamps or heartbeat logs.

With --count-tokens, totals are calculated from the saved tokenizer after the
saved prefix and maximum-length truncation. Token counting is a separate audit
and never enters the reported optimizer duration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUT = RUNS / "optimization_scaling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-tokens", action="store_true")
    parser.add_argument(
        "--dataset", action="append", default=[], help="Restrict token counting only."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"id": str})
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    raise ValueError(f"Unsupported input table: {path}")


def dataset_from_input(path: Path) -> str:
    parts = list(path.parts)
    if "data" in parts and parts.index("data") + 1 < len(parts):
        return parts[parts.index("data") + 1]
    return path.stem


def canonical_runs() -> list[tuple[Path, dict[str, Any]]]:
    result = []
    metadata_paths = list(RUNS.glob("*/metadata.json"))
    benchmark_root = OUT / "benchmarks"
    if benchmark_root.is_dir():
        metadata_paths.extend(benchmark_root.glob("*/metadata.json"))
    for metadata_path in sorted(metadata_paths):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if (
            metadata.get("mode") == "train"
            and metadata.get("input")
            and metadata.get("model_name")
        ):
            result.append((metadata_path.parent, metadata))
    return result


def load_tokens() -> dict[str, dict[str, Any]]:
    path = OUT / "token_counts.csv"
    if not path.is_file():
        return {}
    return {
        str(row["run_dir"]): row.dropna().to_dict()
        for _, row in pd.read_csv(path).iterrows()
    }


def token_count_key(metadata: dict[str, Any]) -> str:
    payload = {
        "input": str(Path(metadata["input"]).resolve()),
        "model_name": metadata["model_name"],
        "text_prefix": str(metadata.get("text_prefix", "")),
        "max_length": int(metadata.get("max_length", 128)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def count_tokens(run_dir: Path, metadata: dict[str, Any], batch_size: int) -> dict[str, Any]:
    input_path = Path(metadata["input"])
    tokenizer_path = run_dir / "best"
    if not input_path.is_file():
        return {"status": "input_missing"}
    if not tokenizer_path.is_dir():
        return {"status": "tokenizer_missing"}
    table = read_table(input_path)
    if "text" not in table:
        return {"status": "text_column_missing"}
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=bool(metadata.get("trust_remote_code", True))
    )
    prefix = str(metadata.get("text_prefix", ""))
    maximum_length = int(metadata.get("max_length", 128))
    total = 0
    for start in range(0, len(table), batch_size):
        texts = [prefix + str(value) for value in table["text"].iloc[start : start + batch_size]]
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=maximum_length,
            padding=False,
            return_attention_mask=False,
        )
        total += sum(len(value) for value in encoded["input_ids"])
    return {
        "status": "measured",
        "token_count_key": token_count_key(metadata),
        "documents": int(len(table)),
        "tokens_after_prefix_and_truncation": int(total),
        "tokens_per_document": float(total / len(table)) if len(table) else None,
        "max_length": maximum_length,
        "tokenizer_source": str(tokenizer_path.resolve()),
    }


def timing(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "bit_generation_timing.json"
    if not path.is_file():
        return {
            "optimization_timing_status": "not_recorded_legacy_run",
            "optimization_seconds": None,
            "optimizer_steps_per_second": None,
            "document_updates_per_second": None,
        }
    try:
        record = json.loads(path.read_text(encoding="utf-8")).get("optimization", {})
    except json.JSONDecodeError:
        record = {}
    if record.get("status") != "completed":
        return {
            "optimization_timing_status": str(record.get("status", "invalid_timing_artifact")),
            "optimization_seconds": None,
            "optimizer_steps_per_second": None,
            "document_updates_per_second": None,
        }
    return {
        "optimization_timing_status": "measured",
        "optimization_seconds": record.get("elapsed_seconds"),
        "optimizer_steps_per_second": record.get("optimizer_steps_per_second"),
        "document_updates_per_second": record.get("document_updates_per_second"),
    }


def inventory_row(
    run_dir: Path, metadata: dict[str, Any], tokens: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    input_path = Path(metadata["input"])
    token_record = tokens.get(str(run_dir.resolve()), {})
    return {
        "run_scope": (
            "optimizer timing replay"
            if run_dir.parent == OUT / "benchmarks"
            else "scientific run"
        ),
        "dataset": dataset_from_input(input_path),
        "run": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "backbone": metadata.get("model_name"),
        "documents": metadata.get("training_embedding_cache", {}).get(
            "rows", token_record.get("documents")
        ),
        "embedding_dim": metadata.get("embedding_dim"),
        "k_factors": metadata.get("k_factors"),
        "target_active": metadata.get("target_active"),
        "epochs": metadata.get("epochs"),
        "batch_size": metadata.get("batch_size"),
        "device": metadata.get("device"),
        "cached_embeddings_used": metadata.get("training_embedding_cache_used"),
        "tokens_status": token_record.get("status", "not_counted"),
        "tokens_after_prefix_and_truncation": token_record.get(
            "tokens_after_prefix_and_truncation"
        ),
        "tokens_per_document": token_record.get("tokens_per_document"),
        **timing(run_dir),
    }


def display(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(frame: pd.DataFrame) -> None:
    lines = [
        "# SAE optimizer-scaling inventory",
        "",
        "Optimizer time means the synchronized wall-clock interval from first to last optimizer step. It excludes embedding generation, model loading, calibration, checkpointing, encoding, and downstream analyses. Legacy heartbeat logs can only bracket activity coarsely, so they are not treated as exact timings.",
        "",
        "`optimizer timing replay` rows are isolated reruns from validated cached embeddings. They preserve the saved SAE configuration but write only under `runs/optimization_scaling/benchmarks/`; the original scientific artifacts remain unchanged. The AG News replay queue explicitly requires MPS and disables CPU fallback.",
        "",
        "Tokens, when measured, are post-prefix and post-truncation input tokens. They are a scale covariate only, not a component of optimizer time.",
        "",
        "## Dataset index",
        "",
        "| Dataset | Runs | Optimizer timings | Token totals |",
        "|---|---:|---:|---:|",
    ]
    for dataset, group in frame.groupby("dataset", sort=True):
        lines.append(
            f"| {dataset} | {len(group)} | "
            f"{group['optimization_timing_status'].eq('measured').sum()} | "
            f"{group['tokens_status'].eq('measured').sum()} |"
        )
    for dataset, group in frame.groupby("dataset", sort=True):
        lines.extend(
            [
                "",
                f"## {dataset}",
                "",
        "| Scope | Run | Backbone | Docs | Tokens | K | Active | Epochs | Device | Optimizer seconds | Steps/s | Doc-updates/s | Timing provenance |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
            ]
        )
        for _, row in group.sort_values(
            ["backbone", "k_factors", "target_active", "run"]
        ).iterrows():
            lines.append(
                f"| {row['run_scope']} | {row['run']} | {row['backbone']} | {display(row['documents'], 0)} | "
                f"{display(row['tokens_after_prefix_and_truncation'], 0)} | "
                f"{display(row['k_factors'], 0)} | {display(row['target_active'], 1)} | "
                f"{display(row['epochs'], 0)} | {row['device']} | "
                f"{display(row['optimization_seconds'])} | "
                f"{display(row['optimizer_steps_per_second'], 3)} | "
                f"{display(row['document_updates_per_second'])} | "
                f"{row['optimization_timing_status']} |"
            )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tokens = load_tokens()
    selected = set(args.dataset)
    if args.count_tokens:
        measured_by_key = {
            record.get("token_count_key"): record
            for record in tokens.values()
            if record.get("status") == "measured" and record.get("token_count_key")
        }
        for run_dir, metadata in canonical_runs():
            if selected and dataset_from_input(Path(metadata["input"])) not in selected:
                continue
            key = str(run_dir.resolve())
            if tokens.get(key, {}).get("status") == "measured":
                continue
            shared_key = token_count_key(metadata)
            if shared_key in measured_by_key:
                tokens[key] = {
                    **measured_by_key[shared_key],
                    "run_dir": key,
                    "token_count_reused_from_equivalent_configuration": True,
                }
                continue
            print(f"counting tokens: {run_dir.name}", flush=True)
            tokens[key] = {"run_dir": key, **count_tokens(run_dir, metadata, args.batch_size)}
            measured_by_key[shared_key] = tokens[key]
        pd.DataFrame(tokens.values()).sort_values("run_dir").to_csv(
            OUT / "token_counts.csv", index=False
        )
    frame = pd.DataFrame(
        [inventory_row(run_dir, metadata, tokens) for run_dir, metadata in canonical_runs()]
    ).sort_values(["dataset", "backbone", "k_factors", "target_active", "run"])
    frame.to_csv(OUT / "optimization_scaling.csv", index=False)
    write_markdown(frame)
    print(f"Wrote {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
