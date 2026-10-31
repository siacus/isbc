#!/usr/bin/env python3
"""Replay AG News SAE optimization from cached embeddings for timing only.

Each replay is isolated under ``runs/optimization_scaling/benchmarks``.  The
source cache is hard-linked read-only in practice (the trainer memory-maps it),
and the scientific run directories are never passed as ``--output_dir``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
BENCHMARK_ROOT = RUNS / "optimization_scaling" / "benchmarks"
STATUS_PATH = RUNS / "optimization_scaling" / "benchmark_status.json"
LOG_PATH = RUNS / "optimization_scaling" / "benchmark_queue.log"
INPUT_STEM = "agnews_all_ready"

BACKBONES = ("qwen", "minilm", "nvidia", "harrier", "embeddinggemma2")
CONFIGURATIONS = ((16, 6),)  # Common reference configuration reported in the paper.


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_log(message: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(f"{utc_now()}\t{message}\n")


def source_run_dir(backbone: str, k_factors: int, target_active: int) -> Path:
    suffix = "fixed_balance" if (k_factors, target_active) == (16, 6) else (
        f"k{k_factors}_target{target_active}_fixed_balance"
    )
    return RUNS / f"agnews_{backbone}_all_{suffix}"


def benchmark_run_dir(source: Path) -> Path:
    return BENCHMARK_ROOT / source.name


def cache_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    prefix = f"training_embeddings_{INPUT_STEM}"
    return (
        run_dir / f"{prefix}.npy",
        run_dir / f"{prefix}_ids.csv",
        run_dir / f"{prefix}_metadata.json",
    )


def materialize_cache(source: Path, destination: Path) -> None:
    source_matrix, source_ids, source_metadata = cache_paths(source)
    destination_matrix, destination_ids, destination_metadata = cache_paths(destination)
    missing = [str(path) for path in (source_matrix, source_ids, source_metadata) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing validated source cache: " + ", ".join(missing))
    destination.mkdir(parents=True, exist_ok=True)
    for origin, target in ((source_matrix, destination_matrix), (source_ids, destination_ids)):
        if target.exists():
            if not target.samefile(origin):
                raise RuntimeError(f"Refusing to replace non-identical benchmark cache: {target}")
        else:
            os.link(origin, target)
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    metadata["matrix"] = str(destination_matrix.resolve())
    metadata["ids"] = str(destination_ids.resolve())
    metadata["benchmark_cache_source"] = str(source.resolve())
    atomic_json(destination_metadata, metadata)


def common_command(source: Path, destination: Path) -> list[str]:
    config = json.loads((source / "best" / "run_config.json").read_text(encoding="utf-8"))
    arguments = [
        sys.executable,
        str(ROOT / "binary_factors_train.py"),
        "--input", config["input"],
        "--output_dir", str(destination),
        "--model_name", config["model_name"],
        "--pooling_mode", config["pooling_mode"],
        "--text_prefix", config["text_prefix"],
        "--precision", config["precision"],
        "--device", "mps",
        "--objective", config["objective"],
        "--decoder_intercept_init", config["decoder_intercept_init"],
        "--require_cached_embeddings",
        "--k_factors", str(config["k_factors"]),
        "--target_active", str(config["target_active"]),
        "--head_hidden_dim", str(config["head_hidden_dim"]),
        "--temperature", str(config["temperature"]),
        "--threshold", str(config["threshold"]),
        "--lambda_recon", str(config["lambda_recon"]),
        "--lambda_tweet_sparse", str(config["lambda_tweet_sparse"]),
        "--lambda_balance", str(config["lambda_balance"]),
        "--lambda_decor", str(config["lambda_decor"]),
        "--lambda_entropy", str(config["lambda_entropy"]),
        "--entropy_epsilon", str(config["entropy_epsilon"]),
        "--calibration_samples", str(config["calibration_samples"]),
        "--recon_scale_epsilon", str(config["recon_scale_epsilon"]),
        "--adaptive_interval", str(config["adaptive_interval"]),
        "--adaptive_ema_beta", str(config["adaptive_ema_beta"]),
        "--adaptive_rate", str(config["adaptive_rate"]),
        "--adaptive_tolerance", str(config["adaptive_tolerance"]),
        "--coefficient_monitor_interval", str(config["coefficient_monitor_interval"]),
        "--lambda_min", str(config["lambda_min"]),
        "--lambda_max", str(config["lambda_max"]),
        "--max_length", str(config["max_length"]),
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--num_workers", "0",
        "--lr", str(config["lr"]),
        "--weight_decay", str(config["weight_decay"]),
        "--warmup_ratio", str(config["warmup_ratio"]),
        "--grad_clip", str(config["grad_clip"]),
        "--seed", str(config["seed"]),
        "--lightweight_checkpoints",
    ]
    if not config.get("trust_remote_code", True):
        arguments.append("--no_trust_remote_code")
    if not config.get("freeze_backbone", True):
        arguments.append("--unfreeze_backbone")
    return arguments


def already_measured(destination: Path) -> bool:
    path = destination / "bit_generation_timing.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("optimization", {}).get("status") == "completed"
    except json.JSONDecodeError:
        return False


def main() -> int:
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    queue = [
        source_run_dir(backbone, k_factors, target_active)
        for k_factors, target_active in CONFIGURATIONS
        for backbone in BACKBONES
    ]
    for position, source in enumerate(queue, start=1):
        destination = benchmark_run_dir(source)
        if already_measured(destination):
            append_log(f"{position}/{len(queue)} skip completed {source.name}")
            continue
        materialize_cache(source, destination)
        status = {
            "status": "running",
            "position": position,
            "total": len(queue),
            "source_run": str(source.resolve()),
            "benchmark_run": str(destination.resolve()),
            "started_at": utc_now(),
            "device": "mps",
            "cached_embeddings_only": True,
        }
        atomic_json(STATUS_PATH, status)
        append_log(f"{position}/{len(queue)} start {source.name}")
        started = time.monotonic()
        with (destination / "optimizer_replay.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                common_command(source, destination),
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"},
                check=False,
            )
        elapsed = time.monotonic() - started
        if result.returncode:
            status.update({"status": "failed", "finished_at": utc_now(), "wall_seconds": elapsed, "exit_code": result.returncode})
            atomic_json(STATUS_PATH, status)
            append_log(f"{position}/{len(queue)} failed {source.name}: exit {result.returncode}")
            return result.returncode
        append_log(f"{position}/{len(queue)} complete {source.name}: launcher {elapsed:.1f}s")
    atomic_json(STATUS_PATH, {"status": "complete", "total": len(queue), "finished_at": utc_now(), "device": "mps", "cached_embeddings_only": True})
    append_log("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
