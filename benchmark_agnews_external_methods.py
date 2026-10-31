#!/usr/bin/env python3
"""Compare external binary-representation packages on cached Qwen AG News data.

The learned models come from the official HypotheSAEs and BinarySentEmb
checkouts.  This file is only a data/device/evaluation adapter: it supplies the
same cached Qwen embeddings and downstream split used by the main AG News
report, invokes the upstream model classes, and exports comparable artifacts.

BinarySentEmb's 2019 code hard-codes a 4,096-dimensional InferSent input and
CUDA calls.  The adapter retains its upstream ``LinearAutoEncoder`` class and
training objective while replacing only the input/output layer dimensions with
the observed Qwen dimension and redirecting its tensor ``.cuda()`` calls to
Apple MPS in this isolated process.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predictive_modeling import (
    COMMON_CLASSIFIERS,
    _fit_one,
    add_balanced_accuracy_intervals,
    probability_log_odds,
)
from information_retention_extended import posthoc_linear_decoder_metrics_many


EXPECTED_HYPOTHESAES_COMMIT = "706d8979c71befe8e3151f51a8d4dbf87d4d7e41"
EXPECTED_BINARYSENTEMB_COMMIT = "2aaaae8062ea68fdce700c5765822a09d525eaf1"
sys.dont_write_bytecode = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypothesaes_repo",
        type=Path,
        default=Path(".external_methods/HypotheSAEs"),
    )
    parser.add_argument(
        "--binarysentemb_repo",
        type=Path,
        default=Path(".external_methods/BinarySentEmb"),
    )
    parser.add_argument(
        "--raw_embeddings",
        type=Path,
        default=Path(
            "runs/agnews_qwen_all_fixed_balance/"
            "raw_embeddings_agnews_all_ready_best.npy"
        ),
    )
    parser.add_argument(
        "--raw_ids",
        type=Path,
        default=Path(
            "runs/agnews_qwen_all_fixed_balance/"
            "raw_embeddings_agnews_all_ready_best_ids.csv"
        ),
    )
    parser.add_argument(
        "--our_encoded",
        type=Path,
        default=Path(
            "runs/agnews_qwen_all_fixed_balance/encoded_agnews_all_ready_best.csv"
        ),
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("data/agnews/agnews_all.csv")
    )
    parser.add_argument(
        "--predictive_split",
        type=Path,
        default=Path(
            "runs/agnews_embedding_comparison/all_fixed_balance_label_name/"
            "predictive_split.csv"
        ),
    )
    parser.add_argument(
        "--existing_metrics",
        type=Path,
        default=Path(
            "runs/agnews_embedding_comparison/all_fixed_balance_label_name/"
            "predictive_modeling_metrics.csv"
        ),
    )
    parser.add_argument(
        "--existing_predictions",
        type=Path,
        default=Path(
            "runs/agnews_embedding_comparison/all_fixed_balance_label_name/"
            "predictive_modeling_predictions.csv"
        ),
    )
    parser.add_argument(
        "--existing_information",
        type=Path,
        default=Path(
            "runs/agnews_embedding_comparison/all_fixed_balance_label_name/"
            "information_retention_summary.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "runs/agnews_embedding_comparison/external_sparse_encoders_qwen_k16"
        ),
    )
    parser.add_argument("--n_bits", type=int, default=16)
    parser.add_argument("--target_active", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hypothesaes_epochs", type=int, default=100)
    parser.add_argument("--binarysentemb_epochs", type=int, default=20)
    parser.add_argument("--binarysentemb_learning_rate", type=float, default=1e-5)
    parser.add_argument("--binarysentemb_struct_coef", type=float, default=0.8)
    parser.add_argument(
        "--bootstrap_draws",
        type=int,
        default=2_000,
        help="Shared class-stratified bootstrap replicates for all held-out BA intervals.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def verify_checkout(repo: Path, expected_commit: str, required_file: str) -> dict:
    repo = repo.resolve()
    source = repo / required_file
    if not source.is_file():
        raise FileNotFoundError(f"Missing upstream source: {source}")
    commit = git_commit(repo)
    if commit != expected_commit:
        raise ValueError(
            f"Unexpected upstream commit for {repo}: {commit}; expected {expected_commit}"
        )
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(f"Upstream checkout must be clean: {repo}")
    return {
        "repo": str(repo),
        "commit": commit,
        "source": str(source),
        "source_sha256": sha256_file(source),
    }


def load_module(source: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load upstream module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_mps() -> torch.device:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is required; CPU fallback is intentionally disabled")
    return torch.device("mps")


def load_aligned_inputs(args: argparse.Namespace) -> dict[str, Any]:
    raw = np.load(args.raw_embeddings, mmap_mode="r")
    raw = np.asarray(raw, dtype=np.float32)
    ids_frame = pd.read_csv(args.raw_ids, dtype={"id": str})
    if raw.ndim != 2 or raw.shape[1] != 1024:
        raise ValueError(f"Expected 1,024-dimensional Qwen embeddings, found {raw.shape}")
    if len(raw) != 127_600:
        raise ValueError(f"Expected the full 127,600-document AG News corpus, found {len(raw)}")
    if "id" not in ids_frame or len(ids_frame) != len(raw):
        raise ValueError("Raw embedding IDs do not align with the matrix")
    ids = ids_frame["id"].astype(str).to_numpy()
    if pd.Series(ids).duplicated().any():
        raise ValueError("Duplicate IDs in raw embedding cache")

    labels = pd.read_csv(args.labels, dtype={"id": str}).set_index("id")
    split = pd.read_csv(args.predictive_split, dtype={"id": str}).set_index("id")
    encoded = pd.read_csv(args.our_encoded, dtype={"id": str}).set_index("id")
    for name, frame in (("labels", labels), ("split", split), ("encoded", encoded)):
        if frame.index.duplicated().any():
            raise ValueError(f"Duplicate IDs in {name}")
        missing = set(ids) - set(frame.index)
        extra = set(frame.index) - set(ids)
        if missing or extra:
            raise ValueError(
                f"{name} ID set differs from raw embeddings: {len(missing)} missing, {len(extra)} extra"
            )

    labels = labels.loc[ids]
    split = split.loc[ids]
    encoded = encoded.loc[ids]
    y = labels["label_name"].astype(str).to_numpy()
    assignment = split["predictive_split"].astype(str).to_numpy()
    train_indices = np.flatnonzero(assignment == "train")
    test_indices = np.flatnonzero(assignment == "test")
    if len(train_indices) + len(test_indices) != len(ids):
        raise ValueError("Predictive split contains values other than train/test")
    if len(train_indices) != 102_080 or len(test_indices) != 25_520:
        raise ValueError(
            f"Expected 102,080/25,520 split, found {len(train_indices)}/{len(test_indices)}"
        )

    bit_columns = sorted(column for column in encoded if column.startswith("bit_"))
    probability_columns = sorted(
        column for column in encoded if column.startswith("prob_")
    )
    if len(bit_columns) != args.n_bits or len(probability_columns) != args.n_bits:
        raise ValueError("Our Qwen encoded artifact does not have the requested width")
    our_hard = encoded[bit_columns].to_numpy(dtype=np.uint8)
    our_probabilities = encoded[probability_columns].to_numpy(dtype=np.float32)
    our_soft = probability_log_odds(our_probabilities)
    return {
        "raw": raw,
        "ids": ids,
        "y": y,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "our_hard": our_hard,
        "our_soft": our_soft,
        "validation": {
            "n_documents": int(len(ids)),
            "embedding_dim": int(raw.shape[1]),
            "unique_ids": int(len(np.unique(ids))),
            "id_sets_identical": True,
            "train_documents": int(len(train_indices)),
            "test_documents": int(len(test_indices)),
            "our_bit_columns": bit_columns,
            "our_probability_columns": probability_columns,
        },
    }


def reconstruction_summary(raw: np.ndarray, reconstructed: np.ndarray) -> dict:
    observed = np.asarray(raw, dtype=np.float64)
    predicted = np.asarray(reconstructed, dtype=np.float64)
    residual_sse = float(np.square(observed - predicted).sum())
    baseline_sse = float(np.square(observed - observed.mean(axis=0)).sum())
    denominator = np.linalg.norm(observed, axis=1) * np.linalg.norm(predicted, axis=1)
    valid = denominator > 1e-12
    cosine = np.full(len(observed), np.nan)
    cosine[valid] = np.einsum(
        "ij,ij->i", observed[valid], predicted[valid]
    ) / denominator[valid]
    normalized_mse = residual_sse / baseline_sse
    return {
        "mse": residual_sse / observed.size,
        "mean_baseline_mse": baseline_sse / observed.size,
        "normalized_mse": normalized_mse,
        "reconstruction_r2": 1.0 - normalized_mse,
        "mean_cosine_similarity": float(np.nanmean(cosine)),
    }


def code_summary(hard: np.ndarray) -> dict:
    bits = np.asarray(hard, dtype=np.uint8)
    strings = np.asarray(["".join(map(str, row.tolist())) for row in bits])
    counts = pd.Series(strings).value_counts()
    shares = counts.to_numpy(dtype=float) / len(bits)
    entropy = float(-(shares * np.log2(shares)).sum())
    active = bits.sum(axis=1)
    marginal = bits.mean(axis=0)
    interior = (marginal > 0) & (marginal < 1)
    marginal_entropy = np.zeros_like(marginal, dtype=float)
    q = marginal[interior]
    marginal_entropy[interior] = -q * np.log2(q) - (1 - q) * np.log2(1 - q)
    return {
        "n_documents": int(len(bits)),
        "n_bits": int(bits.shape[1]),
        "mean_active_bits": float(active.mean()),
        "sd_active_bits": float(active.std(ddof=0)),
        "min_active_bits": int(active.min()),
        "max_active_bits": int(active.max()),
        "observed_profiles": int(len(counts)),
        "profile_entropy_bits": entropy,
        "effective_profiles": float(2**entropy),
        "most_common_profile_share": float(shares[0]),
        "mean_marginal_bit_entropy": float(marginal_entropy.mean()),
        "constant_bits": int(((marginal <= 0.0) | (marginal >= 1.0)).sum()),
    }


def save_codes(
    path: Path, ids: np.ndarray, hard: np.ndarray, latent: np.ndarray
) -> None:
    frame = pd.DataFrame({"id": ids})
    for index in range(hard.shape[1]):
        frame[f"bit_{index:02d}"] = hard[:, index].astype(np.uint8)
        frame[f"latent_{index:02d}"] = latent[:, index].astype(np.float32)
    frame.to_csv(path, index=False)


def train_hypothesaes(
    raw: np.ndarray,
    ids: np.ndarray,
    upstream: dict,
    output_dir: Path,
    *,
    n_bits: int,
    target_active: int,
    epochs: int,
    seed: int,
    device: torch.device,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]], pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    codes_path = output_dir / "codes.csv"
    history_path = output_dir / "training_history.csv"
    reconstruction_path = output_dir / "reconstruction.json"
    if codes_path.is_file() and history_path.is_file() and reconstruction_path.is_file() and not force:
        frame = pd.read_csv(codes_path, dtype={"id": str}).set_index("id").loc[ids]
        hard = frame.filter(regex=r"^bit_").to_numpy(dtype=np.uint8)
        latent = frame.filter(regex=r"^latent_").to_numpy(dtype=np.float32)
        diagnostics_path = output_dir / "training_diagnostics.json"
        return (
            hard,
            latent,
            json.loads(reconstruction_path.read_text()),
            pd.read_csv(history_path),
            json.loads(diagnostics_path.read_text()) if diagnostics_path.is_file() else {"reused": True},
        )

    module = load_module(Path(upstream["source"]), "official_hypothesaes_sae")
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = module.SparseAutoencoder(
        input_dim=raw.shape[1],
        m_total_neurons=n_bits,
        k_active_neurons=target_active,
        device=str(device),
    )
    started = time.monotonic()
    history = model.fit(
        X_train=torch.from_numpy(np.array(raw, dtype=np.float32, copy=True)),
        save_dir=str(output_dir / "checkpoint"),
        batch_size=512,
        learning_rate=5e-4,
        n_epochs=epochs,
        aux_coef=1 / 32,
        multi_coef=0.0,
        patience=3,
        clip_grad=1.0,
        show_progress=True,
    )
    n_epochs_completed = len(history["train_loss"])
    history_columns: dict[str, Any] = {
        "epoch": np.arange(1, n_epochs_completed + 1, dtype=int),
        "train_loss": history["train_loss"],
        "dead_neuron_ratio": history["dead_neuron_ratio"],
    }
    if len(history.get("val_loss", [])) == n_epochs_completed:
        history_columns["val_loss"] = history["val_loss"]
    history_frame = pd.DataFrame(history_columns)
    history_frame.to_csv(history_path, index=False)

    latent_batches = []
    reconstruction_batches = []
    hard_reconstruction_batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(raw), 1024):
            batch = torch.from_numpy(np.asarray(raw[start : start + 1024])).to(device)
            reconstruction, info = model(batch)
            latent_batches.append(info["activations"].cpu().numpy())
            reconstruction_batches.append(reconstruction.cpu().numpy())
            hard_support = (info["activations"] > 0).to(dtype=batch.dtype)
            hard_reconstruction_batches.append(
                (model.decoder(hard_support) + model.input_bias).cpu().numpy()
            )
    latent = np.concatenate(latent_batches).astype(np.float32)
    hard = (latent > 0).astype(np.uint8)
    reconstruction = np.concatenate(reconstruction_batches).astype(np.float32)
    hard_reconstruction = np.concatenate(hard_reconstruction_batches).astype(np.float32)
    reconstruction_metrics = {
        "continuous_sparse_activations_learned_decoder": reconstruction_summary(raw, reconstruction),
        "hard_support_learned_decoder": reconstruction_summary(raw, hard_reconstruction),
    }
    save_codes(codes_path, ids, hard, latent)
    reconstruction_path.write_text(
        json.dumps(reconstruction_metrics, indent=2, sort_keys=True) + "\n"
    )
    diagnostics = {
        "reused": False,
        "requested_epochs": int(epochs),
        "completed_epochs": int(n_epochs_completed),
        "elapsed_seconds": float(time.monotonic() - started),
        "finite_train_loss": bool(np.isfinite(history_frame["train_loss"]).all()),
        "initial_train_loss": float(history_frame["train_loss"].iloc[0]),
        "final_train_loss": float(history_frame["train_loss"].iloc[-1]),
        "dead_neuron_ratio_final": float(history_frame["dead_neuron_ratio"].iloc[-1]),
    }
    (output_dir / "training_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return hard, latent, reconstruction_metrics, history_frame, diagnostics


@contextlib.contextmanager
def redirect_tensor_cuda_to_mps(device: torch.device) -> Iterator[None]:
    original = torch.Tensor.cuda

    def redirected(tensor, *args, **kwargs):
        return tensor.to(device)

    torch.Tensor.cuda = redirected
    try:
        yield
    finally:
        torch.Tensor.cuda = original


def train_binarysentemb(
    raw: np.ndarray,
    ids: np.ndarray,
    upstream: dict,
    output_dir: Path,
    *,
    n_bits: int,
    epochs: int,
    learning_rate: float,
    struct_coef: float,
    device: torch.device,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]], pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    codes_path = output_dir / "codes.csv"
    history_path = output_dir / "training_history.csv"
    reconstruction_path = output_dir / "reconstruction.json"
    if codes_path.is_file() and history_path.is_file() and reconstruction_path.is_file() and not force:
        frame = pd.read_csv(codes_path, dtype={"id": str}).set_index("id").loc[ids]
        hard = frame.filter(regex=r"^bit_").to_numpy(dtype=np.uint8)
        latent = frame.filter(regex=r"^latent_").to_numpy(dtype=np.float32)
        diagnostics_path = output_dir / "training_diagnostics.json"
        return (
            hard,
            latent,
            json.loads(reconstruction_path.read_text()),
            pd.read_csv(history_path),
            json.loads(diagnostics_path.read_text()) if diagnostics_path.is_file() else {"reused": True},
        )

    module = load_module(Path(upstream["source"]), "official_binarysentemb_encoders")
    seed = 2019  # upstream default
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = module.LinearAutoEncoder(n_bits)
    # The upstream implementation fixes InferSent's 4096 dimensions.  Only the
    # two boundary dimensions are adapted; its encoder, ST binarization, and
    # decoder execution remain the official class implementation.
    model.encoder = torch.nn.Sequential(torch.nn.Linear(raw.shape[1], n_bits))
    model.decoder = torch.nn.Sequential(torch.nn.Linear(n_bits, raw.shape[1]))
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    rng = np.random.RandomState(seed)
    history_rows = []
    effective_batch_size = 128  # four equal groups, matching upstream pair comparison
    started = time.monotonic()
    with redirect_tensor_cuda_to_mps(device):
        for epoch in range(1, epochs + 1):
            model.train()
            reconstruction_losses = []
            structural_losses = []
            total_losses = []
            permutation = rng.permutation(len(raw))
            for start in range(0, len(raw), effective_batch_size):
                selected = permutation[start : start + effective_batch_size]
                usable = len(selected) - len(selected) % 4
                if usable < 4:
                    continue
                batch = torch.from_numpy(np.asarray(raw[selected[:usable]])).to(device)
                code_groups = torch.chunk(batch, 4, dim=0)
                outputs = [model(group) for group in code_groups]
                discrete = [model.encode(group) for group in code_groups]
                continuous_distance_1 = 1.0 - torch.nn.functional.cosine_similarity(
                    code_groups[0], code_groups[1]
                )
                continuous_distance_2 = 1.0 - torch.nn.functional.cosine_similarity(
                    code_groups[2], code_groups[3]
                )
                discrete_distance_1 = torch.nn.functional.pairwise_distance(
                    discrete[0], discrete[1]
                )
                discrete_distance_2 = torch.nn.functional.pairwise_distance(
                    discrete[2], discrete[3]
                )
                structural_sign = (
                    (continuous_distance_1 - continuous_distance_2) > 0
                ).float()
                structural_loss = (
                    torch.nn.functional.relu(
                        (discrete_distance_2 - discrete_distance_1)
                        * (2 * structural_sign - 1).detach()
                    )
                    ** 2
                ).mean()
                reconstruction_loss = torch.stack(
                    [
                        torch.mean((observed - predicted) ** 2)
                        for observed, predicted in zip(code_groups, outputs)
                    ]
                ).mean()
                loss = reconstruction_loss + struct_coef * structural_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                reconstruction_losses.append(float(reconstruction_loss.item()))
                structural_losses.append(float(structural_loss.item()))
                total_losses.append(float(loss.item()))
            history_rows.append(
                {
                    "epoch": epoch,
                    "reconstruction_loss": float(np.mean(reconstruction_losses)),
                    "semantic_preserving_loss": float(np.mean(structural_losses)),
                    "weighted_semantic_preserving_loss": float(
                        struct_coef * np.mean(structural_losses)
                    ),
                    "total_loss": float(np.mean(total_losses)),
                }
            )
            print(
                f"BinarySentEmb epoch {epoch}/{epochs}: "
                f"reconstruction={history_rows[-1]['reconstruction_loss']:.6f}, "
                f"semantic={history_rows[-1]['semantic_preserving_loss']:.6f}"
            )

        model.eval()
        hard_batches = []
        latent_batches = []
        reconstruction_batches = []
        continuous_reconstruction_batches = []
        with torch.no_grad():
            for start in range(0, len(raw), 1024):
                batch = torch.from_numpy(np.asarray(raw[start : start + 1024])).to(device)
                hard_batches.append(model.encode(batch).cpu().numpy())
                latent_batches.append(model.encoder(batch).cpu().numpy())
                reconstruction_batches.append(model(batch).cpu().numpy())
                continuous_reconstruction_batches.append(
                    model.decoder(2.0 * model.encoder(batch) - 1.0).cpu().numpy()
                )

    hard = np.rint(np.concatenate(hard_batches)).astype(np.uint8)
    latent = np.concatenate(latent_batches).astype(np.float32)
    reconstruction = np.concatenate(reconstruction_batches).astype(np.float32)
    continuous_reconstruction = np.concatenate(continuous_reconstruction_batches).astype(np.float32)
    reconstruction_metrics = {
        "hard_bits_learned_decoder": reconstruction_summary(raw, reconstruction),
        "continuous_logits_counterfactual_learned_decoder": reconstruction_summary(
            raw, continuous_reconstruction
        ),
    }
    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(history_path, index=False)
    save_codes(codes_path, ids, hard, latent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(raw.shape[1]),
            "n_bits": int(n_bits),
            "upstream_commit": upstream["commit"],
        },
        output_dir / "model.pt",
    )
    reconstruction_path.write_text(
        json.dumps(reconstruction_metrics, indent=2, sort_keys=True) + "\n"
    )
    diagnostics = {
        "reused": False,
        "requested_epochs": int(epochs),
        "completed_epochs": int(len(history_frame)),
        "elapsed_seconds": float(time.monotonic() - started),
        "finite_reconstruction_loss": bool(np.isfinite(history_frame["reconstruction_loss"]).all()),
        "finite_structural_loss": bool(np.isfinite(history_frame["semantic_preserving_loss"]).all()),
        "finite_total_loss": bool(np.isfinite(history_frame["total_loss"]).all()),
        "initial_total_loss": float(history_frame["total_loss"].iloc[0]),
        "final_total_loss": float(history_frame["total_loss"].iloc[-1]),
    }
    (output_dir / "training_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return hard, latent, reconstruction_metrics, history_frame, diagnostics


def evaluate_new_representations(
    representations: dict[tuple[str, str], np.ndarray],
    data: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_rows = []
    importance_rows = []
    for (method, representation), matrix in representations.items():
        feature_names = [f"feature_{index:02d}" for index in range(matrix.shape[1])]
        for classifier in COMMON_CLASSIFIERS:
            print(f"modeling: {method}/{representation}/{classifier}")
            metrics, prediction, _, model, true_probability = _fit_one(
                matrix,
                data["y"],
                data["train_indices"],
                data["test_indices"],
                classifier,
                seed,
            )
            metric_rows.append(
                {
                    "run": method,
                    "representation": representation,
                    "n_features": int(matrix.shape[1]),
                    "classifier": classifier,
                    "train_rows": int(len(data["train_indices"])),
                    "test_rows": int(len(data["test_indices"])),
                    **metrics,
                }
            )
            test_ids = data["ids"][data["test_indices"]]
            test_y = data["y"][data["test_indices"]]
            for index, (doc_id, truth, predicted) in enumerate(
                zip(test_ids, test_y, prediction)
            ):
                prediction_rows.append(
                    {
                        "id": doc_id,
                        "true_label": truth,
                        "run": method,
                        "representation": representation,
                        "classifier": classifier,
                        "predicted_label": predicted,
                        "true_class_probability": (
                            float(true_probability[index])
                            if true_probability is not None
                            else np.nan
                        ),
                    }
                )
            if hasattr(model, "coef_"):
                importance = np.mean(np.abs(np.asarray(model.coef_)), axis=0)
                importance_kind = "mean absolute class coefficient"
            elif hasattr(model, "feature_importances_"):
                importance = np.asarray(model.feature_importances_)
                importance_kind = "random-forest impurity importance"
            else:
                continue
            for feature, value in zip(feature_names, importance):
                importance_rows.append(
                    {
                        "run": method,
                        "representation": representation,
                        "classifier": classifier,
                        "feature": feature,
                        "importance": float(value),
                        "importance_definition": importance_kind,
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    importance = pd.DataFrame(importance_rows)
    return metrics, predictions, importance


def add_shared_balanced_accuracy_intervals(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    seed: int,
    bootstrap_draws: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use identical class-stratified document resamples within each classifier.

    Unlike independent per-row bootstrap seeds, this makes every representation in a
    classifier panel share the same bootstrap document indices.  It is therefore
    suitable both for individual BA intervals and paired visual comparison.
    """
    enriched = metrics.copy()
    enriched["balanced_accuracy_ci_low"] = np.nan
    enriched["balanced_accuracy_ci_high"] = np.nan
    protocol: dict[str, Any] = {
        "method": "shared class-stratified percentile bootstrap",
        "draws": int(bootstrap_draws),
        "seed": int(seed),
        "shared_within": "classifier across all representations",
        "confidence_level": 0.95,
    }
    for classifier in COMMON_CLASSIFIERS:
        frame = predictions[predictions["classifier"].astype(str).eq(classifier)].copy()
        if frame.empty:
            continue
        frame["id"] = frame["id"].astype(str)
        keys = list(dict.fromkeys(zip(frame["run"].astype(str), frame["representation"].astype(str))))
        indexed = {
            key: part.set_index("id").sort_index()
            for key, part in frame.groupby(["run", "representation"], sort=False)
        }
        reference = indexed[keys[0]]
        ids = reference.index
        truth = reference["true_label"].astype(str).to_numpy()
        correct = []
        for key in keys:
            part = indexed[key].reindex(ids)
            if part["true_label"].isna().any() or not np.array_equal(
                part["true_label"].astype(str).to_numpy(), truth
            ):
                raise ValueError(f"Unpaired held-out predictions for {classifier}/{key}")
            correct.append(part["predicted_label"].astype(str).to_numpy() == truth)
        correct_array = np.asarray(correct, dtype=np.float64)
        labels = sorted(set(truth))
        digest = hashlib.sha256(f"{seed}|{classifier}|shared-ba".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        bootstrap = np.zeros((bootstrap_draws, len(keys)), dtype=np.float64)
        for label in labels:
            positions = np.flatnonzero(truth == label)
            class_correct = correct_array[:, positions]
            for start in range(0, bootstrap_draws, 64):
                stop = min(bootstrap_draws, start + 64)
                sampled = rng.integers(0, len(positions), size=(stop - start, len(positions)))
                bootstrap[start:stop] += class_correct[:, sampled].mean(axis=2).T / len(labels)
        lows, highs = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        for key, low, high in zip(keys, lows, highs):
            mask = (
                enriched["run"].astype(str).eq(key[0])
                & enriched["representation"].astype(str).eq(key[1])
                & enriched["classifier"].astype(str).eq(classifier)
            )
            enriched.loc[mask, "balanced_accuracy_ci_low"] = float(low)
            enriched.loc[mask, "balanced_accuracy_ci_high"] = float(high)
    return enriched, protocol


def add_binary_to_raw_ratios(metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach BA(binary code) / BA(raw Qwen) for matched downstream probes."""
    frame = metrics.copy()
    frame["ba_binary_over_raw_embedding"] = np.nan
    for classifier in COMMON_CLASSIFIERS:
        raw = frame[
            frame["run"].astype(str).eq("qwen_baseline")
            & frame["representation"].astype(str).eq("raw_embeddings")
            & frame["classifier"].astype(str).eq(classifier)
        ]
        if len(raw) != 1:
            raise ValueError(f"Need one raw Qwen baseline for {classifier}")
        denominator = float(raw.iloc[0]["balanced_accuracy"])
        mask = (
            frame["representation"].astype(str).eq("hard_bits")
            & frame["classifier"].astype(str).eq(classifier)
        )
        frame.loc[mask, "ba_binary_over_raw_embedding"] = (
            frame.loc[mask, "balanced_accuracy"].astype(float) / denominator
        )
    return frame


def load_existing_baselines(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(args.existing_metrics)
    predictions = pd.read_csv(args.existing_predictions, dtype={"id": str})
    representation_map = {
        "hard_bits": ("ours", "hard_bits"),
        "head_log_odds": ("ours", "continuous_latent"),
        "raw_embeddings": ("qwen_baseline", "raw_embeddings"),
    }
    metrics = metrics[
        (metrics["run"] == "qwen")
        & metrics["classifier"].isin(COMMON_CLASSIFIERS)
        & metrics["representation"].isin(representation_map)
    ].copy()
    predictions = predictions[
        (predictions["run"] == "qwen")
        & predictions["classifier"].isin(COMMON_CLASSIFIERS)
        & predictions["representation"].isin(representation_map)
    ].copy()
    for old, (method, representation) in representation_map.items():
        metric_mask = metrics["representation"] == old
        prediction_mask = predictions["representation"] == old
        metrics.loc[metric_mask, "run"] = method
        metrics.loc[metric_mask, "representation"] = representation
        predictions.loc[prediction_mask, "run"] = method
        predictions.loc[prediction_mask, "representation"] = representation
    return metrics, predictions


def plot_balanced_accuracy(metrics: pd.DataFrame, output_path: Path) -> None:
    classifiers = list(COMMON_CLASSIFIERS)
    series = [
        ("qwen_baseline", "raw_embeddings", "Raw Qwen", "#222222", "o"),
        ("ours", "hard_bits", "Ours hard", "#1F77B4", "o"),
        ("ours", "continuous_latent", "Ours soft", "#1F77B4", "s"),
        ("hypothesaes", "hard_bits", "HypotheSAEs support", "#D62728", "o"),
        ("hypothesaes", "continuous_latent", "HypotheSAEs activations", "#D62728", "s"),
        ("binarysentemb", "hard_bits", "BinarySentEmb hard", "#2CA02C", "o"),
        ("binarysentemb", "continuous_latent", "BinarySentEmb logits", "#2CA02C", "s"),
    ]
    offsets = np.linspace(-0.30, 0.30, len(series))
    figure, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    bounds = []
    for offset, (method, representation, label, color, marker) in zip(offsets, series):
        x_values = []
        values = []
        lower = []
        upper = []
        for index, classifier in enumerate(classifiers):
            selected = metrics[
                (metrics["run"] == method)
                & (metrics["representation"] == representation)
                & (metrics["classifier"] == classifier)
            ]
            if selected.empty:
                continue
            row = selected.iloc[0]
            value = float(row["balanced_accuracy"])
            low = float(row["balanced_accuracy_ci_low"])
            high = float(row["balanced_accuracy_ci_high"])
            x_values.append(index + offset)
            values.append(value)
            lower.append(value - low)
            upper.append(high - value)
            bounds.extend((low, high))
        axis.errorbar(
            x_values,
            values,
            yerr=np.asarray([lower, upper]),
            fmt=marker,
            color=color,
            label=label,
            capsize=3,
            linewidth=1.2,
            markersize=6,
        )
    axis.set_xticks(range(len(classifiers)), [name.replace("_", "\n") for name in classifiers])
    axis.set_ylabel("Held-out balanced accuracy (95% stratified-bootstrap CI)")
    axis.set_title("AG News, cached Qwen embeddings: external binary methods")
    axis.grid(axis="y", alpha=0.25)
    if bounds:
        axis.set_ylim(max(0, min(bounds) - 0.04), min(1, max(bounds) + 0.04))
    axis.legend(ncol=2, fontsize=9)
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_tradeoff(metrics: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    colors = {"ours": "#1F77B4", "hypothesaes": "#D62728", "binarysentemb": "#2CA02C", "qwen_baseline": "#222222"}
    markers = {"hard_bits": "o", "continuous_latent": "s", "raw_embeddings": "D"}
    for _, row in metrics.iterrows():
        axis.scatter(
            row["balanced_accuracy"],
            row["balanced_accuracy_loss_pp"],
            color=colors[row["run"]],
            marker=markers[row["representation"]],
            s=65,
            alpha=0.85,
        )
        axis.annotate(
            f"{row['run']}\n{row['classifier']}",
            (row["balanced_accuracy"], row["balanced_accuracy_loss_pp"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
        )
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Held-out balanced accuracy (higher is better)")
    axis.set_ylabel("Train-test balanced-accuracy loss, percentage points (lower is better)")
    axis.set_title("Predictive utility versus generalization gap")
    axis.grid(alpha=0.25)
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_training(
    hypothesaes_history: pd.DataFrame,
    binarysentemb_history: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(
        np.arange(1, len(hypothesaes_history) + 1),
        hypothesaes_history["train_loss"],
        color="#D62728",
    )
    axes[0].set_title("HypotheSAEs official training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    for column, label in (
        ("reconstruction_loss", "reconstruction"),
        ("weighted_semantic_preserving_loss", "weighted semantic-preserving"),
        ("total_loss", "total"),
    ):
        axes[1].plot(binarysentemb_history["epoch"], binarysentemb_history[column], label=label)
    axes[1].set_title("BinarySentEmb official objective terms")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss contribution")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_confusions(predictions: pd.DataFrame, output_path: Path) -> None:
    classes = sorted(predictions["true_label"].astype(str).unique())
    rows = list(
        dict.fromkeys(
            zip(predictions["run"].astype(str), predictions["representation"].astype(str))
        )
    )
    classifiers = list(COMMON_CLASSIFIERS)
    figure, axes = plt.subplots(
        len(rows), len(classifiers),
        figsize=(4.8 * len(classifiers), 4.0 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row_index, (method, representation) in enumerate(rows):
        for column_index, classifier in enumerate(classifiers):
            axis = axes[row_index, column_index]
            selected = predictions[
                (predictions["run"] == method)
                & (predictions["representation"] == representation)
                & (predictions["classifier"] == classifier)
            ]
            matrix = confusion_matrix(
                selected["true_label"].astype(str),
                selected["predicted_label"].astype(str),
                labels=classes,
                normalize="true",
            )
            image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    axis.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if matrix[i, j] >= 0.55 else "black",
                    )
            axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=7)
            axis.set_yticks(range(len(classes)), classes, fontsize=7)
            axis.set_title(f"{method} / {representation}\n{classifier}", fontsize=9)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.35, label="Share within true class")
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    code_summaries: pd.DataFrame,
    reconstructions: pd.DataFrame,
    hard_ols: pd.DataFrame,
    metadata: dict,
) -> None:
    hard = metrics[metrics["representation"] == "hard_bits"]
    best_hard = hard.sort_values("balanced_accuracy", ascending=False).iloc[0]
    logistic_hard = hard[hard["classifier"] == "logistic_regression"].set_index("run")
    best_continuous = metrics[
        metrics["representation"] == "continuous_latent"
    ].sort_values("balanced_accuracy", ascending=False).iloc[0]
    best_raw = metrics[metrics["representation"] == "raw_embeddings"].sort_values(
        "balanced_accuracy", ascending=False
    ).iloc[0]
    lines = [
        "# AG News external binary-method comparison (Qwen only)",
        "",
        "## TL;DR",
        "",
        (
            f"All methods use the same {metadata['n_documents']:,} AG News documents, "
            "the same cached raw Qwen embeddings, and the exact same held-out IDs. "
            f"The strongest hard-code result is **{best_hard['run']} / "
            f"{best_hard['classifier']}** with balanced accuracy "
            f"**{best_hard['balanced_accuracy']:.3f}**."
        ),
        "",
        (
            "With the common logistic probe, hard-code balanced accuracy is "
            f"**{logistic_hard.loc['ours', 'balanced_accuracy']:.3f}** for ours, "
            f"**{logistic_hard.loc['hypothesaes', 'balanced_accuracy']:.3f}** for "
            "HypotheSAEs, and "
            f"**{logistic_hard.loc['binarysentemb', 'balanced_accuracy']:.3f}** for "
            "BinarySentEmb. The best non-binary sparse result is "
            f"{best_continuous['run']} / {best_continuous['classifier']} at "
            f"{best_continuous['balanced_accuracy']:.3f}; the strongest raw-Qwen "
            f"result is {best_raw['balanced_accuracy']:.3f}. This distinction is central: "
            "a useful continuous bottleneck does not guarantee that its thresholded support "
            "retains the same information."
        ),
        "",
        "This benchmark compares representation algorithms, not backbones: Qwen is held fixed. "
        "HypotheSAEs is run with M=16 and K=6. BinarySentEmb uses 16 thresholded bits but "
        "does not impose a fixed number of active bits, so its empirical activity is reported rather "
        "than assumed.",
        "",
        "## Reproducibility and scope",
        "",
        "The SAE fitting is unsupervised on all documents. Labels enter only the already-frozen "
        "80/20 predictive evaluation. No embeddings are recomputed. Every downstream representation, "
        "including raw Qwen and the reused proposed SAE representations, is refit with the same "
        "classifier implementation and preprocessing. The external model classes "
        "are loaded from clean, commit-pinned official repositories; the BinarySentEmb adapter "
        "changes only its hard-coded InferSent input dimension and CUDA destination so that the "
        "official class can consume Qwen vectors on Apple MPS.",
        "BinarySentEmb uses the paper's Section 4.2 settings (learning rate 1e-5 and "
        "validation-selected lambda_sp=0.8), both accepted by the official training interface. "
        "The released train.py has different no-argument defaults; the exact choice is recorded "
        "in metadata.json.",
        "The original BinarySentEmb experiment learns from InferSent encodings of paired NLI "
        "sentences. AG News has no premise-hypothesis pairing, so this transfer benchmark "
        "feeds the official loss seeded random document-pair comparisons formed from the fixed "
        "Qwen vectors. Consequently, its result measures transfer of the published compressor "
        "to this corpus and backbone, not a reproduction of the paper's InferSent/SNLI result.",
        "",
        f"- HypotheSAEs commit: `{metadata['upstream']['hypothesaes']['commit']}`",
        f"- BinarySentEmb commit: `{metadata['upstream']['binarysentemb']['commit']}`",
        f"- Qwen raw-embedding SHA-256: `{metadata['raw_embeddings_sha256']}`",
        f"- Split SHA-256: `{metadata['predictive_split_sha256']}`",
        "",
        "## Predictive performance",
        "",
        "Balanced accuracy is the primary metric: higher is better and chance is 0.25 for the four "
        "balanced AG News categories. Brackets are class-stratified 95% bootstrap intervals using "
        "the same document resamples within a classifier across representations. The "
        "generalization loss is train balanced accuracy minus test balanced accuracy in percentage "
        "points; values nearer zero indicate a smaller train-test gap.",
        "",
        "![Balanced-accuracy comparison](balanced_accuracy_comparison.png)",
        "",
        "![Utility versus generalization gap](balanced_accuracy_tradeoff.png)",
        "",
        "| Method | Representation | Classifier | Test balanced accuracy [95% CI] | Macro-F1 | BA binary / raw Qwen | Train-test loss (pp) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in metrics.sort_values(
        ["classifier", "representation", "run"], kind="stable"
    ).iterrows():
        lines.append(
            f"| {row['run']} | {row['representation']} | {row['classifier']} | "
            f"{row['balanced_accuracy']:.3f} [{row['balanced_accuracy_ci_low']:.3f}, "
            f"{row['balanced_accuracy_ci_high']:.3f}] | {row['macro_f1']:.3f} | "
            f"{fmt(row.get('ba_binary_over_raw_embedding'))} | "
            f"{row['balanced_accuracy_loss_pp']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Code capacity and sparsity",
            "",
            "Profile entropy and the effective number of profiles quantify how much of the 16-bit "
            "code space is actually used; higher values mean a more diverse code, but do not by "
            "themselves imply better semantics. Mean active bits measures empirical sparsity.",
            "",
            "| Method | Mean active bits | SD | Range | Constant bits | Profiles observed | Profile entropy (bits) | Effective profiles |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in code_summaries.iterrows():
        lines.append(
            f"| {row['method']} | {row['mean_active_bits']:.3f} | "
            f"{row['sd_active_bits']:.3f} | {int(row['min_active_bits'])}-"
            f"{int(row['max_active_bits'])} | {int(row['constant_bits'])} | {int(row['observed_profiles'])} | "
            f"{row['profile_entropy_bits']:.3f} | {row['effective_profiles']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Reconstruction",
            "",
            "Normalized MSE divides decoder error by the error of predicting the global mean; lower "
            "is better. Reconstruction R2 is `1 - normalized MSE`, so higher is better and negative "
            "values are worse than the mean baseline. Mean cosine similarity is higher-is-better. "
            "These are in-sample diagnostics because the representations are fitted on all documents.",
            "",
            "The continuous representation used by the classifier is: proposed SAE = bit-probability "
            "log-odds; HypotheSAEs = nonnegative sparse top-K activations; BinarySentEmb = pre-threshold "
            "encoder logits. The following *learned-decoder* rows state separately which values actually "
            "entered each decoder; they are not all comparable to the classifier-side continuous values.",
            "",
            "| Method | Representation entering learned decoder | Normalized MSE | Reconstruction R2 | Mean cosine similarity |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in reconstructions.iterrows():
        lines.append(
            f"| {row['method']} | {row['decoder_input']} | {row['normalized_mse']:.3f} | "
            f"{row['reconstruction_r2']:.3f} | {row['mean_cosine_similarity']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Fixed-hard-code affine OLS decoder",
            "",
            "To compare reconstructive information in the binary codes independently of each method's "
            "trained decoder, the following affine OLS decoder is refitted on the complete 127,600-document "
            "unsupervised fitting corpus with each hard code held fixed. This is in-sample and is therefore "
            "a code-content diagnostic, not a generalization score. Higher R² and lower normalized MSE are better.",
            "",
            "| Method | OLS code rank | OLS normalized MSE | OLS reconstruction R² | Condition number |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in hard_ols.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['code_rank'])} | "
            f"{row['posthoc_linear_decoder_normalized_mse']:.3f} | "
            f"{row['posthoc_linear_decoder_r2']:.3f} | {fmt(row['code_condition_number'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "![Training losses](training_diagnostics.png)",
            "",
            "The BinarySentEmb panel separates its raw reconstruction term and the weighted "
            "semantic-preserving contribution. The plot makes visible whether the published "
            "lambda_sp setting materially affects the total objective and whether either term "
            "collapses during fitting.",
            "",
            "![Row-normalized confusion matrices](confusion_matrices.png)",
            "",
            "Each confusion matrix conditions on the true class by row. A strong model concentrates "
            "mass on the diagonal; off-diagonal cells identify which news categories become "
            "indistinguishable after compression.",
            "",
            "## Artifacts",
            "",
            "- `predictive_metrics.csv`: all downstream metrics and uncertainty intervals.",
            "- `predictive_predictions.csv`: document-level held-out predictions.",
            "- `feature_importance.csv`: all coefficients/importances for newly fitted probes.",
            "- `code_summary.csv`: empirical sparsity and profile-capacity statistics.",
            "- `reconstruction_summary.csv`: original learned-decoder diagnostics, with decoder input stated per row.",
            "- `hard_code_affine_ols_reconstruction.csv`: common post-hoc affine OLS reconstruction of fixed hard codes.",
            "- `input_validation.json`: row, dimension, uniqueness, ID-alignment, split-count, and input-hash checks.",
            "- `ours/`, `hypothesaes/`, and `binarysentemb/`: aligned codes; external-method directories also contain checkpoints, training histories, and reconstruction metrics.",
            "- `labels.csv`, `predictive_split.csv`, and `factor_collection_comparison.json`: self-contained matching and evaluation inputs.",
            "- `metadata.json`: exact inputs, hashes, upstream commits, hyperparameters, and device.",
            "- `reproduce.sh`: benchmark and directional-matching commands; every top-level PNG figure also has a vector PDF counterpart.",
        ]
    )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_bits <= 0 or not 0 < args.target_active <= args.n_bits:
        raise ValueError("Require 0 < target_active <= n_bits")
    device = require_mps()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hypothesaes = verify_checkout(
        args.hypothesaes_repo,
        EXPECTED_HYPOTHESAES_COMMIT,
        "hypothesaes/sae.py",
    )
    binarysentemb = verify_checkout(
        args.binarysentemb_repo,
        EXPECTED_BINARYSENTEMB_COMMIT,
        "discrete_encoders.py",
    )
    binarysentemb_train_source = args.binarysentemb_repo.resolve() / "train.py"
    if not binarysentemb_train_source.is_file():
        raise FileNotFoundError(
            f"Missing upstream training source: {binarysentemb_train_source}"
        )
    binarysentemb["training_source"] = str(binarysentemb_train_source)
    binarysentemb["training_source_sha256"] = sha256_file(
        binarysentemb_train_source
    )
    data = load_aligned_inputs(args)
    ours_config = args.our_encoded.parent / "best" / "run_config.json"
    if not ours_config.is_file():
        raise FileNotFoundError(f"Missing proposed-SAE provenance config: {ours_config}")
    ours_dir = output_dir / "ours"
    ours_dir.mkdir(parents=True, exist_ok=True)
    save_codes(
        ours_dir / "codes.csv",
        data["ids"],
        data["our_hard"],
        data["our_soft"],
    )
    pd.DataFrame({"id": data["ids"], "label_name": data["y"]}).to_csv(
        output_dir / "labels.csv", index=False
    )
    shutil.copyfile(args.predictive_split, output_dir / "predictive_split.csv")

    hs_hard, hs_latent, hs_reconstruction, hs_history, hs_diagnostics = train_hypothesaes(
        data["raw"],
        data["ids"],
        hypothesaes,
        output_dir / "hypothesaes",
        n_bits=args.n_bits,
        target_active=args.target_active,
        epochs=args.hypothesaes_epochs,
        seed=args.seed,
        device=device,
        force=args.force,
    )
    bs_hard, bs_latent, bs_reconstruction, bs_history, bs_diagnostics = train_binarysentemb(
        data["raw"],
        data["ids"],
        binarysentemb,
        output_dir / "binarysentemb",
        n_bits=args.n_bits,
        epochs=args.binarysentemb_epochs,
        learning_rate=args.binarysentemb_learning_rate,
        struct_coef=args.binarysentemb_struct_coef,
        device=device,
        force=args.force,
    )

    new_metrics, new_predictions, feature_importance = evaluate_new_representations(
        {
            ("qwen_baseline", "raw_embeddings"): np.asarray(data["raw"], dtype=np.float32),
            ("ours", "hard_bits"): data["our_hard"],
            ("ours", "continuous_latent"): data["our_soft"],
            ("hypothesaes", "hard_bits"): hs_hard,
            ("hypothesaes", "continuous_latent"): hs_latent,
            ("binarysentemb", "hard_bits"): bs_hard,
            ("binarysentemb", "continuous_latent"): bs_latent,
        },
        data,
        args.seed,
    )
    metrics, bootstrap_protocol = add_shared_balanced_accuracy_intervals(
        new_metrics,
        new_predictions,
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    metrics = add_binary_to_raw_ratios(metrics)
    predictions = new_predictions
    metrics.to_csv(output_dir / "predictive_metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictive_predictions.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    code_summaries = pd.DataFrame(
        [
            {"method": "ours", **code_summary(data["our_hard"])},
            {"method": "hypothesaes", **code_summary(hs_hard)},
            {"method": "binarysentemb", **code_summary(bs_hard)},
        ]
    )
    code_summaries.to_csv(output_dir / "code_summary.csv", index=False)

    our_information = pd.read_csv(args.existing_information)
    our_row = our_information[our_information["run"] == "qwen"].iloc[0]
    reconstructions = pd.DataFrame(
        [
            {
                "method": "ours",
                "decoder_input": "hard bits (saved SAE decoder)",
                "normalized_mse": float(our_row["hard_normalized_mse"]),
                "reconstruction_r2": float(our_row["hard_reconstruction_r2"]),
                "mean_cosine_similarity": float(our_row["hard_mean_cosine_similarity"]),
            },
            {
                "method": "ours",
                "decoder_input": "soft probabilities (saved SAE decoder)",
                "normalized_mse": float(our_row["soft_normalized_mse"]),
                "reconstruction_r2": float(our_row["soft_reconstruction_r2"]),
                "mean_cosine_similarity": float(our_row["soft_mean_cosine_similarity"]),
            },
            *[
                {"method": "hypothesaes", "decoder_input": description, **values}
                for description, values in hs_reconstruction.items()
            ],
            *[
                {"method": "binarysentemb", "decoder_input": description, **values}
                for description, values in bs_reconstruction.items()
            ],
        ]
    )
    reconstructions.to_csv(output_dir / "reconstruction_summary.csv", index=False)
    ols = posthoc_linear_decoder_metrics_many(
        data["raw"],
        {"ours": data["our_hard"], "hypothesaes": hs_hard, "binarysentemb": bs_hard},
    )
    ols_rows = pd.DataFrame(
        [
            {
                "method": method,
                "decoder_input": "fixed hard bits",
                "decoder": "post-hoc affine OLS",
                **values,
            }
            for method, values in ols.items()
        ]
    )
    ols_rows.to_csv(output_dir / "hard_code_affine_ols_reconstruction.csv", index=False)

    plot_balanced_accuracy(metrics, output_dir / "balanced_accuracy_comparison.png")
    plot_tradeoff(metrics, output_dir / "balanced_accuracy_tradeoff.png")
    plot_training(hs_history, bs_history, output_dir / "training_diagnostics.png")
    plot_confusions(predictions, output_dir / "confusion_matrices.png")

    metadata = {
        "schema_version": 2,
        "dataset": "agnews",
        "backbone": "Qwen/Qwen3-Embedding-0.6B",
        "n_documents": int(len(data["ids"])),
        "embedding_dim": int(data["raw"].shape[1]),
        "n_bits": int(args.n_bits),
        "target_active": int(args.target_active),
        "seed": int(args.seed),
        "device": str(device),
        "raw_embeddings": str(args.raw_embeddings.resolve()),
        "raw_embeddings_sha256": sha256_file(args.raw_embeddings),
        "raw_ids_sha256": sha256_file(args.raw_ids),
        "predictive_split": str(args.predictive_split.resolve()),
        "predictive_split_sha256": sha256_file(args.predictive_split),
        "labels_sha256": sha256_file(args.labels),
        "unsupervised_training_scope": "all 127,600 aligned AG News documents",
        "predictive_evaluation": "existing seeded stratified 80/20 ID assignment",
        "upstream": {
            "hypothesaes": hypothesaes,
            "binarysentemb": binarysentemb,
        },
        "hypothesaes": {
            "M": int(args.n_bits),
            "K": int(args.target_active),
            "epochs": int(args.hypothesaes_epochs),
            "batch_size": 512,
            "learning_rate": 5e-4,
            "aux_coef": 1 / 32,
            "multi_coef": 0.0,
        },
        "binarysentemb": {
            "architecture": "official LinearAutoEncoder with Qwen boundary dimensions",
            "epochs": int(args.binarysentemb_epochs),
            "upstream_seed": 2019,
            "effective_document_batch_size": 128,
            "learning_rate": float(args.binarysentemb_learning_rate),
            "weight_decay": 1e-5,
            "semantic_preserving_coefficient": float(
                args.binarysentemb_struct_coef
            ),
            "hyperparameter_source": (
                "paper Section 4.2 (lr=1e-5; validation-selected lambda_sp=0.8); "
                "official train.py exposes these as CLI parameters but has different defaults"
            ),
            "mps_adapter": "tensor .cuda() redirected to .to(mps) in isolated process",
        },
        "input_validation": {
            **data["validation"],
            "raw_embeddings_sha256": sha256_file(args.raw_embeddings),
            "raw_ids_sha256": sha256_file(args.raw_ids),
            "our_encoded_sha256": sha256_file(args.our_encoded),
            "labels_sha256": sha256_file(args.labels),
            "predictive_split_sha256": sha256_file(args.predictive_split),
        },
        "ours_reused": {
            "encoded_codes": str(args.our_encoded.resolve()),
            "encoded_codes_sha256": sha256_file(args.our_encoded),
            "checkpoint_config": str(ours_config.resolve()),
            "checkpoint_config_sha256": sha256_file(ours_config),
            "training_scope": "full 127,600-document AG News corpus; reused without retraining",
        },
        "predictive_protocol": {
            "all_representations_refit": True,
            "preprocessing": "The same predictive_modeling._fit_one classifier implementation is applied directly to every representation; no classifier-side L2 normalization or feature scaling is added.",
            "bootstrap": bootstrap_protocol,
            "representations": {
                "ours_continuous_latent": "log-odds of the proposed SAE's stored bit probabilities",
                "hypothesaes_continuous_latent": "official sparse nonnegative top-K activations",
                "binarysentemb_continuous_latent": "official encoder logits before straight-through thresholding",
            },
        },
        "training_diagnostics": {
            "hypothesaes": hs_diagnostics,
            "binarysentemb": bs_diagnostics,
        },
        "hard_code_ols_decoder": {
            "scope": "in-sample, complete unsupervised fitting corpus",
            "protocol": "affine ordinary least-squares decoder refitted with hard codes fixed; not an original learned decoder",
            "artifact": "hard_code_affine_ols_reconstruction.csv",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "input_validation.json").write_text(
        json.dumps(metadata["input_validation"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "reproduce.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "source .venv/bin/activate\n"
        "MPLCONFIGDIR=/tmp/latent-mpl python benchmark_agnews_external_methods.py "
        f"--output_dir {output_dir} --bootstrap_draws {args.bootstrap_draws}\n"
        "MPLCONFIGDIR=/tmp/latent-mpl python compare_external_sparse_profiles.py "
        f"--run_dir {output_dir} --max_profiles 10 --null_draws 1000 --min_docs 30\n",
        encoding="utf-8",
    )
    matching_manifest = {
        "schema_version": 1,
        "label_column": "label_name",
        "run_order": ["ours", "hypothesaes", "binarysentemb"],
        "labels": {"path": str((output_dir / "labels.csv").resolve())},
        "runs": {
            method: {
                "encoded": {
                    "path": str((output_dir / method / "codes.csv").resolve())
                }
            }
            for method in ("ours", "hypothesaes", "binarysentemb")
        },
    }
    (output_dir / "factor_collection_comparison.json").write_text(
        json.dumps(matching_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir, metrics, code_summaries, reconstructions, ols_rows, metadata)
    print(f"External method comparison complete: {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
