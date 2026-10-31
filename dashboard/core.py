"""Data and projection operations. No embedding files or model checkpoints are read."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

BIT = re.compile(r"^bit_(\d+)$")
MISSING = {"", "unknown", "nan", "none", "null", "n/a", "na", "not applicable"}
MDS_LIMIT = 2500


def fingerprint(path: str | Path) -> tuple[str, int, int]:
    p = Path(path).expanduser().resolve()
    stat = p.stat()
    return str(p), stat.st_mtime_ns, stat.st_size


def discover_runs(root: str) -> list[Path]:
    """Find encoded tables, never traverse or open embedding caches."""
    p = Path(root).expanduser().resolve()
    if p.is_file():
        return [p] if p.suffix == ".csv" and p.name.startswith("encoded_") else []
    if not p.is_dir():
        raise ValueError("Choose an existing runs directory or encoded CSV file.")
    return sorted(f for f in p.rglob("encoded_*.csv")
                  if not any(part.startswith(".") for part in f.relative_to(p).parts))


def read_metadata(encoded: Path) -> dict:
    p = encoded.parent / "metadata.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
        # Whitelist fields: embedding-cache manifests are deliberately ignored.
        return {k: data[k] for k in ("input", "model_name", "k_factors", "target_active",
                                   "checkpoint", "seed") if k in data}
    except (ValueError, OSError):
        return {}


def annotation_candidates(encoded: str) -> list[str]:
    p = Path(encoded).resolve()
    meta = read_metadata(p)
    candidates: list[Path] = []
    source = Path(meta.get("input", "__absent__"))
    dirs = [p.parent]
    if source.is_file():
        dirs.insert(0, source.parent)
        candidates.extend([source.with_name(source.name.replace("_ready", "")), source])
    # Relocated repositories: recover data/<dataset>/ from a stale source path.
    if "data" in source.parts:
        tail = source.parts[source.parts.index("data"):]
        for ancestor in p.parents:
            relocated = ancestor.joinpath(*tail)
            if relocated.is_file():
                dirs.insert(0, relocated.parent)
                candidates.insert(0, relocated.with_name(relocated.name.replace("_ready", "")))
                break
    for d in dirs:
        candidates = sorted(d.glob("*labels*.csv")) + candidates
        candidates.extend(sorted(d.glob("*_all.csv")))
    found = []
    for f in candidates:
        f = f.resolve()
        if not f.is_file() or f == p or str(f) in found:
            continue
        columns = pd.read_csv(f, nrows=0).columns
        if "id" in columns and any(c not in {"id", "text"} and not BIT.match(c)
                                   and not c.startswith("prob_") for c in columns):
            found.append(str(f))
    return found


@dataclass
class Dataset:
    path: str
    frame: pd.DataFrame
    bit_columns: list[str]
    bits: np.ndarray
    labels: list[str]
    metadata: dict
    notes: list[str]
    signature: str


def load_dataset(encoded: str, annotations: str = "") -> Dataset:
    return _load(fingerprint(encoded), fingerprint(annotations) if annotations else None)


@lru_cache(maxsize=3)
def _load(encoded_key: tuple, label_key: tuple | None) -> Dataset:
    path = Path(encoded_key[0])
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    bits = sorted((c for c in cols if BIT.match(c)), key=lambda x: int(BIT.match(x)[1]))
    if not bits or "id" not in cols:
        raise ValueError("The encoded CSV must contain id and bit_00, bit_01, … columns.")
    if len({int(BIT.match(c)[1]) for c in bits}) != len(bits):
        raise ValueError("Ambiguous duplicate bit numbers.")
    # Read no probability coordinates; preserve IDs and text literally.
    use = [c for c in cols if not c.startswith("prob_")]
    frame = pd.read_csv(path, usecols=use, dtype=str, keep_default_na=False)
    if frame.empty or frame.id.eq("").any() or frame.id.duplicated().any():
        raise ValueError("Document IDs must be nonempty and unique; use a composite ID if needed.")
    numeric = frame[bits].apply(pd.to_numeric, errors="coerce")
    if not np.isin(numeric.to_numpy(), [0, 1]).all():
        raise ValueError("Every bit must be exactly 0 or 1; probabilities are not binary codes.")
    frame[bits] = numeric.astype(np.uint8)
    notes = []
    if label_key:
        lp = Path(label_key[0])
        if lp.suffix == ".parquet":
            extra = pd.read_parquet(lp).fillna("")
            for c in extra:
                extra[c] = extra[c].map(lambda v: json.dumps(v.tolist() if isinstance(v, np.ndarray) else v)
                                       if isinstance(v, (list, np.ndarray)) else str(v))
        elif lp.suffix == ".csv":
            extra = pd.read_csv(lp, dtype=str, keep_default_na=False,
                                usecols=lambda c: not c.startswith("prob_") and not BIT.match(c))
        else:
            raise ValueError("Annotations must be a CSV or Parquet table.")
        if "id" not in extra or extra.id.eq("").any() or extra.id.duplicated().any():
            raise ValueError("The annotation table must have unique, nonempty id values.")
        missing = (~frame.id.isin(extra.id)).sum()
        notes.append(f"Annotation join: {len(frame) - missing:,}/{len(frame):,} document IDs matched.")
        if missing == len(frame):
            raise ValueError("No annotation IDs match the encoded documents. Check the selected source.")
        new = [c for c in extra if c not in frame.columns and not BIT.match(c)
               and not c.startswith("prob_")]
        frame = frame.merge(extra[["id", *new]], on="id", how="left", validate="one_to_one").fillna("")
    if "text" not in frame:
        frame["text"] = ""
        notes.append("No text column found. Choose a source table with id and text to read documents.")
    frame = frame.sort_values("id", kind="stable").reset_index(drop=True)
    excluded = {"id", "text", "n_active", "source_id", "source_partition", "split", "index"}
    labels = [c for c in frame if c not in excluded and c not in bits and not c.startswith("prob_")
              and not c.endswith("_id")]
    signature = hashlib.sha256(repr((encoded_key, label_key)).encode()).hexdigest()[:16]
    return Dataset(str(path), frame, bits, frame[bits].to_numpy(dtype=np.uint8), labels,
                   read_metadata(path), notes, signature)


def select_documents(data: Dataset, selected: list[str], size: int, seed: int, match_mode="unrestricted"):
    if not set(selected).issubset(data.bit_columns):
        raise ValueError("The bit selection does not belong to this run.")
    if size < 1 or not 0 <= seed < 2**32:
        raise ValueError("Choose a positive sample size and a seed between 0 and 4294967295.")
    columns = [data.bit_columns.index(c) for c in sorted(set(selected))]
    if match_mode == "exact":
        target = np.zeros(data.bits.shape[1], dtype=np.uint8)
        target[columns] = 1
        mask = (data.bits == target).all(axis=1)
    elif match_mode == "unrestricted":
        mask = data.bits[:, columns].all(axis=1) if columns else np.ones(len(data.bits), dtype=bool)
    else:
        raise ValueError("Choose unrestricted or exact-profile matching.")
    eligible = np.flatnonzero(mask)
    sample = np.sort(np.random.default_rng(seed).choice(eligible, min(size, len(eligible)), replace=False))
    return eligible, sample


def profile_keys(bits: np.ndarray) -> list[str]:
    return ["".join(map(str, row)) for row in bits]


def group_profiles(bits: np.ndarray):
    return np.unique(bits, axis=0, return_inverse=True, return_counts=True)


def project(codes: np.ndarray, method: str, metric: str, dimensions: int,
            seed: int, neighbors: int = 15, min_dist: float = .1):
    profiles, _, counts = group_profiles(codes)
    coordinates, info = _project_cached(profiles.tobytes(), profiles.shape, tuple(counts.tolist())
                                        if method == "pca" else (), method, metric,
                                        dimensions, seed, neighbors, min_dist)
    return profiles, coordinates, dict(info)


@lru_cache(maxsize=12)
def _project_cached(raw: bytes, shape: tuple, counts: tuple, method: str, metric: str,
                    dimensions: int, seed: int, neighbors: int, min_dist: float):
    x = np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(float)
    n, k = x.shape
    info = {"method": method, "metric": "Euclidean on unscaled bits" if method == "pca" else metric,
            "unique_profiles_fitted": n, "weighting": "document frequency" if method == "pca"
            else "equal weight per unique profile"}
    if n < 2:
        info["note"] = "Only one complete profile: no spatial variation."
        return np.zeros((n, dimensions)), info
    if method == "pca":
        weights = np.asarray(counts, dtype=float)
        centered = x - np.average(x, axis=0, weights=weights)
        covariance = (centered.T * weights) @ centered / weights.sum()
        vals, vecs = eigh(covariance)
        order = np.argsort(vals)[::-1][:dimensions]
        axes = vecs[:, order]
        # Deterministic axis signs.
        axes *= np.where(axes[np.argmax(np.abs(axes), axis=0), np.arange(axes.shape[1])] < 0, -1, 1)
        result = centered @ axes
        total = np.maximum(vals, 0).sum()
        info["variance_explained"] = (np.maximum(vals[order], 0) / total).tolist() if total else []
    elif method == "mds" or (method == "umap" and n < 4):
        if n > MDS_LIMIT:
            raise ValueError(f"Classical MDS is limited to {MDS_LIMIT:,} unique profiles. "
                             "Choose Selected sample, reduce the sample, or use PCA/UMAP globally.")
        if metric not in {"hamming", "jaccard"}:
            raise ValueError("Choose Hamming or Jaccard distance.")
        distances = squareform(pdist(x.astype(bool), metric=metric))
        distances = np.nan_to_num(distances, nan=0.0)
        d2 = distances**2
        gram = -.5 * (d2 - d2.mean(axis=0)[None, :] - d2.mean(axis=1)[:, None] + d2.mean())
        vals, vecs = eigh(gram, overwrite_a=True)
        positive = np.flatnonzero(vals > 1e-10)[::-1][:dimensions]
        result = vecs[:, positive] * np.sqrt(vals[positive])
        mass = np.abs(vals).sum()
        info["negative_eigenvalue_mass_fraction"] = float(np.abs(vals[vals < -1e-10]).sum() / mass) if mass else 0
        if method == "umap":
            info["note"] = "Fewer than four profiles: classical MDS used instead of UMAP."
    elif method == "umap":
        from umap import UMAP
        effective_neighbors = max(2, min(int(neighbors), n - 1))
        result = UMAP(n_components=dimensions, metric=metric, n_neighbors=effective_neighbors,
                      min_dist=float(min_dist), random_state=seed, n_jobs=1,
                      init="random", disconnection_distance=np.inf).fit_transform(x)
        info["effective_neighbors"] = effective_neighbors
        info["min_dist"] = min_dist
    else:
        raise ValueError("Unknown projection method.")
    result = np.pad(result, ((0, 0), (0, max(0, dimensions - result.shape[1]))))
    if not np.isfinite(result).all():
        raise ValueError("Projection produced nonfinite coordinates. Try PCA or another distance.")
    return result, info


def _tokens(value, multilabel: bool) -> list[str]:
    text = str(value).strip()
    if text.casefold() in MISSING:
        return []
    if not multilabel:
        return [text]
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        parsed = None
    raw = parsed if isinstance(parsed, list) else re.split(r"[|;]", text)
    return sorted({str(v).strip() for v in raw if str(v).strip().casefold() not in MISSING})


def label_distribution(data: Dataset, indices: np.ndarray, column: str, multilabel=False):
    all_tokens = data.frame[column].map(lambda v: _tokens(v, multilabel))
    selected = all_tokens.iloc[indices]
    valid = int(selected.map(bool).sum())
    total_valid = int(all_tokens.map(bool).sum())
    selected_counts = selected.explode().dropna().value_counts()
    total_counts = all_tokens.explode().dropna().value_counts()
    rows = [{"category": str(cat), "count": int(count), "percent": float(100 * count / valid),
             "corpus_percent": float(100 * total_counts.get(cat, 0) / total_valid) if total_valid else 0.0}
            for cat, count in selected_counts.items()] if valid else []
    return {"field": column, "valid": valid, "missing": int(len(indices) - valid),
            "corpus_valid": total_valid, "multilabel": multilabel, "categories": rows}


def profile_label_summary(data: Dataset, members: list[list[int]], column: str, multilabel=False):
    """Describe sampled members of each complete profile; never change coordinates."""
    tokens = data.frame[column].map(lambda value: _tokens(value, multilabel))
    categories = sorted({label for labels in tokens for label in labels})
    summaries = []
    for indices in members:
        selected = tokens.iloc[indices]
        counts = Counter(label for labels in selected for label in labels)
        valid = int(selected.map(bool).sum())
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        winners = [label for label, count in ordered if count == ordered[0][1]] if ordered else []
        summaries.append({"kind": "missing" if not winners else "tie" if len(winners) > 1 else "label",
                          "label": winners[0] if len(winners) == 1 else None,
                          "valid": valid, "missing": len(indices) - valid,
                          "counts": dict(ordered), "multilabel": multilabel})
    return summaries, categories


def profile_outcome_summary(data: Dataset, members: list[list[int]], column: str, colour_scale="linear"):
    """Continuous colour overlay: sampled profile medians, fixed corpus scale."""
    if column not in data.labels:
        raise ValueError("Choose an outcome available in this dataset.")
    if colour_scale not in {"linear", "log"}:
        raise ValueError("Choose a linear or logarithmic outcome colour scale.")
    values = pd.to_numeric(data.frame[column], errors="coerce").to_numpy(dtype=float)
    values[~np.isfinite(values)] = np.nan
    observed = values[np.isfinite(values)]
    summaries = []
    for indices in members:
        selected = values[indices]
        valid = selected[np.isfinite(selected)]
        summaries.append(dict(valid=len(valid), missing=len(indices) - len(valid),
                              median=float(np.median(valid)) if len(valid) else None,
                              mean=float(np.mean(valid)) if len(valid) else None))
    low = float(observed.min()) if len(observed) else None
    high = float(observed.max()) if len(observed) else None
    divergent = low is not None and low < 0 < high
    cmin, cmax = low, high
    if divergent:
        cmin, cmax = -max(abs(low), abs(high)), max(abs(low), abs(high))
    elif low is not None and low == high:
        # A nonzero display interval for constant outcomes; never imply a sign change.
        pad = max(abs(low) * .05, 1e-6)
        cmin, cmax = (max(0., low - pad), high + pad) if low >= 0 else (low - pad, min(0., high + pad))
    # Aggregate on the original scale, then transform only the displayed colour.
    def transform(value):
        if value is None:
            return None
        return float(np.sign(value) * np.log1p(abs(value))) if colour_scale == "log" else value

    cmin, cmax = transform(cmin), transform(cmax)
    ticks = np.linspace(cmin, cmax, 5) if cmin is not None and colour_scale == "log" else None
    return dict(field=column, kind="continuous", aggregation="median", profiles=summaries,
                color_values=[transform(s["median"]) for s in summaries],
                corpus_valid=len(observed), corpus_missing=len(values) - len(observed),
                corpus_min=low, corpus_max=high,
                scale=dict(type="diverging" if divergent else "sequential", cmin=cmin, cmax=cmax,
                           midpoint=0. if divergent else None,
                           transform="sign(x) * log1p(abs(x))" if colour_scale == "log" else "identity",
                           mode=colour_scale,
                           tickvals=ticks.tolist() if ticks is not None else None,
                           ticktext=[f"{np.sign(v) * np.expm1(abs(v)):,.6g}" for v in ticks] if ticks is not None else None,
                           colorscale="RdBu_r" if divergent else "Viridis"))


def label_count_summary(data: Dataset, eligible: np.ndarray, sample: np.ndarray, column: str, multilabel=False):
    """Document counts, independent of profile aggregation and dominant-label colouring."""
    tokens = data.frame[column].map(lambda value: _tokens(value, multilabel))
    populations = {"corpus": tokens, "matching": tokens.iloc[eligible], "sample": tokens.iloc[sample]}
    counts = {name: Counter(label for labels in rows for label in labels)
              for name, rows in populations.items()}
    observed = {name: int(rows.map(bool).sum()) for name, rows in populations.items()}
    totals = {name: len(rows) for name, rows in populations.items()}
    categories = sorted(counts["corpus"], key=lambda label: (-counts["matching"][label], label))
    rows = [{"category": label, "matching": counts["matching"][label], "sample": counts["sample"][label],
             "corpus": counts["corpus"][label],
             "matching_percent": 100 * counts["matching"][label] / observed["matching"]
             if observed["matching"] else None} for label in categories]
    return {"field": column, "multilabel": bool(multilabel), "categories": rows, "totals": totals,
            "observed": observed, "missing": {name: totals[name] - observed[name] for name in totals}}


def clear_caches():
    _load.cache_clear()
    _project_cached.cache_clear()
