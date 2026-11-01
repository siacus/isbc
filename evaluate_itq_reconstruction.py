"""Evaluate inverse-transform and refitted affine reconstruction of fixed ITQ codes.

Run after recovering the fitted PCA/ITQ transformation. No embeddings or codes
are refitted. The inverse transform uses the signed alphabet {-1,+1}, including
for top-k readouts, and introduces no fitted scale or intercept correction.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/agnews_qwen_all_fixed_balance'
OUT = ROOT / 'runs/agnews_embedding_comparison/itq_activity_interpretation'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def evaluate(raw, row_indices, codes, rotation, components, mean, chunk_size):
    n, width = codes.shape
    # Each row of components is a PCA direction. For scores V=(X-mu)P^T,
    # the ITQ objective is ||VR-S||_F^2 with S=2B-1.
    inverse = rotation.T @ components
    gram = np.zeros((width + 1, width + 1))
    cross = np.zeros((width + 1, raw.shape[1]))
    total = np.zeros(raw.shape[1])
    inverse_sse = 0.0
    pca_sse = 0.0
    quantization_sse = 0.0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        x = np.asarray(raw[row_indices[start:stop]], dtype=np.float64)
        b = codes[start:stop].astype(np.float64)
        signed = 2 * b - 1
        design = np.column_stack([np.ones(len(b)), b])
        gram += design.T @ design
        cross += design.T @ x
        total += x.sum(axis=0)
        prediction = mean + signed @ inverse
        inverse_sse += float(np.square(x - prediction).sum())
        projected = (x - mean) @ components.T
        pca_sse += float(np.square((x - mean) - projected @ components).sum())
        quantization_sse += float(np.square(projected @ rotation - signed).sum())
    # The intercept makes the affine class identical for 0/1 and signed codes.
    # A pseudoinverse also handles the intercept/activity dependence of top-k.
    coefficients = np.linalg.lstsq(gram, cross, rcond=None)[0]
    corpus_mean = total / n
    baseline_sse = 0.0
    ols_sse = 0.0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        x = np.asarray(raw[row_indices[start:stop]], dtype=np.float64)
        b = codes[start:stop].astype(np.float64)
        design = np.column_stack([np.ones(len(b)), b])
        baseline_sse += float(np.square(x - corpus_mean).sum())
        ols_sse += float(np.square(x - design @ coefficients).sum())
    if baseline_sse <= 0:
        raise ValueError('The mean-baseline reconstruction error must be positive.')
    # Independent geometric check of the inverse reconstruction calculation.
    decomposition_error = abs(inverse_sse - pca_sse - quantization_sse) / baseline_sse
    if decomposition_error > 1e-6:
        raise ValueError(f'PCA/quantization error decomposition failed: {decomposition_error}')
    if ols_sse > inverse_sse + 1e-6 * baseline_sse:
        raise ValueError('OLS fit should not be worse than the fixed affine inverse.')
    return dict(n_documents=n, n_bits=width, mean_activity=float(codes.sum(1).mean()),
                inverse_nmse=inverse_sse / baseline_sse,
                inverse_r2=1 - inverse_sse / baseline_sse,
                ols_nmse=ols_sse / baseline_sse, ols_r2=1 - ols_sse / baseline_sse,
                pca_r2=1 - pca_sse / baseline_sse,
                quantization_error_over_baseline=quantization_sse / baseline_sse,
                decomposition_error_over_baseline=decomposition_error,
                baseline_sse=baseline_sse, inverse_sse=inverse_sse, ols_sse=ols_sse)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-embeddings', type=Path, default=RUN/'raw_embeddings_agnews_all_ready_best.npy')
    ap.add_argument('--raw-ids', type=Path, default=RUN/'raw_embeddings_agnews_all_ready_best_ids.csv')
    ap.add_argument('--model', type=Path, default=OUT/'itq_scores.npz')
    ap.add_argument('--output-dir', type=Path, default=OUT/'inverse_reconstruction')
    ap.add_argument('--top-k', type=int, default=6)
    ap.add_argument('--expected-documents', type=int, default=127600)
    ap.add_argument('--chunk-size', type=int, default=4096)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    if args.chunk_size < 1 or args.threads < 1:
        ap.error('chunk-size and threads must be positive')
    with np.load(args.model, allow_pickle=False) as saved:
        ids = saved['ids'].astype(str)
        rotation = saved['rotation'].astype(np.float64)
        components = saved['pca_components'].astype(np.float64)
        mean = saved['pca_mean'].astype(np.float64)
        scores = saved['scores'].astype(np.float64)
        native = saved['native_codes'].astype(np.uint8)
        if 'cached_codes' in saved and not np.array_equal(native, saved['cached_codes']):
            raise ValueError('Recovered ITQ codes differ from the original reference codes.')
    raw = np.load(args.raw_embeddings, mmap_mode='r', allow_pickle=False)
    raw_ids = pd.Index(pd.read_csv(args.raw_ids, dtype={'id':str})['id'])
    if not raw_ids.is_unique or not pd.Index(ids).is_unique:
        raise ValueError('Document IDs must be unique.')
    ix = raw_ids.get_indexer(ids)
    if (ix < 0).any() or len(ids) != args.expected_documents or len(raw_ids) != len(raw):
        raise ValueError('Input IDs or corpus size do not match.')
    k = native.shape[1]
    if not (0 < args.top_k <= k):
        raise ValueError('top-k must be between 1 and code width.')
    assert scores.shape == native.shape == (len(ids), k)
    assert components.shape == (k, raw.shape[1]) and rotation.shape == (k, k)
    assert np.isfinite(scores).all() and np.isfinite(mean).all()
    assert np.array_equal(native, (scores >= 0).astype(np.uint8))
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(k), atol=1e-10)
    np.testing.assert_allclose(components @ components.T, np.eye(k), atol=5e-6)
    top = np.zeros_like(native)
    order = np.argsort(-scores, axis=1, kind='stable')
    np.put_along_axis(top, order[:, :args.top_k], 1, axis=1)
    rows = []
    max_projection_error = 0.0
    with threadpool_limits(limits=args.threads):
        for start in range(0, len(ids), args.chunk_size):
            stop = min(start + args.chunk_size, len(ids))
            x = np.asarray(raw[ix[start:stop]], dtype=np.float64)
            # Match the float32 score cache used during rotation estimation.
            check = ((x - mean) @ components.T).astype(np.float32).astype(np.float64) @ rotation
            max_projection_error = max(max_projection_error, float(np.max(np.abs(check - scores[start:stop]))))
        if max_projection_error > 1e-5:
            raise ValueError(f'Raw embeddings do not reproduce saved scores: {max_projection_error}')
        for name, code in [('ITQ', native), (f'ITQ_top{args.top_k}', top)]:
            row = dict(method=name, **evaluate(raw, ix, code, rotation, components, mean, args.chunk_size))
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    protocol = dict(n_documents=len(ids), n_bits=k, top_k=args.top_k,
                    inverse_transform='X_hat = mu + (2 B - 1) R.T P; V=(X-mu)P.T; P rows are PCA directions.',
                    decoder='Fixed inverse of the fitted rotation and PCA projection; signed unit amplitudes, no fitted rescaling or intercept correction.',
                    top_k_rule='Use the same fixed inverse with exactly the top-k signed rotated scores active; stable sorting resolves ties by coordinate index.',
                    fitting='No new encoder or PCA/rotation fit. OLS refits an affine decoder to each fixed code on all documents.',
                    normalization='NMSE=SSE / sum_i ||X_i - corpus_mean||^2; R2=1-NMSE. Raw unnormalized embeddings.',
                    max_saved_score_reproduction_error=max_projection_error,
                    pca_orthogonality_error=float(np.max(np.abs(components @ components.T-np.eye(k)))),
                    source_files={p.name:sha256(p) for p in [args.raw_embeddings,args.raw_ids,args.model]},
                    numpy_version=np.__version__, pandas_version=pd.__version__)
    (args.output_dir/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    report = ['# ITQ reconstruction on AG News', '',
              f'All {len(ids):,} embeddings; the existing {k}-bit PCA/ITQ fit is held fixed.', '',
              '| Readout | Inverse NMSE | Inverse R2 | Affine OLS R2 |',
              '|---|---:|---:|---:|']
    report += [f"| {r['method']} | {r['inverse_nmse']:.6f} | {r['inverse_r2']:.6f} | {r['ols_r2']:.6f} |" for r in rows]
    report += ['', 'The fixed inverse uses signed codes (2B-1), reverses the learned rotation,',
               'projects back through the PCA directions and restores the PCA mean.',
               'Top-6 changes only the hard code. No decoder coefficients are optimized',
               'for either inverse-transform score. OLS is the separate common affine reference.',
               'The calculations reproduce all saved rotated scores and check the decomposition',
               'of inverse-transform error into PCA residual error and quantization error.', '']
    (args.output_dir/'report.md').write_text('\n'.join(report))


if __name__ == '__main__':
    main()
