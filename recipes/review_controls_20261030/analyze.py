from pathlib import Path
import sys, json, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import benchmark_agnews_external_methods as b
from predictive_modeling import build_classifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from information_retention_extended import _chunked_projection

OUT = Path(__file__).resolve().parent
BASE = ROOT / 'runs/agnews_embedding_comparison'
args = b.parse_args()
data = b.load_aligned_inputs(args)
from reproduction_paths import qwen_reference_cache
cache = qwen_reference_cache(ROOT)
saved = np.load(cache)
cache_ids = pd.read_csv(args.predictive_split, dtype={'id':str}).id.to_numpy()
cache_to_raw = pd.Index(cache_ids).get_indexer(data['ids'])
assert (cache_to_raw >= 0).all() and len(np.unique(cache_to_raw)) == 127600
reference = pd.read_csv(BASE / 'all_fixed_balance_label_name/information_retention_quantizer_baselines.csv')
reference = reference[reference.run.eq('qwen')].set_index('method')
metrics, predictions = [], []
with threadpool_limits(limits=4):
    print('Fitting PCA on all frozen embeddings, without labels.', flush=True)
    pca = PCA(n_components=16, svd_solver='randomized', random_state=1234,
              n_oversamples=32, iterated_power=7).fit(data['raw'])
    scores = _chunked_projection(data['raw'], pca.mean_, pca.components_)
    np.savez_compressed(OUT / 'pca16.npz', scores=scores, mean=pca.mean_, components=pca.components_, explained_variance_ratio=pca.explained_variance_ratio_)
    matrices = {name: saved[name][cache_to_raw] for name in ['pca_sign', 'itq', 'random_hyperplane', 'pca_top_s', 'random_top_s']}
    matrices['pca16'] = scores
    print('PCA variance explained', pca.explained_variance_ratio_.sum(), 'PCA-sign agreement with ID-aligned cache', np.mean((scores >= 0) == matrices['pca_sign']), flush=True)
    for name, matrix in matrices.items():
        model = build_classifier('logistic_regression', 1234)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(matrix[data['train_indices']], data['y'][data['train_indices']])
        assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), name
        pred = model.predict(matrix[data['test_indices']])
        truth = data['y'][data['test_indices']]
        row = dict(run=name, representation='continuous' if name == 'pca16' else 'hard_bits', classifier='logistic_regression', balanced_accuracy=balanced_accuracy_score(truth,pred), macro_f1=f1_score(truth,pred,average='macro'), n_iter=int(model.n_iter_.max()))
        if name == 'pca16':
            row.update(mean_active_bits=np.nan, posthoc_linear_decoder_r2=float(pca.explained_variance_ratio_.sum()))
        else:
            row.update(mean_active_bits=float(matrix.sum(1).mean()), posthoc_linear_decoder_r2=float(reference.loc[name,'posthoc_linear_decoder_r2']))
            row['ba_delta_from_saved'] = float(row['balanced_accuracy'] - reference.loc[name,'balanced_accuracy'])
            assert abs(row['ba_delta_from_saved']) < .001, (name, row)
        metrics.append(row)
        predictions.append(pd.DataFrame(dict(id=data['ids'][data['test_indices']], true_label=truth, predicted_label=pred, run=name, representation=row['representation'], classifier='logistic_regression')))
        print(row, flush=True)
predictions = pd.concat(predictions, ignore_index=True)
metrics, protocol = b.add_shared_balanced_accuracy_intervals(pd.DataFrame(metrics), predictions, seed=1234, bootstrap_draws=2000)
raw = pd.read_csv(BASE/'external_sparse_encoders_qwen_k16_full127600/predictive_metrics.csv')
raw_ba = float(raw.loc[raw.run.eq('qwen_baseline') & raw.representation.eq('raw_embeddings') & raw.classifier.eq('logistic_regression'),'balanced_accuracy'].iloc[0])
metrics['relative_ba'] = 100 * metrics.balanced_accuracy / raw_ba
metrics.to_csv(OUT/'linear_reference_metrics.csv',index=False)
predictions.to_csv(OUT/'linear_reference_predictions.csv',index=False)
provenance = dict(seed=1234, n_documents=127600, train=102080,test=25520, raw_ba=raw_ba, pca='Randomized SVD; 16 components, 32 oversamples, 7 power iterations; full corpus fit; centered raw vectors, no whitening or feature scaling.', classifier='Same class-balanced logistic regression as external comparison; max_iter=3000; no convergence warnings.', bootstrap=protocol, sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.raw_embeddings,args.raw_ids,args.labels,args.predictive_split,cache]})
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
print(metrics.to_string(index=False), flush=True)
