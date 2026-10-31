"""Compare complete AG News profile partitions, without labels or bit matching."""
from pathlib import Path
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd
import scipy
from scipy.special import gammaln
import sklearn
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, mutual_info_score


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/agnews_embedding_comparison/all_fixed_balance_label_name'


def expected_mi(a, b, n):
    """Exact hypergeometric EMI, aggregating repeated contingency margin sizes.

    Each cell with margins a_i,b_j contributes E[x/n log(nx/(a_i b_j))].
    Identically sized margins have identical expectations, avoiding an expensive
    loop over every pair of profile groups. Zero cells contribute zero.
    """
    aa, af = np.unique(a, return_counts=True)
    bb, bf = np.unique(b, return_counts=True)
    lf = gammaln(np.arange(n + 1, dtype=float) + 1)
    total = 0.0
    for size, freq in zip(aa, af):
        x = np.arange(1, min(size, bb.max()) + 1)[None, :]
        bcol = bb[:, None]
        valid = (x <= bcol) & (n - size - bcol + x >= 0)
        bi, xi = np.nonzero(valid)
        bv, xv = bb[bi], x[0, xi]
        logp = (lf[size] - lf[xv] - lf[size-xv]
                + lf[n-size] - lf[bv-xv] - lf[n-size-bv+xv]
                - lf[n] + lf[bv] + lf[n-bv])
        terms = (xv / n) * np.log(n * xv / (float(size) * bv)) * np.exp(logp)
        total += freq * np.sum(bf[bi] * terms)
    return float(total)


def measures(left, right):
    n = len(left)
    _, a = np.unique(left, return_counts=True)
    _, b = np.unique(right, return_counts=True)
    h1 = -np.sum(a/n * np.log(a/n))
    h2 = -np.sum(b/n * np.log(b/n))
    mi = mutual_info_score(left, right)
    emi = expected_mi(a, b, n)
    return dict(ari=float(adjusted_rand_score(left, right)),
                ami=float((mi-emi)/((h1+h2)/2-emi)),
                mi_nats=float(mi), expected_mi_nats=emi,
                entropy_left_nats=float(h1), entropy_right_nats=float(h2))


def main():
    source = OUT / 'factor_collection_comparison.json'
    manifest = json.loads(source.read_text())
    codes, stats, provenance = {}, [], {}
    reference_ids = None
    for name in manifest['run_order']:
        item = manifest['runs'][name]['encoded']
        path = Path(item['path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item['sha256'], (name, 'source changed')
        frame = pd.read_csv(path, usecols=lambda c: c == 'id' or c.startswith('bit_'), dtype={'id': str})
        assert len(frame) == 127600 and frame.id.is_unique
        frame = frame.sort_values('id').reset_index(drop=True)
        if reference_ids is None:
            reference_ids = frame.id.to_numpy()
        assert np.array_equal(reference_ids, frame.id.to_numpy())
        cols = sorted(c for c in frame if c.startswith('bit_'))
        bits = frame[cols].to_numpy()
        assert bits.shape == (127600, 16) and np.isin(bits, [0, 1]).all()
        packed = bits.astype(np.uint32) @ (1 << np.arange(16, dtype=np.uint32))
        _, code, counts = np.unique(packed, return_inverse=True, return_counts=True)
        codes[name] = code
        stats.append(dict(backbone=name, n_documents=len(code), n_profiles=len(counts),
                          mean_profile_size=len(code)/len(counts),
                          singleton_profiles=int(np.sum(counts == 1)),
                          singleton_document_percent=100*np.sum(counts == 1)/len(code)))
        provenance[name] = dict(path=str(path), sha256=digest)
    # Independent reference checks of grouped EMI and arithmetic AMI, including
    # imbalanced margins and profiles observed only once.
    rng = np.random.default_rng(1234)
    checks = []
    for n in [50, 300, 2000]:
        ix = rng.choice(127600, n, replace=False)
        a, b = codes['qwen'][ix], codes['minilm'][ix]
        calc = measures(a, b)['ami']
        ref = adjusted_mutual_info_score(a, b, average_method='arithmetic')
        assert abs(calc-ref) < 1e-9, (calc, ref)
        checks.append(dict(n=n, grouped_ami=calc, sklearn_ami=float(ref)))
    # Also verify highly imbalanced arbitrary partitions with large overlaps.
    a = np.repeat([0, 1, 2], [240, 40, 20])
    b = np.roll(np.repeat([0, 1, 2], [220, 60, 20]), 17)
    assert abs(measures(a, b)['ami'] - adjusted_mutual_info_score(a, b)) < 1e-9
    rows = []
    for left, right in combinations(manifest['run_order'], 2):
        result = measures(codes[left], codes[right])
        rows.append(dict(left=left, right=right, n_documents=127600, **result))
        print(left, right, result['ari'], result['ami'], flush=True)
    pd.DataFrame(stats).to_csv(OUT / 'backbone_profile_partition_sizes.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'backbone_profile_partition_agreement.csv', index=False)
    metadata = dict(n_documents=127600, n_bits=16, target_active=6,
                    partition='Distinct complete hard codes; all observed profiles retained',
                    alignment='Exact ID-set equality and lexicographic ID order',
                    ami_normalization='arithmetic mean of partition entropies',
                    chance_model='Permutation model with fixed contingency margins',
                    expected_mi='Exact hypergeometric sum aggregated over repeated margin sizes',
                    uses_category_labels=False, uses_bit_matching=False,
                    validation=checks, sources=provenance,
                    versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
    (OUT / 'backbone_profile_partition_agreement_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(pd.DataFrame(stats).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
