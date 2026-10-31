#!/usr/bin/env python3
"""Neighbor/profile agreement and embedding distances within exact profiles.

Uses all same-profile partners of each eligible anchor, with equal anchor
weighting. The random reference is the exact mean over all other documents.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analyze_qwen_compression_extensions import (
    DEFAULT_CONFIGURATIONS, load_raw, read_codes, sha256_file,
)


def main():
    out = Path('runs/agnews_embedding_comparison/qwen_compression_extensions')
    ids, _, _, _ = read_codes(DEFAULT_CONFIGURATIONS[0])
    raw, _, raw_path = load_raw(DEFAULT_CONFIGURATIONS[0], ids)
    raw_hash = sha256_file(raw_path)
    sample = pd.read_csv(out / 'geometry_sample_documents.csv', dtype={'id': str})
    anchors = pd.Index(ids).get_indexer(sample.id)
    assert len(anchors) == 2000 and np.all(anchors >= 0)
    configurations, sources = {}, []
    for config in DEFAULT_CONFIGURATIONS:
        current_ids, bits, _, encoded = read_codes(config)
        _, _, current_raw = load_raw(config, current_ids)
        assert np.array_equal(ids, current_ids) and sha256_file(current_raw) == raw_hash
        _, groups, counts = np.unique(np.packbits(bits, axis=1), axis=0,
                                      return_inverse=True, return_counts=True)
        order = np.argsort(groups, kind='stable')
        ends = np.cumsum(counts)
        starts = ends - counts
        partners = []
        for anchor in anchors:
            group = groups[anchor]
            indices = order[starts[group]:ends[group]]
            partners.append(indices[indices != anchor])
        configurations[config.name] = (config, partners)
        sources.append({'configuration': config.name, 'encoded': str(encoded.resolve()),
                        'encoded_sha256': sha256_file(encoded)})
    x = np.asarray(raw, dtype=np.float64)
    norms2 = np.einsum('ij,ij->i', x, x)
    norms = np.sqrt(norms2)
    assert np.all(norms > 0)
    records = []
    with threadpool_limits(limits=4):
        for start in range(0, len(anchors), 64):
            batch = anchors[start:start + 64]
            dot = x[batch] @ x.T
            for metric in ['cosine', 'euclidean']:
                if metric == 'cosine':
                    distances = np.clip(1 - dot / (norms[batch, None] * norms[None, :]), 0, 2)
                else:
                    distances = np.sqrt(np.maximum(norms2[batch, None] + norms2[None, :] - 2 * dot, 0))
                distances[np.arange(len(batch)), batch] = 0
                random_means = distances.sum(axis=1) / (len(ids) - 1)
                for name, (config, partners) in configurations.items():
                    for j, anchor in enumerate(batch):
                        selected = partners[start + j]
                        if len(selected):
                            values = distances[j, selected]
                            mean = float(values.mean())
                            quantiles = np.quantile(values, [.25, .5, .75])
                        else:
                            mean = float('nan')
                            quantiles = [float('nan')] * 3
                        records.append({'configuration': name, 'K': config.k,
                                        'target_active': config.target, 'embedding_distance': metric,
                                        'id': ids[anchor], 'same_profile_partners': len(selected),
                                        'mean_within_profile_distance': mean,
                                        'within_profile_q25': quantiles[0],
                                        'within_profile_median': quantiles[1],
                                        'within_profile_q75': quantiles[2],
                                        'mean_random_distance': random_means[j]})
            if start % 512 == 0:
                print(f'Reverse distances: {min(start + 64, len(anchors))}/2000 anchors', flush=True)
    detail = pd.DataFrame(records)
    detail.to_csv(out / 'within_profile_embedding_distances_documents.csv', index=False)
    forward = pd.read_csv(out / 'nearest_profile_agreement.csv')
    summaries = []
    for (name, metric), group in detail.groupby(['configuration', 'embedding_distance'], sort=False):
        eligible = group[group.same_profile_partners > 0]
        assert len(group) == 2000
        inside = eligible.mean_within_profile_distance.mean()
        random = eligible.mean_random_distance.mean()
        prior = forward[(forward.configuration == name) & (forward.embedding_distance == metric)]
        assert len(prior) == 1
        row = prior.iloc[0].to_dict()
        row.update({'eligible_anchors': len(eligible), 'eligible_anchor_percent': len(eligible) / 20,
                    'mean_within_profile_distance': inside, 'mean_random_distance': random,
                    'within_to_random_distance_ratio': inside / random,
                    'distance_reduction_percent': 100 * (1 - inside / random),
                    'directed_same_profile_pairs': int(eligible.same_profile_partners.sum())})
        summaries.append(row)
    result = pd.DataFrame(summaries)
    result.to_csv(out / 'bidirectional_local_geometry.csv', index=False)
    metadata = {'anchor_source': 'geometry_sample_documents.csv', 'anchors': 2000, 'anchor_seed': 1234,
                'population_documents': len(ids), 'self_excluded': True,
                'forward': 'Nearest embedding neighbor among all other corpus documents; exact hard-profile equality',
                'reverse': 'All same-profile partners for every anchor with at least one such partner',
                'weighting': 'Mean of per-anchor mean distances; equal weight for eligible anchors',
                'random_distance_reference': 'Exact mean distance to all other corpus documents for the same eligible anchors',
                'singletons': 'Excluded only from reverse-distance analysis, retained in forward analysis; coverage reported',
                'distance_ratio': 'Mean within-profile distance divided by mean random-reference distance',
                'caliper': None, 'labels_used': False, 'binary_neighbor_ranking': False,
                'raw_path': str(raw_path.resolve()), 'raw_sha256': raw_hash, 'sources': sources}
    (out / 'bidirectional_local_geometry.json').write_text(json.dumps(metadata, indent=2) + '\n')
    table = '| K/s | NN cosine: same profile (%) | NN Euclidean: same profile (%) | Random: same profile (%) | Eligible reverse anchors (%) | Cosine distance: profile / random | Euclidean distance: profile / random |\n'
    table += '|' + '---|' * 7 + '\n'
    for config in DEFAULT_CONFIGURATIONS:
        subset = result[result.configuration == config.name].set_index('embedding_distance')
        c, e = subset.loc['cosine'], subset.loc['euclidean']
        table += (f'| {config.k}/{config.target} | {c.same_profile_percent:.2f} | {e.same_profile_percent:.2f} | '
                  f'{c.random_same_profile_percent:.4f} | {c.eligible_anchor_percent:.2f} | '
                  f'{c.mean_within_profile_distance:.3f} / {c.mean_random_distance:.3f} | '
                  f'{e.mean_within_profile_distance:.3f} / {e.mean_random_distance:.3f} |\n')
    text = ('# Bidirectional local geometry on Qwen / AG News\n\n'
            'These diagnostics avoid selecting an arbitrary top-k list among tied binary codes. They are not the standard trustworthiness and continuity scores.\n\n'
            'Forward: for the same 2,000 seeded random anchors, find the nearest embedding among all other 127,599 documents and test exact binary-profile equality. '
            'The random same-profile percentage is the anchor-averaged exact probability for a uniformly selected other document.\n\n'
            'Reverse: for each anchor, calculate embedding distances to ALL other documents sharing its exact hard profile. '
            'Average within each anchor, then across eligible anchors with equal weights. Compare with the exact mean distance to all other corpus documents for those SAME eligible anchors. '
            'Singleton profiles have no within-profile partners and are excluded only from this reverse analysis. Coverage is reported explicitly. '
            'All distances use original unnormalized embeddings. No caliper, labels, or binary-space ranking is used.\n\n'
            + table + '\n'
            'In the final two columns, the first number is the within-profile mean distance and the second is its matched random-reference mean distance; smaller within-profile distances indicate greater proximity. '
            'Ratios and percentage reductions, partner counts, and anchor-level distance quartiles are supplied in the CSV files. '
            'Coverage changes across configurations, so the reverse summaries describe different eligible subsets of the common anchor sample.\n\n'
            'Reproduce: `python analyze_qwen_bidirectional_local_geometry.py` after `analyze_qwen_nearest_profile.py`.\n')
    (out / 'bidirectional_local_geometry.md').write_text(text)
    print(table, flush=True)


if __name__ == '__main__':
    main()
