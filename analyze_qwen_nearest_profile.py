#!/usr/bin/env python3
"""Full-corpus embedding nearest neighbors and exact binary-profile agreement."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analyze_qwen_compression_extensions import (
    DEFAULT_CONFIGURATIONS, load_raw, read_codes, sha256_file,
)


def main():
    output = Path('runs/agnews_embedding_comparison/qwen_compression_extensions')
    ids, _, _, _ = read_codes(DEFAULT_CONFIGURATIONS[0])
    raw, _, raw_path = load_raw(DEFAULT_CONFIGURATIONS[0], ids)
    sample = pd.read_csv(output / 'geometry_sample_documents.csv', dtype={'id': str})
    anchors = pd.Index(ids).get_indexer(sample.id)
    assert len(anchors) == 2000 and np.all(anchors >= 0) and len(np.unique(anchors)) == 2000
    x = np.asarray(raw, dtype=np.float64)
    norms2 = np.einsum('ij,ij->i', x, x)
    norms = np.sqrt(norms2)
    assert np.all(norms > 0)
    neighbors = {metric: [] for metric in ['cosine', 'euclidean']}
    distances = {metric: [] for metric in neighbors}
    tie_counts = {metric: [] for metric in neighbors}
    # Exact exhaustive search, sharing the dot products between the two metrics.
    with threadpool_limits(limits=4):
        for start in range(0, len(anchors), 64):
            batch = anchors[start:start + 64]
            dot = x[batch] @ x.T
            for metric in neighbors:
                if metric == 'cosine':
                    d = np.clip(1 - dot / (norms[batch, None] * norms[None, :]), 0, 2)
                else:
                    d = np.sqrt(np.maximum(norms2[batch, None] + norms2[None, :] - 2 * dot, 0))
                d[np.arange(len(batch)), batch] = np.inf
                minimum = d.min(axis=1)
                # Resolve numerical ties deterministically by original corpus order.
                ties = np.isclose(d, minimum[:, None], atol=1e-12, rtol=0)
                chosen = ties.argmax(axis=1)
                assert np.all(chosen != batch)
                neighbors[metric].extend(chosen.tolist())
                distances[metric].extend(d[np.arange(len(batch)), chosen].tolist())
                tie_counts[metric].extend(ties.sum(axis=1).tolist())
            if start % 512 == 0:
                print(f'Neighbors: {min(start + 64, len(anchors))}/{len(anchors)} anchors', flush=True)
    # Independently verify one full-corpus Euclidean search using direct differences.
    check = np.empty(len(x))
    for start in range(0, len(x), 4096):
        diff = x[start:start + 4096] - x[anchors[0]]
        check[start:start + len(diff)] = np.linalg.norm(diff, axis=1)
    check[anchors[0]] = np.inf
    np.testing.assert_allclose(check[neighbors['euclidean'][0]], check.min(), atol=1e-9)
    nearest = pd.DataFrame({'id': ids[anchors]})
    for metric in neighbors:
        nearest[f'{metric}_neighbor_id'] = ids[neighbors[metric]]
        nearest[f'{metric}_distance'] = distances[metric]
        nearest[f'{metric}_ties'] = tie_counts[metric]
    nearest.to_csv(output / 'nearest_embedding_neighbors.csv', index=False)
    raw_hash = sha256_file(raw_path)
    rows, details, sources = [], [], []
    for config in DEFAULT_CONFIGURATIONS:
        current_ids, bits, _, encoded_path = read_codes(config)
        _, _, current_raw_path = load_raw(config, current_ids)
        assert np.array_equal(ids, current_ids) and sha256_file(current_raw_path) == raw_hash
        _, group, counts = np.unique(np.packbits(bits, axis=1), axis=0, return_inverse=True, return_counts=True)
        # Exact anchor-conditional probability for one uniform other document.
        chance = (counts[group[anchors]] - 1) / (len(ids) - 1)
        for metric, indices in neighbors.items():
            differing_bits = (bits[anchors] != bits[indices]).sum(axis=1)
            matched = differing_bits == 0
            rows.append({'configuration': config.name, 'K': config.k, 'target_active': config.target,
                         'embedding_distance': metric, 'anchors': len(anchors), 'matched': int(matched.sum()),
                         'same_profile_percent': float(100 * matched.mean()),
                         'random_same_profile_percent': float(100 * chance.mean()),
                         'mean_differing_bits': float(differing_bits.mean()),
                         'anchors_with_embedding_ties': int((np.array(tie_counts[metric]) > 1).sum())})
            details.append(pd.DataFrame({'configuration': config.name, 'embedding_distance': metric,
                                         'id': ids[anchors], 'neighbor_id': ids[indices],
                                         'embedding_nn_distance': distances[metric],
                                         'differing_bits': differing_bits, 'same_profile': matched.astype(int),
                                         'random_same_profile_probability': chance}))
        sources.append({'configuration': config.name, 'encoded': str(encoded_path.resolve()),
                        'encoded_sha256': sha256_file(encoded_path)})
    result = pd.DataFrame(rows)
    result.to_csv(output / 'nearest_profile_agreement.csv', index=False)
    pd.concat(details, ignore_index=True).to_csv(output / 'nearest_profile_agreement_documents.csv', index=False)
    metadata = {'anchors': 2000, 'anchor_seed': 1234,
                'anchor_source': 'geometry_sample_documents.csv; same sample as previous distance comparisons',
                'candidate_documents_per_anchor': len(ids) - 1, 'self_excluded': True,
                'search': 'Exhaustive float64 pairwise distances on unnormalized embeddings',
                'embedding_ties': 'Absolute distance tolerance 1e-12, first in original corpus order',
                'caliper': None, 'profile_match': 'All hard bits equal',
                'random_reference': 'Mean over anchors of (profile_size - 1)/(corpus_size - 1)',
                'raw_path': str(raw_path.resolve()), 'raw_sha256': raw_hash, 'sources': sources}
    (output / 'nearest_profile_agreement.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(result.to_string(index=False), flush=True)
    table = '| K | Target | Cosine NN: same profile (%) | Euclidean NN: same profile (%) | Random other document (%) |\n|---|---|---|---|---|\n'
    for config in DEFAULT_CONFIGURATIONS:
        part = result[result.configuration == config.name].set_index('embedding_distance')
        table += f'| {config.k} | {config.target} | {part.loc["cosine", "same_profile_percent"]:.2f} | {part.loc["euclidean", "same_profile_percent"]:.2f} | {part.iloc[0].random_same_profile_percent:.2f} |\n'
    (output / 'nearest_profile_agreement.md').write_text(
        '# Original nearest neighbors sharing the same binary profile\n\n'
        'For 2,000 seeded random anchors, find the nearest embedding among all other 127,599 corpus documents. '
        'Score 1 if anchor and neighbor have identical hard codes, and 0 otherwise. No binary nearest-neighbor list is constructed. '
        'No labels or distance caliper are used. Both metrics use the same anchors and unnormalized original embeddings.\n\n' + table + '\n'
        'The random reference is the exact expected agreement for one uniformly selected other document, averaged over the same anchors. '
        'It accounts for profile prevalence. Exact agreement is stricter than small binary distance and does not assess separation of distant documents. '
        'Embedding-space ties and per-anchor distances are recorded in the accompanying CSV files.\n\n'
        'Reproduce: `python analyze_qwen_nearest_profile.py`.\n')


if __name__ == '__main__':
    main()
