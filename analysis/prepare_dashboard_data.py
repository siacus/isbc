"""Join locally obtained texts/annotations to the published fitted ISBC codes.

No training or embedding extraction. Outputs are local-only and Git-ignored.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import pandas as pd
from export_dashboard_data import ROOT, SPECS, BITS, read, verify
from prepare_pheme_engagement import extract_engagement

def aligned_merge(left, right, columns):
    missing = set(left.id) - set(right.id)
    if missing: raise ValueError(f'Source table is missing {len(missing)} fitted document IDs')
    return left.merge(right[['id', *columns]], on='id', how='left', validate='one_to_one')

def prepare(key, source_root, bundle_root, output_root):
    spec = SPECS[key]; bundle = bundle_root/spec['directory']
    codes, hashes = verify(bundle, spec)
    if key == 'agnews':
        source = read(source_root/'data/agnews/agnews_all.csv')
        full = aligned_merge(codes, source, ['text', 'label', 'label_name'])
    else:
        source = read(source_root/'data/pheme/pheme_ready.csv')
        full = aligned_merge(codes, source, ['text'])
        labels = read(source_root/'data/pheme/pheme_labels.csv')
        check = codes[['id', *spec['public_labels']]].merge(labels, on='id', suffixes=('_public','_source'), validate='one_to_one')
        if len(check) != len(codes): raise ValueError('Missing PHEME annotations')
        for col in spec['public_labels']:
            if not check[col+'_public'].equals(check[col+'_source']):
                raise ValueError(f'PHEME source annotation differs: {col}')
    # Match the whitespace normalization used in the saved encoded texts.
    full["text"] = full.text.map(lambda value: re.sub(r"\s+", " ", value).strip())
    # Require matching texts, not just matching row counts or numeric IDs.
    check = full[['id', 'text']].merge(hashes, on='id', validate='one_to_one')
    actual = check.text.map(lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest())
    mismatch = actual.ne(check.text_sha256)
    if mismatch.any():
        raise ValueError(f'{mismatch.sum()} texts differ from the fitted corpus. Check source version and preparation; no output written.')
    columns = ['id', 'text', *BITS]
    if key == 'agnews': columns += ['label', 'label_name']
    else:
        columns += spec['public_labels']
        archive = source_root/'data/pheme/PHEME_veracity.tar.bz2'
        metrics = extract_engagement(archive, set(codes.id)).fillna('')
        keep = ['retweet_count','favorite_count','followers_count','friends_count',
                'statuses_count','author_favourites_count','listed_count','created_at','text_length']
        full = aligned_merge(full, metrics, keep); columns += keep
    full = full[columns].sort_values('id', kind='stable').reset_index(drop=True)
    output = output_root/spec['directory']; file = output/f'encoded_{key}_qwen.csv'
    if bundle.resolve() == output.resolve():
        raise ValueError('Local text-bearing output must not replace the public code bundle')
    if file.exists():
        saved = read(file)
        pd.testing.assert_frame_equal(saved, full.astype(str), check_dtype=False)
        print(f'{file}: already matches; unchanged.'); return
    output.mkdir(parents=True, exist_ok=True); full.to_csv(file, index=False)
    metadata = json.loads((bundle/'metadata.json').read_text())
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (output/'README.md').write_text('Locally reconstructed source content for ISBC Explorer.\n'
        'Not part of the public code release. See docs/data_licenses.md for source terms.\n')
    print(f'Prepared {file}: {len(full):,} documents. No model was trained.')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=[*SPECS, 'all'], default='all')
    p.add_argument('--source-root', type=Path, default=ROOT)
    p.add_argument('--bundle-root', type=Path, default=ROOT/'dashboard/data')
    p.add_argument('--output-root', type=Path, default=ROOT/'dashboard/local_data')
    a = p.parse_args()
    for key in SPECS if a.dataset == 'all' else [a.dataset]:
        prepare(key, a.source_root, a.bundle_root, a.output_root)
if __name__ == '__main__': main()
