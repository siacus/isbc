"""Export or verify public dashboard codes, without third-party text or counters."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
BITS = [f'bit_{k:02d}' for k in range(16)]
SPECS = {
    'agnews': dict(directory='agnews_qwen', dataset='AG News', rows=127600,
        source='https://huggingface.co/datasets/fancyzhx/ag_news',
        codes='runs/agnews_qwen_all_fixed_balance/encoded_agnews_all_ready_best.csv',
        labels='data/agnews/agnews_all.csv', public_labels=[]),
    'pheme': dict(directory='pheme_qwen', dataset='PHEME', rows=5802,
        source='https://doi.org/10.6084/m9.figshare.6392078.v1',
        codes='runs/pheme_qwen_fixed_balance/encoded_pheme_ready_best.csv',
        labels='data/pheme/pheme_labels.csv',
        public_labels=['label', 'veracity', 'event', 'label_fine']),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'id' not in df or df.id.eq('').any() or df.id.duplicated().any():
        raise ValueError(f'Missing, empty or duplicate IDs: {path}')
    return df

def validate(df, spec):
    expected = ['id', *BITS, *spec['public_labels']]
    if list(df.columns) != expected:
        raise ValueError('Public bundle must contain only IDs, hard codes and permitted annotations')
    if len(df) != spec['rows'] or not df.id.is_unique:
        raise ValueError('Unexpected corpus size or duplicate IDs')
    if not np.isin(df[BITS].apply(pd.to_numeric, errors='coerce').to_numpy(), [0, 1]).all():
        raise ValueError('Invalid hard codes')

def verify(directory, spec):
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, checksum in [(manifest['file'], manifest['sha256']),
                            (manifest['text_hash_file'], manifest['text_hash_sha256'])]:
        if sha(directory/name) != checksum:
            raise ValueError(f'Checksum mismatch: {directory/name}')
    df = read(directory/manifest['file']); validate(df, spec)
    hashes = read(directory/manifest['text_hash_file'])
    if list(hashes.columns) != ['id', 'text_sha256'] or set(hashes.id) != set(df.id):
        raise ValueError('Invalid text-hash alignment table')
    if not hashes.text_sha256.str.fullmatch('[0-9a-f]{64}').all():
        raise ValueError('Invalid text checksums')
    print(f'{spec["dataset"]}: {len(df):,} documents; public schema and checksums verified.')
    return df, hashes

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=[*SPECS, 'all'], default='all')
    p.add_argument('--source-root', type=Path, default=ROOT)
    p.add_argument('--output-root', type=Path, default=ROOT/'dashboard/data')
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    for key in SPECS if a.dataset == 'all' else [a.dataset]:
        spec = SPECS[key]; directory = a.output_root/spec['directory']
        if a.verify:
            verify(directory, spec); continue
        codepath = a.source_root/spec['codes']
        config = json.loads((codepath.parent/'metadata.json').read_text())
        metadata = dict(model_name='Qwen/Qwen3-Embedding-0.6B', k_factors=16,
                        target_active=6.0, checkpoint='best', seed=1234)
        for field in ['model_name', 'k_factors', 'target_active', 'seed']:
            if config.get(field) != metadata[field]:
                raise ValueError(f'Unexpected fitted configuration: {field}')
        original = read(codepath).sort_values('id', kind='stable').reset_index(drop=True)
        df = original[['id', *BITS]].copy(); sources = [codepath]
        if spec['public_labels']:
            path = a.source_root/spec['labels']; labels = read(path); sources.append(path)
            if set(labels.id) != set(df.id): raise ValueError('Annotation IDs differ')
            df = df.merge(labels[['id', *spec['public_labels']]], on='id', validate='one_to_one')
        validate(df, spec)
        hashes = original[['id']].copy()
        hashes['text_sha256'] = original.text.map(lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest())
        file = directory/f'encoded_{key}_qwen.csv'
        if file.exists():
            saved, saved_hashes = verify(directory, spec)
            pd.testing.assert_frame_equal(saved, df, check_dtype=False)
            pd.testing.assert_frame_equal(saved_hashes, hashes, check_dtype=False)
            print('Existing public bundle matches the fitted representation; unchanged.'); continue
        directory.mkdir(parents=True, exist_ok=True)
        df.to_csv(file, index=False); hashes.to_csv(directory/'text_hashes.csv', index=False)
        (directory/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        manifest = dict(schema_version=2, dataset=spec['dataset'], dataset_source=spec['source'],
            rows=len(df), bits=BITS, labels=spec['public_labels'], file=file.name,
            bytes=file.stat().st_size, sha256=sha(file), text_hash_file='text_hashes.csv',
            text_hash_sha256=sha(directory/'text_hashes.csv'), license_notice='DATA_LICENSE.md',
            content='IDs and fitted binary codes; PHEME annotations only. No source text or platform counters.',
            sources=[dict(repository_relative_path=str(x.relative_to(a.source_root)), sha256=sha(x)) for x in sources])
        (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'Exported {file}')
if __name__ == '__main__': main()
