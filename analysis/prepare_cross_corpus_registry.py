"""Create Qwen-only analysis descriptors from the curated target registry."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    targets=json.loads((ROOT/'reproduction/cross_corpus_targets.json').read_text())
    for row in targets:
        for field in ['encoded','labels']:
            if not (ROOT/row[field]).is_file():raise FileNotFoundError(ROOT/row[field])
        if not (ROOT/row['destination']).with_name('predictive_split.csv').is_file():
            raise FileNotFoundError('Restore the recorded splits before preparing descriptors: '+row['destination'])
    for row in targets:
        p=ROOT/row['destination'];p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():continue # preserve a full comparison if already regenerated
        info=lambda rel:{'path':rel,'sha256':hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()}
        value={'runs':{'qwen':{'training':{'k_factors':row['k_factors'],'target_active':row['target_active']},'encoded':info(row['encoded'])}},'labels':info(row['labels']),'label_column':row['label_column']}
        p.write_text(json.dumps(value,indent=2)+'\n')
    print('Prepared',len(targets),'Qwen-only annotation descriptors.')
if __name__=='__main__':main()
