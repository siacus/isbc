"""Recover the historical Mixed draw by source IDs, never resample it."""
from pathlib import Path
import argparse,sys,json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from build_mixed_dataset import SOURCE_SPECS,read_source,UNKNOWN

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--output-dir',type=Path,default=ROOT/'data/mixed');p.add_argument('--check-only',action='store_true');a=p.parse_args()
    membership=pd.read_csv(ROOT/'reproduction/mixed_membership.csv',dtype=str,keep_default_na=False)
    labels=list(dict.fromkeys(k for spec in SOURCE_SPECS for _,k in spec.labels))
    blocks=[]
    for spec in SOURCE_SPECS:
        wanted=membership[membership.component.eq(spec.key)]
        frame,_=read_source(spec,a.data_dir)
        frame['text_sha256']=frame.text.map(lambda text:hashlib.sha256(text.encode()).hexdigest())
        frame=frame.set_index('source_id',drop=False)
        chosen_rows=[]
        for original in wanted.to_dict('records'):
            current=frame.loc[[original['source_id']]] if original['source_id'] in frame.index else frame.iloc[:0]
            current=current[current.text_sha256.eq(original['text_sha256'])]
            if current.empty:
                # AG News preparation reassigned source IDs when replacing the older subset.
                current=frame[frame.text_sha256.eq(original['text_sha256'])]
                current=current.drop(columns=['source_id']).drop_duplicates()
            if len(current)!=1:
                raise ValueError(f"Missing or ambiguous content for {spec.key}, historical ID {original['source_id']}")
            values=current.iloc[0].to_dict()
            values.update(original) # retain original Mixed ID and historical source ID
            chosen_rows.append(values)
        chosen=pd.DataFrame(chosen_rows)
        for col in labels:
            if col not in chosen:chosen[col]=UNKNOWN
        for _,col in spec.labels:chosen[col]=chosen[col].astype(str).str.strip().replace('',UNKNOWN)
        blocks.append(chosen[['id','source_id','text','source','type']+labels])
    combined=pd.concat(blocks).set_index('id').loc[membership.id].reset_index()
    assert len(combined)==111000 and combined.id.is_unique
    ready=combined[['id','text']];annotation=combined[['id','source_id','source','type']+labels]
    original=json.loads((ROOT/'reproduction/mixed_membership.json').read_text())
    payloads={'mixed_ready.csv':ready.to_csv(index=False),'mixed_labels.csv':annotation.to_csv(index=False)}
    for name,payload in payloads.items():
        expected=original['ready_sha256' if name=='mixed_ready.csv' else 'labels_sha256']
        if hashlib.sha256(payload.encode()).hexdigest()!=expected:
            raise ValueError('Reconstructed values/order/serialization differ from the original '+name+'. No files written; inspect source preparation/version.')
        dest=a.output_dir/name
        if dest.exists() and dest.read_bytes()!=payload.encode():raise ValueError('Refusing to replace '+str(dest))
    if not a.check_only:
        a.output_dir.mkdir(parents=True,exist_ok=True)
        for name,payload in payloads.items():(a.output_dir/name).write_text(payload)
    print('Verified all 111,000 historical memberships, texts and CSV hashes.'+(' No files written.'if a.check_only else ' Files saved.'))
if __name__=='__main__':main()
