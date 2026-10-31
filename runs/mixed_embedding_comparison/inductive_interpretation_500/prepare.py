from pathlib import Path
import sys,os,json,hashlib,html,unicodedata,re,itertools,shutil
os.environ['MPLCONFIGDIR']='/tmp/isbc-signed-mpl'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np
from transformers import AutoTokenizer
from build_mixed_dataset import SOURCE_SPECS,read_source,derived_seed
from binary_factors_train import basic_clean
OUT=Path(__file__).resolve().parent;RUN=ROOT/'runs/mixed_qwen_all_fixed_balance';SEED=20261030
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canon=lambda s:re.sub(r'\s+',' ',unicodedata.normalize('NFKC',html.unescape(s)).casefold()).strip()
key=lambda s:hashlib.sha256(canon(s).encode()).hexdigest()
fit=pd.read_csv(ROOT/'data/mixed/mixed_ready.csv',dtype=str,keep_default_na=False)
lab=pd.read_csv(ROOT/'data/mixed/mixed_labels.csv',dtype=str,keep_default_na=False)
enc=pd.read_csv(RUN/'encoded_mixed_ready_best.csv',dtype={'id':str})
ref=fit.merge(lab,on='id',validate='one_to_one').merge(enc.drop(columns=['text']),on='id',validate='one_to_one')
assert len(ref)==111000
manifest=json.loads((ROOT/'runs/mixed_embedding_comparison/frozen_profiles_qwen_seed1234/manifest.json').read_text())
for v in manifest['inputs'].values():
 if v.get('path') and v.get('sha256'): assert sha(ROOT/v['path'])==v['sha256']
bits=[f'bit_{k:02d}' for k in range(16)]
# Freeze reference evidence before sampling or inspecting new documents.
refrecords=[];examples=[]
for k,b in enumerate(bits):
 mask=ref[b].eq(1);g=ref[mask]
 refrecords.append(dict(bit=k,n=len(g),source=g.source.value_counts(normalize=True).to_dict(),type=g.type.value_counts(normalize=True).to_dict()))
 sample=g.sample(n=12,random_state=SEED+k)
 for r in sample.to_dict('records'): examples.append({'bit':k,'fit_id':r['id'],'source':r['source'],'type':r['type'],'text':r['text']})
(OUT/'reference_bits.json').write_text(json.dumps(refrecords,indent=2))
pd.DataFrame(examples).to_csv(OUT/'reference_bit_texts.csv',index=False)
(OUT/'design.json').write_text(json.dumps(dict(seed=SEED,target_per_component=500,qualitative_per_component=20,
 design='Frozen original Qwen encoder; 12 components, nine sources; exact text and model-input overlap excluded globally. Original-corpus bit interpretations and ten profile definitions frozen before inspecting new documents. No encoder refit. All available eligible texts if fewer than 500. No classifier in the primary analysis. Qualitative cases selected randomly, not by successful interpretation.',
 inputs={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/mixed/mixed_ready.csv',ROOT/'data/mixed/mixed_labels.csv',RUN/'encoded_mixed_ready_best.csv',RUN/'best/model.pt',RUN/'best/run_config.json']},profiles=manifest['profiles']),indent=2))
print('Reference atlas frozen.',flush=True)
tok=AutoTokenizer.from_pretrained(RUN/'best',local_files_only=True)
def tokenkeys(texts):
 seqs=tok([basic_clean(t) for t in texts],padding=False,truncation=True,max_length=256)['input_ids']
 return [hashlib.sha256(np.asarray(x,dtype='<i4').tobytes()).hexdigest() for x in seqs]
seen={key(t)for t in fit.text};seen_tokens=set()
for start in range(0,len(fit),1000):seen_tokens.update(tokenkeys(fit.text.iloc[start:start+1000].tolist()))
print('Fit token inputs indexed:',len(seen_tokens),flush=True)
allrows=[];stats=[]
for spec in SOURCE_SPECS:
 frame,prov=read_source(spec,ROOT/'data');frame['text_hash']=frame.text.map(key)
 frame=frame[~frame.text_hash.isin(seen)].drop_duplicates('text_hash').sample(frac=1,random_state=derived_seed(SEED,spec.key)%(2**32))
 selected=[];removed=0
 for start in range(0,len(frame),128):
  batch=frame.iloc[start:start+128]; hs=tokenkeys(batch.text.tolist())
  for r,th in zip(batch.to_dict('records'),hs):
   if th in seen_tokens:removed+=1;continue
   r.update(id=f'new_{spec.key}_{len(selected)+1:04d}',source=spec.source,component=spec.key,type=spec.data_type,token_hash=th)
   selected.append(r);seen.add(r['text_hash']);seen_tokens.add(th)
   if len(selected)==500:break
  if len(selected)==500:break
 allrows.extend(selected);stats.append(dict(component=spec.key,n=len(selected),token_duplicates_skipped=removed,**prov));print(spec.key,len(selected),flush=True)
new=pd.DataFrame(allrows).fillna('unknown');assert new.id.is_unique and new.text_hash.is_unique and new.token_hash.is_unique
new.to_csv(OUT/'sample_with_labels.csv',index=False)
new[['id','text']].to_csv(OUT/'new_ready.csv',index=False)
qual=pd.concat([g.sample(n=min(20,len(g)),random_state=derived_seed(SEED,'qual-'+c)%(2**32))for c,g in new.groupby('component')])
qual[['id','component']].to_csv(OUT/'qualitative_ids.csv',index=False)
controls=fit.sample(64,random_state=SEED).copy();controls['original_id']=controls.id;controls.id='control_'+controls.id
controls.to_csv(OUT/'controls.csv',index=False)
pd.concat([controls[['id','text']],new[['id','text']]]).to_csv(OUT/'encoding_input.csv',index=False)
shutil.copytree(RUN/'best',OUT/'best',dirs_exist_ok=False)
(OUT/'sampling.json').write_text(json.dumps(dict(rows=len(new),components=stats,fit_distinct_token_inputs=len(set(tokenkeys(fit.text.iloc[:0].tolist()))) if False else 'see prepare log',sample_sha256=sha(OUT/'sample_with_labels.csv'),qualitative_ids_sha256=sha(OUT/'qualitative_ids.csv')),indent=2))
print('Prepared',len(new),'new documents plus 64 reference controls.',flush=True)
