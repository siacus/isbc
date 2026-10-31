from pathlib import Path
import json,itertools
import numpy as np,pandas as pd
D=Path(__file__).resolve().parent;ROOT=D.parents[2];bits=[f'bit_{k:02d}'for k in range(16)]
r=pd.read_csv(ROOT/'runs/mixed_qwen_all_fixed_balance/encoded_mixed_ready_best.csv',usecols=['id']+bits,dtype={'id':str}).merge(pd.read_csv(ROOT/'data/mixed/mixed_labels.csv',dtype=str),on='id',validate='one_to_one')
n=pd.read_csv(D/'new_documents_encoded_with_labels.csv',dtype={'id':str})
rows=[]
for src,pool,variable,col,value in [('airline',[0,5,8,10,13,14],11,'airline_sentiment','negative'),('climmig',[1,6,8,12,13,15],0,'climmig_migration','yes')]:
 # Candidate context consists of 3 shared active coordinates of the pre-existing pair.
 # Selection uses only the reference corpus and source, never new-document labels.
 candidates=[]
 for combo in itertools.combinations(pool,3):
  sel=r[[bits[k]for k in combo]].all(axis=1);source_sel=sel&r.source.eq(src)
  if source_sel.sum()<.25*r.source.eq(src).sum():continue
  candidates.append((float(source_sel.sum()/sel.sum()),int(source_sel.sum()),combo))
 purity,support,combo=max(candidates,key=lambda v:(v[0],v[1],tuple(-x for x in v[2])))
 for dataset,f in [('fit',r),('new',n)]:
  context=f[[bits[k]for k in combo]].all(axis=1);source=context&f.source.eq(src)
  for state in [0,1]:
   sel=source&f[bits[variable]].eq(state);cnt=int(sel.sum());yes=int((sel&f[col].eq(value)).sum())
   rows.append(dict(source=src,context=list(combo),varying_bit=variable,state=state,dataset=dataset,context_all_n=int(context.sum()),context_source_n=int(source.sum()),context_source_pct=float(100*source.sum()/context.sum()),n=cnt,attribute_count=yes,attribute_pct=float(100*yes/cnt)if cnt else None,attribute=col+'='+value,new_complete_profiles=int(f.loc[sel,'fit_profile_n'].eq(0).sum())if dataset=='new'else None))
(D/'three_bit_contexts.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
