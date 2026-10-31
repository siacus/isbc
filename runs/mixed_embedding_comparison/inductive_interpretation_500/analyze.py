from pathlib import Path
import os,json,hashlib,itertools
os.environ.setdefault('MPLCONFIGDIR','/tmp/isbc-signed-mpl')
import pandas as pd,numpy as np
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
bits=[f'bit_{k:02d}'for k in range(16)]
meta=json.loads((D/'design.json').read_text());profiles=meta['profiles'];oldpath=ROOT/'runs/mixed_qwen_all_fixed_balance/encoded_mixed_ready_best.csv'
ref=pd.read_csv(oldpath,usecols=['id','text']+bits,dtype={'id':str}).merge(pd.read_csv(ROOT/'data/mixed/mixed_labels.csv',dtype=str),on='id',validate='one_to_one')
e=pd.read_csv(D/'encoded_encoding_input_best.csv',dtype={'id':str});new=pd.read_csv(D/'sample_with_labels.csv',dtype=str,keep_default_na=False).merge(e.drop(columns=['text']),on='id',validate='one_to_one')
assert len(new)==6000 and len(ref)==111000
for f in [ref,new]:f['profile']=f[bits].astype(str).agg(''.join,axis=1)
counts=ref.profile.value_counts();new['fit_profile_n']=new.profile.map(counts).fillna(0).astype(int)
profileid={p['profile_code']:p['profile_id']for p in profiles};new['frequent_fit_profile']=new.profile.map(profileid).fillna('other')
ref['component']=ref.source
for c in ['sentiment','emotion','hate','offensive']:
 mask=(ref.source=='tweeteval')&ref[c].notna()&ref[c].ne('unknown');ref.loc[mask,'component']='tweeteval_'+c
assert set(ref.component)==set(new.component)
summary=[];activation=[]
for c,g in new.groupby('component'):
 r=ref[ref.component==c];diff=g[bits].mean()-r[bits].mean()
 summary.append(dict(component=c,n=len(g),mean_active=float(g[bits].sum(axis=1).mean()),seen_profile_pct=float(100*g.fit_profile_n.gt(0).mean()),supported_profile_pct=float(100*g.fit_profile_n.ge(100).mean()),top10_profile_pct=float(100*g.frequent_fit_profile.ne('other').mean()),mean_absolute_activation_change_pp=float(100*abs(diff).mean()),max_absolute_activation_change_pp=float(100*abs(diff).max())))
 for b in bits:activation.append(dict(component=c,bit=b,fit_pct=float(100*r[b].mean()),new_pct=float(100*g[b].mean()),difference_pp=float(100*diff[b])))
pd.DataFrame(summary).to_csv(D/'source_summary.csv',index=False);pd.DataFrame(activation).to_csv(D/'activation_transfer.csv',index=False)
# Previously frozen profile descriptions; annotations only evaluate their transfer.
attrs={'P01':('airline','airline_sentiment','negative'),'P07':('airline','airline_sentiment','negative'),
 'P02':('climmig','climmig_migration','yes'),'P09':('climmig','climmig_migration','yes'),
 'P03':('20ng','newsgroups_label_coarse','comp'),'P05':('agnews','agnews_label','Sports')}
print('ANNOTATION VALUES',ref.climmig_migration.value_counts().to_dict(),ref.newsgroups_label_coarse.value_counts().to_dict())
prs=[];examples=[]
for p in profiles:
 code=p['profile_code'];pid=p['profile_id'];out={'profile_id':pid,'code':code}
 for dataset,f in [('fit',ref),('new',new)]:
  g=f[f.profile==code];out[dataset+'_n']=len(g);out[dataset+'_source_counts']=g.source.value_counts().to_dict()
  if pid in attrs:
   src,col,val=attrs[pid];assert val in set(ref.loc[ref.source==src,col]), (col,val);q=g[g.source==src];out['attribute']=col+'='+val;out['attribute_source']=src
   out[dataset+'_attribute_n']=len(q);out[dataset+'_attribute_positive']=int(q[col].eq(val).sum());out[dataset+'_attribute_pct']=float(100*q[col].eq(val).mean())if len(q)else None
  if len(g):
   sample=g.sample(min(3,len(g)),random_state=20261030+int(pid[1:]))
   for x in sample.to_dict('records'):examples.append({'dataset':dataset,'profile_id':pid,'id':x['id'],'source':x['source'],'text':x['text'],**{k:x.get(k)for k in ['airline_sentiment','climmig_migration','newsgroups_label','newsgroups_label_coarse','agnews_label']}})
 prs.append(out)
(D/'profile_transfer.json').write_text(json.dumps(prs,indent=2));pd.DataFrame(examples).to_csv(D/'profile_transfer_examples.csv',index=False)
# General single-bit/source and type associations under the new source prior.
# Reweight original components to the predetermined new sampling allocation.
weights=ref.component.map((new.component.value_counts(normalize=True)/ref.component.value_counts(normalize=True)).to_dict()).to_numpy()
association=[]
for b in bits:
 for target in ['source','type']:
  for val in sorted(ref[target].unique()):
   a=ref[b].to_numpy(bool);n=new[b].to_numpy(bool)
   association.append(dict(bit=b,target=target,value=val,fit_unweighted_pct=float(100*ref.loc[a,target].eq(val).mean()),fit_reweighted_pct=float(100*weights[a&ref[target].eq(val).to_numpy()].sum()/weights[a].sum()),new_pct=float(100*new.loc[n,target].eq(val).mean())))
pd.DataFrame(association).to_csv(D/'bit_annotation_transfer.csv',index=False)
# Prepare audit cards for the fixed random qualitative subsample. No inference labels added.
qids=pd.read_csv(D/'qualitative_ids.csv',dtype=str);q=new.set_index('id').loc[qids.id].reset_index();q.to_csv(D/'qualitative_encoded_documents.csv',index=False)
refs=pd.read_csv(D/'reference_bit_texts.csv',dtype={'fit_id':str});atlas=json.loads((D/'reference_bits.json').read_text());cards=[]
for r in q.to_dict('records'):
 active=[k for k,b in enumerate(bits)if r[b]==1];cards.extend([f"## {r['id']}",f"Source: {r['source']}; component: {r['component']}; active bits: {active}; profile: {r['profile']}; reference documents with exact profile: {r['fit_profile_n']}",r['text'],''])
 for k in active:
  # Fixed reference examples sampled before the new texts were encoded.
  rr=refs[refs.bit==k].head(2)
  cards.append(f'Bit {k:02d}, fixed reference examples:')
  for z in rr.itertuples():cards.append(f'- Fit {z.fit_id} ({z.source}): {z.text}')
 cards.append('')
(D/'qualitative_evidence_cards.md').write_text('\n'.join(cards))
new.to_csv(D/'new_documents_encoded_with_labels.csv',index=False)
results=dict(new_documents=len(new),known_profile_documents=int(new.fit_profile_n.gt(0).sum()),known_profile_pct=float(100*new.fit_profile_n.gt(0).mean()),new_distinct_profiles=int(new.profile.nunique()),previously_unobserved_distinct_profiles=int(new.loc[new.fit_profile_n.eq(0),'profile'].nunique()),supported_profile_documents=int(new.fit_profile_n.ge(100).sum()),top10_documents=int(new.frequent_fit_profile.ne('other').sum()),mean_active=float(new[bits].sum(axis=1).mean()),mean_abs_component_bit_change_pp=float(pd.DataFrame(activation).difference_pp.abs().mean()),largest_abs_component_bit_change_pp=float(pd.DataFrame(activation).difference_pp.abs().max()),qualitative_cards=len(q),classifiers_fitted=False)
(D/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));print(pd.DataFrame(summary).to_string(index=False))
for r in prs:print(r)
