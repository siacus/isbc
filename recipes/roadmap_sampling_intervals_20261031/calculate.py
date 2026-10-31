from pathlib import Path
import json,csv,math,random,hashlib
from statistics import NormalDist
import pandas as pd
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
z=NormalDist().inv_cdf(.975)
def wilson(x,n):
 p=x/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-h,c+h]
C=R/'runs/mixed_embedding_comparison/inductive_interpretation_500'
rows=json.loads((C/'three_bit_contexts.json').read_text())
new=pd.read_csv(C/'new_documents_encoded_with_labels.csv',dtype={'id':str})
for r in rows:
 if r['dataset']!='new':continue
 m=new.source.eq(r['source'])&new[[f'bit_{k:02d}' for k in r['context']]].eq(1).all(axis=1)&new[f"bit_{r['varying_bit']:02d}"].eq(r['state'])
 col,val=r['attribute'].split('='); assert int(m.sum())==r['n'] and int((m&new[col].eq(val)).sum())==r['attribute_count']
 r['wilson95']=wilson(r['attribute_count'],r['n'])
diffs=[]
for src in ['airline','climmig']:
 a,b=sorted([r for r in rows if r['source']==src and r['dataset']=='new'],key=lambda r:r['state'])
 p0=a['attribute_count']/a['n'];p1=b['attribute_count']/b['n'];l0,u0=a['wilson95'];l1,u1=b['wilson95'];d=p1-p0
 diffs.append(dict(source=src,active_minus_inactive=d,newcombe95=[d-math.sqrt((p1-l1)**2+(u0-p0)**2),d+math.sqrt((u1-p1)**2+(p0-l0)**2)]))
counts=[]
for case,target in [('technology_market','Sci_Tech'),('political_economy','World')]:
 p=R/'runs/agnews_qwen_all_fixed_balance/bit_structure/conjunction_profile_maps'/f'{case}_profiles.csv'
 with p.open() as f:r=list(csv.DictReader(f))
 b=[x for x in r if int(x['Business'])>0];dom=[x for x in b if int(x[target])>max(int(x[k]) for k in ['Business','Sci_Tech','World','Sports'] if k!=target)]
 counts.append(dict(case=case,profiles=len(r),business_profiles=len(b),business_documents=sum(int(x['Business'])for x in b),target_dominant_business_profiles=len(dom),business_documents_in_target_dominant_profiles=sum(int(x['Business'])for x in dom)))
checks={}
A=R/'runs/profile_single_bit_contrasts_qwen_20261030'
groups=json.loads((A/'agnews_full_selected_groups.json').read_text())
saved=pd.read_csv(A/'agnews_inspection_sample.csv',dtype=str)
for code in ['0110100001000110','0110100011000110']:
 selected=random.Random(1234).sample(sorted(groups[code],key=lambda x:int(x['id'])),12)
 assert [r['id'] for r in selected]==saved.loc[saved.profile.eq(code),'id'].tolist()
checks['agnews_sports_24_ids_reproduced']=True
P=R/'runs/pheme_embedding_comparison/engagement/interpretation_audit_20261030'
d=pd.read_csv(P/'joined.csv',dtype={'id':str});saved=pd.read_csv(P/'sydney_random_texts.csv',dtype={'id':str})
for state in [0,1]:
 ids=d[d.event.eq('sydneysiege')&d.predictive_split.eq('train')&d.bit_07.eq(state)].sort_values('id').sample(12,random_state=20261030).id.tolist()
 assert ids==saved.loc[saved.bit07.eq(state),'id'].tolist()
checks['pheme_sydney_24_ids_reproduced']=True
M=R/'runs/mixed_embedding_comparison/frozen_profiles_qwen_seed1234'
man=json.loads((M/'manifest.json').read_text()); enc=pd.read_csv(R/'runs/mixed_qwen_all_fixed_balance/encoded_mixed_ready_best.csv',usecols=['id']+[f'bit_{k:02d}'for k in range(16)],dtype={'id':str})
ready=pd.read_csv(R/'data/mixed/mixed_ready.csv',usecols=['id'],dtype=str);enc=ready.merge(enc,on='id',sort=False,validate='one_to_one')
for profile in man['profiles']:
 mask=enc[[f'bit_{k:02d}'for k in range(16)]].eq([int(c) for c in profile['profile_code']]).all(axis=1)
 pos=random.Random(int(profile['derived_seed'])).sample(mask[mask].index.tolist(),500)
 expected=pd.read_csv(R/profile['csv']['path'],usecols=['id'],dtype=str).id.tolist()
 assert enc.iloc[pos].id.tolist()==expected
checks['mixed_10_profile_samples_5000_ids_reproduced']=True
qual=pd.read_csv(C/'qualitative_ids.csv',dtype=str);sample=pd.read_csv(C/'sample_with_labels.csv',dtype=str)
for component,g in sample.groupby('component'):
 payload=f'latent-mixed-v1:20261030:qual-{component}'.encode();seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')%(2**32)
 assert g.sample(20,random_state=seed).id.tolist()==qual.loc[qual.component.eq(component),'id'].tolist()
checks['mixed_new_240_inspection_ids_reproduced']=True
result={'new_rows':[r for r in rows if r['dataset']=='new'],'differences':diffs,'business_counts':counts,'sampling_checks':checks,'interval_method':'Wilson and Newcombe hybrid-score without continuity correction; 95%; independent-document binomial approximation conditional on frozen encoder/context and observed group sizes; no finite population correction.'}
(O/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'differences':diffs,'business_counts':counts,'sampling_checks':checks},indent=2))
