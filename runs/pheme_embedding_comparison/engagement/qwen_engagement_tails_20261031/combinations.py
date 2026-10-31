"""Exhaustive one-, two-, three-bit conjunctions and conditional contrasts.
No resampling; no model fitting; all remaining coordinates unrestricted.
"""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
R=P.parents[3]
d=pd.read_csv(P/'all_documents_with_tail_membership.csv',dtype={'id':str,'profile':str})
B=d[[f'bit_{i:02d}' for i in range(16)]].to_numpy(int)
keys=[()] + [c for k in [1,2,3] for c in combinations(range(16),k)]
lookup={c:i for i,c in enumerate(keys)}
names=['+'.join(f'{i:02d}' for i in c) or 'all' for c in keys]
M=np.asarray([np.all(B[:,c]==1,axis=1) if c else np.ones(len(d),bool) for c in keys])
Mi=M.astype(np.int64)
assert len(keys)==697
groupnames=['corpus']+[x for x in d if x.startswith(('global_','within_event_'))]
G=np.asarray([np.ones(len(d),bool) if x=='corpus' else d[x].to_numpy(bool) for x in groupnames])
counts=Mi@G.astype(np.int64).T
X=np.load(R/'runs/pheme_qwen_fixed_balance/raw_embeddings_pheme_ready_best.npy').astype(float)
ids=pd.read_csv(R/'runs/pheme_qwen_fixed_balance/raw_embeddings_pheme_ready_best_ids.csv',dtype=str).id
assert list(d.id)==list(ids)
U=X/np.linalg.norm(X,axis=1,keepdims=True)
# Aggregate vectors by complete profile before evaluating all conjunctions.
profiles,pi=np.unique(d.profile.to_numpy(),return_inverse=True)
PB=np.asarray([[int(v) for v in profile] for profile in profiles])
PM=np.asarray([np.all(PB[:,c]==1,axis=1) if c else np.ones(len(PB),bool) for c in keys],float)
rows=[]
for j,g in enumerate(groupnames):
 mask=G[j]; n=counts[:,j]
 sums=np.zeros((len(profiles),U.shape[1]));np.add.at(sums,pi[mask],U[mask])
 selected_sums=PM@sums
 cos=np.divide(np.square(selected_sums).sum(1)-n,n*(n-1),out=np.full(len(n),np.nan),where=n>1)
 for i,k in enumerate(keys[1:],1):
  freq=n[i]/n[0]
  rows.append(dict(group=g,combination=names[i],size=len(k),n=int(n[i]),group_n=int(n[0]),prevalence=freq,corpus_n=int(counts[i,0]),enrichment=freq/(counts[i,0]/len(d)) if counts[i,0] else np.nan,mean_cosine=cos[i]))
pd.DataFrame(rows).to_csv(P/'all_conjunctions.csv',index=False)
print('All conjunctions and cosine summaries completed.',flush=True)

events=sorted(d.event.unique()); E=np.asarray([d.event.eq(e).to_numpy() for e in events])
corpus_e=Mi@E.astype(np.int64).T
contrasts=[];bitset_diffs=[]
for scope in ['global','within_event']:
 for outcome in ['retweet_count','favorite_count']:
  for q in [10,25]:
   low=d[f'{scope}_{outcome}_{q}_low'].to_numpy(bool); high=d[f'{scope}_{outcome}_{q}_high'].to_numpy(bool)
   lc=Mi@low.astype(np.int64);hc=Mi@high.astype(np.int64)
   le=Mi@(E&low).astype(np.int64).T;he=Mi@(E&high).astype(np.int64).T
   for i,k in enumerate(keys[1:],1):
    nl=int(lc[i]);nh=int(hc[i]);fl=nl/lc[0];fh=nh/hc[0]
    bitset_diffs.append(dict(scope=scope,outcome=outcome,fraction=q/100,combination=names[i],size=len(k),n_low=nl,n_high=nh,low_prevalence=fl,high_prevalence=fh,difference_pp=100*(fh-fl),exclusive='low' if nl>0 and nh==0 else 'high' if nh>0 and nl==0 else 'absent' if nh==nl==0 else 'both',mean_prevalence=(fl+fh)/2))
   for core in [c for c in keys if len(c)<=2]:
    ci=lookup[core]
    for bit in range(16):
     if bit in core:continue
     ki=lookup[tuple(sorted((*core,bit)))];weights=corpus_e[ci].astype(float)
     if not weights.sum():continue
     weights/=weights.sum()
     valid=(le[ci]>0)&(he[ci]>0)
     dl=np.divide(le[ki],le[ci],out=np.full(len(events),np.nan),where=le[ci]>0)
     dh=np.divide(he[ki],he[ci],out=np.full(len(events),np.nan),where=he[ci]>0)
     coverage=weights[valid].sum()
     delta=float(np.sum(weights[valid]*(dh[valid]-dl[valid]))/coverage) if coverage else np.nan
     pl=lc[ki]/lc[ci] if lc[ci] else np.nan; ph=hc[ki]/hc[ci] if hc[ci] else np.nan
     base=dict(scope=scope,outcome=outcome,fraction=q/100,context=names[ci],context_size=len(core),bit=f'{bit:02d}',n_low_context=int(lc[ci]),n_high_context=int(hc[ci]),n_low_on=int(lc[ki]),n_high_on=int(hc[ki]),n_low_off=int(lc[ci]-lc[ki]),n_high_off=int(hc[ci]-hc[ki]),low_on_fraction=pl,high_on_fraction=ph,raw_difference_pp=100*(ph-pl),event_standardized_difference_pp=100*delta,event_weight_coverage=coverage,context_corpus_n=int(corpus_e[ci].sum()),support_weighted_score=abs(delta)*min(lc[ci]/lc[0],hc[ci]/hc[0]) if np.isfinite(delta) else np.nan)
     contrasts.append(dict(**base,event='standardized'))
     for e,event in enumerate(events):
      contrasts.append({**base,'event':event,'n_low_context':int(le[ci,e]),'n_high_context':int(he[ci,e]),'n_low_on':int(le[ki,e]),'n_high_on':int(he[ki,e]),'n_low_off':int(le[ci,e]-le[ki,e]),'n_high_off':int(he[ci,e]-he[ki,e]),'low_on_fraction':dl[e],'high_on_fraction':dh[e],'raw_difference_pp':100*(dh[e]-dl[e])})
pd.DataFrame(contrasts).to_csv(P/'conditional_bit_contrasts.csv',index=False)
df=pd.DataFrame(bitset_diffs);df.to_csv(P/'conjunction_tail_differences.csv',index=False)
cc=pd.DataFrame(contrasts);z=cc[(cc.event=='standardized')&(cc.scope=='within_event')]
a=z[z.fraction.eq(.1)];b=z[z.fraction.eq(.25)]
both=a.merge(b,on=['scope','outcome','context','context_size','bit'],suffixes=('_10','_25'))
both['same_direction']=(both.event_standardized_difference_pp_10*both.event_standardized_difference_pp_25)>0
both['absolute_contrast_change_pp']=both.event_standardized_difference_pp_10.abs()-both.event_standardized_difference_pp_25.abs()
both.sort_values('support_weighted_score_10',ascending=False).to_csv(P/'conditional_contrast_ranking.csv',index=False)
(P/'combination_method.json').write_text(json.dumps(dict(conjunctions='All 16 single bits, 120 pairs, 560 triples; every required bit active; remaining coordinates unrestricted. Complete profiles analyzed separately.',contexts='Every shared context of zero, one or two active bits, crossed with each other bit at 0 versus 1.',sampling='None; complete observed groups, all ties retained.',standardization='Within each context, event weights equal event shares among all corpus documents satisfying that context. Events with no context document in either tail are unavailable; weights renormalized, coverage explicitly reported.',ranking='For readable examples only: |event-standardized difference| times min(context share of low tail, context share of high tail), separately at 10% and 25%; no requirement of persistence or concordance across thresholds. All results exported without minimum-support filtering. Rankings are exploratory, not significance tests.',entropy='Bernoulli entropy of a single bit measures its variability, not predictive discrimination; joint profile entropy does not establish coherence of individual combinations.',dependence='10% tails are subsets of 25% tails; consistency across thresholds is sensitivity analysis, not independent replication.'),indent=2))
print(both[both.context_size.gt(0)].sort_values('support_weighted_score_10',ascending=False)[['outcome','context','bit','n_low_context_10','n_high_context_10','raw_difference_pp_10','event_standardized_difference_pp_10','event_standardized_difference_pp_25','absolute_contrast_change_pp','event_weight_coverage_10']].head(30).to_string(index=False),flush=True)
