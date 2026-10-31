from pathlib import Path
import json
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
c=pd.read_csv(P/'all_conjunctions.csv',dtype={'combination':str})
p=pd.read_csv(P/'all_profile_statistics.csv',dtype={'profile':str})
summary=pd.read_csv(P/'group_summary.csv').set_index('group')
allrows=[]
for kind,frame,key in [('conjunction',c,'combination'),('profile',p,'profile')]:
 for scope in ['global','within_event']:
  for outcome in ['retweet_count','favorite_count']:
   for tail in ['low','high']:
    g10=f'{scope}_{outcome}_10_{tail}';g25=f'{scope}_{outcome}_25_{tail}'
    a=frame[frame.group.eq(g10)].set_index(key);b=frame[frame.group.eq(g25)].set_index(key)
    allkeys=a.index.union(b.index)
    for k in allkeys:
     n10=int(a.loc[k,'n']) if k in a.index else 0;n25=int(b.loc[k,'n']) if k in b.index else 0
     assert n10<=n25
     if not n25:continue
     N10=int(summary.loc[g10,'n']);N25=int(summary.loc[g25,'n'])
     corpus_n=int(b.loc[k,'corpus_n']);f10=n10/N10;f25=n25/N25
     allrows.append(dict(kind=kind,scope=scope,outcome=outcome,tail=tail,pattern=k,size=len(k.split('+')) if kind=='conjunction' else 16,n10=n10,n25=n25,N10=N10,N25=N25,prevalence10=f10,prevalence25=f25,enrichment10=f10/(corpus_n/5802),enrichment25=f25/(corpus_n/5802),extreme_fraction=n10/n25,tail_extreme_fraction=N10/N25,concentration_ratio=(n10/n25)/(N10/N25),frequency_change_pp=100*(f10-f25)))
select=pd.DataFrame(allrows);select.to_csv(P/'selectivity_10_vs_25.csv',index=False)
for y in ['retweet_count','favorite_count']:
 z=select[(select.scope=='within_event')&(select.outcome==y)&(select['tail']=='high')&(select.kind=='conjunction')].copy()
 print('\n',y,': largest positive frequency changes, all supports retained')
 print(z.sort_values('frequency_change_pp',ascending=False)[['pattern','n10','n25','prevalence10','prevalence25','concentration_ratio']].head(12).to_string(index=False))
diff=pd.read_csv(P/'conjunction_tail_differences.csv',dtype={'combination':str})
a=diff[diff.fraction.eq(.1)];b=diff[diff.fraction.eq(.25)]
joined=a.merge(b,on=['scope','outcome','combination','size'],suffixes=('_10','_25'))
joined['absolute_contrast_change_pp']=joined.difference_pp_10.abs()-joined.difference_pp_25.abs()
joined.to_csv(P/'conjunction_contrast_10_vs_25.csv',index=False)
joint=[]
for scope in ['global','within_event']:
 for q in [10,25]:
  for left,right in [('both','retweet_only'),('both','favorite_only'),('favorite_only','retweet_only')]:
   l=c[c.group.eq(f'{scope}_joint_{q}_high_{left}')].set_index('combination');r=c[c.group.eq(f'{scope}_joint_{q}_high_{right}')].set_index('combination')
   for k in l.index:
    joint.append(dict(scope=scope,fraction=q/100,left_group=left,right_group=right,combination=k,size=int(l.loc[k,'size']),left_n=int(l.loc[k,'n']),right_n=int(r.loc[k,'n']),left_prevalence=l.loc[k,'prevalence'],right_prevalence=r.loc[k,'prevalence'],difference_pp=100*(l.loc[k,'prevalence']-r.loc[k,'prevalence'])))
pd.DataFrame(joint).to_csv(P/'joint_conjunction_contrasts.csv',index=False)
(P/'selectivity_method.json').write_text(json.dumps(dict(definition='For the same outcome, direction and pattern: C=(n10/n25)/(N10/N25)=prevalence10/prevalence25. C>1 means disproportionate concentration at the 10% extreme compared with the corresponding 25% tail.',ties='Uses actual tail sizes, not the nominal 0.4 ratio.',interpretation='No cross-threshold persistence requirement. Report changes, disappearance, exclusivity and direction reversals. Counts accompany ratios because rare patterns can produce extreme ratios.',scope='All observed documents; exhaustive conjunctions of up to three positive bits plus all observed complete profiles; no held-out discovery validation.'),indent=2))
