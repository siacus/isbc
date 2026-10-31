from pathlib import Path
import pandas as pd,numpy as np,re,html,json
from sklearn.feature_extraction.text import CountVectorizer,ENGLISH_STOP_WORDS
P=Path(__file__).resolve().parent
d=pd.read_csv(P.parent/'all_documents_with_tail_membership.csv',dtype={'id':str,'profile':str})
d['pair_state']=d.bit_05.astype(str)+d.bit_11.astype(str)
composition=pd.crosstab(d.pair_state,d.event).reindex(['00','01','10','11'])
composition.insert(0,'N',composition.sum(1))
composition.to_csv(P/'pair_event_composition.csv')
triple=pd.crosstab([d.bit_05,d.bit_11,d.bit_02],d.event);triple.to_csv(P/'triple_event_composition.csv')
texts=d.text.map(lambda t:re.sub(r'https?://\S+|(?<!\w)@\w+',' ',html.unescape(str(t)).lower()))
cv=CountVectorizer(binary=True,stop_words=sorted(set(ENGLISH_STOP_WORDS)|{'rt','amp'}),ngram_range=(1,2),token_pattern=r'(?u)\b[a-z][a-z_-]{2,}\b')
W=cv.fit_transform(texts);terms=cv.get_feature_names_out()
kw=[];stats=[];reading=[];group_rows=[];masks={}
for event in sorted(d.event.unique()):
 for state in ['00','01','10','11']:
  for b in ['all',0,1]:
   mask=d.event.eq(event)&d.pair_state.eq(state)
   if b!='all':mask=mask&d.bit_02.eq(b)
   n=int(mask.sum());name=f'{event}_{state}_bit02_{b}';masks[name]=mask.to_numpy()
   if not n:continue
   x=d[mask];cnt=W[mask.to_numpy()].sum(0).A1
   group_rows.append(dict(group=name,event=event,pair_state=state,bit02=b,n=n))
   for j in np.flatnonzero(cnt):kw.append(dict(group=name,event=event,pair_state=state,bit02=b,n=n,term=terms[j],count=int(cnt[j]),percent=100*cnt[j]/n))
   if event=='charliehebdo':
    x.to_csv(P/(name+'_documents.csv'),index=False)
    stats.append(dict(pair_state=state,bit02=b,n=n,median_favorite=float(x.favorite_count.median()),low10=int(x.within_event_favorite_count_10_low.sum()),high10=int(x.within_event_favorite_count_10_high.sum()),low25=int(x.within_event_favorite_count_25_low.sum()),high25=int(x.within_event_favorite_count_25_high.sum())))
    if b!='all':
     # Complete small states, deterministic sample otherwise; statistical groups unchanged.
     seed=20261100+int(state,2)*2+b
     chosen=x.sort_values('id').sample(n=min(12,n),random_state=seed).sort_values('id')
     reading.extend(dict(group=name,id=r.id,text=r.text,seed=seed) for r in chosen.itertuples())
k=pd.DataFrame(kw);k.to_csv(P/'event_stratified_keywords.csv',index=False)
pd.DataFrame(group_rows).to_csv(P/'groups.csv',index=False)
pd.DataFrame(stats).to_csv(P/'charlie_state_outcomes.csv',index=False)
pd.DataFrame(reading).to_csv(P/'reading_documents.csv',index=False)
contrasts=[]
for event in sorted(d.event.unique()):
 for state in ['00','01','10']:
  left=masks[f'{event}_11_bit02_all'];right=masks[f'{event}_{state}_bit02_all'];nl=int(left.sum());nr=int(right.sum())
  if not nl or not nr:continue
  lc=W[left].sum(0).A1;rc=W[right].sum(0).A1
  for j in np.flatnonzero(lc+rc):contrasts.append(dict(event=event,comparison='11_vs_'+state,term=terms[j],left_count=int(lc[j]),left_n=nl,right_count=int(rc[j]),right_n=nr,difference_pp=100*(lc[j]/nl-rc[j]/nr)))
pd.DataFrame(contrasts).to_csv(P/'pair_complement_keyword_contrasts.csv',index=False)
lines=[]
for group in dict.fromkeys(r['group'] for r in reading):
 lines.append('\n## '+group)
 z=k[k.group.eq(group)].sort_values(['count','term'],ascending=[False,True]).head(16)
 lines.append('FULL GROUP TERMS: '+ '; '.join(f'{r.term} {r.count}/{r.n}' for r in z.itertuples()))
 lines.extend(r['id']+' | '+r['text'].replace('\n',' ') for r in reading if r['group']==group)
(P/'reading_dossier.md').write_text('\n'.join(lines))
(P/'method.json').write_text(json.dumps(dict(scope='All 5802 Qwen-coded corpus documents; separate four exhaustive states of bits 05 and 11, then bit02. Event stratification is descriptive using the dataset construction field, not training supervision.',keywords='Full group document presence; lowercase; URL/@mention removal; English stopword removal; unigrams and post-stopword bigrams. Event-stratified groups, all observed terms exported.',qualitative_reading='Within Charlie Hebdo: up to 12 texts per pair-state x bit02 state. Sorted IDs, pandas sample without replacement, seed=20261100+int(pair_state,2)*2+bit02. Not used for quantitative counts. Small groups read in full.',outcome='Existing within-event favorite tails; all-document descriptive analysis, no retraining or additional held-out evaluation.'),indent=2))
print(composition.to_string());print(pd.DataFrame(stats).to_string(index=False))
