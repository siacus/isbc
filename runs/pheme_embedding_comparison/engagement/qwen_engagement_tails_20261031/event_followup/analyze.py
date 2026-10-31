from pathlib import Path
import pandas as pd,re,html
from sklearn.feature_extraction.text import CountVectorizer,ENGLISH_STOP_WORDS
out=Path(__file__).resolve().parent;p=out.parent
d=pd.read_csv(p/'all_documents_with_tail_membership.csv',dtype={'id':str})
clean=lambda t:re.sub(r'https?://\S+|(?<!\w)@\w+',' ',html.unescape(str(t)).lower())
v=CountVectorizer(binary=True,stop_words=sorted(set(ENGLISH_STOP_WORDS)|{'rt','amp'}),token_pattern=r'(?u)\b[a-z][a-z_-]{2,}\b')
W=v.fit_transform(d.text.map(clean));terms=v.get_feature_names_out();rows=[]
for event,bit in [('sydneysiege',7),('ferguson',7),('ottawashooting',14),('germanwings-crash',2)]:
 for y in ['retweet_count','favorite_count']:
  for q in [10,25]:
   for state in ['all',0,1]:
    base=d.event.eq(event).to_numpy(copy=True)
    if state!='all':base=base&d[f'bit_{bit:02d}'].eq(state).to_numpy()
    low=base&d[f'within_event_{y}_{q}_low'].to_numpy();high=base&d[f'within_event_{y}_{q}_high'].to_numpy()
    lc=W[low].sum(0).A1;hc=W[high].sum(0).A1
    for i in (lc+hc).nonzero()[0]:
     rows.append(dict(event=event,outcome=y,percent=q,bit=bit,state=state,term=terms[i],low_count=int(lc[i]),low_n=int(low.sum()),high_count=int(hc[i]),high_n=int(high.sum()),difference_pp=100*(hc[i]/high.sum()-lc[i]/low.sum()) if high.sum() and low.sum() else None))
r=pd.DataFrame(rows);r.to_csv(out/'event_keyword_contrasts.csv',index=False)
for event,y in [('sydneysiege','favorite_count'),('ferguson','retweet_count'),('ferguson','favorite_count'),('ottawashooting','favorite_count')]:
 z=r[(r.event==event)&(r.outcome==y)&(r.percent==10)&(r.state=='all')]
 print('\n',event,y)
 for side in [1,-1]:
  a=z[(z.high_count if side==1 else z.low_count)>=5].copy();a['score']=side*a.difference_pp
  print(a[a.score>0].nlargest(7,'score')[['term','low_count','low_n','high_count','high_n','difference_pp']].round(2).to_string(index=False))
# Retain all event-level contrasts, including rare cases and reversals.
c=pd.read_csv(p/'conditional_bit_contrasts.csv',dtype={'context':str,'bit':str})
c=c[(c.scope=='within_event')&~c.event.isin(['standardized','charliehebdo'])]
c.to_csv(out/'event_conditional_bit_contrasts.csv',index=False)
