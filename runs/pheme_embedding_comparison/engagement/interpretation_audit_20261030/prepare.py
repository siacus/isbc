from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,re
from collections import Counter
root=Path(__file__).resolve().parents[4];out=Path(__file__).resolve().parent
paths=[root/'data/pheme/pheme_ready.csv',root/'data/pheme/pheme_labels.csv',root/'runs/pheme_embedding_comparison/engagement/pheme_engagement_extracted.csv',root/'runs/pheme_embedding_comparison/engagement/pheme_engagement_split.csv',root/'runs/pheme_qwen_fixed_balance/encoded_pheme_ready_best.csv']
d=pd.read_csv(paths[0],dtype={'id':str})
for p in paths[1:]:
 t=pd.read_csv(p,dtype={'id':str});cols=['id']+[c for c in t if c not in d];d=d.merge(t[cols],on='id',validate='one_to_one')
d=d.sort_values('id').reset_index(drop=True);assert len(d)==5802 and (d.predictive_split=='train').sum()==4641
d.to_csv(out/'joined.csv',index=False)
conditions={'b07_off':d.bit_07==0,'b07_on':d.bit_07==1,'b07off_b06off':(d.bit_07==0)&(d.bit_06==0),'b07off_b06on':(d.bit_07==0)&(d.bit_06==1),'b07off_b08off':(d.bit_07==0)&(d.bit_08==0),'b07off_b08on':(d.bit_07==0)&(d.bit_08==1)}
rows=[];samples=[];lines=[]
for name,mask in conditions.items():
 for split in ['train','test']:
  x=d[mask&(d.predictive_split==split)];row={'group':name,'split':split,'n':len(x),'events':x.event.value_counts().to_dict(),'rumour_share':float((x.label=='rumour').mean())}
  for y in ['retweet_count','favorite_count']:row[y+'_mean']=float(x[y].mean());row[y+'_median']=float(x[y].median())
  rows.append(row)
 x=d[mask&(d.predictive_split=='train')].sample(n=12,random_state=20261030)
 for r in x.itertuples():samples.append({'group':name,'id':r.id,'text':r.text})
 lines += ['\n## '+name]+[r.id+' | '+r.text.replace('\n',' ') for r in x.itertuples()]
(out/'random_texts.md').write_text('\n'.join(lines));pd.DataFrame(samples).to_csv(out/'random_texts.csv',index=False)
(out/'groups.json').write_text(json.dumps(rows,indent=2))
(out/'provenance.json').write_text(json.dumps({'sampling':'12 random training documents per group, pandas sample random_state20261030, sorted IDs, no outcome or label selection; groups fixed from previously fitted tree splits; overlapping samples','sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2))
print((out/'random_texts.md').read_text())
# Follow-up within-event reading: same fixed seed, no outcome-based selection.
sydney_samples=[];sydney_lines=[]
for b in [0,1]:
 x=d[(d.event=='sydneysiege')&(d.predictive_split=='train')&(d.bit_07==b)].sort_values('id').sample(12,random_state=20261030)
 sydney_lines.append('\n## Sydney bit07='+str(b))
 for r in x.itertuples():
  sydney_samples.append({'bit07':b,'id':r.id,'text':r.text})
  sydney_lines.append(r.id+' | '+r.text.replace('\n',' '))
pd.DataFrame(sydney_samples).to_csv(out/'sydney_random_texts.csv',index=False)
(out/'sydney_random_texts.md').write_text('\n'.join(sydney_lines))
