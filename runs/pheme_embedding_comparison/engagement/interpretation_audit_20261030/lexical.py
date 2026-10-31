from pathlib import Path
import pandas as pd,json
out=Path(__file__).resolve().parent;d=pd.read_csv(out/'joined.csv',dtype={'id':str})
patterns={'incident_terms':r'\b(?:hostages?|gunman|police|cafe|café)\b','reaction_terms':r'\b(?:muslims?|islam|islamophobia|racis\w*|solidarity|pray\w*|illridewithyou)\b'}
rows=[]
for split in ['train','test']:
 for b in [0,1]:
  x=d[(d.event=='sydneysiege')&(d.predictive_split==split)&(d.bit_07==b)];row={'split':split,'bit07':b,'n':len(x)}
  for key,pat in patterns.items():
   hit=x.text.str.contains(pat,case=False,regex=True)
   row[key+'_count']=int(hit.sum());row[key+'_pct']=float(100*hit.mean())
  for y in ['retweet_count','favorite_count']:row[y+'_mean']=float(x[y].mean());row[y+'_median']=float(x[y].median())
  rows.append(row)
(out/'sydney_lexical_outcomes.json').write_text(json.dumps({'patterns':patterns,'selection':'Lexicons specified after reading fixed random training samples; evaluated separately on all training and test Sydney documents. Descriptive lexical indicators, not semantic labels.','results':rows},indent=2))
