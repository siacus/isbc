from pathlib import Path
import csv,json,collections,hashlib
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
manifest=json.loads((root/'runs/mixed_embedding_comparison/frozen_profiles_qwen_seed1234/manifest.json').read_text())
p=root/manifest['inputs']['labels']['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['inputs']['labels']['sha256']
labels={r['id']:r for r in csv.DictReader(p.open())}
mp=root/'runs/mixed_embedding_comparison/frozen_profiles_qwen_seed1234/profile_membership.csv'
rows=list(csv.DictReader(mp.open()));print('Columns',list(rows[0]))
d=collections.defaultdict(collections.Counter)
for r in rows:
 if r['profile_id'] not in ['P01','P07']:continue
 l=labels[r['id']]
 if l['source']=='airline':d[(r['profile_id'],l['airline_name'])][l['airline_sentiment']]+=1
res=[]
for a in sorted({a for p,a in d}):
 v=[]
 for p in ['P01','P07']:
  n=sum(d[p,a].values());neg=d[p,a]['negative'];v.append(dict(profile=p,n=n,negative=neg,share=neg/n))
 res.append(dict(airline=a,profiles=v));print(a,v)
(out/'mixed_airline_conditioned.json').write_text(json.dumps(res,indent=2))
