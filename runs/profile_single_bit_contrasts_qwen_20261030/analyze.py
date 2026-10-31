from pathlib import Path
import csv,json,hashlib,collections
ROOT=Path(__file__).resolve().parents[1]; INPUT=ROOT/'runs/cross_corpus_binary_qwen'; OUT=Path(__file__).resolve().parent

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
all_data={};source=[]
for p in sorted(INPUT.glob('*__*__flows.csv')):
 corpus,target,_=p.stem.split('__')
 if corpus=='ira_io':continue
 d=collections.defaultdict(collections.Counter)
 for r in read(p):d[r['profile']][r[target]]+=int(r['documents'])
 all_data.setdefault(corpus,{})[target]=dict(d)
 source.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
pairs=[]; summaries=[]
for corpus,tasks in all_data.items():
 first=next(iter(tasks.values())); counts={p:sum(v.values()) for p,v in first.items()}
 for d in tasks.values():assert counts=={p:sum(v.values()) for p,v in d.items()},corpus
 eligible={p:n for p,n in counts.items() if n>=50}; cp=[]
 for p,n in sorted(eligible.items()):
  for k,b in enumerate(p):
   if b!='0':continue
   q=p[:k]+'1'+p[k+1:]
   if q not in eligible:continue
   row=dict(corpus=corpus,bit=k,profile0=p,profile1=q,n0=n,n1=eligible[q],min_n=min(n,eligible[q]),total=n+eligible[q])
   cp.append(row)
   for target,d in tasks.items():
    a=d[p];b=d[q];categories=sorted(a.keys()|b.keys());delta={c:b.get(c,0)/row['n1']-a.get(c,0)/n for c in categories}
    largest=max(categories,key=lambda c:abs(delta[c])); dom0=max(a,key=a.get);dom1=max(b,key=b.get)
    pairs.append(row|dict(target=target,tvd=.5*sum(abs(x) for x in delta.values()),largest_category=largest,share0=a.get(largest,0)/n,share1=b.get(largest,0)/row['n1'],delta=delta[largest],dominant0=dom0,dominant1=dom1,dominant_share0=a[dom0]/n,dominant_share1=b[dom1]/row['n1']))
 used={r[key] for r in cp for key in ['profile0','profile1']}
 summaries.append(dict(corpus=corpus,N=sum(counts.values()),profiles=len(counts),pairs_min50=len(cp),pairs_min100=sum(r['min_n']>=100 for r in cp),unique_documents_min50=sum(counts[p] for p in used)))
for name,rows in [('contrasts.csv',pairs),('coverage.csv',summaries)]:
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'provenance.json').write_text(json.dumps(source,indent=2))
print('COVERAGE',json.dumps(summaries))
for corpus in ['agnews','airline','climmig','20ng','pheme','goemotions']:
 print('\nCORPUS',corpus)
 ts=all_data[corpus]
 targets={'agnews':['label_name'],'airline':['airline_sentiment'],'climmig':['about_migration','quant_methods'],'20ng':['label_name'],'pheme':['label'],'goemotions':['label_grouped']}[corpus]
 for target in targets:
  pp=[r for r in pairs if r['corpus']==corpus and r['target']==target and r['min_n']>=100]
  for mode,key in [('SUPPORT',lambda r:(r['min_n'],r['total'])),('CONTRAST',lambda r:r['tvd'])]:
   for r in sorted(pp,key=key,reverse=True)[:3]:
    print(mode,target,json.dumps(r))
    for t,d in ts.items():
     if t in ['airline','airline_sentiment','about_migration','quant_methods','label_coarse','label_grouped']:
      print(t,dict(d[r['profile0']]),dict(d[r['profile1']]))
