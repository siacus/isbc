from pathlib import Path
import csv,collections,hashlib,json,re,html
csv.field_size_limit(10000000)
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
read=lambda p:list(csv.DictReader(Path(p).open()))
rates={r['corpus']:r for r in read(root/'runs/cross_corpus_binary_qwen/code_use.csv')}
prov=json.loads((root/'runs/cross_corpus_binary_qwen/provenance.json').read_text())
chosen={
'agnews':[(8,'0110100001000110'),(6,'0001110001000000'),(6,'1000010000101010')],
'airline':[(1,'1011110000001000'),(14,'0010100000001101')],
'20ng':[(3,'0000011110001010')],
'climmig':[(9,'0100001100010011')],
'pheme':[(7,'1000100010001111')]}
res={};sources=[]
for corpus,pairs in chosen.items():
 ep=Path(rates[corpus]['encoded_path']); eraw=ep.read_bytes(); assert hashlib.sha256(eraw).hexdigest()==rates[corpus]['encoded_sha256']
 meta=next(r for r in prov if r['corpus']==corpus); lp=Path(meta['labels']);assert hashlib.sha256(lp.read_bytes()).hexdigest()==meta['labels_sha256']
 codes={p for k,p in pairs}|{p[:k]+'1'+p[k+1:] for k,p in pairs}
 egroups={p:[] for p in codes}; counts=collections.Counter();ids=set()
 for r in read(ep):
  assert r['id'] not in ids;ids.add(r['id']);p=''.join(r[f'bit_{k:02d}'] for k in range(16));counts[p]+=1
  if p in codes:egroups[p].append({'id':r['id'],'text':r.get('text','')})
 assert len(ids)==int(rates[corpus]['n_documents'])
 labels={r['id']:r for r in read(lp)}; assert len(ids&labels.keys())==len(ids)
 res[corpus]={}
 for p,g in egroups.items():
  res[corpus][p]={'n':len(g),'active_bits':[k for k,b in enumerate(p) if b=='1'],'labels':{}}
  for fp in (root/'runs/cross_corpus_binary_qwen').glob(corpus+'__*__flows.csv'):
   target=fp.stem.split('__')[1]; actual=dict(collections.Counter(labels[r['id']][target] for r in g));saved={r[target]:int(r['documents']) for r in read(fp) if r['profile']==p}
   assert actual==saved,(corpus,p,target);res[corpus][p]['labels'][target]=actual
  if corpus=='agnews':
   patterns={'touchdown':r'\btouchdowns?\b','quarterback':r'\bquarterbacks?\b','touchdown_or_quarterback':r'\b(?:touchdowns?|quarterbacks?)\b','home_run_or_inning':r'\b(?:home runs?|innings?)\b'}
   res[corpus][p]['lexical_document_counts']={name:sum(bool(re.search(pat,html.unescape(r['text']).lower())) for r in g) for name,pat in patterns.items()}
  if corpus=='airline':
   res[corpus][p]['negative_within_airline']={a:{'n':sum(labels[r['id']]['airline']==a for r in g),'negative':sum(labels[r['id']]['airline']==a and labels[r['id']]['airline_sentiment']=='negative' for r in g)} for a in sorted({labels[r['id']]['airline'] for r in g})}
 sources.append({'corpus':corpus,'encoded':str(ep),'encoded_sha256':rates[corpus]['encoded_sha256'],'labels':str(lp),'labels_sha256':meta['labels_sha256']})
(out/'verified_examples.json').write_text(json.dumps(res,indent=2))
(out/'verified_sources.json').write_text(json.dumps(sources,indent=2))
print(json.dumps(res,indent=2))
# Audit the already discussed Mixed comparisons, preserving source-specific denominators.
mp=root/'runs/mixed_embedding_comparison/paper_case_study_qwen'
profiles={r['profile']:r for r in read(mp/'profiles.csv')}; labs=read(mp/'profile_labels.csv')
mixed=[]
for pid0,pid1 in [('P07','P01'),('P09','P02')]:
 a=profiles[pid0];b=profiles[pid1];diff=[i for i,(x,y) in enumerate(zip(a['code'],b['code'])) if x!=y];assert len(diff)==1
 mixed.append({'pair':[pid0,pid1],'bit':diff[0],'states':[a['code'][diff[0]],b['code'][diff[0]]],'counts':[a['documents'],b['documents']],'labels':[r for r in labs if r['profile'] in [pid0,pid1] and r['target'] in ['airline_sentiment','airline_name','climmig_migration']]})
(out/'mixed_verified_comparisons.json').write_text(json.dumps(mixed,indent=2))
print('MIXED',json.dumps(mixed))
