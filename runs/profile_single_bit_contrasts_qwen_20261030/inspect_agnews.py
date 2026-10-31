from pathlib import Path
import csv,hashlib,json,random,collections
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
p=root/'runs/agnews_qwen_all_fixed_balance/encoded_agnews_all_ready_best.csv'
expected=next(r for r in csv.DictReader((root/'runs/cross_corpus_binary_qwen/code_use.csv').open()) if r['corpus']=='agnews')['encoded_sha256']
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
chosen=['0110100001000110','0110100011000110','1000010000101010','1000011000101010','0001110001000000','0001111001000000']
groups={c:[] for c in chosen};counts=collections.Counter()
with p.open() as f:
 for r in csv.DictReader(f):
  code=''.join(r[f'bit_{k:02d}'] for k in range(16));counts[code]+=1
  if code in groups:groups[code].append({'id':r['id'],'text':r['text'],'profile':code})
label={r['id']:r['label_name'] for r in csv.DictReader((root/'data/agnews/agnews_all.csv').open())}
flow=collections.defaultdict(collections.Counter)
for r in csv.DictReader((root/'runs/cross_corpus_binary_qwen/agnews__label_name__flows.csv').open()):flow[r['profile']][r['label_name']]=int(r['documents'])
assert counts=={p:sum(c.values()) for p,c in flow.items()}
rows=[]
for c,g in groups.items():
 assert collections.Counter(label[r['id']] for r in g)==flow[c]
 sample=random.Random(1234).sample(sorted(g,key=lambda r:int(r['id'])),min(12,len(g)))
 print('\nPROFILE',c,'N',len(g),'LABELS',dict(flow[c]))
 for r in sample:
  r['label']=label[r['id']];rows.append(r);print(r['id'],r['label'],r['text'][:380].replace('\n',' '))
with (out/'agnews_inspection_sample.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['id','text','profile','label']);w.writeheader();w.writerows(rows)
(out/'agnews_full_selected_groups.json').write_text(json.dumps(groups))
