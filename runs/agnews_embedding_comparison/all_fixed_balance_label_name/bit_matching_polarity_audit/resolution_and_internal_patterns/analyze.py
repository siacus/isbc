from pathlib import Path
import sys,os,json,hashlib,itertools
os.environ.setdefault('MPLCONFIGDIR','/tmp/isbc-signed-mpl')
D=Path(__file__).resolve().parent;A=D.parent;BASE=A.parent;ROOT=BASE.parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,networkx as nx,igraph as ig,leidenalg as la
from sklearn.metrics import adjusted_rand_score
from plot_backbone_bit_matching_network import canonical_membership
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected=[A/'positive_leiden_negative_overlay.pdf',A/'positive_leiden_negative_overlay.png',A/'positive_community_nodes.csv']
before={str(p):sha(p)for p in protected}
nodes=pd.read_csv(A/'positive_community_nodes.csv').sort_values('node');names=nodes.node.tolist();ix={n:i for i,n in enumerate(names)}
edges=pd.read_csv(A/'signed_edges.csv').query("matching=='activation_nmi_hungarian'");pos=edges[edges.phi>0]
G=nx.Graph();G.add_nodes_from(names)
for r in pos.itertuples():G.add_edge(r.model_a+'::'+r.bit_a,r.model_b+'::'+r.bit_b,weight=r.activation_nmi)
g=ig.Graph(n=80,edges=[(ix[a],ix[b])for a,b in G.edges]);g.es['weight']=[d['weight']for a,b,d in G.edges(data=True)]
runs=[];partitions={};summary=[]
for gamma in [1.,1.5,2.,3.,4.,6.,8.]:
 cur=[]
 for seed in range(1234,1254):
  p=la.find_partition(g,la.RBConfigurationVertexPartition,weights='weight',resolution_parameter=gamma,seed=seed,n_iterations=-1)
  cur.append(dict(gamma=gamma,seed=seed,quality=p.quality()/(2*sum(g.es['weight'])),membership=canonical_membership(p.membership,names)))
 runs.extend(cur);best=max(cur,key=lambda r:(r['quality'],-r['seed']));mem=np.array(best['membership']);partitions[gamma]=mem
 sizes=np.bincount(mem)[1:];aris=[adjusted_rand_score(a['membership'],b['membership'])for i,a in enumerate(cur)for b in cur[i+1:]]
 row=dict(gamma=gamma,groups=len(sizes),multinode=int(sum(sizes>1)),singletons=int(sum(sizes==1)),mean_seed_ari=float(np.mean(aris)),min_seed_ari=float(min(aris)),seed=best['seed'])
 for c in (6,10):
  mask=nodes.positive_community.eq(c).to_numpy();row[f'C{c}_destination_groups']=len(set(mem[mask]));row[f'C{c}_memberships']={str(k):nodes.loc[mask&(mem==k),'display_bit'].tolist()for k in sorted(set(mem[mask]))}
 summary.append(row)
# Labels and code matrices are loaded only after all graph partitions have been selected.
meta=json.loads((BASE/'factor_collection_comparison.json').read_text());frames={};bits=[f'bit_{k:02d}'for k in range(16)]
for m in meta['run_order']:
 p=Path(meta['runs'][m]['encoded']['path']);assert sha(p)==meta['runs'][m]['encoded']['sha256'];frames[m]=pd.read_csv(p,usecols=['id']+bits,dtype={'id':str}).set_index('id').sort_index()
idx=frames['qwen'].index;assert all(f.index.equals(idx)for f in frames.values())
p=Path(meta['labels']['path']);assert sha(p)==meta['labels']['sha256'];y=pd.read_csv(p,usecols=['id','label_name'],dtype=str).set_index('id').loc[idx].label_name
cats=['Business','Sci_Tech','Sports','World'];X=np.column_stack([frames[r.model][r.bit].to_numpy(np.int8)for r in nodes.itertuples()])
def composition(sel):
 counts=y[sel].value_counts();n=int(sel.sum());return dict(n=n,**{c:int(counts.get(c,0))for c in cats},**{c+'_pct':float(100*counts.get(c,0)/n)if n else None for c in cats})
def votes_for(mask):
 nn=nodes[mask];v=np.zeros(len(idx),np.int8)
 for m,ns in nn.groupby('model'):v+=frames[m][ns.bit.tolist()].any(axis=1).to_numpy()
 return v
composition_rows=[]
for gamma,mem in partitions.items():
 for group in sorted(set(mem)):
  mask=mem==group;ns=nodes[mask];v=votes_for(mask);sel=v>=(1 if len(ns)==1 else 2)
  composition_rows.append(dict(gamma=gamma,group=int(group),bits=len(ns),backbones=ns.model.nunique(),members=' '.join(ns.display_bit),**composition(sel)))
pd.DataFrame(composition_rows).to_csv(D/'resolution_document_compositions.csv',index=False)
bit_rows=[];consensus=[];patterns=[];conjunctions=[]
for c in (6,10):
 mask=nodes.positive_community.eq(c).to_numpy();ns=nodes[mask];x=X[:,mask];v=votes_for(mask);base=v>=2;cols=ns.display_bit.tolist()
 for k,name in enumerate(cols):
  for state in (0,1):bit_rows.append(dict(community=c,bit=name,state=state,scope='all_documents',**composition(x[:,k]==state)))
  bit_rows.append(dict(community=c,bit=name,state=1,scope='community_selected',**composition(base&(x[:,k]==1))))
 for threshold in range(2,ns.model.nunique()+1):consensus.append(dict(community=c,rule=f'at_least_{threshold}_backbones',**composition(v>=threshold)))
 consensus.append(dict(community=c,rule='all_member_bits_active',**composition(x.all(1))))
 codes=(x*(1<<np.arange(x.shape[1]))).sum(1)
 for code in sorted(set(codes)):
  states=[int(code>>i&1)for i in range(x.shape[1])];sel=codes==code
  patterns.append(dict(community=c,bit_order=' '.join(cols),pattern=''.join(map(str,states)),active=' '.join(name for name,s in zip(cols,states)if s),inactive=' '.join(name for name,s in zip(cols,states)if not s),selected_by_community_rule=bool(base[np.flatnonzero(sel)[0]]),**composition(sel)))
 for size in (2,3):
  for comb in itertools.combinations(range(x.shape[1]),size):conjunctions.append(dict(community=c,bits=' '.join(cols[i]for i in comb),**composition(x[:,comb].all(1))))
for fn,rows in [('individual_bits.csv',bit_rows),('consensus_rules.csv',consensus),('within_community_patterns.csv',patterns),('positive_conjunctions.csv',conjunctions)]:pd.DataFrame(rows).to_csv(D/fn,index=False)
(D/'resolution_sweep.json').write_text(json.dumps(dict(summary=summary,runs=runs,method='Fixed positive selected-edge graph, weighted Leiden RBConfiguration, no exact group-count constraint. 20 seeds, convergence, highest objective at each gamma selected without labels. Only then describe category composition. No encoder refit. Document counts use any active member in >=2 backbones; singleton activity reported separately.',protected_files_before=before),indent=2))
assert all(sha(Path(p))==v for p,v in before.items())
print('RESOLUTION',json.dumps(summary,indent=2));print('BITS',pd.DataFrame(bit_rows).query("state==1 and scope=='all_documents'").to_string(index=False));print('CONSENSUS',pd.DataFrame(consensus).to_string(index=False))
for c,pair in [(6,['Sports','World']),(10,['Business','Sci_Tech'])]:
 pp=pd.DataFrame(patterns);pp=pp[(pp.community==c)&(pp.n>=100)&pp.selected_by_community_rule]
 for cat in pair:print('TOP PATTERNS',c,cat,pp.sort_values(cat+'_pct',ascending=False).head(4).to_string(index=False))
