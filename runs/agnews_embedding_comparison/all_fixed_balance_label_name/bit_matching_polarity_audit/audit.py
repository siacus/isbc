from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
meta=json.loads((BASE/'factor_collection_comparison.json').read_text())
nodes=pd.read_csv(BASE/'backbone_bit_matching_nodes.csv'); edges=pd.read_csv(BASE/'backbone_bit_matching_edges.csv')
bits=[f'bit_{k:02d}' for k in range(16)];blocks={};sources=[]
for m in meta['run_order']:
 rec=meta['runs'][m]['encoded'];p=Path(rec['path']);sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==rec['sha256']
 df=pd.read_csv(p,usecols=['id']+bits,dtype={'id':str}).set_index('id').sort_index();assert df.index.is_unique
 if blocks:assert df.index.equals(next(iter(blocks.values())).index)
 blocks[m]=df; sources.append({'model':m,'path':str(p),'sha256':sha})
N=len(df);assert N==127600
X=np.column_stack([blocks[r.model][r.bit].to_numpy(dtype=float) for r in nodes.itertuples()]);index={r.node:i for i,r in enumerate(nodes.itertuples())}
p=X.mean(0);p11=X.T@X/N
phi=(p11-p[:,None]*p[None,:])/np.sqrt((p*(1-p))[:,None]*(p*(1-p))[None,:])
comm={r.node:int(r.activation_community) for r in nodes.itertuples()};out=[]
for r in edges.to_dict('records'):
 a=index[r['model_a']+'::'+r['bit_a']];b=index[r['model_b']+'::'+r['bit_b']]
 u=p[a];v=p[b];both=p11[a,b];table=np.array([[1-u-v+both,v-both],[u-both,both]])
 exp=np.array([1-u,u])[:,None]*np.array([1-v,v])[None,:]
 nz=table>0;mi=float((table[nz]*np.log2(table[nz]/exp[nz])).sum())
 h=lambda z:-z*np.log2(z)-(1-z)*np.log2(1-z)
 nmi=2*mi/(h(u)+h(v));assert abs(nmi-r['activation_nmi'])<1e-9
 out.append(r|{'phi':phi[a,b],'joint_active_share':both,'independence_joint_share':u*v,'negative_phi':bool(phi[a,b]<0),'activation_community_a':comm[nodes.iloc[a]['node']],'activation_community_b':comm[nodes.iloc[b]['node']]})
e=pd.DataFrame(out);e.to_csv(OUT/'signed_edges.csv',index=False)
summary=[]
for matching,g in e.groupby('matching'):
 neg=g[g.phi<0];summary.append({'matching':matching,'edges':len(g),'negative_edges':len(neg),'negative_nmi_weight_share':neg.activation_nmi.sum()/g.activation_nmi.sum(),'median_phi':g.phi.median(),'median_nmi_negative':neg.activation_nmi.median(),'median_nmi_positive':g.loc[g.phi>0,'activation_nmi'].median()})
act=e[e.matching=='activation_nmi_hungarian'];community=[];flips=[]
labelsp=Path(meta['labels']['path']);labels=pd.read_csv(labelsp,usecols=['id',meta['label_column']],dtype=str).set_index('id').reindex(df.index)
for c,ns in nodes.groupby('activation_community'):
 ids=ns.node.tolist();es=act[(act.activation_community_a==c)&(act.activation_community_b==c)]
 j={node:i for i,node in enumerate(ids)};M=len(ids)
 states=np.ones((2**(M-1),M),dtype=np.int8)
 states[:,1:]=1-2*((np.arange(len(states))[:,None]>>np.arange(M-1))&1)
 penalty=np.zeros(len(states));nbad=np.zeros(len(states),int)
 for r in es.itertuples():
  a=j[r.model_a+'::'+r.bit_a];b=j[r.model_b+'::'+r.bit_b];bad=states[:,a]*states[:,b] != (1 if r.phi>0 else -1)
  penalty+=bad*r.activation_nmi;nbad+=bad
 best=int(np.argmin(penalty));votes=np.zeros(N,int);rates={}
 for m,ms in ns.groupby('model'):
  active=blocks[m][ms.bit.tolist()].to_numpy(bool).any(axis=1);votes+=active;rates[m]=float(active.mean())
 sel=votes>=2;cat=labels.loc[sel,meta['label_column']].value_counts(normalize=True).to_dict()
 expected=pd.read_csv(BASE/'backbone_bit_matching_activation_community_summary.csv')
 row={'community':int(c),'nodes':M,'internal_edges':len(es),'negative_internal_edges':int((es.phi<0).sum()),'negative_internal_nmi_share':float(es.loc[es.phi<0,'activation_nmi'].sum()/es.activation_nmi.sum()),'minimum_frustrated_nmi_share':float(penalty[best]/es.activation_nmi.sum()),'unsatisfied_edges_at_min_weight':int(nbad[best]),'exact_sign_alignment_possible':bool(np.min(nbad)==0),'documents_original_rule':int(sel.sum()),'coverage_original_rule':float(sel.mean()),'per_backbone_any_member_rates':rates,'category_shares':cat}
 community.append(row)
 for i,node in enumerate(ids):flips.append({'community':int(c),'node':node,'suggested_flip_for_diagnostic':bool(states[best,i]<0)})
result={'N':N,'edge_summary':summary,'communities':community,'method':'Phi recomputed from saved hard activations aligned by document ID. Edge NMI independently verified. Exact minimum NMI-weighted sign frustration enumerates orientations within each saved activation community, fixing its first node orientation. Orientation is diagnostic only; saved codes and memberships are unchanged.','sources':sources}
(OUT/'audit.json').write_text(json.dumps(result,indent=2));pd.DataFrame(flips).to_csv(OUT/'diagnostic_orientations.csv',index=False)
print(json.dumps(summary,indent=2))
print(pd.DataFrame(community).drop(columns=['per_backbone_any_member_rates','category_shares']).to_string(index=False))
print('ANY MEMBER RATES',json.dumps({r['community']:r['per_backbone_any_member_rates'] for r in community}))
