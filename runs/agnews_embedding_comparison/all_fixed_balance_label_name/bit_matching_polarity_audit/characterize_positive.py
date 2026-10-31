from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
D=Path(__file__).resolve().parent;BASE=D.parent
meta=json.loads((BASE/'factor_collection_comparison.json').read_text());nodes=pd.read_csv(D/'positive_community_nodes.csv')
bits=[f'bit_{i:02d}'for i in range(16)];frames={};provenance={}
for m in meta['run_order']:
 p=Path(meta['runs'][m]['encoded']['path']);h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==meta['runs'][m]['encoded']['sha256']
 frames[m]=pd.read_csv(p,usecols=['id']+bits,dtype={'id':str}).set_index('id').sort_index();provenance[str(p)]=h
idx=frames['qwen'].index
assert all(x.index.equals(idx)for x in frames.values())
p=Path(meta['labels']['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['labels']['sha256']
labels=pd.read_csv(p,dtype={'id':str}).set_index('id').loc[idx];assert len(labels)==127600
# Use saved encoded texts, ID-aligned to the same documents.
p=Path(meta['runs']['qwen']['encoded']['path']);texts=pd.read_csv(p,usecols=['id','text'],dtype=str).set_index('id').loc[idx].text.fillna('')
rows=[];selections={};samples=[]
for c,g in nodes.groupby('positive_community'):
 votes=np.zeros(len(idx),int)
 for m,mg in g.groupby('model'):votes+=frames[m][mg.bit.tolist()].astype(bool).any(axis=1).to_numpy()
 singleton=len(g)==1;sel=votes>= (1 if singleton else 2);selections[int(c)]=sel
 counts=labels.loc[sel,'label_name'].value_counts();order=counts.index.tolist()
 r=dict(community=int(c),nodes=len(g),backbones=g.model.nunique(),singleton=singleton,n=int(sel.sum()),coverage=float(sel.mean()),categories={x:dict(n=int(counts[x]),pct=float(100*counts[x]/sel.sum()))for x in order},members=g.display_bit.tolist())
 rows.append(r)
 rng=np.random.default_rng(20261030+int(c));inds=rng.choice(np.flatnonzero(sel),min(30,sel.sum()),replace=False)
 for j in inds:samples.append(dict(community=int(c),id=idx[j],label=labels.iloc[j].label_name,text=texts.iloc[j]))
(D/'positive_document_compositions.json').write_text(json.dumps(rows,indent=2))
pd.DataFrame(samples).to_csv(D/'positive_document_samples.csv',index=False)
# All cross-backbone associations for each isolated node, not only selected matches.
X=np.column_stack([frames[r.model][r.bit].to_numpy(float)for r in nodes.itertuples()]);p=X.mean(0)
H=lambda u: -np.where(u>0,u*np.log2(np.maximum(u,1e-300)),0)-np.where(u<1,(1-u)*np.log2(np.maximum(1-u,1e-300)),0)
edges=pd.read_csv(D/'signed_edges.csv').query("matching=='activation_nmi_hungarian'")
assoc=[];single=[]
for c in [12,13]:
 nr=nodes[nodes.positive_community==c].iloc[0];i=nodes.index[nodes.node==nr.node][0];a=X[:,i];u=p[i]
 for j,r in enumerate(nodes.itertuples()):
  if r.model==nr.model:continue
  v=p[j];both=float(a@X[:,j]/len(a));phi=(both-u*v)/np.sqrt(u*(1-u)*v*(1-v))
  joint=np.array([[1-u-v+both,v-both],[u-both,both]]);expected=np.outer([1-u,u],[1-v,v]);nz=joint>0;mi=float((joint[nz]*np.log2(joint[nz]/expected[nz])).sum());nmi=2*mi/(H(u)+H(v))
  selected=((edges.model_a.eq(nr.model)&edges.bit_a.eq(nr.bit)&edges.model_b.eq(r.model)&edges.bit_b.eq(r.bit))|(edges.model_b.eq(nr.model)&edges.bit_b.eq(nr.bit)&edges.model_a.eq(r.model)&edges.bit_a.eq(r.bit))).any()
  assoc.append(dict(node=nr.node,other=r.node,other_display=r.display_bit,phi=phi,nmi=float(nmi),selected=bool(selected)))
 sel=selections[c];act=labels.loc[sel,'label_name'].value_counts(normalize=True)
 lengths=texts.str.split().str.len().to_numpy()
 single.append(dict(community=c,node=nr.node,display=nr.display_bit,n=int(sel.sum()),coverage=float(sel.mean()),category_shares=act.to_dict(),median_words_active=float(np.median(lengths[sel])),median_words_inactive=float(np.median(lengths[~sel]))))
pd.DataFrame(assoc).to_csv(D/'singleton_all_cross_backbone_associations.csv',index=False)
# Full-corpus document-frequency contrasts; terms are exploratory, selected after the graph.
v=CountVectorizer(stop_words='english',ngram_range=(1,2),min_df=100,binary=True,max_features=60000)
T=v.fit_transform(texts);vocab=v.get_feature_names_out();terms=[]
for c in [12,13]:
 sel=selections[c];na=sel.sum();nb=(~sel).sum();aa=np.asarray(T[sel].sum(0)).ravel();bb=np.asarray(T[~sel].sum(0)).ravel()
 ratio=((aa+.5)/(na+1))/((bb+.5)/(nb+1));candidates=np.flatnonzero(aa>=100)
 top=sorted(candidates,key=lambda j:(-ratio[j],-aa[j],vocab[j]))[:30]
 for j in top:terms.append(dict(community=c,term=vocab[j],active_count=int(aa[j]),inactive_count=int(bb[j]),active_pct=float(100*aa[j]/na),inactive_pct=float(100*bb[j]/nb),ratio=float(ratio[j])))
pd.DataFrame(terms).to_csv(D/'singleton_enriched_terms.csv',index=False)
(D/'singleton_summary.json').write_text(json.dumps(single,indent=2))
(D/'characterization_provenance.json').write_text(json.dumps(dict(encoded_hashes=provenance,label_hash=meta['labels']['sha256'],sample_seed='20261030 + community ID',documents='At least two distinct backbones with any active member bit for multi-node communities; active bit for singletons.',terms='Document frequency: sklearn English stopwords, unigrams/bigrams, min_df100, max_features60000, at least100 active occurrences, ranked by smoothed active/inactive prevalence ratio.'),indent=2))
for r in rows:print(r)
for r in single:
 a=pd.DataFrame(assoc).query('node==@r["node"]') if False else pd.DataFrame(assoc)[lambda x:x.node==r['node']]
 print('SINGLETON',r);print('SELECTED',a[a.selected].to_dict('records'));print('TOP POSITIVE',a[a.phi>0].sort_values('nmi',ascending=False).head(3).to_dict('records'))
print(pd.DataFrame(terms).groupby('community').head(12).to_string(index=False))
