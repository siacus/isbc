from pathlib import Path
import sys,json,hashlib,warnings
from itertools import combinations
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score,f1_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import benchmark_agnews_external_methods as b
from predictive_modeling import build_classifier
from information_retention_extended import posthoc_linear_decoder_metrics_many
from analyze_agnews_kmeans_bit_residuals import binary_entropy,cluster_bit_statistics,entropy_from_counts
OUT=Path(__file__).resolve().parent;RUN=ROOT/'runs/agnews_qwen_all_fixed_balance'
data=b.load_aligned_inputs(b.parse_args())
saved=np.load(OUT/'itq_scores.npz');ix=pd.Index(saved['ids']).get_indexer(data['ids'])
assert (ix>=0).all() and np.array_equal(saved['native_codes'],saved['cached_codes'])
scores=saved['scores'][ix]
native=saved['native_codes'][ix]
order=np.argsort(-scores,axis=1,kind='stable')
top6=np.zeros(native.shape,dtype=np.uint8);np.put_along_axis(top6,order[:,:6],1,axis=1)
assert (top6.sum(1)==6).all()
codes={'ISBC':data['our_hard'],'ITQ':native,'ITQ_top6':top6}
np.savez_compressed(OUT/'codes.npz',ids=data['ids'].astype(str),**codes)
usage=[];perbits=[];metrics=[];predictions=[]
with threadpool_limits(limits=4):
 ols=posthoc_linear_decoder_metrics_many(data['raw'],codes)
 for name,bits in codes.items():
  profile=(bits.astype(np.uint32)*(1<<np.arange(16,dtype=np.uint32))).sum(1)
  unique,inverse,counts=np.unique(profile,return_inverse=True,return_counts=True)
  probs=bits.mean(0);phi=np.corrcoef(bits,rowvar=False)
  H=entropy_from_counts(counts)
  usage.append(dict(method=name,mean_activity=float(bits.sum(1).mean()),sd_activity=float(bits.sum(1).std()),profiles=len(counts),entropy=H,effective_profiles=2**H,constant_bits=int(((probs==0)|(probs==1)).sum()),largest_profile_share=float(counts.max()/len(bits)),median_profile_size=float(np.median(counts)),median_document_profile_size=float(np.median(counts[inverse])),singleton_document_share=float((counts[inverse]==1).mean()),mean_pair_phi=float(phi[np.triu_indices(16,1)].mean()),min_bit_prevalence=float(probs.min()),max_bit_prevalence=float(probs.max()),r2_ols=ols[name]['posthoc_linear_decoder_r2']))
  perbits.extend(dict(method=name,bit=k,prevalence=float(p),entropy=float(binary_entropy(np.array([p]))[0])) for k,p in enumerate(probs))
  model=build_classifier('logistic_regression',1234)
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always',ConvergenceWarning)
   model.fit(bits[data['train_indices']],data['y'][data['train_indices']])
  assert not any(issubclass(w.category,ConvergenceWarning) for w in caught),name
  test=data['test_indices'];truth=data['y'][test];pred=model.predict(bits[test])
  metrics.append(dict(run=name,representation='hard_bits',classifier='logistic_regression',balanced_accuracy=balanced_accuracy_score(truth,pred),macro_f1=f1_score(truth,pred,average='macro'),n_iter=int(model.n_iter_.max())))
  predictions.append(pd.DataFrame(dict(id=data['ids'][test],true_label=truth,predicted_label=pred,run=name,representation='hard_bits',classifier='logistic_regression')))
  print('CODE',usage[-1],flush=True)
metrics,bootstrap=b.add_shared_balanced_accuracy_intervals(pd.DataFrame(metrics),pd.concat(predictions,ignore_index=True),seed=1234,bootstrap_draws=2000)
metrics.to_csv(OUT/'predictive_metrics.csv',index=False);pd.concat(predictions).to_csv(OUT/'predictive_predictions.csv',index=False)
pd.DataFrame(usage).to_csv(OUT/'code_usage.csv',index=False);pd.DataFrame(perbits).to_csv(OUT/'per_bit.csv',index=False)
print('PREDICTION\n',metrics.to_string(index=False),flush=True)

labels=sorted(np.unique(data['y']));yi=pd.Categorical(data['y'],categories=labels).codes
cluster_assign={};cluster_rows=[]
for M in [32,64,256]:
 p=RUN/f'kmeans{M}_tech_market_case_study/document_assignments.csv'
 f=pd.read_csv(p,dtype={'id':str}).set_index('id').loc[data['ids']]
 c=f[f'kmeans{M}_cluster'].to_numpy(int);cluster_assign[M]=c
 for j in range(M):
  ct=np.bincount(yi[c==j],minlength=4)
  cluster_rows.append(dict(M=M,cluster=j,support=int(ct.sum()),purity=float(ct.max()/ct.sum())))
cluster_frame=pd.DataFrame(cluster_rows);cluster_frame.to_csv(OUT/'cluster_label_purity.csv',index=False)
bitrows=[];conjrows=[];summary=[];pairings=list(combinations(range(16),3))
for mi,(name,bits) in enumerate(codes.items()):
 p=bits.mean(0);h=binary_entropy(p)
 for M,c in cluster_assign.items():
  sizes=np.bincount(c,minlength=M)
  _,hc,_,_=cluster_bit_statistics(bits,c,M)
  for k in range(16):
   draws=np.random.default_rng(np.random.SeedSequence([1234,mi,M,k])).multivariate_hypergeometric(sizes,int(bits[:,k].sum()),size=1000,method='marginals')
   null=(binary_entropy(draws/sizes)*(sizes/len(bits))).sum(1)/h[k] if h[k]>0 else np.full(1000,np.nan)
   bitrows.append(dict(method=name,M=M,bit=k,entropy=float(h[k]),conditional_entropy=float(hc[k]),residual_fraction=float(hc[k]/h[k]) if h[k]>0 else np.nan,mutual_information=float(h[k]-hc[k]),null_mean=float(np.nanmean(null))))
 # membership reused across resolutions; no annotation-based group selection
 for group,selected in enumerate(pairings):
  member=np.all(bits[:,selected]==1,axis=1);support=int(member.sum())
  ct=np.bincount(yi[member],minlength=4);purity=float(ct.max()/support) if support else np.nan
  for M,c in cluster_assign.items():
   counts=np.bincount(c[member],minlength=M);ordered=np.sort(counts)[::-1]
   needed=int(1+(np.cumsum(ordered)<.8*support).sum()) if support else np.nan
   conjrows.append(dict(method=name,M=M,group=group,bits=','.join(map(str,selected)),support=support,purity=purity,dominant_label=labels[int(ct.argmax())],clusters_for_80pct=needed,top_cluster_share=float(counts.max()/support) if support else np.nan))
 print('CLUSTER METRICS',name,flush=True)
bf=pd.DataFrame(bitrows);cf=pd.DataFrame(conjrows)
bf.to_csv(OUT/'bit_cluster_information.csv',index=False);cf.to_csv(OUT/'conjunctions.csv',index=False)

# Same support-window description as the existing paper, plus nearest-size pairs.
for (name,M),f in cf.groupby(['method','M'],sort=False):
 clusters=cluster_frame[cluster_frame.M.eq(M)];lo,hi=clusters.support.min(),clusters.support.max()
 window=f[f.support.between(lo,hi)]
 dist=np.abs(np.log(f.support.to_numpy()[:,None]/clusters.support.to_numpy()[None,:]))
 nearest=dist.argmin(1);ratio=f.support.to_numpy()/clusters.support.to_numpy()[nearest]
 valid=(ratio>=.95)&(ratio<=1.05)
 bsub=bf[bf.method.eq(name)&bf.M.eq(M)]
 summary.append(dict(method=name,M=M,mean_bit_residual=bsub.residual_fraction.mean(),null_mean_bit_residual=bsub.null_mean.mean(),median_conjunction_support=f.support.median(),median_clusters_80=f.clusters_for_80pct.median(),support_window_lower=int(lo),support_window_upper=int(hi),n_conjunctions_in_window=len(window),window_mean_purity=window.purity.mean(),cluster_mean_purity=clusters.purity.mean(),n_size_matched=int(valid.sum()),size_matched_conjunction_mean_purity=float(f.purity.to_numpy()[valid].mean()) if valid.any() else np.nan,size_matched_cluster_mean_purity=float(clusters.purity.to_numpy()[nearest][valid].mean()) if valid.any() else np.nan))
pd.DataFrame(summary).to_csv(OUT/'cluster_summary.csv',index=False)
print('CLUSTER SUMMARY\n',pd.DataFrame(summary).to_string(index=False),flush=True)

# Match each ISBC conjunction to nearest-size groups of both ITQ readouts.
# With replacement, all three supports must be within 5%; labels and outcomes unused.
matched=[]
for M in [32,64,256]:
 ref=cf[cf.method.eq('ISBC')&cf.M.eq(M)]
 others={n:cf[cf.method.eq(n)&cf.M.eq(M)].reset_index(drop=True) for n in ['ITQ','ITQ_top6']}
 for _,r in ref.iterrows():
  selections={'ISBC':r};valid=True
  for name,f in others.items():
   ix=np.abs(np.log(f.support.to_numpy()/r.support)).argmin();match=f.iloc[ix]
   if not .95<=match.support/r.support<=1.05:valid=False
   selections[name]=match
  if valid:
   for name,row in selections.items():matched.append(dict(M=M,reference_group=int(r.group),method=name,group=int(row.group),support=int(row.support),purity=float(row.purity),clusters_for_80pct=int(row.clusters_for_80pct)))
matched=pd.DataFrame(matched);matched.to_csv(OUT/'support_matched_method_groups.csv',index=False)
if not matched.empty:
 ms=matched.groupby(['M','method']).agg(n_reference_groups=('reference_group','size'),n_unique_groups=('group','nunique'),median_support=('support','median'),mean_purity=('purity','mean'),median_clusters_80=('clusters_for_80pct','median')).reset_index()
 ms.to_csv(OUT/'support_matched_method_summary.csv',index=False)
 print('MATCHED SUMMARY\n',ms.to_string(index=False),flush=True)

gap=np.take_along_axis(scores,order,axis=1);gap=gap[:,5]-gap[:,6]
protocol=dict(n_documents=127600,seed=1234,methods=['ISBC original','ITQ original','ITQ top-6'],isbc_unchanged=True,itq_exact_reproduction=json.loads((OUT/'recovery.json').read_text()),top6='Six largest signed rotated scores; stable sort, original coordinate index breaks ties; no new rotation optimization.',top6_boundary_ties=int((gap==0).sum()),top6_gap_median=float(np.median(gap)),bootstrap=bootstrap,bit_null_draws=1000,bit_null='Marginal bit prevalence and cluster sizes fixed; multivariate hypergeometric.',conjunctions='All 560 three-bit active conjunctions, other bits unrestricted.',purity_window='Original support-window restriction, not exact support matching.',nearest_cluster_matching='Nearest size with replacement, within 5% of cluster support.',between_method_matching='ISBC conjunctions as reference; nearest support in each ITQ readout with replacement, retained only if both within 5%; labels unused.',input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'itq_scores.npz',b.parse_args().our_encoded,b.parse_args().predictive_split]})
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
print('COMPLETE',flush=True)
