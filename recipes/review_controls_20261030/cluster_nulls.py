from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/agnews_qwen_all_fixed_balance'
CASE=RUN/'kmeans_business_crosscutting_case_study'
coverage=pd.read_csv(CASE/'conjunction_cluster_coverage_80pct.csv')
recovery=pd.read_csv(CASE/'recovery_summary.csv')
occupancy=[]; conj=[]; summaries=[]; sources={}
for k in [32,64,256]:
    path=RUN/f'kmeans{k}_tech_market_case_study/document_assignments.csv'
    assignments=pd.read_csv(path,dtype={'id':str})
    sizes=assignments[f'kmeans{k}_cluster'].value_counts().sort_index().to_numpy()
    assert len(sizes)==k and sizes.sum()==127600
    sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for _,row in recovery[recovery.k.eq(k)].iterrows():
        n=int(row['target documents'])
        expected=float((-np.expm1(hypergeom.logpmf(0,127600,sizes,n))).sum())
        occupancy.append(dict(k=k,target=row['SAE-defined subset'],support=n,observed=int(row['clusters containing target documents']),expected_occupied_clusters=expected))
    selected=coverage[coverage.k.eq(k)&coverage.degree.eq(3)].reset_index(drop=True)
    assert len(selected)==560 and selected.support.min()>0
    nulls=[]
    for j,row in selected.iterrows():
        n=int(row.support)
        rng=np.random.default_rng(np.random.SeedSequence([1234,k,j]))
        draws=rng.multivariate_hypergeometric(sizes,n,size=2000,method='marginals')
        ordered=np.sort(draws,axis=1)[:,::-1]
        needed=1+(np.cumsum(ordered,axis=1)<.8*n).sum(1)
        expected=float(needed.mean()); nulls.append(expected)
        conj.append(dict(k=k,conjunction=row.conjunction,support=n,observed=int(row.clusters_for_80pct),null_mean=expected,null_sd=float(needed.std()),null_mc_se=float(needed.std()/np.sqrt(2000))))
    summaries.append(dict(k=k,n_conjunctions=len(selected),observed_median=float(selected.clusters_for_80pct.median()),median_null_mean=float(np.median(nulls))))
    print('Completed',summaries[-1],flush=True)
pd.DataFrame(occupancy).to_csv(OUT/'cluster_occupancy_null.csv',index=False)
pd.DataFrame(conj).to_csv(OUT/'conjunction_coverage_null.csv',index=False)
pd.DataFrame(summaries).to_csv(OUT/'conjunction_null_summary.csv',index=False)
(OUT/'cluster_null_protocol.json').write_text(json.dumps(dict(seed=1234,draws_per_conjunction=2000,n_documents=127600,occupancy='Exact expectation sum_c [1-C(N-n_c,m)/C(N,m)] for uniformly reassigning a fixed-size target among documents.',coverage='Marginal permutation reference for each of 560 three-bit conjunctions: multivariate hypergeometric counts preserve cluster sizes and conjunction support. Report median of 560 null means, not the expectation or interval of a joint median statistic. No joint p-value.',sources=sources),indent=2))
print(pd.DataFrame(occupancy).to_string(index=False),flush=True)
