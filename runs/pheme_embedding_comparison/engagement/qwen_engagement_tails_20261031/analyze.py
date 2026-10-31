from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SRC = ROOT/'runs/pheme_embedding_comparison/engagement/interpretation_audit_20261030/joined.csv'
EMB = ROOT/'runs/pheme_qwen_fixed_balance'
d = pd.read_csv(SRC, dtype={'id':str})
ids = pd.read_csv(EMB/'raw_embeddings_pheme_ready_best_ids.csv', dtype=str).id
assert d.id.is_unique and ids.is_unique and set(ids)==set(d.id)
d = d.set_index('id').loc[ids].reset_index()
X = np.load(EMB/'raw_embeddings_pheme_ready_best.npy').astype(float)
assert len(X)==len(d) and np.isfinite(X).all()
U = X/np.linalg.norm(X,axis=1,keepdims=True)
bits = [f'bit_{i:02d}' for i in range(16)]
B = d[bits].to_numpy(dtype=int)
assert np.isin(B,[0,1]).all()
d['profile'] = [''.join(map(str,row)) for row in B]
d['active_bits'] = [','.join(f'{i:02d}' for i in np.flatnonzero(row)) for row in B]
N = len(d)
events = sorted(d.event.unique())
outcomes = ['retweet_count','favorite_count']
groups, specs, thresholds, profile_rows, bit_rows, summaries = {}, {}, [], [], [], []

def cosmean(index):
    n=len(index)
    return float((np.square(U[index].sum(axis=0)).sum()-n)/(n*(n-1))) if n>1 else np.nan

def add(name, mask, **spec):
    groups[name]=np.asarray(mask,dtype=bool)
    specs[name]=spec

add('corpus',np.ones(N,bool),scope='corpus',outcome='both',fraction=1,tail='all')
for scope in ['global','within_event']:
    for outcome in outcomes:
        for q in [.10,.25]:
            masks={t:np.zeros(N,bool) for t in ['low','high']}
            strata=[('all',np.arange(N))] if scope=='global' else [(e,np.flatnonzero(d.event.eq(e))) for e in events]
            for event,index in strata:
                y=d.iloc[index][outcome]
                lo,hi=np.quantile(y,[q,1-q],method='linear')
                masks['low'][index]=y.to_numpy()<=lo
                masks['high'][index]=y.to_numpy()>=hi
                thresholds.append(dict(scope=scope,outcome=outcome,fraction=q,event=event,n_eligible=len(index),low_threshold=lo,high_threshold=hi,n_low=int(masks['low'][index].sum()),n_high=int(masks['high'][index].sum()),overlap=int((masks['low'][index]&masks['high'][index]).sum())))
            for tail,mask in masks.items():
                add(f'{scope}_{outcome}_{int(q*100)}_{tail}',mask,scope=scope,outcome=outcome,fraction=q,tail=tail)

overlaps=[]
for scope in ['global','within_event']:
    for q in [.1,.25]:
        for tail in ['high','low']:
            r=groups[f'{scope}_retweet_count_{int(q*100)}_{tail}']
            f=groups[f'{scope}_favorite_count_{int(q*100)}_{tail}']
            both=r&f
            overlaps.append(dict(scope=scope,fraction=q,tail=tail,retweets=int(r.sum()),favorites=int(f.sum()),both=int(both.sum()),retweet_only=int((r&~f).sum()),favorite_only=int((f&~r).sum()),jaccard=float(both.sum()/(r|f).sum()),share_retweet_also_favorite=float(both.sum()/r.sum()),share_favorite_also_retweet=float(both.sum()/f.sum())))
            for kind,mask in [('both',both),('retweet_only',r&~f),('favorite_only',f&~r)]:
                add(f'{scope}_joint_{int(q*100)}_{tail}_{kind}',mask,scope=scope,outcome='joint',fraction=q,tail=tail,intersection=kind)

base_counts=d.profile.value_counts()
for name,mask in groups.items():
    index=np.flatnonzero(mask); sub=d.iloc[index]; n=len(index)
    counts=sub.profile.value_counts(); p=counts/n
    pair_count=int((counts*(counts-1)).sum())
    entropy=float(-(p*np.log2(p)).sum())
    one=B[index].sum(axis=0)
    hamming=float((2*one*(n-one)).sum()/(n*(n-1)*16)) if n>1 else np.nan
    within_sum=0.; within_pairs=0
    for profile, ix in sub.groupby('profile').groups.items():
        ix=np.asarray(list(ix)); m=len(ix); c=cosmean(ix)
        if m>1:
            within_sum+=c*m*(m-1); within_pairs+=m*(m-1)
        profile_rows.append(dict(group=name,profile=profile,active_bits=d.loc[ix[0],'active_bits'],n=m,share=m/n,corpus_n=int(base_counts[profile]),enrichment=(m/n)/(base_counts[profile]/N),mean_cosine=c,events=json.dumps(d.loc[ix,'event'].value_counts().to_dict()),retweet_median=float(d.loc[ix,'retweet_count'].median()),favorite_median=float(d.loc[ix,'favorite_count'].median())))
    rec=dict(group=name,**specs[name],n=n,corpus_share=n/N,profiles=len(counts),effective_profiles=2**entropy,entropy_bits=entropy,largest_profile_share=float(p.max()),same_profile_probability=pair_count/(n*(n-1)) if n>1 else np.nan,mean_hamming=hamming,mean_cosine=cosmean(index),within_profile_cosine=within_sum/within_pairs if within_pairs else np.nan,within_profile_pair_count=within_pairs//2,mean_active_bits=float(B[index].sum(axis=1).mean()),event_shares=json.dumps(sub.event.value_counts(normalize=True).to_dict()))
    summaries.append(rec)
    for i,bit in enumerate(bits):
        bit_rows.append(dict(group=name,bit=bit,n=n,active=int(one[i]),prevalence=one[i]/n,corpus_prevalence=float(B[:,i].mean()),difference_pp=100*(one[i]/n-B[:,i].mean())))

summary=pd.DataFrame(summaries); profiles=pd.DataFrame(profile_rows); bitdf=pd.DataFrame(bit_rows)
summary.to_csv(OUT/'group_summary.csv',index=False)
profiles.to_csv(OUT/'all_profile_statistics.csv',index=False)
bitdf.to_csv(OUT/'bit_prevalence.csv',index=False)
pd.DataFrame(thresholds).to_csv(OUT/'thresholds.csv',index=False)
pd.DataFrame(overlaps).to_csv(OUT/'outcome_overlap.csv',index=False)
# All documents are retained. Membership columns make every calculation reproducible.
for name,mask in groups.items():
    if name!='corpus': d[name]=mask
d.to_csv(OUT/'all_documents_with_tail_membership.csv',index=False)

# Comparisons within events, and standardized aggregate contrasts using full-corpus event weights.
contrasts=[]
for scope in ['global','within_event']:
    for outcome in outcomes:
        for q in [.1,.25]:
            low=groups[f'{scope}_{outcome}_{int(q*100)}_low']; high=groups[f'{scope}_{outcome}_{int(q*100)}_high']
            for i,bit in enumerate(bits):
                differences=[]; weights=[]
                for event in events:
                    e=d.event.eq(event).to_numpy(); l=low&e; h=high&e
                    diff=float(B[h,i].mean()-B[l,i].mean()) if l.any() and h.any() else np.nan
                    contrasts.append(dict(scope=scope,outcome=outcome,fraction=q,event=event,bit=bit,n_low=int(l.sum()),n_high=int(h.sum()),low_prevalence=float(B[l,i].mean()) if l.any() else np.nan,high_prevalence=float(B[h,i].mean()) if h.any() else np.nan,difference_pp=diff*100))
                    differences.append(diff); weights.append(e.mean())
                contrasts.append(dict(scope=scope,outcome=outcome,fraction=q,event='standardized',bit=bit,n_low=int(low.sum()),n_high=int(high.sum()),low_prevalence=np.nan,high_prevalence=np.nan,difference_pp=float(np.dot(weights,differences)*100)))
pd.DataFrame(contrasts).to_csv(OUT/'within_event_bit_contrasts.csv',index=False)

fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for row,outcome in enumerate(outcomes):
    for col,(metric,title,scale) in enumerate([('same_profile_probability','Same-profile pairs (%)',100),('mean_cosine','Mean Qwen cosine similarity',1)]):
        ax=axes[row,col]
        for j,q in enumerate([.1,.25]):
            vals=[summary.set_index('group').loc[f'global_{outcome}_{int(q*100)}_{t}',metric]*scale for t in ['low','high']]
            ax.plot([0,1],vals,'o-',label=f'{int(q*100)}% tails')
        ax.axhline(summary.iloc[0][metric]*scale,color='gray',ls='--',label='Full corpus')
        ax.set_xticks([0,1],['Lower tail','Upper tail']); ax.set_title(outcome.replace('_count','').capitalize()+': '+title); ax.legend(fontsize=8)
fig.savefig(OUT/'tail_homogeneity.pdf');fig.savefig(OUT/'tail_homogeneity.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for row,outcome in enumerate(outcomes):
    for col,q in enumerate([.1,.25]):
        ax=axes[row,col]
        for tail,color in [('low','#2563a6'),('high','#c84b40')]:
            z=bitdf[bitdf.group.eq(f'global_{outcome}_{int(q*100)}_{tail}')]
            ax.plot(range(16),100*z.prevalence,'o-',label=tail.capitalize(),color=color)
        ax.set_xticks(range(16),[f'{i:02d}' for i in range(16)]);ax.set_ylim(0,100);ax.set_ylabel('Active documents (%)');ax.set_xlabel('Bit');ax.set_title(f'{outcome.replace("_count", "").capitalize()}: {int(q*100)}% tails');ax.legend()
fig.savefig(OUT/'tail_bit_prevalence.pdf');fig.savefig(OUT/'tail_bit_prevalence.png',dpi=180);plt.close(fig)

manifest=dict(encoder='Qwen/Qwen3-Embedding-0.6B, saved ISBC K=16',n=N,embedding_shape=X.shape,quantiles='numpy linear empirical quantiles; inclusive comparisons, all boundary ties retained',sampling='None: all documents in every defined group',profile_order='bit_00 through bit_15',same_profile_probability='sum n_p(n_p-1)/(n(n-1)); ordered distinct-document pairs',mean_hamming='Average fraction of differing coordinates over all distinct-document pairs',mean_cosine='Mean cosine of raw Qwen embeddings over all distinct-document pairs; computed exactly via vector sums',within_profile_cosine='Pair-weighted cosine over distinct-document pairs sharing the same complete profile, within the selected group; singleton profiles contribute no pairs',scope='Exploratory full corpus, train and test included; no outcome generalization claim',event_caution='Corpus retweet inclusion thresholds differ by event (25 Germanwings; 100 others). Within-event analyses use event-specific quantiles.',hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC,EMB/'raw_embeddings_pheme_ready_best.npy',EMB/'raw_embeddings_pheme_ready_best_ids.csv']})
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(summary[summary.scope.isin(['global','corpus']) & summary.outcome.ne('joint')][['group','n','profiles','same_profile_probability','mean_cosine','within_profile_cosine']].to_string(index=False))
print(pd.DataFrame(overlaps).to_string(index=False))
