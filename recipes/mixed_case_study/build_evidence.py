"""Verify and summarize the frozen Mixed Qwen profiles for the paper.

No models are trained. Population counts use all profile members, not the
500-document inspection samples. All label denominators remain explicit.
"""
from pathlib import Path
import sys, json, hashlib, os
os.environ.setdefault('MPLCONFIGDIR','/tmp/isbc-matplotlib')
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/mixed_embedding_comparison/paper_case_study_qwen'
FROZEN=ROOT/'runs/mixed_embedding_comparison/frozen_profiles_qwen_seed1234'
OUT.mkdir(exist_ok=True)

def main():
    meta=json.loads((FROZEN/'manifest.json').read_text())
    for key in ['ready','labels','encoded']:
        item=meta['inputs'][key]
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    bits=[f'bit_{i:02d}' for i in range(16)]
    enc=pd.read_csv(ROOT/meta['inputs']['encoded']['path'],usecols=['id']+bits,dtype=str,keep_default_na=False)
    labels=pd.read_csv(ROOT/meta['inputs']['labels']['path'],dtype=str,keep_default_na=False)
    d=enc.merge(labels,on='id',validate='one_to_one')
    assert len(d)==111000
    d['profile']=d[bits].agg(''.join,axis=1)
    membership=pd.read_csv(FROZEN/'profile_membership.csv',dtype={'id':str,'profile_code':str})
    selected=membership.merge(d,on='id',validate='one_to_one')
    assert (selected.profile_code==selected.profile).all()
    counts=d.profile.value_counts()
    assert set(counts.head(10).index)==set(selected.profile_code)
    assert len(selected)==12739
    summary=[];label_rows=[];source_rows=[]
    for pid,g in selected.groupby('profile_id',sort=True):
        n=len(g);code=g.profile.iloc[0]
        assert counts[code]==n
        sources=g.source.value_counts()
        for src,num in sources.items():source_rows.append(dict(profile=pid,source=src,documents=int(num),share=num/n))
        summary.append(dict(profile=pid,code=code,active_bits=','.join(str(i) for i,v in enumerate(code) if v=='1'),
            documents=n,corpus_share=n/len(d),dominant_source=sources.index[0],
            dominant_source_n=int(sources.iloc[0]),dominant_source_share=float(sources.iloc[0]/n)))
        for target in labels.columns.difference(['id','source_id','source','type']):
            valid=g.loc[~g[target].isin(['unknown','']),target]
            for val,num in valid.value_counts().items():
                label_rows.append(dict(profile=pid,target=target,label=val,documents=int(num),
                    annotated_documents=len(valid),share_within_annotated=num/len(valid)))
    pd.DataFrame(summary).to_csv(OUT/'profiles.csv',index=False)
    pd.DataFrame(source_rows).to_csv(OUT/'profile_sources.csv',index=False)
    pd.DataFrame(label_rows).to_csv(OUT/'profile_labels.csv',index=False)
    evaluation=[]
    for target in ['source','type']:
        comp=ROOT/f'runs/mixed_embedding_comparison/all_fixed_balance_{target}'
        r=pd.read_csv(comp/'predictive_modeling_metrics.csv')
        r=r[(r.run=='qwen')&(r.representation=='hard_bits')&(r.classifier=='logistic_regression')].iloc[0]
        ami=adjusted_mutual_info_score(d[target],d.profile,average_method='arithmetic')
        saved=pd.read_csv(comp/'information_retention_profile_label.csv').query("run=='qwen'").iloc[0]
        assert abs(ami-saved.adjusted_mutual_information)<1e-10
        evaluation.append(dict(target=target,n_classes=d[target].nunique(),ami=ami,
            ba=r.balanced_accuracy,ba_low=r.balanced_accuracy_ci_low,ba_high=r.balanced_accuracy_ci_high,macro_f1=r.macro_f1))
        flows=d.groupby(['profile',target]).size().rename('documents').reset_index()
        display_names={'airline':'Airline','fakenewsnet':'FakeNewsNet','climmig':'Climate Migration',
            '20ng':'20 Newsgroups','agnews':'AG News','tweeteval':'TweetEval','ira_io':'IRA',
            'goemotions':'GoEmotions','pheme':'PHEME','tweet':'Tweet','news':'News',
            'abstract':'Abstract','post':'Discussion post','comment':'Comment'}
        flows[target]=flows[target].replace(display_names)
        flows.to_csv(OUT/f'mixed__{target}__flows.csv',index=False)
    pd.DataFrame(evaluation).to_csv(OUT/'evaluation.csv',index=False)
    probs=counts/counts.sum();entropy=float(-(probs*np.log2(probs)).sum())
    rate=dict(n_documents=len(d),mean_active=float(d[bits].astype(int).sum(axis=1).mean()),
        observed_profiles=len(counts),entropy=entropy,effective_profiles=2**entropy,
        displayed_documents=len(selected),displayed_share=len(selected)/len(d),
        source_manifest=str(FROZEN/'manifest.json'),selection='Ten frozen most frequent exact profiles; threshold 0.0045')
    (OUT/'summary.json').write_text(json.dumps(rate,indent=2))

    # Two illustrative source documents, identified during inspection of the
    # existing seeded P05 sample. Their membership is independently checked.
    example_ids=['37042','89685']
    ready=pd.read_csv(ROOT/meta['inputs']['ready']['path'],dtype=str,keep_default_na=False)
    examples=selected[selected.id.isin(example_ids)].merge(ready[['id','text']],on='id',validate='one_to_one')
    assert len(examples)==2 and examples.profile_id.eq('P05').all()
    examples[['id','source_id','source','profile_id','text']].to_csv(OUT/'illustrative_documents.csv',index=False)

    from recipes.cross_corpus_binary import plot_profiles
    import matplotlib.pyplot as plt
    plot_profiles.OUT=OUT
    fig,axes=plt.subplots(1,2,figsize=(13.4,5.2))
    panels=[plot_profiles.panel(ax,'mixed',target,title) for ax,target,title in zip(axes,['source','type'],['Source collection','Document type'])]
    fig.subplots_adjust(left=.02,right=.98,top=.78,bottom=.065,wspace=.11)
    fig.suptitle('Mixed corpus: Qwen binary profiles, source, and document type',fontsize=15,y=.985)
    fig.text(.5,.012,'P1–P10: most frequent complete profiles; numbers list active bit indices (0–15). Ribbon width = document count.',ha='center',fontsize=10)
    fig.savefig(OUT/'mixed_qwen_profile_sources.pdf')
    fig.savefig(OUT/'mixed_qwen_profile_sources.png',dpi=140)
    plt.close(fig)
    (OUT/'figure_manifest.json').write_text(json.dumps(panels,indent=2))
    print(json.dumps(rate,indent=2))
    print('Verified population counts, exact membership, AMIs, and illustrative documents; generated PDF.')

if __name__=='__main__':main()
