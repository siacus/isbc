"""Audit saved Qwen hard-code evidence and fill three missing label probes.

No encoder is retrained. Existing predictions/metrics are never overwritten.
Outputs live in runs/cross_corpus_binary_qwen; all joins use document IDs.
"""
from pathlib import Path
import sys, json, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from predictive_modeling import build_classifier, add_balanced_accuracy_intervals

OUT = ROOT / 'runs/cross_corpus_binary_qwen'
OUT.mkdir(exist_ok=True)
BITS = [f'bit_{i:02d}' for i in range(16)]
NAMES = {'agnews':'AG News','20ng':'20 Newsgroups','airline':'Airline',
         'goemotions':'GoEmotions','fakenewsnet':'FakeNewsNet','pheme':'PHEME',
         'climmig':'Climate Migration'}
ATTR = {'label_name':'Topic', 'label_coarse':'Topic group', 'airline':'Airline',
        'airline_sentiment':'Sentiment','sentiment_fine':'Sentiment refinement',
        'sentiment_x_airline':'Sentiment x airline','label_primary':'Emotion',
        'label_grouped':'Grouped affect','source':'Source collection',
        'label_x_source':'Veracity x source', 'event':'Event',
        'label_fine':'Status and veracity','account_category':'Account category',
        'account_type':'Account type','category_x_language':'Category x language',
        'language':'Language','about_migration':'Migration relevance',
        'climate_hazards':'Hazard combinations','discipline':'Discipline',
        'migration_drivers':'Driver combinations','quant_methods':'Quantitative methods',
        'sentiment':'Sentiment'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path, cols):
    return pd.read_csv(path, usecols=cols, dtype=str, keep_default_na=False)

def main():
    all_results=[]; rates={}; sources=[]; cache={}; full_frames={}
    for root in ['agnews','20ng','airline','goemotions','tweeteval','fakenewsnet','pheme','climmig']:
        paths=sorted((ROOT/'runs'/f'{root}_embedding_comparison').glob('*fixed_balance*/factor_collection_comparison.json'))
        if root=='agnews': paths=[p for p in paths if p.parent.name=='all_fixed_balance_label_name']
        for path in paths:
            meta=json.loads(path.read_text()); run=meta['runs']['qwen']; comp=path.parent
            assert run['training']['k_factors']==16 and run['training']['target_active']==6
            corpus=root if root!='tweeteval' else 'tweeteval_'+comp.name.split('_')[0]
            dataset=NAMES.get(root, 'TweetEval '+comp.name.split('_')[0])
            encoded=Path(run['encoded']['path']); labels=Path(meta['labels']['path']); target=meta['label_column']
            if str(encoded) not in cache:
                frame=read_csv(encoded,['id']+BITS)
                assert frame.id.is_unique
                cache[str(encoded)]=frame
                binary=frame[BITS].to_numpy(dtype=np.uint8)
                assert np.isin(binary,[0,1]).all()
                _,counts=np.unique(binary,axis=0,return_counts=True)
                probs=counts/counts.sum(); entropy=float(-(probs*np.log2(probs)).sum())
                rates[corpus]=dict(corpus=corpus,dataset=dataset,n_documents=len(frame),K=16,A=6,
                    mean_active=float(binary.sum(1).mean()),sd_active=float(binary.sum(1).std()),
                    observed_profiles=len(counts),entropy=entropy,effective_profiles=2**entropy,
                    top10_share=float(np.sort(counts)[-10:].sum()/len(frame)),
                    encoded_path=str(encoded),encoded_sha256=sha(encoded))
            frame=cache[str(encoded)]
            label_frame=read_csv(labels,['id',target]); assert label_frame.id.is_unique
            full=frame.merge(label_frame,on='id',validate='one_to_one')
            assert len(full)==len(frame)
            full['profile']=full[BITS].agg(''.join,axis=1)
            full_frames[(corpus,target)]=full
            counts=full[target].value_counts()
            eligible=full[full[target].map(counts)>=2].copy()
            split_path=comp/'predictive_split.csv'
            split=read_csv(split_path,['id','predictive_split'])
            assert set(split.id)==set(eligible.id)
            data=split.merge(eligible,on='id',validate='one_to_one')
            metrics_path=comp/'predictive_modeling_metrics.csv'
            new_probe=not metrics_path.exists()
            if not new_probe:
                metrics=pd.read_csv(metrics_path)
                row=metrics[(metrics.run=='qwen')&(metrics.representation=='hard_bits')&(metrics.classifier=='logistic_regression')].iloc[0].to_dict()
                association=pd.read_csv(comp/'information_retention_profile_label.csv')
                association=association[association.run=='qwen'].iloc[0]
                assert int(association.n_documents)==len(data)
                ami=float(association.adjusted_mutual_information)
            else:
                row=fit_missing(corpus,target,data)
                ami=float(adjusted_mutual_info_score(data[target],data.profile,average_method='arithmetic'))
            test=data[data.predictive_split=='test']
            assert len(test)==int(row['test_rows']) and len(data)-len(test)==int(row['train_rows'])
            attr=ATTR.get(target,target)
            if target=='label': attr='Veracity' if root=='fakenewsnet' else 'Rumour status'
            if root=='tweeteval':attr=comp.name.split('_')[0].capitalize()
            result=dict(corpus=corpus,dataset=dataset,target=target,attribute=attr,
                n_documents=len(data),n_classes=data[target].nunique(),n_test_classes=test[target].nunique(),
                train_rows=int(row['train_rows']),test_rows=int(row['test_rows']),ami=ami,
                ba=float(row['balanced_accuracy']),ba_low=float(row['balanced_accuracy_ci_low']),
                ba_high=float(row['balanced_accuracy_ci_high']),macro_f1=float(row['macro_f1']),
                new_probe=new_probe,source_dir=str(comp))
            all_results.append(result)
            sources.append(dict(corpus=corpus,target=target,metadata=str(path),
                metadata_sha256=sha(path),labels=str(labels),labels_sha256=sha(labels),
                split=str(split_path),split_sha256=sha(split_path),
                metrics=str(metrics_path) if not new_probe else str(OUT/f'{corpus}_{target}_new_probe.json')))
            print(dataset,attr, 'BA',round(100*result['ba'],2),'AMI',round(ami,3),flush=True)

    # A recorded cross-classification lacking a saved comparison directory.
    base=next(s for s in sources if s['corpus']=='fakenewsnet')
    target='label_x_source'; corpus='fakenewsnet'
    full=cache[rates[corpus]['encoded_path']].merge(read_csv(base['labels'],['id',target]),on='id',validate='one_to_one')
    full['profile']=full[BITS].agg(''.join,axis=1)
    train,test=train_test_split(np.arange(len(full)),test_size=.2,random_state=1234,stratify=full[target])
    full['predictive_split']='train';full.loc[test,'predictive_split']='test'
    full[['id','predictive_split']].to_csv(OUT/'fakenewsnet_label_x_source_split.csv',index=False)
    row=fit_missing(corpus,target,full)
    ami=float(adjusted_mutual_info_score(full[target],full.profile,average_method='arithmetic'))
    all_results.append(dict(corpus=corpus,dataset='FakeNewsNet',target=target,attribute=ATTR[target],
        n_documents=len(full),n_classes=full[target].nunique(),n_test_classes=full.loc[test,target].nunique(),
        train_rows=len(train),test_rows=len(test),ami=ami,ba=row['balanced_accuracy'],
        ba_low=row['balanced_accuracy_ci_low'],ba_high=row['balanced_accuracy_ci_high'],
        macro_f1=row['macro_f1'],new_probe=True,source_dir=str(OUT)))
    sources.append(dict(corpus=corpus,target=target,labels=base['labels'],labels_sha256=base['labels_sha256'],
        split=str(OUT/'fakenewsnet_label_x_source_split.csv'),split_sha256=sha(OUT/'fakenewsnet_label_x_source_split.csv')))
    full_frames[(corpus,target)]=full
    pd.DataFrame(all_results).to_csv(OUT/'label_evidence.csv',index=False)
    pd.DataFrame(rates.values()).to_csv(OUT/'code_use.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(sources,indent=2))
    # Small sufficient tables for plotting, without redistributing source text.
    for (corpus,target),frame in full_frames.items():
        frame.groupby(['profile',target],sort=True).size().rename('documents').reset_index().to_csv(OUT/f'{corpus}__{target}__flows.csv',index=False)
    print('Saved',len(rates),'corpora and',len(all_results),'label evaluations.',flush=True)

def fit_missing(corpus,target,data):
    dest=OUT/f'{corpus}_{target}_new_probe.json'
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(data[['id',target,'predictive_split']+BITS],index=False).to_numpy().tobytes()).hexdigest()
    if dest.exists():
        saved=json.loads(dest.read_text())
        if saved['fingerprint']==fingerprint:return saved['metrics']
    train=data.predictive_split.eq('train').to_numpy(); test=~train
    X=data[BITS].to_numpy(dtype=np.float32); y=data[target].to_numpy()
    model=build_classifier('logistic_regression',1234)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(X[train],y[train])
    yp=model.predict(X[test]); truth=y[test]
    row=dict(run='qwen',representation='hard_bits',classifier='logistic_regression',
        train_rows=int(train.sum()),test_rows=int(test.sum()),
        balanced_accuracy=float(balanced_accuracy_score(truth,yp)),
        macro_f1=float(f1_score(truth,yp,average='macro',zero_division=0)))
    predictions=pd.DataFrame(dict(id=data.loc[test,'id'].to_numpy(),true_label=truth,predicted_label=yp,
        run='qwen',representation='hard_bits',classifier='logistic_regression'))
    predictions.to_csv(OUT/f'{corpus}_{target}_new_predictions.csv',index=False)
    row=add_balanced_accuracy_intervals(pd.DataFrame([row]),predictions,seed=1234).iloc[0].to_dict()
    dest.write_text(json.dumps(dict(fingerprint=fingerprint,metrics=row,iterations=model.n_iter_.tolist(),
        warnings=[str(w.message) for w in caught],classifier_parameters=model.get_params()),indent=2))
    print('Completed additional hard-code probe:',corpus,target,flush=True)
    return row

if __name__=='__main__':main()
