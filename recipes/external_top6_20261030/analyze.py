from pathlib import Path
import sys,json,hashlib,subprocess
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import benchmark_agnews_external_methods as b
import compare_external_sparse_profiles as pm
BASE=ROOT/'runs/agnews_embedding_comparison/external_sparse_encoders_qwen_k16_full127600'
OUT=BASE/'top6_extension';OUT.mkdir(exist_ok=True)
args=b.parse_args();data=b.load_aligned_inputs(args)
frames={m:pd.read_csv(BASE/m/'codes.csv',dtype={'id':str}).set_index('id').loc[data['ids']] for m in ['ours','hypothesaes','binarysentemb']}
z=frames['binarysentemb'][[f'latent_{k:02d}' for k in range(16)]].to_numpy(np.float32)
assert np.isfinite(z).all()
order=np.argsort(-z,axis=1,kind='stable');ordered=np.take_along_axis(z,order,axis=1)
hard=np.zeros(z.shape,np.uint8);np.put_along_axis(hard,order[:,:6],1,axis=1)
assert (hard.sum(1)==6).all()
gap=ordered[:,5]-ordered[:,6]
(OUT/'binarysentemb_top6').mkdir(exist_ok=True)
b.save_codes(OUT/'binarysentemb_top6/codes.csv',data['ids'],hard,z)
pd.DataFrame({'id':data['ids'],'sixth_logit':ordered[:,5],'seventh_logit':ordered[:,6],'gap':gap}).to_csv(OUT/'document_gaps.csv',index=False)
stat={'n':len(gap),'min':float(gap.min()),'mean':float(gap.mean()),'median':float(np.median(gap)),'max':float(gap.max()),'exact_ties':int((gap==0).sum()),'q01':float(np.quantile(gap,.01)),'q99':float(np.quantile(gap,.99))}
summary=b.code_summary(hard);(OUT/'diagnostics.json').write_text(json.dumps({'gap':stat,'code':summary,'bit_prevalences':hard.mean(0).tolist(),'tie_rule':'Descending saved float32 logits; lower coordinate index wins exact ties.'},indent=2))
print('DIAGNOSTICS',stat,summary,flush=True)
old=pd.read_csv(BASE/'code_summary.csv');pd.concat([old,pd.DataFrame([{'method':'binarysentemb_top6',**summary}])]).to_csv(OUT/'code_summary.csv',index=False)
if not (OUT/'new_predictive_metrics.csv').exists():
    metrics,pred,importance=b.evaluate_new_representations({('binarysentemb_top6','hard_bits'):hard},data,1234)
    metrics,protocol=b.add_shared_balanced_accuracy_intervals(metrics,pred,seed=1234,bootstrap_draws=2000)
    metrics.to_csv(OUT/'new_predictive_metrics.csv',index=False);pred.to_csv(OUT/'new_predictive_predictions.csv',index=False);importance.to_csv(OUT/'new_feature_importance.csv',index=False)
metrics=pd.concat([pd.read_csv(BASE/'predictive_metrics.csv'),pd.read_csv(OUT/'new_predictive_metrics.csv')],ignore_index=True)
metrics=b.add_binary_to_raw_ratios(metrics);metrics.to_csv(OUT/'predictive_metrics.csv',index=False)
print(metrics[metrics.run.eq('binarysentemb_top6')].to_string(index=False),flush=True)
ckpt=torch.load(BASE/'binarysentemb/model.pt',map_location='cpu',weights_only=False);sd=ckpt['state_dict']
w=sd['decoder.0.weight'].numpy().astype(np.float64);bias=sd['decoder.0.bias'].numpy().astype(np.float64)
raw=data['raw'];mean=raw.mean(0,dtype=np.float64);sse=sst=0.
for start in range(0,len(raw),2048):
    x=np.asarray(raw[start:start+2048],np.float64);pred=(2*hard[start:start+2048].astype(float)-1)@w.T+bias
    sse+=np.square(x-pred).sum();sst+=np.square(x-mean).sum()
ols=b.posthoc_linear_decoder_metrics_many(raw,{'binarysentemb_top6':hard})['binarysentemb_top6']
recon={'method':'binarysentemb_top6','normalized_mse':sse/sst,'r2':1-sse/sst,**ols};(OUT/'reconstruction.json').write_text(json.dumps(recon,indent=2));print('RECONSTRUCTION',recon,flush=True)
manifest=json.loads((BASE/'factor_collection_comparison.json').read_text());manifest['run_order'].append('binarysentemb_top6');manifest['runs']['binarysentemb_top6']={'encoded':{'path':str(OUT/'binarysentemb_top6/codes.csv')}};(OUT/'factor_collection_comparison.json').write_text(json.dumps(manifest,indent=2))
pm.DISPLAY.update(ours='ISBC',binarysentemb_top6='BinarySentEmb top-6')
rows=[]
for method in manifest['run_order']:
    dest=OUT/'profile_matching'/f'from_{method}'
    if not (dest/'invariance.csv').exists():
        print('MATCHING',method,flush=True)
        subprocess.run([sys.executable,'profile_matching/profile_matching.py','--run-dir',str(OUT),'--reference',method,'--min-share','0','--min-top-share','0','--top-reference-profiles','10','--tail-reference-profiles','0','--min-docs','30','--null-draws','1000','--out',str(dest)],cwd=ROOT,check=True)
    rows.append(pm.summarize_reference(method,dest,len(raw)))
pd.DataFrame(rows).to_csv(OUT/'profile_matching_summary.csv',index=False)
inputs=[BASE/m/'codes.csv' for m in frames]+[args.raw_embeddings,args.raw_ids,args.predictive_split,BASE/'binarysentemb/model.pt']
provenance={'inputs':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs},'n':len(raw),'seed':1234,'training':'No encoder refit; classifiers and affine OLS decoder only','tie_rule':'Stable descending sort; lower original coordinate wins ties','bootstrap_draws':2000,'profile_null_draws':1000}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2));print('COMPLETE',flush=True)
