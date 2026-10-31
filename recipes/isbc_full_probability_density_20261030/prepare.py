from pathlib import Path
import os,json,hashlib,difflib
os.environ['MPLCONFIGDIR']='/tmp/isbc-matplotlib'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
source=ROOT/'runs/agnews_qwen_all_fixed_balance/encoded_agnews_all_ready_best.csv';comparison=ROOT/'runs/agnews_embedding_comparison/external_sparse_encoders_qwen_k16_full127600/ours/codes.csv'
cols=[f'prob_{k:02d}' for k in range(16)];df=pd.read_csv(source,usecols=['id']+cols,dtype={'id':str}).set_index('id');saved=pd.read_csv(comparison,dtype={'id':str});df=df.loc[saved.id];pvals=df[cols].to_numpy(np.float32);assert pvals.shape==(127600,16);assert np.isfinite(pvals).all() and (pvals>=0).all() and (pvals<=1).all()
clipped=np.clip(pvals,1e-7,1-1e-7);logodds=np.log(clipped/(1-clipped));old=saved[[f'latent_{k:02d}' for k in range(16)]].to_numpy();assert np.allclose(logodds,old,rtol=1e-6,atol=1e-6)
order=np.argsort(-old.mean(0),kind='stable');colors=plt.get_cmap('tab20')(np.linspace(0,1,16));edges=np.linspace(0,1,2001);x=(edges[:-1]+edges[1:])/2;dx=edges[1]-edges[0]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(9.4,4.8),layout='constrained');stats=[]
for k in order:
 v=pvals[:,k].astype(float);bw=np.std(v,ddof=1)*len(v)**(-.2);h=np.histogram(v,edges)[0]/(len(v)*dx);y=gaussian_filter1d(h,bw/dx,mode='reflect');assert abs(y.sum()*dx-1)<1e-10
 # Extend the half-bin centres to the exact support boundaries for display.
 ax.plot(np.r_[0,x,1],np.r_[y[0],y,y[-1]],color=colors[k],lw=1.5,label=f'Bit {k:02d}');ax.fill_between(np.r_[0,x,1],np.r_[y[0],y,y[-1]],color=colors[k],alpha=.20)
 stats.append({'bit':f'bit_{k:02d}','mean_probability':float(v.mean()),'bandwidth':bw,'p_le_001':float((v<=.01).mean()),'p_ge_099':float((v>=.99).mean()),'exact_zero':float((v==0).mean()),'exact_one':float((v==1).mean()),'density_integral':float(y.sum()*dx)})
ax.set(xlim=(0,1),ylim=(0,None),xlabel=r'Activation probability $p_{ik}$',ylabel='Smoothed density');ax.set_xticks(np.linspace(0,1,11));ax.grid(axis='y',alpha=.18);ax.legend(ncol=8,fontsize=8,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,borderaxespad=0);fig.suptitle('ISBC: full distributions of activation probabilities',fontsize=13)
for ext in ['pdf','png']:fig.savefig(OUT/f'isbc_full_probability_densities.{ext}',dpi=180)
plt.close(fig);pd.DataFrame(stats).to_csv(OUT/'bit_probability_summary.csv',index=False)
(OUT/'provenance.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'n_documents':127600,'source_probabilities':'Original saved probabilities, without log-odds clipping; IDs and clipped log-odds checked against comparison artifact','density':'Gaussian KDE, Scott bandwidth, 2000-bin approximation, reflection at 0 and 1; all observations including exact endpoints; density integral one for each bit','order':'Same as existing ISBC fixed-bit plot, descending mean stored log-odds','no_fit':True},indent=2))
