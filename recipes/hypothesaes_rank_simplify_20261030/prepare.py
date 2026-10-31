from pathlib import Path
import os,json,hashlib,difflib
os.environ['MPLCONFIGDIR']='/tmp/isbc-matplotlib'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];BASE=ROOT/'runs/agnews_embedding_comparison/external_sparse_encoders_qwen_k16_full127600'
z=pd.read_csv(BASE/'hypothesaes/codes.csv').filter(like='latent_').to_numpy();z=np.sort(z,axis=1)[:,::-1];assert z.shape==(127600,16);assert np.all(z[:,6:]==0)
edges=np.linspace(0,z.max(),1801);x=(edges[:-1]+edges[1:])/2;dx=edges[1]-edges[0];colors=plt.get_cmap('tab20')(np.linspace(0,1,16))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(8,4.1),layout='constrained')
for k in range(6):
 v=z[:,k];v=v[v>0];bw=v.std(ddof=1)*len(v)**(-.2);h=np.histogram(v,edges)[0]/(len(z)*dx);y=gaussian_filter1d(h,max(bw/dx,.5),mode='reflect');ax.plot(x,y,color=colors[k],lw=1.6,label=f'Rank {k+1}');ax.fill_between(x,y,color=colors[k],alpha=.2)
ax.set(xlim=(0,edges[-1]),xlabel='Nonnegative continuous activation',ylabel='Continuous density');ax.grid(axis='y',alpha=.18);ax.legend(ncol=3,frameon=False,loc='upper right');ax.set_title('HypotheSAEs: six largest activations',fontsize=13)
for ext in ['pdf','png']:fig.savefig(OUT/f'external_rank_densities_hypothesaes.{ext}',dpi=180)
plt.close(fig)
