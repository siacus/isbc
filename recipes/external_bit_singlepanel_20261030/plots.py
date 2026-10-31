from pathlib import Path
import os,json,hashlib
os.environ['MPLCONFIGDIR']='/tmp/isbc-matplotlib'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/agnews_embedding_comparison/external_sparse_encoders_qwen_k16_full127600';OUT=Path(__file__).resolve().parent;FIG=OUT
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors=plt.get_cmap('tab20')(np.linspace(0,1,16));rows=[];sources={};orders={}
for method,title,xlabel in [('hypothesaes','HypotheSAEs','Nonnegative continuous activation'),('binarysentemb','BinarySentEmb','Encoder logit')]:
    file=BASE/method/'codes.csv';sources[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    frame=pd.read_csv(file);cols=[f'latent_{k:02d}' for k in range(16)];z=frame[cols].to_numpy();assert z.shape==(127600,16) and np.isfinite(z).all()
    means=z.mean(0);order=np.argsort(-means,kind='stable');orders[method]=order.tolist();lo,hi=z.min(),z.max();atoms=method!='binarysentemb'
    if atoms:
        fig=plt.figure(figsize=(10.4,5.5),layout='constrained');gs=fig.add_gridspec(1,2,width_ratios=[3.7,1.5]);axes=[fig.add_subplot(gs[j,0]) for j in range(1)];massaxes=[fig.add_subplot(gs[j,1]) for j in range(1)]
    else:fig,axes=plt.subplots(1,1,figsize=(9.4,4.8),layout='constrained');axes=[axes]
    pad=4 if not atoms else 0;edges=np.linspace(lo-pad,hi+pad,1801);x=(edges[:-1]+edges[1:])/2;dx=edges[1]-edges[0];ymax=0
    for j,selected in enumerate([order]):
        ax=axes[j];mls=[];mhs=[]
        for k in selected:
            values=z[:,k];mask=np.ones(len(values),bool);ml=mh=0.
            if method=='ours':ml=(values==lo).mean();mh=(values==hi).mean();mask=(values!=lo)&(values!=hi)
            elif method=='hypothesaes':ml=(values==0).mean();mask=values!=0
            v=values[mask];density=np.zeros(len(x))
            if len(v)>1 and np.std(v)>0:
                bw=v.std(ddof=1)*len(v)**(-.2);hist=np.histogram(v,edges)[0]/(len(values)*dx)
                density=gaussian_filter1d(hist,max(bw/dx,.5),mode='reflect' if atoms else 'constant')
            assert abs(density.sum()*dx-mask.mean())<.001
            ymax=max(ymax,float(density.max()));ax.plot(x,density,color=colors[k],lw=1.5,label=f'Bit {k:02d}');ax.fill_between(x,density,color=colors[k],alpha=.20)
            mls.append(ml);mhs.append(mh)
            rows.append({'method':method,'bit':f'bit_{k:02d}','order_by_mean':int(np.flatnonzero(order==k)[0])+1,'mean':float(means[k]),'median':float(np.median(values)),'min':float(values.min()),'max':float(values.max()),'lower_or_zero_mass':float(ml),'upper_mass':float(mh),'continuous_mass':float(mask.mean())})
        ax.set(xlim=(edges[0],edges[-1]),xlabel=xlabel,ylabel='Continuous density')
        ax.legend(ncol=8,fontsize=7,loc='lower left',bbox_to_anchor=(0,1.005),frameon=False,borderaxespad=0);ax.grid(axis='y',alpha=.18)
        if atoms:
            ma=massaxes[j];pos=np.arange(16);ma.barh(pos,mls,color=colors[selected],alpha=.55);ma.barh(pos,mhs,left=mls,color=colors[selected],alpha=.85,hatch='///')
            ma.set(yticks=pos,yticklabels=[f'{k:02d}' for k in selected],xlim=(0,1.02),xlabel='Probability mass',ylabel='Bit');ma.invert_yaxis()
            ma.set_title('Clipping endpoints' if method=='ours' else 'Exact zeros',fontsize=10,pad=38)
            if method=='ours':ma.legend(handles=[Patch(facecolor='gray',alpha=.55,label='Lower'),Patch(facecolor='gray',hatch='///',alpha=.85,label='Upper')],loc='lower left',bbox_to_anchor=(0,1.005),ncol=2,fontsize=8,frameon=False,borderaxespad=0)
            for posi,l,h in zip(pos,mls,mhs):ma.text(.99,posi,f'{100*(l+h):.1f}%',ha='right',va='center',fontsize=8)
    for ax in axes:ax.set_ylim(0,ymax*1.08)
    fig.suptitle(f'{title}: fixed coordinates, ordered by mean',fontsize=13)
    for ext in ['pdf','png']:fig.savefig(FIG/f'external_bit_densities_{method}.{ext}',dpi=180)
    plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'bit_distribution_statistics.csv',index=False)
(OUT/'bit_density_protocol.json').write_text(json.dumps({'n_documents':127600,'inputs_sha256':sources,'bit_orders':orders,'ordering':'Descending full-corpus mean; ties by coordinate index; fixed identity across documents','density':'Binned Gaussian KDE with Scott bandwidth; finite support reflection; opacity .20; same axes within encoder','atoms':'Clipping endpoints for ISBC and exact zeros for HypotheSAEs shown separately; densities integrate to continuous mass','verification':'32 columns and density integrals checked; no encoder or probe fit'},indent=2))
print('Generated two fixed-coordinate figures (PDF/PNG), 32 summaries, verified density masses.')
