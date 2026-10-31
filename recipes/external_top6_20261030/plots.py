from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/tmp/isbc-matplotlib'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/agnews_embedding_comparison/external_sparse_encoders_qwen_k16_full127600';OUT=BASE/'top6_extension';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors=plt.get_cmap('tab20')(np.linspace(0,1,16))
def kde(values,edges,weight=1.,reflect=False):
    centers=(edges[:-1]+edges[1:])/2;dx=edges[1]-edges[0]
    if len(values)<2 or np.std(values)==0:return centers,np.zeros(len(centers))
    bw=np.std(values,ddof=1)*len(values)**(-.2)
    hist=np.histogram(values,edges)[0].astype(float)/(len(values)*dx)
    dens=gaussian_filter1d(hist,max(bw/dx,.5),mode='reflect' if reflect else 'constant')*weight
    return centers,dens

g=pd.read_csv(OUT/'document_gaps.csv').gap.to_numpy();edges=np.linspace(0,g.max()+3*np.std(g)*len(g)**(-.2),1401)
x,y=kde(g,edges,reflect=True);fig,ax=plt.subplots(figsize=(7.4,3.8),layout='constrained');ax.fill_between(x,y,color='#4477aa',alpha=.28);ax.plot(x,y,color='#225588',lw=1.8);ax.axvline(np.median(g),color='#bb3333',ls='--',label=f'Median = {np.median(g):.3f}');ax.set(xlabel=r'Sixth minus seventh logit, $\Delta_i=z_{i,(6)}-z_{i,(7)}$',ylabel='Density',xlim=(0,edges[-1]));ax.legend(frameon=False);ax.text(.98,.68,f'{len(g):,} documents\nExact ties: {(g==0).sum()}\nMean = {g.mean():.3f}',transform=ax.transAxes,ha='right',va='top');ax.grid(axis='y',alpha=.18)
for ext in ['pdf','png']:fig.savefig(FIG/f'binarysentemb_top6_gap.{ext}',dpi=180)
plt.close(fig)
rows=[]
for method,title,xlabel in [('ours','ISBC','Stored log-odds (clipped probabilities)'),('hypothesaes','HypotheSAEs','Nonnegative continuous activation'),('binarysentemb','BinarySentEmb','Encoder logit')]:
    z=pd.read_csv(BASE/method/'codes.csv').filter(like='latent_').to_numpy();z=np.sort(z,axis=1)[:,::-1];lo,hi=z.min(),z.max();nonnegative=method=='hypothesaes';atoms=method!='binarysentemb'
    if atoms:
        fig=plt.figure(figsize=(10,6.4),layout='constrained');gs=fig.add_gridspec(2,2,width_ratios=[3.7,1.65]);axs=[fig.add_subplot(gs[j,0]) for j in range(2)];masses=[fig.add_subplot(gs[j,1]) for j in range(2)]
    else:
        fig,axs=plt.subplots(2,1,figsize=(9.2,6.4),sharex=True,layout='constrained')
    if method=='binarysentemb':edges=np.linspace(lo-4,hi+4,1801)
    else:edges=np.linspace(lo,hi,1801)
    for panel,ranks in enumerate([range(6),range(6,16)]):
        ax=axs[panel];masslo=[];masshi=[]
        for k in ranks:
            values=z[:,k];mask=np.ones(len(values),bool)
            if method=='ours':mask=(values!=lo)&(values!=hi);ml=(values==lo).mean();mh=(values==hi).mean()
            elif method=='hypothesaes':mask=values!=0;ml=(values==0).mean();mh=0.
            else:ml=mh=0.
            v=values[mask];x,y=kde(v,edges,weight=mask.mean(),reflect=atoms)
            ax.plot(x,y,color=colors[k],lw=1.5,label=f'Rank {k+1}');ax.fill_between(x,y,color=colors[k],alpha=.20)
            masslo.append(ml);masshi.append(mh)
            rows.append({'method':method,'rank':k+1,'min':float(values.min()),'mean':float(values.mean()),'median':float(np.median(values)),'max':float(values.max()),'lower_mass':ml,'upper_mass':mh,'continuous_mass':float(mask.mean())})
        ax.set_title('Ranks 1–6 (largest values)' if panel==0 else 'Ranks 7–16 (remaining values)',loc='left',fontsize=12)
        ax.set(xlim=(edges[0],edges[-1]),ylabel='Continuous density',xlabel=xlabel);ax.grid(axis='y',alpha=.18)
        ax.set_ylim(0,max(ax.get_ylim()[1]*1.25,1e-6))
        ax.legend(ncol=3 if panel==0 else 5,fontsize=8,loc='upper left',frameon=False)
        if atoms:
            ma=masses[panel];rr=np.array(list(ranks))+1
            ma.barh(rr,masslo,color=[colors[k] for k in ranks],alpha=.5,label='Lower limit' if method=='ours' else 'Zero')
            ma.barh(rr,masshi,left=masslo,color=[colors[k] for k in ranks],hatch='///',alpha=.8,label='Upper limit')
            ma.set(yticks=rr,yticklabels=[str(k) for k in rr],xlim=(0,1.03),xlabel='Probability mass',ylabel='Rank');ma.invert_yaxis();ma.set_title('Clipping limits' if method=='ours' else 'Exact zeros',fontsize=11)
            if method=='ours':
                from matplotlib.patches import Patch
                ma.legend(handles=[Patch(facecolor='gray',alpha=.5,label='Lower'),Patch(facecolor='gray',hatch='///',alpha=.8,label='Upper')],loc='lower right',fontsize=7,ncol=2,frameon=False,bbox_to_anchor=(1,1.0))
                for r,ml,mh in zip(rr,masslo,masshi):ma.text(.99,r,f'{100*(ml+mh):.1f}%',ha='right',va='center',fontsize=8)
            if method=='hypothesaes' and panel==1:
                ax.set_ylim(0,1);ax.text(.5,.48,'All ten ranks are identically zero.\nTheir distributions are point masses,\nshown separately at right.',ha='center',va='center',transform=ax.transAxes)
    fig.suptitle(title,fontsize=14)
    for ext in ['pdf','png']:fig.savefig(FIG/f'external_rank_densities_{method}.{ext}',dpi=180)
    plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'rank_distribution_statistics.csv',index=False)
(FIG/'plot_protocol.json').write_text(json.dumps({'density':'Gaussian KDE on a 1400/1800-bin histogram grid; Scott bandwidth; reflection at finite support boundaries; alpha=.20','atoms':'ISBC clipping endpoints and HypotheSAEs exact zeros excluded from continuous density and shown as probability masses; densities integrate to remaining probability','ranks':'Descending values within each document; coordinate identity can differ between documents','gap':'Nonnegative reflected KDE; no exact sixth-seventh ties'},indent=2))
print('Figures saved',flush=True)
