"""Regenerate the two AG News profile maps from all conjunction members.

This portable replacement implements the paper's reported visualization protocol;
it is not a recovered copy of the unavailable historical plotting script.
Install requirements-figures.txt (Matplotlib and UMAP dependencies).
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from umap import UMAP
ROOT=Path(__file__).resolve().parents[1]
CASES={
 'technology_market':{'bits':[0,5,6,10,12],'ids':['2079','2822','30040','85857','3785'],'n':2432,'profiles':116},
 'political_economy':{'bits':[3,7,8,11,15],'ids':['40589','62621','104216','125602'],'n':1568,'profiles':108},
}
def main():
    out=ROOT/'runs/agnews_qwen_all_fixed_balance/bit_structure/conjunction_profile_maps';out.mkdir(parents=True,exist_ok=True)
    encoded=ROOT/'runs/agnews_qwen_all_fixed_balance/encoded_agnews_all_ready_best.csv'
    labels=ROOT/'data/agnews/agnews_all.csv'
    bits=[f'bit_{i:02d}'for i in range(16)]
    d=pd.read_csv(encoded,usecols=['id']+bits,dtype={'id':str}).merge(pd.read_csv(labels,usecols=['id','label_name'],dtype={'id':str}),on='id',validate='one_to_one')
    assert len(d)==127600 and np.isin(d[bits],[0,1]).all()
    colors={'Business':'#287bb5','Sci_Tech':'#c35186','World':'#55aa9b','Sports':'#e9a332','Tie':'#999999'}
    for name,c in CASES.items():
        sub=d[d[[bits[i]for i in c['bits']]].eq(1).all(axis=1)].copy()
        sub['profile']=sub[bits].astype(str).agg(''.join,axis=1)
        tab=pd.crosstab(sub.profile,sub.label_name).reindex(columns=list(colors)[:4],fill_value=0).sort_index()
        assert len(sub)==c['n'] and len(tab)==c['profiles']
        tab['documents']=tab.sum(axis=1)
        maxima=tab[list(colors)[:4]].max(axis=1)
        tab['dominant']=tab[list(colors)[:4]].idxmax(axis=1)
        tab.loc[tab[list(colors)[:4]].eq(maxima,axis=0).sum(axis=1)>1,'dominant']='Tie'
        matrix=np.array([[int(x)for x in p]for p in tab.index],dtype=float)
        xy=UMAP(n_components=2,metric='hamming',n_neighbors=15,min_dist=.1,init='random',random_state=1234,n_jobs=1,disconnection_distance=np.inf).fit_transform(matrix)
        tab['x']=xy[:,0];tab['y']=xy[:,1]
        fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
        ax.scatter(tab.x,tab.y,s=12*np.sqrt(tab.documents),c=tab.dominant.map(colors),alpha=.85,linewidths=.35,edgecolors='white')
        selected=sub.set_index('id').loc[c['ids']]
        for j,(profile,g)in enumerate(selected.groupby('profile',sort=True)):
            row=tab.loc[profile];ax.scatter([row.x],[row.y],s=12*np.sqrt(row.documents)+22,facecolors='none',edgecolors='black',linewidths=1)
            ax.annotate(', '.join(g.index),xy=(row.x,row.y),xytext=(12,15+13*(j%3)),textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','lw':.6},bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
        ax.set(xlabel='UMAP 1',ylabel='UMAP 2',title=name.replace('_',' ').capitalize())
        ax.legend(handles=[Line2D([],[],color=v,marker='o',linestyle='',label=k.replace('_','/'))for k,v in colors.items()],loc='best',frameon=False)
        for ext in ['pdf','png']:fig.savefig(out/f'agnews_{name}_profile_map.{ext}',dpi=200)
        plt.close(fig);tab.to_csv(out/f'{name}_profiles.csv');selected.to_csv(out/f'{name}_annotated_documents.csv')
    (out/'protocol.json').write_text(json.dumps({'cases':CASES,'seed':1234,'metric':'hamming','n_neighbors':15,'min_dist':.1,'init':'random','weighting':'one observation per distinct complete profile','inputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [encoded,labels]},'status':'portable replacement; exact historical layout not verified'},indent=2))
if __name__=='__main__':main()
