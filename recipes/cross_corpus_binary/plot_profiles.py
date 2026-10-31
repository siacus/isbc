"""Vector Sankey diagrams for the ten most frequent Qwen profiles.

Profile selection is label-blind. Category widths refer only to the displayed
documents. Category aggregation affects plots only, never reported metrics.
"""
from pathlib import Path
import os, textwrap
os.environ.setdefault('MPLCONFIGDIR','/tmp/isbc-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/cross_corpus_binary_qwen'
GROUPS=[
 ('agnews','AG News',[('agnews','label_name','News topic')]),
 ('20ng','20 Newsgroups',[('20ng','label_coarse','Topic group'),('20ng','label_name','Newsgroup')]),
 ('airline','Airline',[('airline','airline','Airline'),('airline','airline_sentiment','Sentiment'),('airline','sentiment_fine','Sentiment refinement')]),
 ('goemotions','GoEmotions',[('goemotions','label_primary','Emotion'),('goemotions','label_grouped','Grouped affect')]),
 ('tweeteval','TweetEval',[(f'tweeteval_{t}','label_name',t.capitalize()) for t in ['sentiment','emotion','hate','offensive']]),
 ('fakenewsnet','FakeNewsNet',[('fakenewsnet','source','Source collection'),('fakenewsnet','label','Veracity')]),
 ('pheme','PHEME',[('pheme','event','Event'),('pheme','label','Rumour status'),('pheme','label_fine','Status and veracity')]),
 ('climmig','Climate Migration',[('climmig','about_migration','Migration relevance'),('climmig','quant_methods','Quantitative methods'),('climmig','discipline','Discipline'),('climmig','sentiment','Sentiment')]),
]

def ribbon(ax,x0,x1,y0,y1,height,color):
    d=(x1-x0)*.48
    pts=[(x0,y0),(x0+d,y0),(x1-d,y1),(x1,y1),(x1,y1+height),
         (x1-d,y1+height),(x0+d,y0+height),(x0,y0+height),(x0,y0)]
    codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,
           MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(pts,codes),facecolor=color,alpha=.55,edgecolor='none'))

def panel(ax,corpus,target,title):
    flows=pd.read_csv(OUT/f'{corpus}__{target}__flows.csv',dtype={'profile':str,target:str},keep_default_na=False)
    counts=flows.groupby('profile').documents.sum().reset_index().sort_values(['documents','profile'],ascending=[False,True])
    profiles=counts.head(10).profile.tolist(); total=int(counts.documents.sum())
    selected=flows[flows.profile.isin(profiles)].copy(); n=int(selected.documents.sum())
    lab_counts=selected.groupby(target).documents.sum().sort_values(ascending=False)
    # Up to ten named categories plus the remaining categories pooled.
    keep=lab_counts.index[:10].tolist(); pooled=len(lab_counts)>10
    selected['label']=selected[target].where(selected[target].isin(keep),'Other labels')
    matrix=selected.pivot_table(index='profile',columns='label',values='documents',aggfunc='sum',fill_value=0).reindex(profiles)
    labs=keep+(['Other labels'] if pooled else [])
    matrix=matrix.reindex(columns=labs,fill_value=0)
    scale=.79/n; gap=.021
    def starts(values):
        used=sum(values)*scale+gap*(len(values)-1)
        y=(1+used)/2; positions=[]
        for value in values:
            height=value*scale;positions.append(y-height);y-=height+gap
        return np.array(positions)
    left=starts(matrix.sum(axis=1).to_numpy()); right=starts(matrix.sum(axis=0).to_numpy())
    palette=plt.get_cmap('tab20'); colors=[palette(i if i>=10 else i*2) for i in range(len(labs))]
    if pooled:colors[-1]='#999999'
    right_offset=right.copy()
    for i,profile in enumerate(profiles):
        offset=left[i]
        for j,label in enumerate(labs):
            height=float(matrix.loc[profile,label])*scale
            if height:
                ribbon(ax,.37,.67,offset,right_offset[j],height,colors[j])
                offset+=height;right_offset[j]+=height
        height=float(matrix.loc[profile].sum())*scale
        ax.add_patch(Rectangle((.35,left[i]),.02,height,color='#526273',lw=0))
        active=','.join(str(k) for k,b in enumerate(profile) if b=='1')
        ax.text(.337,left[i]+height/2, f'P{i+1}: {active}',ha='right',va='center',fontsize=10.2)
    short={'comp.sys.ibm.pc.hardware':'IBM PC hardware','comp.sys.mac.hardware':'Mac hardware',
           'comp.os.ms-windows.misc':'MS Windows','comp.windows.x':'X Windows',
           'soc.religion.christian':'Christianity','talk.politics.mideast':'Middle East politics',
           'rec.sport.hockey':'Hockey','rec.sport.baseball':'Baseball',
           'comp.graphics':'Computer graphics','misc.forsale':'Items for sale',
           'comp':'Computing','talk':'Discussion','rec':'Recreation','sci':'Science','misc':'Sales'}
    pretty_labels=[textwrap.fill(short.get(label,label.replace('_',' ')),width=20,
                                 break_long_words=False,break_on_hyphens=False) for label in labs]
    heights=matrix.sum(axis=0).to_numpy()*scale
    centers=right+heights/2
    text_y=centers.copy()
    for j in range(1,len(labs)):
        separation=.057+.023*(pretty_labels[j-1].count('\n')+pretty_labels[j].count('\n'))
        text_y[j]=min(text_y[j],text_y[j-1]-separation)
    if text_y[-1]<.015:text_y+=.015-text_y[-1]
    for j,label in enumerate(labs):
        height=float(matrix[label].sum())*scale
        ax.add_patch(Rectangle((.67,right[j]),.02,height,color=colors[j],lw=0))
        if abs(text_y[j]-centers[j])>.006:
            ax.plot([.69,.706],[centers[j],text_y[j]],color='#777777',lw=.5)
        ax.text(.713,text_y[j],pretty_labels[j],ha='left',va='center',fontsize=10.2)
    ax.set(xlim=(-.04,1.05),ylim=(-.04,1.06));ax.axis('off')
    ax.set_title(f'{title}\n{n:,} / {total:,} documents ({100*n/total:.1f}%)',fontsize=12,pad=13)
    return dict(corpus=corpus,target=target,total=total,displayed=n,coverage=n/total,
                named_categories=len(keep),other_categories=pooled,
                profiles=[dict(rank=i+1,bits=b,documents=int(matrix.loc[b].sum())) for i,b in enumerate(profiles)])

def main():
    import json
    (OUT/'figures').mkdir(exist_ok=True)
    manifest=[]
    for key,title,panels in GROUPS:
        ncols=1 if len(panels)==1 else 2;nrows=int(np.ceil(len(panels)/ncols))
        fig,axes=plt.subplots(nrows,ncols,figsize=(6.7*ncols,5.2*nrows),squeeze=False)
        for ax,args in zip(axes.ravel(),panels): manifest.append(dict(figure=key,**panel(ax,*args)))
        for ax in axes.ravel()[len(panels):]:ax.axis('off')
        fig.subplots_adjust(left=.02,right=.98,top=.78 if nrows==1 else .88,bottom=.065,wspace=.11,hspace=.36)
        fig.suptitle(f'{title}: Qwen binary profiles and recorded labels',fontsize=13 if ncols==1 else 15,y=.985)
        fig.text(.5,.012,'P1–P10: most frequent complete profiles; numbers list active bit indices (0–15). Ribbon width = document count.',
                 ha='center',fontsize=10)
        dest=OUT/'figures'/f'qwen_corpus_profiles_{key}'
        fig.savefig(dest.with_suffix('.pdf'))
        fig.savefig(dest.with_suffix('.png'),dpi=140)
        plt.close(fig)
    (OUT/'figures/manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Saved nine PDF figures and PNG previews.')

if __name__=='__main__':main()
