from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg as la
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
Q=ROOT/'runs/agnews_qwen_all_fixed_balance/bit_structure'
T=ROOT/'runs/agnews_embedding_comparison/qwen_capacity_profile_matching'
N=127600
nodes=pd.read_csv(Q/'bit_community_nodes.csv').sort_values('bit')
e=pd.read_csv(Q/'bit_coactivation.csv')
prev=nodes.set_index('bit').activation_prevalence.to_dict()
mem=nodes.set_index('bit').community_resolution_1.to_dict()
p=e.bit_a.map(prev).to_numpy();q=e.bit_b.map(prev).to_numpy()
e['phi']=(e.joint_activation_count/N-p*q)/np.sqrt(p*(1-p)*q*(1-q))
assert np.allclose(e.phi,e.phi_correlation,equal_nan=False)
assert ((e.pmi_bits>0)==(e.phi>0)).all()
e['internal']=e.bit_a.map(mem)==e.bit_b.map(mem)
pos=e[e.phi>0];neg=e[e.phi<0]
summary={'n_documents':N,'qwen':{'positive':len(pos),'negative':len(neg),'zero':int((e.phi==0).sum()),'positive_internal':int(pos.internal.sum()),'negative_internal':int(neg.internal.sum()),'negative_external':int((~neg.internal).sum()),'most_negative':neg.nsmallest(8,'phi').to_dict('records'),'internal_negative':neg[neg.internal].to_dict('records')}}
# Test changing weights separately from retaining the same positive edge set.
keys=nodes.bit.tolist();ix={b:i for i,b in enumerate(keys)}
g=ig.Graph(n=len(keys),edges=[(ix[r.bit_a],ix[r.bit_b]) for r in pos.itertuples()])
runs=[]
for seed in range(1234,1254):
    part=la.find_partition(g,la.RBConfigurationVertexPartition,weights=pos.phi.tolist(),resolution_parameter=1,seed=seed,n_iterations=-1)
    runs.append({'seed':seed,'quality':part.quality(),'membership':part.membership,'ari_original':adjusted_rand_score(nodes.community_resolution_1,part.membership)})
best=max(runs,key=lambda x:x['quality'])
summary['qwen']['phi_weight_sensitivity']={'best':best,'mean_ari_original':float(np.mean([r['ari_original'] for r in runs])),'mean_seed_ari':float(np.mean([adjusted_rand_score(a['membership'],b['membership']) for i,a in enumerate(runs) for b in runs[i+1:]]))}
G=nx.Graph();G.add_nodes_from(keys)
for r in pos.itertuples():G.add_edge(r.bit_a,r.bit_b,weight=r.pmi_bits)
layout=nx.spring_layout(G,seed=20261023,weight='weight',iterations=500)
colors=['#3389bd','#e2af35','#55a783']
fig,axs=plt.subplots(1,2,figsize=(15,6.7))
for ax,negative,title in zip(axs,[False,True],['Positive associations only','Same communities, with negative associations']):
    drawn=e if negative else pos
    for r in drawn.sort_values('phi',key=lambda s:s.abs()).itertuples():
        x0,y0=layout[r.bit_a];x1,y1=layout[r.bit_b]
        ax.plot([x0,x1],[y0,y1],color='#bd3434' if r.phi<0 else '#3e5263',lw=.25+4*abs(r.phi),alpha=.10+.75*abs(r.phi),zorder=1)
    nx.draw_networkx_nodes(G,layout,ax=ax,node_color=[colors[mem[b]] for b in keys],node_size=[220+700*prev[b] for b in keys],edgecolors='white',linewidths=1)
    nx.draw_networkx_labels(G,layout,ax=ax,labels={b:b.replace('bit_','b') for b in keys},font_size=9)
    ax.set_title(title,fontsize=13);ax.margins(.12);ax.axis('off')
fig.suptitle('Qwen / AG News: 16 bits, target activity 6',fontsize=16)
fig.legend(handles=[Line2D([],[],color=colors[i],marker='o',lw=0,label=f'C{i+1}') for i in range(3)]+[Line2D([],[],color='#3e5263',label='Positive phi'),Line2D([],[],color='#bd3434',label='Negative phi')],loc='lower center',ncol=5,frameon=False)
fig.text(.5,.065,'Original positive-PMI Leiden communities retained. Edge width and opacity show |phi|; no edges filtered by magnitude.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.10,1,.95))
for ext in ['png','pdf']:fig.savefig(OUT/f'qwen_positive_negative_overlay.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
e.to_csv(OUT/'qwen_pairs.csv',index=False)

bn=pd.read_csv(T/'three_way_bit_nodes.csv');be=pd.read_csv(T/'three_way_cross_capacity_bit_edges.csv')
bp=bn.set_index(['configuration','bit']).activation_prevalence.to_dict()
p=np.array([bp[(r.source_configuration,r.source_bit)] for r in be.itertuples()]);q=np.array([bp[(r.target_configuration,r.target_bit)] for r in be.itertuples()])
be['phi']=(be.joint_activation_count/N-p*q)/np.sqrt(p*(1-p)*q*(1-q))
finite=np.isfinite(be.phi_correlation)
assert np.allclose(be.loc[finite,'phi'],be.loc[finite,'phi_correlation'])
assert ((be.pmi_bits>0)==(be.phi>0)).all()
summary['tripartite']={'positive':int((be.phi>0).sum()),'negative':int((be.phi<0).sum()),'zero':int((be.phi==0).sum()),'undefined_phi_in_original':int((~finite).sum()),'pairs':[]}
for (a,b),part in be.groupby(['source_configuration','target_configuration'],sort=False):
    summary['tripartite']['pairs'].append({'source':a,'target':b,'positive':int((part.phi>0).sum()),'negative':int((part.phi<0).sum()),'min_phi':float(part.phi.min()),'max_phi':float(part.phi.max()),'strongest_negative':part.nsmallest(3,'phi').to_dict('records')})
# Separate pairwise panels avoid the 8-to-32 lines crossing the middle node column.
configs=['k8_target6','k16_target6','k32_target6'];names={a:a.replace('k','').replace('_target','/') for a in configs}
order={a:bn[bn.configuration==a].sort_values(['community_resolution_1','activation_prevalence'],ascending=[True,False]).bit.tolist() for a in configs}
cmap=plt.get_cmap('tab20')
fig,axs=plt.subplots(1,3,figsize=(15,10.5))
for ax,((a,b),part) in zip(axs,be.groupby(['source_configuration','target_configuration'],sort=False)):
    positions={}
    for x,c in enumerate([a,b]):
        for i,bit in enumerate(order[c]):positions[(c,bit)]=(x,1-(i+.5)/len(order[c]))
    for r in part.sort_values('phi',key=lambda s:s.abs()).itertuples():
        x0,y0=positions[(a,r.source_bit)];x1,y1=positions[(b,r.target_bit)]
        ax.plot([x0,x1],[y0,y1],color='#bd3434' if r.phi<0 else '#3e5263',lw=.15+3*abs(r.phi),alpha=.06+.7*abs(r.phi),zorder=1)
    for x,c in enumerate([a,b]):
        for r in bn[bn.configuration==c].itertuples():
            px,py=positions[(c,r.bit)]
            ax.scatter(px,py,s=35+110*r.activation_prevalence,color='white',edgecolor=cmap(r.community_resolution_1),lw=1.6,zorder=3)
            ax.text(px+(-.045 if x==0 else .045),py,r.bit.replace('bit_','b'),ha='right' if x==0 else 'left',va='center',fontsize=8)
        ax.text(x,1.035,names[c],ha='center',fontsize=13,fontweight='bold')
    ax.set_xlim(-.22,1.22);ax.set_ylim(-.025,1.07);ax.axis('off')
    ax.text(.5,-.012,f'{(part.phi>0).sum()} positive / {(part.phi<0).sum()} negative edges',ha='center',fontsize=10)
fig.suptitle('Cross-capacity associations: positive edges and negative overlay',fontsize=16)
fig.text(.5,.04,'All 896 cross-capacity pairs shown; widths and opacity encode |phi|. Grey = positive, red = negative.',ha='center',fontsize=10)
fig.text(.5,.018,'Node outlines retain the original within-encoder communities. Community colours do not identify matches across encoders.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.07,1,.96))
for ext in ['png','pdf']:fig.savefig(OUT/f'capacity_positive_negative_overlay.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
be.to_csv(OUT/'cross_capacity_pairs.csv',index=False)
summary['inputs_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Q/'bit_coactivation.csv',Q/'bit_community_nodes.csv',T/'three_way_bit_nodes.csv',T/'three_way_cross_capacity_bit_edges.csv']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
