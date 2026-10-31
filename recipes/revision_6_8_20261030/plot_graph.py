from pathlib import Path
import pandas as pd,networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
n=pd.read_csv(root/'runs/agnews_qwen_all_fixed_balance/bit_structure/bit_community_nodes.csv').sort_values('bit')
e=pd.read_csv(root/'runs/agnews_embedding_comparison/positive_graph_overlay_audit/qwen_pairs.csv')
p=e[e.phi>0];prev=n.set_index('bit').activation_prevalence.to_dict();mem=n.set_index('bit').community_resolution_1.to_dict();keys=n.bit.tolist()
g=nx.Graph();g.add_nodes_from(keys)
for r in p.itertuples():g.add_edge(r.bit_a,r.bit_b,weight=r.pmi_bits)
layout=nx.spring_layout(g,seed=20261023,weight='weight',iterations=500);colors=['#3389bd','#e2af35','#55a783']
fig,axs=plt.subplots(1,2,figsize=(15,6.8))
for ax,include_negative,title in zip(axs,[False,True],['Positive associations','Positive and negative associations']):
 if include_negative:
  for r in e[e.phi<0].sort_values('phi',key=lambda s:s.abs()).itertuples():
   a,b=layout[r.bit_a],layout[r.bit_b];ax.plot([a[0],b[0]],[a[1],b[1]],color='#ba3434',ls=(0,(4,3)),lw=.3+4*abs(r.phi),alpha=.17+.85*abs(r.phi),zorder=1)
 for r in p.sort_values('phi').itertuples():
  a,b=layout[r.bit_a],layout[r.bit_b];ax.plot([a[0],b[0]],[a[1],b[1]],color='#2274A5',lw=.3+4*abs(r.phi),alpha=.17+.85*abs(r.phi),zorder=1)
 nx.draw_networkx_nodes(g,layout,ax=ax,node_color=[colors[mem[b]] for b in keys],node_size=[220+700*prev[b] for b in keys],edgecolors='white',linewidths=1)
 nx.draw_networkx_labels(g,layout,ax=ax,labels={b:b.replace('bit_','b') for b in keys},font_size=10)
 ax.axis('off');ax.margins(.12);ax.set_title(title,fontsize=14)
handles=[Line2D([],[],color=colors[i],marker='o',lw=0,label=f'C{i+1}') for i in range(3)]+[Line2D([],[],color='#2274A5',label=r'$\phi>0$ (43 edges)'),Line2D([],[],color='#ba3434',ls=(0,(4,3)),label=r'$\phi<0$ (77 edges)')]
fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,fontsize=11)
fig.tight_layout(rect=(0,.08,1,1))
for ext in ['png','pdf']:fig.savefig(out/f'qwen_bit_community_signed_overlay.{ext}',dpi=200,bbox_inches='tight')
