"""Leiden on positive selected matches; negative matches retained for display."""
from pathlib import Path
import sys,os,json,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/isbc-signed-mpl')
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,networkx as nx,igraph as ig,leidenalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score
from plot_backbone_bit_matching_network import canonical_membership,community_packed_layout
D=Path(__file__).resolve().parent
nodes=pd.read_csv(D.parent/'backbone_bit_matching_nodes.csv').sort_values('node');names=nodes.node.tolist();ix={v:i for i,v in enumerate(names)}
e=pd.read_csv(D/'signed_edges.csv').query("matching=='activation_nmi_hungarian'")
G=nx.Graph();G.add_nodes_from(names)
for r in e.itertuples():G.add_edge(r.model_a+'::'+r.bit_a,r.model_b+'::'+r.bit_b,activation_nmi=r.activation_nmi,phi=r.phi)
P=nx.Graph();P.add_nodes_from(names);P.add_edges_from((a,b,d)for a,b,d in G.edges(data=True)if d['phi']>0)
g=ig.Graph(n=len(names),edges=[(ix[a],ix[b])for a,b in P.edges]);g.es['weight']=[d['activation_nmi']for a,b,d in P.edges(data=True)]
old=nodes.activation_community.tolist()
def stats(mem):
 ds={n:c for n,c in zip(names,mem)};group=[set(n for n in names if ds[n]==c)for c in sorted(set(mem))]
 pos=[d for a,b,d in G.edges(data=True)if d['phi']>0];neg=[d for a,b,d in G.edges(data=True)if d['phi']<0]
 ip=[d for a,b,d in G.edges(data=True)if d['phi']>0 and ds[a]==ds[b]];ineg=[d for a,b,d in G.edges(data=True)if d['phi']<0 and ds[a]==ds[b]]
 return dict(groups=len(group),singletons=sum(len(v)==1 for v in group),disconnected_groups=sum(not nx.is_connected(P.subgraph(v))for v in group),positive_internal_edges=len(ip),negative_internal_edges=len(ineg),positive_weight_inside=sum(v['activation_nmi']for v in ip)/sum(v['activation_nmi']for v in pos),negative_weight_inside=sum(v['activation_nmi']for v in ineg)/sum(v['activation_nmi']for v in neg),ari_original=adjusted_rand_score(old,mem))
runs=[];summary=[];bestmaps={}
for gamma in [.5,1.,2.]:
 cur=[]
 for seed in range(1234,1254):
  p=la.find_partition(g,la.RBConfigurationVertexPartition,weights='weight',resolution_parameter=gamma,seed=seed,n_iterations=-1)
  mem=canonical_membership(p.membership,names)
  q=nx.algorithms.community.modularity(P,[{n for n,c in zip(names,mem)if c==v}for v in set(mem)],weight='activation_nmi',resolution=gamma)
  assert abs(q-p.quality()/(2*sum(g.es['weight'])))<1e-10
  row=dict(gamma=gamma,seed=seed,quality=q,membership=mem);cur.append(row);runs.append(row)
 best=max(cur,key=lambda r:(r['quality'],-r['seed']));aris=[adjusted_rand_score(a['membership'],b['membership'])for i,a in enumerate(cur)for b in cur[i+1:]]
 summary.append(dict(gamma=gamma,seed=best['seed'],quality=best['quality'],mean_seed_ari=float(np.mean(aris)),min_seed_ari=float(min(aris)),**stats(best['membership'])))
 bestmaps[gamma]=dict(zip(names,best['membership']))
mem=bestmaps[1.];nodes['positive_community']=nodes.node.map(mem);nodes.to_csv(D/'positive_community_nodes.csv',index=False)
rows=[]
for c,ns in nodes.groupby('positive_community'):
 nsids=set(ns.node);internal=[(a,b,d)for a,b,d in G.edges(data=True)if a in nsids and b in nsids]
 rows.append(dict(community=int(c),nodes=len(ns),backbones=ns.model.nunique(),positive_components=nx.number_connected_components(P.subgraph(nsids)),positive_edges=sum(d['phi']>0 for a,b,d in internal),negative_edges=sum(d['phi']<0 for a,b,d in internal),members=' '.join(ns.display_bit)))
pd.DataFrame(rows).to_csv(D/'positive_community_summary.csv',index=False)
result=dict(method='Leiden RBConfiguration on positive-phi selected activation matches only, NMI weights, gamma 1 primary, sensitivity .5 and 2, 20 seeds 1234--1253, convergence. All 80 nodes retained. All 51 negative edges displayed but unused by clustering. No label use; no Hungarian rematching; no encoder refit.',positive_graph_components=nx.number_connected_components(P),isolated_nodes=list(nx.isolates(P)),summary=summary,runs=runs,communities=rows,versions={'leidenalg':la.__version__,'igraph':ig.__version__},inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [D/'signed_edges.csv',D.parent/'backbone_bit_matching_nodes.csv']})
(D/'positive_clustering.json').write_text(json.dumps(result,indent=2));pd.DataFrame(summary).to_csv(D/'positive_resolution_sensitivity.csv',index=False)
# Grid-packed local layouts make all groups legible; no metric meaning.
colors={c:plt.get_cmap('tab20')((c-1)%20)for c in sorted(set(mem.values()))}
positions={};groups=sorted(set(mem.values()));ncols=4;nrows=int(np.ceil(len(groups)/ncols))
for j,c in enumerate(groups):
 ns=sorted(n for n in names if mem[n]==c);sub=P.subgraph(ns)
 pos=nx.spring_layout(sub,seed=20261030+c,weight='activation_nmi',iterations=500,scale=1.2)if len(ns)>1 else {ns[0]:np.zeros(2)}
 from plot_backbone_bit_matching_network import resolve_position_collisions
 pos=resolve_position_collisions(pos,.62)
 center=np.array([(j%ncols)*4.,-(j//ncols)*4.])
 if c in (12,13): center=np.array([12.+(-.65 if c==12 else .65),-8.])
 for n in ns:positions[n]=pos[n]+center
labels=nodes.set_index('node').display_bit.to_dict();blue='#2274A5';red='#C44343'
def draw(ax,subset=None):
 ns=set(names)if subset is None else subset
 for a,b,d in sorted(G.edges(data=True),key=lambda x:x[2]['activation_nmi']):
  if a not in ns or b not in ns:continue
  neg=d['phi']<0
  ax.plot([positions[a][0],positions[b][0]],[positions[a][1],positions[b][1]],color=red if neg else blue,ls='--'if neg else '-',lw=.4+5*d['activation_nmi'],alpha=.55 if neg else .85,zorder=1)
 for n in ns:
  x,y=positions[n];ax.scatter(x,y,s=570,c=[colors[mem[n]]],edgecolors='white',linewidths=1.5,zorder=3)
  ax.text(x,y,labels[n],ha='center',va='center',fontsize=9,fontweight='bold',zorder=4,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.5))
 ax.set_aspect('equal');ax.axis('off');ax.margins(.13)
legend=[Line2D([],[],color=blue,lw=2,label='Positive: used for clustering'),Line2D([],[],color=red,ls='--',lw=2,label='Negative: displayed only')]
fig,ax=plt.subplots(figsize=(16,12));draw(ax)
for c in groups:
 xy=np.array([positions[n]for n in names if mem[n]==c]);ax.text(xy[:,0].mean(),xy[:,1].max()+.5,'C'+str(c),ha='center',fontsize=14,fontweight='bold')
r=next(x for x in summary if x['gamma']==1.)
ax.set_title(f"Communities fitted to positive associations\n11 connected communities and 2 isolated bits; {r['negative_internal_edges']}/51 negative edges inside",fontsize=17,pad=25)
fig.legend(handles=legend,loc='lower center',ncol=2,bbox_to_anchor=(.5,.035),frameon=False)
fig.text(.5,.015,'Q: Qwen   M: MiniLM   N: NVIDIA   H: Harrier   G: EmbeddingGemma. Width: NMI. Layout distances are not statistical.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(D/'positive_leiden_negative_overlay.png',dpi=140);fig.savefig(D/'positive_leiden_negative_overlay.pdf');plt.close(fig)
# Isolated panels allow inspection of within-community signs without cross-edge clutter.
fig,axes=plt.subplots(nrows,ncols,figsize=(16,4*nrows),squeeze=False)
for ax,c in zip(axes.flat,groups):
 ns={n for n in names if mem[n]==c};draw(ax,ns);row=next(x for x in rows if x['community']==c)
 ax.set_title(f"C{c}: {len(ns)} bits, {row['backbones']} backbones\n{row['negative_edges']} internal negative edges",fontsize=11)
for ax in list(axes.flat)[len(groups):]:ax.axis('off')
fig.suptitle('Positive-edge communities: internal connections',fontsize=17)
fig.tight_layout(rect=[0,.04,1,.96]);fig.legend(handles=legend,loc='lower center',ncol=2,frameon=False)
fig.savefig(D/'positive_communities_detail.png',dpi=140);fig.savefig(D/'positive_communities_detail.pdf');plt.close(fig)
print(json.dumps({'summary':summary,'communities':rows,'isolated':result['isolated_nodes']},indent=2))
