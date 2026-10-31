from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
p=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
m=pd.read_csv(p/'metrics.csv'); c=pd.read_csv(p/'paired_contrasts.csv'); imp=pd.read_csv(p/'rf_permutation_importance.csv')
specs=['bits_only_no_controls','bits','veracity','veracity_plus_bits','rumour','rumour_plus_bits','event','event_plus_bits','all_three_labels','all_three_labels_plus_bits']
names=['Bits only','Bits + controls','Veracity','Veracity + bits','Rumour status','Rumour status + bits','Event','Event + bits','All labels','All labels + bits']
def save(fig,stem):
 fig.savefig(p/(stem+'.pdf'),bbox_inches='tight');fig.savefig(p/(stem+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11.5,6),sharey=True)
for ax,out,title in zip(axs,['retweet_count','favorite_count'],['Retweets','Favorites']):
 for est,colour,shift in [('OLS','#2171b5',-.13),('RF log','#d95f02',.13)]:
  d=m[(m.outcome==out)&(m.estimator==est)].set_index('specification').loc[specs]
  ax.errorbar(d.mse,np.arange(10)+shift,xerr=[d.mse-d.ci_low,d.ci_high-d.mse],fmt='o',ms=4,capsize=2,color=colour,label=est)
 base=float(m[(m.outcome==out)&(m.estimator=='Constant')].mse.iloc[0]);ax.axvline(base,color='.4',ls='--',lw=1,label='Constant reference')
 ax.set_title(title);ax.set_xlabel('Test MSE on standardized log(1 + count)');ax.grid(axis='x',alpha=.2)
 ax.set_yticks(range(10),names);ax.set_ylim(9.65,-.65)
fig.legend(*axs[1].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.58,.065),ncol=3,fontsize=9,frameon=False)
fig.text(.5,.01,'All rows except Bits only include follower count and text length. Bars: individual 95% bootstrap intervals.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.14,1,1));save(fig,'pheme_log_model_comparison')
# Predictor perturbation importance: blue = increased error, red = decreased error.
for out in ['retweet_count','favorite_count']:
 d=imp[imp.outcome==out];mat=d.pivot(index='predictor',columns='specification',values='importance').reindex(columns=specs)
 order=sorted(mat.index,key=lambda x:(0 if x.startswith('bit') else 1 if x.startswith('control') else 2,x));mat=mat.loc[order]
 a=mat.to_numpy(); vmax=max(.001,float(np.nanmax(abs(a)))); cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#ededed')
 fig,ax=plt.subplots(figsize=(11,10.5));im=ax.imshow(a,cmap=cmap,norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),aspect='auto')
 labs=[x.replace('control_followers_count','Follower count').replace('control_text_length','Text length').replace('label_rumour','Rumour = yes').replace('veracity_','Veracity: ').replace('event_','Event: ').replace('bit','Bit ').replace('.',' ') for x in order]
 ax.set_yticks(range(len(order)),labs);ax.set_xticks(range(10),names,rotation=55,ha='right');ax.tick_params(length=0)
 for i in range(len(order)):
  for j in range(10):
   if np.isfinite(a[i,j]):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if abs(a[i,j])>.5*vmax else 'black')
 ax.set_title(('Retweets' if out=='retweet_count' else 'Favorites')+': RF permutation importance on the log scale',pad=12)
 fig.colorbar(im,ax=ax,shrink=.75,label='Increase in OOB standardized log-outcome MSE')
 fig.tight_layout();save(fig,'pheme_log_rf_importance_'+out)
if (p/'tree_split_improvement.csv').exists():
 t=pd.read_csv(p/'tree_split_improvement.csv');fig,axs=plt.subplots(3,2,figsize=(11,12))
 for i,var in enumerate(['bits_only','bits_controls','all_variables']):
  for j,out in enumerate(['retweet_count','favorite_count']):
   ax=axs[i,j];d=t[(t.variant==var)&(t.outcome==out)].groupby('feature').improvement.sum().sort_values();v=100*d/d.sum()
   cols=['#3182bd' if x.startswith('bit') else '#e6550d' if x.startswith('control') else '#31a354' for x in v.index]
   labs=[x.replace('control_followers_count','Follower count').replace('control_text_length','Text length').replace('bit','Bit ').replace('event_','Event: ').replace('veracity_','Veracity: ') for x in v.index]
   ax.barh(labs,v,color=cols);ax.set_xlim(0,max(v)*1.24);ax.bar_label(ax.containers[0],fmt='%.1f',padding=3,fontsize=8)
   ax.set_title(('Retweets' if j==0 else 'Favorites')+' | '+{'bits_only':'Bits only','bits_controls':'Bits + controls','all_variables':'Bits + labels + controls'}[var],fontsize=10);ax.set_xlabel('Share of realized training RSS reduction (%)')
 fig.tight_layout(h_pad=2.5);save(fig,'pheme_log_tree_split_importance')
lines=['# PHEME: OLS and random forests on log outcomes','','Same frozen Qwen 16/6 codes, 4,641 training tweets and 1,161 test tweets. Outcome: log1p(count), standardized with training mean and sample SD. OLS predictions reproduced to 1e-8. RF: 1,000 variance-split trees, all predictors considered at each split, minimum node size 10; no test-set tuning. Controls remain raw follower count and text length.','','## Results','','| Outcome | Specification | OLS MSE | RF MSE |','|---|---|---:|---:|']
for out in ['retweet_count','favorite_count']:
 for s,n in zip(['constant']+specs,['Constant']+names):
  d=m[(m.outcome==out)&(m.specification==s)]
  if s=='constant': a=b=d.mse.iloc[0]
  else:a=d[d.estimator=='OLS'].mse.iloc[0];b=d[d.estimator=='RF log'].mse.iloc[0]
  lines.append(f'| {out} | {n} | {a:.4f} | {b:.4f} |')
lines+=['','## Interpretation','','Bits-only RF and OLS have nearly equal error. Their paired RF-minus-OLS intervals include zero for both outcomes. Both improve on the constant reference, with paired intervals below zero. With bits and controls, RF improves favorite MSE over OLS by 0.1223 (paired 95% interval [-0.1988, -0.0509]); the retweet difference is -0.0295 [-0.0835, 0.0210]. This supports additional predictive value from the nonlinear model when controls are included, especially for favorites. It does not identify which nonlinearity or interaction explains the improvement.','','Adding bits to label-and-control RF models lowers the point estimate in all eight comparisons. Paired intervals exclude zero except for retweets with event identity alone. For OLS, the corresponding retweet event and joint-label intervals include zero; all favorite comparisons exclude zero.','','All intervals use the original 2,000 shared test-document bootstrap resamples, conditional on the fixed models and split; no multiplicity adjustment. Descriptive trees use the same log outcome, with unchanged depth/split/leaf/complexity settings, and their split importance allocates actual training RSS reduction, excluding surrogate credit. RF permutation importance is an OOB diagnostic; it does not use the test set.','','The original count-model artifacts are preserved, but this report supersedes their predictive comparison. Log transformation does not guarantee normal residuals.']
(p/'report.md').write_text('\n'.join(lines)+'\n')
print(m[m.specification.isin(['constant','bits_only_no_controls','bits'])].to_string(index=False))
