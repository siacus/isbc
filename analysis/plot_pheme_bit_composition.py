from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
src=root/'runs/pheme_embedding_comparison/engagement/qwen_log_outcomes'
out=src
# Display all categories, not only the most common one, using the report's full-corpus counts.
d=pd.read_csv(src/'all_bits_label_distribution_given_active.csv')
families=[('label',['non-rumour','rumour'],['Non-rumour','Rumour'],'Rumour status'),('veracity',['false','not applicable','true','unverified'],['False','Not applicable','True','Unverified'],'Veracity'),('event',['Charlie Hebdo','Ferguson','Germanwings crash','Ottawa shooting','Sydney siege'],['Charlie Hebdo','Ferguson','Germanwings','Ottawa','Sydney'],'Event')]
fig,axes=plt.subplots(1,3,figsize=(12,7.6),gridspec_kw={'width_ratios':[2,4,5]},layout='constrained')
for ax,(family,cats,names,title) in zip(axes,families):
    sub=d[d.label_family==family]
    vals=sub.pivot(index='bit',columns='category',values='probability_label_given_bit_active')[cats]*100
    assert np.allclose(vals.sum(axis=1),100)
    base=sub.groupby('category').probability_label_unconditional.first()[cats]*100
    data=np.vstack([base.values,vals.values])
    im=ax.imshow(data,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(len(cats)),names,rotation=45,ha='right',fontsize=9)
    ax.set_yticks(range(17),['Corpus']+vals.index.tolist() if ax is axes[0] else ['']*17,fontsize=9)
    ax.set_title(title,fontsize=11)
    ax.axhline(.5,color='black',lw=1)
    for i in range(17):
        for j in range(len(cats)):
            ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if data[i,j]>60 else 'black')
fig.colorbar(im,ax=axes,shrink=.65,label='Category share (%)')
fig.savefig(out/'pheme_labelwise_bit_composition.pdf')
fig.savefig(out/'pheme_labelwise_bit_composition.png',dpi=160)
plt.close(fig)
