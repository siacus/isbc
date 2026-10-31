from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
ROOT=P.parents[3]
d=pd.read_csv(P/'all_documents_with_tail_membership.csv',dtype={'id':str,'profile':str})
s=pd.read_csv(P/'group_summary.csv').set_index('group')
b=pd.read_csv(P/'bit_prevalence.csv')
p=pd.read_csv(P/'all_profile_statistics.csv',dtype={'profile':str})
c=pd.read_csv(P/'within_event_bit_contrasts.csv')
o=pd.read_csv(P/'outcome_overlap.csv')
t=pd.read_csv(P/'thresholds.csv')
raw=ROOT/'runs/pheme_qwen_fixed_balance'
ids=pd.read_csv(raw/'raw_embeddings_pheme_ready_best_ids.csv',dtype=str).id
assert list(ids)==list(d.id)
X=np.load(raw/'raw_embeddings_pheme_ready_best.npy').astype(float)
U=X/np.linalg.norm(X,axis=1,keepdims=True)
def cosine(ix):
    n=len(ix)
    return (np.square(U[ix].sum(0)).sum()-n)/(n*(n-1)) if n>1 else np.nan
profile_comparisons=[]; overlap=[]
for scope in ['global','within_event']:
 for outcome in ['retweet_count','favorite_count']:
  for q in [10,25]:
   low=d[f'{scope}_{outcome}_{q}_low']; high=d[f'{scope}_{outcome}_{q}_high']
   lp=set(d.loc[low,'profile']); hp=set(d.loc[high,'profile']); common=lp&hp
   crossnum=0.; crossden=0
   for profile,frame in d.groupby('profile'):
    il=frame.index[low.loc[frame.index]].to_numpy(); ih=frame.index[high.loc[frame.index]].to_numpy()
    if not len(il) and not len(ih):continue
    cross=float(U[il].sum(0)@U[ih].sum(0)/(len(il)*len(ih))) if len(il) and len(ih) else np.nan
    if np.isfinite(cross):crossnum+=cross*len(il)*len(ih);crossden+=len(il)*len(ih)
    profile_comparisons.append(dict(scope=scope,outcome=outcome,fraction=q/100,profile=profile,active_bits=frame.active_bits.iloc[0],n_corpus=len(frame),n_low=len(il),n_high=len(ih),low_cosine=cosine(il),high_cosine=cosine(ih),cross_tail_cosine=cross))
   overlap.append(dict(scope=scope,outcome=outcome,fraction=q/100,low_profiles=len(lp),high_profiles=len(hp),shared_profiles=len(common),low_docs_in_shared_profiles=int((low&d.profile.isin(common)).sum()),low_coverage=float(d.loc[low,'profile'].isin(common).mean()),high_docs_in_shared_profiles=int((high&d.profile.isin(common)).sum()),high_coverage=float(d.loc[high,'profile'].isin(common).mean()),same_profile_cross_tail_cosine=crossnum/crossden))
pd.DataFrame(profile_comparisons).to_csv(P/'profile_low_high_comparisons.csv',index=False)
ov=pd.DataFrame(overlap);ov.to_csv(P/'profile_tail_overlap.csv',index=False)

# Joint-tail bit contrasts: no refitting, no resampling.
fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
for ax,q in zip(axes,[10,25]):
 for kind,label in [('both','Both outcomes'),('retweet_only','Retweet tail only'),('favorite_only','Favorite tail only')]:
  z=b[b.group.eq(f'global_joint_{q}_high_{kind}')]
  ax.plot(range(16),z.prevalence*100,'o-',label=label)
 ax.set_xticks(range(16),[f'{i:02d}' for i in range(16)]);ax.set_ylim(0,100);ax.set_title(f'Upper {q}% tails');ax.set_xlabel('Bit');ax.set_ylabel('Active documents (%)');ax.legend(fontsize=8)
fig.savefig(P/'joint_tail_bits.pdf');fig.savefig(P/'joint_tail_bits.png',dpi=180);plt.close(fig)

def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'

report=['# Qwen / PHEME: analisi completa delle code di engagement\n',
'## Disegno e definizioni\n',
'Sono inclusi tutti i 5.802 documenti, con encoder e codici Qwen salvati: nessun campionamento, riaddestramento o modifica al paper. Le analisi sono esplorative sull’intero corpus, includendo sia training sia test del precedente esercizio predittivo. Non costituiscono una nuova validazione fuori campione.\n',
'Retweet e favorite sono analizzati prima separatamente. Le code inferiori e superiori al 10% e al 25% usano i quantili empirici interpolati linearmente, includendo tutti i pari merito alla soglia. Per questo alcune numerosità superano il taglio nominale. La scala logaritmica non è necessaria per selezionare le code perché preserva l’ordinamento.\n',
'**Omogeneità del codice**: probabilità che due documenti distinti del gruppo condividano il profilo completo, $\\sum_p n_p(n_p-1)/[n(n-1)]$. Valori maggiori indicano maggiore concentrazione su profili ricorrenti. Questa normalizzazione evita la crescita meccanica del conteggio di coppie con la numerosità; non rende invarianti i risultati all’ampliamento della coda. Riportiamo separatamente numero di profili, entropia e profili effettivi nei CSV, senza interpretarli come misure di significato.\n',
'**Omogeneità dei testi nello spazio Qwen**: cosine media su tutte le coppie di documenti distinti. La media entro profilo considera tutte le coppie che condividono il profilo, pesandole allo stesso modo: i profili grandi contribuiscono più coppie, quelli singleton nessuna. È una misura geometrica della somiglianza dei testi, non una validazione manuale della loro coerenza semantica e non un criterio indipendente dal backbone da cui sono appresi i bit. Non ha una soglia universale di “buono”.\n',
'Sono disponibili due definizioni delle code: soglie globali, richieste dall’analisi principale, e soglie separate entro ciascun evento. Le seconde mantengono approssimativamente la composizione del corpus, salvo ties. I contrasti dei bit entro evento sono inoltre mediati usando le quote degli eventi nell’intero corpus. Non viene effettuato alcun controllo per autore, follower o momento di raccolta.\n',
'## 1. Analisi indipendenti degli outcome\n']
rows=[]
for _,r in t[t.scope.eq('global')].iterrows():
 rows.append([r.outcome.replace('_count',''),f'{100*r.fraction:.0f}%',f'≤ {r.low_threshold:g}',int(r.n_low),f'≥ {r.high_threshold:g}',int(r.n_high)])
report.append(table(['Outcome','Taglio','Soglia bassa','N basso','Soglia alta','N alto'],rows))
rows=[]
for outcome in ['retweet_count','favorite_count']:
 for q in [10,25]:
  for tail in ['low','high']:
   z=s.loc[f'global_{outcome}_{q}_{tail}']
   rows.append([outcome.replace('_count',''),f'{q}% {tail}',int(z.n),int(z.profiles),f'{100*z.same_profile_probability:.2f}%',f'{z.mean_cosine:.3f}',f'{z.within_profile_cosine:.3f}'])
report.append(table(['Outcome','Coda','Documenti','Profili','Coppie stesso profilo','Cosine tutti','Cosine entro profilo'],rows))
report.append('Il corpus completo contiene 303 profili; la probabilità di coincidenza è 4,61%, la cosine media 0,376 e quella entro profilo 0,564. Le code non formano un unico tipo di documento. Le code basse al 10% sono molto concentrate nei codici, ma questa concentrazione si attenua passando al 25%. La somiglianza Qwen complessiva non segue necessariamente la concentrazione dei profili.\n')
report.append('### Quanto conta l’evento\nLa coda bassa globale dei retweet al 10% contiene 220 tweet con il profilo `{04,06,09,11,12,13,15}`, tutti Germanwings: 36,36% della coda. Il corpus ammette retweet da 25 per questo evento e da 100 per gli altri. La concentrazione globale combina quindi contenuto/evento e regole di raccolta.\n')
rows=[]
for outcome in ['retweet_count','favorite_count']:
 for q in [10,25]:
  lo=s.loc[f'within_event_{outcome}_{q}_low']; hi=s.loc[f'within_event_{outcome}_{q}_high']
  rows.append([outcome.replace('_count',''),f'{q}%',int(lo.n),int(hi.n),f'{100*lo.same_profile_probability:.2f}%',f'{100*hi.same_profile_probability:.2f}%',f'{lo.mean_cosine:.3f}',f'{hi.mean_cosine:.3f}'])
report.append(table(['Outcome','Taglio entro evento','N basso','N alto','Stesso profilo basso','Stesso profilo alto','Cosine bassa','Cosine alta'],rows))
report.append('Con soglie entro evento, le code alte hanno maggiore concentrazione di profili delle code basse per entrambi gli outcome e tagli. Per i favorite, tuttavia, la cosine complessiva è maggiore nelle code basse: omogeneità dei profili e somiglianza media dei testi misurano aspetti distinti.\n')
report.append('### Frequenze dei bit\nDifferenze alta meno bassa, in punti percentuali, con code definite entro evento e media pesata per la composizione del corpus. Sono mostrate tutte le coordinate, senza selezione in base al risultato.\n')
rows=[]
for bit in [f'bit_{i:02d}' for i in range(16)]:
 row=[bit]
 for outcome in ['retweet_count','favorite_count']:
  for q in [.1,.25]:
   z=c[(c.scope=='within_event')&(c.event=='standardized')&(c.outcome==outcome)&(c.fraction==q)&(c.bit==bit)]
   row.append(f'{z.difference_pp.iloc[0]:+.2f}')
 rows.append(row)
report.append(table(['Bit','Retweet 10%','Retweet 25%','Favorite 10%','Favorite 25%'],rows))
report.append('Nei retweet i contrasti medi entro evento sono piccoli, tutti entro circa 3,5 punti. Nei favorite persistono contrasti più netti e concordi fra 10% e 25%: bit 02 e 03 più frequenti nella coda alta, bit 07 meno frequente. Sono associazioni aggregate condizionate all’evento; non effetti causali o un significato univoco della coordinata. I dettagli per ciascun evento sono nel CSV dedicato.\n')
report.append('### Gli stessi profili compaiono nelle code opposte?\n')
rows=[]
for _,r in ov[ov.scope.eq('global')].iterrows():
 rows.append([r.outcome.replace('_count',''),f'{100*r.fraction:.0f}%',int(r.shared_profiles),f'{100*r.low_coverage:.1f}%',f'{100*r.high_coverage:.1f}%',f'{r.same_profile_cross_tail_cosine:.3f}'])
report.append(table(['Outcome','Taglio','Profili comuni','Documenti bassi coperti','Documenti alti coperti','Cosine tra code, stesso profilo'],rows))
report.append('La copertura conta i documenti della coda il cui profilo è presente anche nella coda opposta. Il confronto tra code entro lo stesso profilo usa tutte le coppie incrociate, senza campionamento. Un profilo ricorrente può contenere sia engagement elevato sia basso: l’interesse è nella diversa frequenza e nel contenuto dei gruppi, non in una separazione deterministica dell’engagement.\n')
report.append('## 2. Sovrapposizione condizionata alle analisi precedenti\nLe code alte già definite sono incrociate senza cambiare soglie. “Solo retweet” significa nella coda alta dei retweet ma fuori dalla coda alta dei favorite; non significa necessariamente pochi favorite. Lo stesso vale nella direzione opposta.\n')
rows=[]
for _,r in o[o.scope.eq('global')&o['tail'].eq('high')].iterrows():
 rows.append([f'{100*r.fraction:.0f}%',int(r.retweets),int(r.favorites),int(r.both),int(r.retweet_only),int(r.favorite_only),f'{100*r.share_retweet_also_favorite:.1f}%'])
report.append(table(['Taglio','Top retweet','Top favorite','Entrambi','Solo retweet','Solo favorite','Quota top retweet in entrambi'],rows))
rows=[]
for q in [10,25]:
 for kind in ['both','retweet_only','favorite_only']:
  z=s.loc[f'global_joint_{q}_high_{kind}']
  rows.append([f'{q}%',kind,int(z.n),int(z.profiles),f'{100*z.same_profile_probability:.2f}%',f'{z.mean_cosine:.3f}',f'{z.within_profile_cosine:.3f}'])
report.append(table(['Taglio','Gruppo','N','Profili','Stesso profilo','Cosine tutti','Cosine entro profilo'],rows))
report.append('Il gruppo alto per entrambi gli outcome non è quello più omogeneo. Il gruppo solo-favorite è più concentrato nei profili e più simile nello spazio Qwen rispetto al gruppo comune, a entrambi i tagli. La composizione per evento è disponibile per tutti i gruppi e profili. Questa differenza descrittiva va letta insieme alla composizione, non come una categoria semantica già identificata.\n')
report.append('### Profili più frequenti nei tre gruppi (10%)\nPrime cinque configurazioni per numerosità in ciascun gruppo; tutti i profili, inclusi i singleton, restano nei file. La selezione serve soltanto a rendere leggibile questa tabella, non entra nei calcoli.\n')
rows=[]
for kind in ['both','retweet_only','favorite_only']:
 for _,r in p[p.group.eq(f'global_joint_10_high_{kind}')].nlargest(5,'n').iterrows():
  rows.append([kind,'{'+r.active_bits+'}',int(r.n),f'{100*r.share:.1f}%',f'{r.mean_cosine:.3f}',r.events])
report.append(table(['Gruppo','Bit attivi','N','Quota','Cosine entro profilo','Eventi'],rows))
report.append('## File e limiti di lettura\n- `all_documents_with_tail_membership.csv`: tutti i documenti e testi, conteggi, bit e appartenenza a ciascuna coda/intersezione.\n- `all_profile_statistics.csv`: tutti i profili in ogni gruppo, numerosità, arricchimento descrittivo rispetto al corpus, composizione per evento e cosine. Nessuna soglia minima di supporto.\n- `profile_low_high_comparisons.csv`: ogni profilo nelle code opposte e cosine entro/tra code.\n- `bit_prevalence.csv`, `within_event_bit_contrasts.csv`: tutte le coordinate e tutti i contrasti per evento.\n- `group_summary.csv`, `thresholds.csv`, `outcome_overlap.csv`, `profile_tail_overlap.csv`, `manifest.json`: statistiche complete, definizioni e provenienza.\n- Figure PNG/PDF: `tail_homogeneity`, `tail_bit_prevalence`, `joint_tail_bits`.\n\nNessun test di significatività, intervallo da ricampionamento o modello predittivo è aggiunto. Le misure descrivono l’intero corpus disponibile, non una popolazione di tweet scelta casualmente. La lettura sistematica dei testi per assegnare temi ai profili è un passo distinto: queste analisi non la sostituiscono.\n')
(P/'report.md').write_text('\n'.join(report))
print(ov.to_string(index=False))
print('Report written:',P/'report.md')
