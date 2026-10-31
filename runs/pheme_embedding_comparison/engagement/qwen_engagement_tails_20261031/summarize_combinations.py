from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent; R=P.parents[3]
d=pd.read_csv(P/'all_documents_with_tail_membership.csv',dtype={'id':str,'profile':str})
c=pd.read_csv(P/'conditional_bit_contrasts.csv',dtype={'context':str,'bit':str})
con=pd.read_csv(P/'conjunction_contrast_10_vs_25.csv',dtype={'combination':str})
sel=pd.read_csv(P/'selectivity_10_vs_25.csv',dtype={'pattern':str})
X=np.load(R/'runs/pheme_qwen_fixed_balance/raw_embeddings_pheme_ready_best.npy').astype(float)
U=X/np.linalg.norm(X,axis=1,keepdims=True)
cases=[('retweet_count',[6],7),('favorite_count',[13],7),('favorite_count',[5,11],2)]
rows=[];numbers=[]
for outcome,core,bit in cases:
 context='+'.join(f'{i:02d}' for i in core)
 base=d[[f'bit_{i:02d}' for i in core]].eq(1).all(axis=1)
 d.loc[base].to_csv(P/(outcome+'_'+context+f'_bit{bit:02d}_all_documents.csv'),index=False)
 for q in [10,25]:
  z=c[(c.scope=='within_event')&(c.outcome==outcome)&(c.context==context)&(c.bit==f'{bit:02d}')&(c.event=='standardized')&(c.fraction==q/100)].iloc[0]
  numbers.append(z.to_dict())
  for tail in ['low','high']:
   for state in [0,1]:
    mask=base&d[f'within_event_{outcome}_{q}_{tail}']&d[f'bit_{bit:02d}'].eq(state)
    ix=np.flatnonzero(mask);n=len(ix);p=d.loc[mask,'profile'].value_counts(normalize=True)
    cos=(np.square(U[ix].sum(0)).sum()-n)/(n*(n-1)) if n>1 else np.nan
    rows.append(dict(outcome=outcome,context=context,bit=f'{bit:02d}',fraction=q/100,tail=tail,state=state,n=n,mean_cosine=cos,profiles=len(p),entropy_bits=float(-(p*np.log2(p)).sum()),events=json.dumps(d.loc[mask,'event'].value_counts().to_dict())))
h=pd.DataFrame(rows);h.to_csv(P/'illustrative_context_homogeneity.csv',index=False)
num=pd.DataFrame(numbers);num.to_csv(P/'illustrative_context_contrasts.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
for ax,(outcome,core,bit) in zip(axes,cases):
 context='+'.join(f'{i:02d}' for i in core)
 for q,color in [(25,'#999999'),(10,'#2166ac')]:
  z=num[(num.outcome==outcome)&(num.context==context)&(num.fraction==q/100)].iloc[0]
  y=[100*z.low_on_fraction,100*z.high_on_fraction]
  ax.plot([0,1],y,'o-',color=color,label=f'{q}% tails')
  other=num[(num.outcome==outcome)&(num.context==context)&(num.fraction==(25 if q==10 else 10)/100)].iloc[0]
  other_y=[100*other.low_on_fraction,100*other.high_on_fraction]
  for x,val in enumerate(y):
   offset=9 if val>=other_y[x] else -15
   ax.annotate(f'{val:.1f}%',(x,val),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=9)
 ax.set_xticks([0,1],['Lower tail','Upper tail']);ax.set_ylim(0,85);ax.set_xlim(-.15,1.15)
 ax.set_title(f'{outcome.replace("_count", "").capitalize()}\nContext {{{context.replace("+", ", ")}}}, bit {bit:02d}')
 ax.set_ylabel('Bit active within context (%)');ax.legend(loc='upper right',fontsize=8)
fig.suptitle('Event-specific tails; remaining coordinates unrestricted',fontsize=11)
fig.savefig(P/'conditional_bit_examples.pdf');fig.savefig(P/'conditional_bit_examples.png',dpi=180);plt.close(fig)

def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)+'\n'

lines=['# Bit distintivi, combinazioni e profili nelle code PHEME / Qwen\n',
'## Obiettivo e perimetro\n',
'Analisi esplorativa completata su tutti i 5.802 documenti e sulle code inclusive al 10% e 25%, separatamente per retweet e favorite, poi sulla loro intersezione. Nessun campionamento, training o modifica a Overleaf. I codici sono quelli Qwen già salvati.\n',
'Sono calcolati tutti i 16 bit, le 120 coppie e le 560 terne di bit attivi, senza fissare le altre coordinate, oltre a tutti i profili completi osservati. Per distinguere contesto comune e bit aggiuntivo, ogni contesto di zero, uno o due bit attivi è confrontato con ciascun altro bit, spento o acceso. Questo non esaurisce le combinazioni di quattro o più bit, né tutti i contesti con zeri prescritti.\n',
'**Il passaggio 25% → 10% è un’analisi di selettività, non un requisito di stabilità.** Segnaliamo concentrazione negli estremi, perdita/acquisizione di esclusività, attenuazioni e inversioni. Tutti i risultati restano nei file, anche quando un pattern è raro o il contrasto scompare. Le code sono annidate, non repliche indipendenti.\n',
'## 1. Bit comuni ed esclusività\n',
'Tutti i 16 bit compaiono in entrambe le code, per entrambi gli outcome e tagli: nessun singolo bit è esclusivo nel corpus osservato. Questo non li rende tutti ugualmente informativi e non comporta alcuna esclusione. La differenza rilevante può emergere soltanto entro un contesto condiviso.\n',
'Alcune combinazioni sono esclusive di una coda al 10% e non al 25%. Per esempio, nei retweet la terna `{02,06,15}` compare in 28 documenti della coda bassa e in zero della coda alta al 10%; al 25% i conteggi diventano 32 e 4. Non è scartata per questo. Con soglie entro evento i corrispondenti conteggi al 10% sono 9 e zero: parte della separazione globale dipende quindi dalla composizione per evento.\n',
'Al 25% le combinazioni ancora strettamente esclusive hanno supporti massimi piccoli: tre documenti per retweet e cinque per favorite nelle code globali. La loro esclusività osservata non dimostra che siano impossibili nell’altra coda nella popolazione.\n',
'## 2. Contrasti entro un contesto condiviso\n',
'La tabella seguente usa le code definite separatamente entro ciascun evento. I numeratori sono documenti con il bit aggiuntivo attivo; i denominatori sono tutti i documenti della coda che soddisfano il contesto. Tutte le altre coordinate restano libere. Il divario standardizzato media i contrasti dei singoli eventi usando le quote degli eventi nel contesto sull’intero corpus.\n']
tr=[]
for _,z in num.iterrows():
 tr.append([z.outcome.replace('_count',''),'{'+z.context.replace('+',',')+'}',z.bit,f'{100*z.fraction:.0f}%',f'{z.n_low_on}/{z.n_low_context} ({100*z.low_on_fraction:.1f}%)',f'{z.n_high_on}/{z.n_high_context} ({100*z.high_on_fraction:.1f}%)',f'{z.event_standardized_difference_pp:+.1f} pp'])
lines.append(table(['Outcome','Contesto attivo','Bit aggiuntivo','Taglio','Coda bassa','Coda alta','Divario entro evento'],tr))
lines.extend([
'**Retweet, contesto `{06}`:** il bit 07 passa dal 45,7% al 54,6% fra coda bassa e alta al 10%. Il contrasto è circa nove punti anche al 25%. Invece, senza fissare il contesto, il suo contrasto medio entro evento è solo +0,8 punti al 10%. È un esempio di informazione condizionale che il confronto marginale nasconde.\n',
'**Favorite, contesto `{13}`:** il bit 07 è meno frequente nella coda alta. Il contrasto standardizzato aumenta in valore assoluto da 24,0 punti al 25% a 28,1 al 10%. Il bit 13 è il contesto comune; lo stato del bit 07 differenzia i sottogruppi.\n',
'**Favorite, contesto `{05,11}`:** il bit 02 è attivo nel 16,5% dei documenti della coda bassa e nel 62,3% della coda alta al 10%. Il contrasto sale da 37,2 punti al 25% a 45,7 al 10%. Questi documenti appartengono tutti a Charlie Hebdo: il confronto distingue contenuti nello stesso evento, ma non isola gli effetti di autore, follower o altri fattori.\n',
'Questi sono tre esempi leggibili, non una selezione prespecificata o una prova di superiorità di tali combinazioni. Sono stati scelti dall’elenco completo per mostrare un contrasto condizionale dei retweet, un contrasto opposto dei favorite e un caso interamente entro evento; i contesti scelti hanno copertura completa dei pesi di evento a entrambi i tagli. Il file di ranking descrittivo ordina i contrasti in base a differenza e supporto al 10%, senza imporre persistenza al 25%. Non vengono effettuati test di significatività dopo questa ricerca esplorativa.\n',
'## 3. Omogeneità: configurazioni e testi\n',
'La cosine media Qwen è calcolata su tutte le coppie di documenti distinti del gruppo. L’entropia misura la distribuzione dei profili completi all’interno del gruppo; un’entropia minore indica più concentrazione nei codici, non necessariamente maggiore coerenza semantica. Sono due descrizioni diverse.\n'])
tr=[]
for _,z in h[h.fraction.eq(.1)].iterrows():
 tr.append([z.outcome.replace('_count',''),'{'+z.context.replace('+',',')+'}',f'{z.bit}={z.state}',z['tail'],z.n,f'{z.mean_cosine:.3f}',f'{z.entropy_bits:.3f}'])
lines.append(table(['Outcome','Contesto','Stato aggiuntivo','Coda 10%','N','Cosine media','Entropia profili (bit)'],tr))
lines.extend([
'Non emerge una regola secondo cui maggiore engagement implica sempre testi più omogenei. Nel contesto `{06}`, i tweet con bit 07 attivo nella coda alta dei retweet hanno cosine 0,474 contro 0,455 nella coda bassa. Per `{05,11,02}` nei favorite, la coda alta è più concentrata nei profili completi (entropia 0,244 contro 0,792), ma ha cosine leggermente minore (0,535 contro 0,556). L’associazione con engagement e l’omogeneità vanno quindi riportate separatamente. La cosine è una misura nello stesso spazio Qwen usato per costruire i bit, non una verifica semantica indipendente.\n',
'## 4. Selettività degli estremi\n',
'Per lo stesso pattern e la stessa direzione di coda definiamo `C = (n10/n25)/(N10/N25)`, equivalente al rapporto fra la frequenza relativa del pattern nella coda al 10% e quella nella coda al 25%. `n` conta i documenti del pattern, `N` tutti i documenti della coda. `C > 1` indica concentrazione sproporzionata nell’estremo; il riferimento usa le dimensioni effettive e quindi rispetta i ties. Non è una probabilità né un test statistico. La differenza alta-bassa e C rispondono a domande diverse: la separazione può aumentare anche perché il pattern diventa meno frequente nella coda opposta.\n'])
tr=[]
for y,pattern in [('retweet_count','06+07'),('favorite_count','07+13'),('favorite_count','02+05+11')]:
 for tail in ['low','high']:
  z=sel[(sel.kind=='conjunction')&(sel.scope=='within_event')&(sel.outcome==y)&(sel.pattern==pattern)&(sel['tail']==tail)].iloc[0]
  tr.append([y.replace('_count',''),'{'+pattern.replace('+',',')+'}',tail,int(z.n25),int(z.n10),f'{100*z.prevalence25:.2f}%',f'{100*z.prevalence10:.2f}%',f'{z.concentration_ratio:.3f}'])
lines.append(table(['Outcome','Combinazione','Coda','N pattern 25%','N pattern 10%','Frequenza 25%','Frequenza 10%','C'],tr))
lines.extend([
'## 5. Intersezione degli outcome\n',
'Le code alte globali al 10% condividono 361 documenti su 582 per ciascun outcome (62,0%). Restano 221 tweet nella sola coda alta dei retweet e 221 nella sola coda alta dei favorite. Al 25% i documenti comuni sono 1.026, circa il 70,5% delle due code. “Solo” significa fuori dall’altra coda alta, non necessariamente basso nell’altro outcome.\n',
'Anche qui il bit 07 distingue i gruppi: al 10% è attivo nel 59,3% dei tweet solo-retweet, contro il 38,0% dei solo-favorite e il 38,5% dei comuni. Al 25% le percentuali sono 65,9%, 41,0% e 39,8%. Il bit 02 è invece più frequente nel gruppo comune: 54,6% al 10%, contro 28,1% nei solo-retweet e 37,1% nei solo-favorite. Sono confronti globali descrittivi; l’intero elenco per bit e combinazioni, anche con soglie entro evento, è salvato.\n',
'## Interpretazione e materiali\n',
'La lettura sostenuta dai dati è che alcune coordinate distinguono le code soprattutto in combinazione con altre, e che alcuni contrasti diventano più marcati negli estremi. Non emerge un singolo profilo del tweet popolare. Non abbiamo ancora attribuito sistematicamente un tema a ogni gruppo, né separato il contributo dei follower: nessuna conclusione causale o sul fattore dominante è implicita. Tutti i testi dei tre contesti illustrativi sono esportati, senza campionamento, per la successiva lettura.\n',
'File principali:\n- `all_conjunctions.csv`: tutti i bit, coppie e terne, frequenze e cosine in ogni gruppo.\n- `conditional_bit_contrasts.csv`: tutti i contesti e contrasti, anche per evento.\n- `conditional_contrast_ranking.csv`: contrasti al 10% e 25%, inclusi inversioni e variazioni.\n- `selectivity_10_vs_25.csv`: concentrazione negli estremi per tutte le combinazioni e i profili.\n- `conjunction_contrast_10_vs_25.csv`: esclusività, conteggi e variazione dei contrasti.\n- `joint_conjunction_contrasts.csv`: tutte le combinazioni nell’intersezione e nei gruppi esclusivi degli outcome.\n- `illustrative_context_homogeneity.csv`: omogeneità dei tre contesti, bit acceso/spento, entrambe le code e tagli.\n- `*_all_documents.csv`: tutti i testi e le appartenenze dei contesti illustrati.\n- `conditional_bit_examples.pdf` e `.png`: figura dei contrasti illustrativi.\n- `report.md`: analisi iniziale completa di soglie, distribuzione dei profili e sovrapposizioni.\n',
'Le analisi usano l’intero corpus disponibile e sono esplorative. Gli esempi non sono validati su dati indipendenti, e le combinazioni non sono indipendenti tra loro. Le soglie di inclusione PHEME differiscono tra eventi; per questo sono conservate entrambe le analisi, globale ed entro evento.\n'])
(P/'bit_combinations_report.md').write_text('\n'.join(lines))
print('Updated report and figure:',P/'bit_combinations_report.md')
