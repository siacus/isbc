from pathlib import Path
import json,re,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
d=pd.read_csv(P.parent/'all_documents_with_tail_membership.csv',dtype={'id':str,'profile':str})
k=pd.read_csv(P/'keyword_contrasts.csv')
reading=pd.read_csv(P/'llm_reading_documents.csv',dtype={'id':str})

interpretations=[
 dict(case='R06',title='Retweet: cronaca e contestazione delle azioni di polizia',
 summary='Il contesto con bit 06 attivo raccoglie più eventi. Il bit 07 acceso seleziona maggiormente polizia, proteste e operazioni con ostaggi. Nei testi letti della coda alta si osservano immagini della forza di polizia, richieste di accountability e commenti sulla militarizzazione; nella coda bassa compaiono anche aggiornamenti su conferenze stampa, ricostruzioni della morte di Michael Brown e presunta rapina. Le due code condividono ampiamente gli argomenti: è una differenza di enfasi, non una separazione netta tra due temi. A bit 07 spento il peso di Germanwings e della cronaca aerea è molto maggiore, quindi parte della distinzione del bit è tra eventi.',
 examples={'R06_10_low_1':['500359983301939200','500287538578587648'], 'R06_10_high_1':['499673142114013186','500345408150065153'], 'R06_10_low_0':['580326473803767808'], 'R06_10_high_0':['580350587289604097']},
 boundary='Le differenze nei retweet sono meno nette di quelle nei favorite. Non emerge dai testi un unico meccanismo di popolarità, né il bit 07 identifica solo indignazione.'),
 dict(case='F13',title='Favorite: aggiornamenti sulla crisi e partecipazione pubblica',
 summary='Nel contesto con bit 13 attivo, la coda bassa contiene più aggiornamenti operativi: persone trattenute nel café, numero di ostaggi, bandiere, gunman e interventi della polizia. La coda alta contiene più messaggi di vicinanza, preghiera, solidarietà e rifiuto della stigmatizzazione religiosa, in particolare relativi a Sydney. Il bit 07 spento è arricchito in questi ultimi testi; il bit acceso resta legato alla cronaca operativa e alla polizia. Tuttavia anche nel gruppo acceso ad alto engagement compaiono giudizi morali, preghiere e solidarietà: la distinzione è contestuale e graduata.',
 examples={'F13_10_low_1':['544279262547935232','544302938374426624'], 'F13_10_high_0':['544499931101548544','544379143136497664'], 'F13_10_high_1':['544472530921480192','544420375224791040'], 'F13_10_low_0':['580699430690656256']},
 boundary='I gruppi combinano eventi diversi, con una maggiore quota di Sydney nella coda alta a bit 07 spento. Le keyword aggregate da sole non isolano una componente emotiva indipendente dall’evento.'),
 dict(case='F0511',title='Favorite / Charlie Hebdo: dalla cronaca alla partecipazione emotiva, simbolica e normativa',
 summary='Il contesto con bit 05 e 11 attivi identifica qui documenti Charlie Hebdo. Nella coda bassa la lettura completa evidenzia numerosi lanci di cronaca: bilanci delle vittime, dettagli dell’attacco, identificazione e ricerca degli aggressori. Nella coda alta emerge un repertorio espressivo: cordoglio e tributi, condivisione di vignette, identificazione collettiva attraverso JeSuisCharlie, difesa della satira e della libertà di espressione. Sono presenti anche indignazione, polemiche religiose e critiche ai doppi standard occidentali. Non si tratta quindi semplicemente di sentiment positivo. Il bit 02 acceso seleziona soprattutto questa dimensione simbolica, artistica e normativa; non è un indicatore esclusivo di emozione.',
 examples={'F0511_10_low_0':['552802128688078848','553184390260338688'], 'F0511_10_high_1':['553118384540643328','553220054184382465','552832738542514176'], 'F0511_10_high_0':['553205372719472640','553157600704204800'], 'F0511_10_low_1':['553126517027262465','553131557892804609']},
 boundary='Tributi ed emozioni compaiono anche con bit 02 spento, e il bit acceso è presente in testi a basso engagement. L’interpretazione sostiene un’associazione locale fra modalità espressive ed engagement, non un effetto causale del contenuto.')
]
evidence=[]
for case in interpretations:
 for group,ids in case['examples'].items():
  source=pd.read_csv(P/(group+'_documents.csv'),dtype={'id':str}).set_index('id')
  for id in ids:
   assert id in source.index
   row=source.loc[id]
   evidence.append(dict(case=case['case'],group=group,id=id,text=row.text,favorite_count=int(row.favorite_count),retweet_count=int(row.retweet_count),followers_count=int(row.followers_count)))
pd.DataFrame(evidence).to_csv(P/'interpreted_examples.csv',index=False)
(P/'llm_interpretation.json').write_text(json.dumps(interpretations,indent=2,ensure_ascii=False))

# All 286 Charlie Hebdo texts at 10% were read, including the addendum.
extra=[]
for tail in ['low','high']:
 for state in [0,1]:
  group=f'F0511_10_{tail}_{state}'
  x=pd.read_csv(P/(group+'_documents.csv'),dtype={'id':str})
  extra.extend(dict(group=group,id=row.id,text=row.text,event=row.event,seed=np.nan) for row in x.itertuples())
reviewed=pd.concat([reading[~reading.group.str.startswith('F0511_')],pd.DataFrame(extra)],ignore_index=True)
reviewed.to_csv(P/'all_llm_reviewed_documents.csv',index=False)
assert reviewed[reviewed.group.str.startswith('F0511_')].id.nunique()==286
method=json.loads((P/'method.json').read_text())
method['llm_reading']='Direct qualitative reading by the assistant in this conversation. For R06 and F13: 20 reproducibly sampled texts per 10% tail x bit-state cell (160 reading rows). For F0511/Charlie Hebdo: all 286 texts in the two 10% tails, both bit states, using the initial dossier plus complete-reading addendum. All-group keyword summaries at 10% and 25% considered. No external API; no independent annotator, blinding, or formal theme coding.'
method['llm_reading_rows']=len(reviewed);method['llm_unique_documents']=reviewed.id.nunique()
(P/'method.json').write_text(json.dumps(method,indent=2))

selected_terms={'R06':['police','ferguson police','shot','chief','robbery','muslim'], 'F13':['thoughts','illridewithyou','prayers','breaking','cafe','hostages'], 'F0511':['jesuischarlie','freedom','tribute','dead','shooting','police']}
fig,axes=plt.subplots(1,3,figsize=(13,5.1),layout='constrained')
for ax,(case,terms) in zip(axes,selected_terms.items()):
 for q,color,offset in [(25,'#bbbbbb',.17),(10,'#2166ac',-.17)]:
  sub=k[k.comparison.eq(f'{case}_{q}_tails_all')].set_index('term')
  vals=[sub.loc[t,'difference_pp'] for t in terms]
  ax.barh(np.arange(len(terms))+offset,vals,height=.3,label=f'{q}% tails',color=color)
 ax.axvline(0,color='black',lw=.7);ax.set_yticks(range(len(terms)),terms);ax.invert_yaxis();ax.set_xlabel('Upper minus lower tail (percentage points)');ax.legend(fontsize=8)
 ax.set_title({'R06':'Retweets: context {06}','F13':'Favorites: context {13}','F0511':'Favorites: context {05, 11}'}[case])
fig.suptitle('Keyword document prevalence: all texts in each context; both bit states',fontsize=11)
fig.savefig(P/'keyword_tail_contrasts.pdf');fig.savefig(P/'keyword_tail_contrasts.png',dpi=180);plt.close(fig)

def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
lines=['# Contenuto dei tre contesti Qwen nelle code di engagement\n',
'## Domanda di ricerca e portata dell’applicazione\n',
'La domanda originaria è quali caratteristiche del testo si associno all’engagement senza fissare prima un elenco di topic, sentiment o altre tassonomie. I bit sono appresi senza quelle annotazioni e senza l’outcome; le loro combinazioni selezionano documenti da interpretare successivamente. Non si presuppone che l’applicazione debba essere specifica per topic.\n',
'PHEME è una dimostrazione circoscritta: 5.802 documenti, cinque eventi selezionati, con soglie di raccolta diverse. Il confronto entro evento limita qui la confusione fra contenuto ed evento; non trasforma questa impostazione in un requisito generale del metodo. La presenza di una dimensione emotiva in un caso locale non implica un significato emotivo universale del bit né una capacità generale di sentiment analysis.\n',
'## Cosa è stato analizzato\n',
'I tre casi sono contesti con bit condivisi, non tre profili completi: `{06}` con bit 07 variabile per retweet, `{13}` con bit 07 variabile per favorite e `{05,11}` con bit 02 variabile per favorite. Le altre coordinate sono libere. Sono mantenute le code definite entro evento dell’analisi precedente. Per ogni caso si confrontano sia le due code sia il bit acceso/spento all’interno delle code.\n',
'Le keyword usano tutti i documenti al 10% e al 25%, senza campioni di dimensione uguale. Le frequenze sono percentuali di documenti contenenti un termine, non conteggi di occorrenze. URL e menzioni sono rimossi, gli hashtag conservati come parole, con unigrammi e bigrammi in minuscolo e stopword inglesi. I bigrammi sono costruiti dopo la rimozione delle stopword. I file includono tutti i termini osservati; il riepilogo automatico mostra termini con almeno cinque documenti sul lato arricchito.\n',
f'La lettura LLM è stata effettuata direttamente dall’assistente in questa conversazione: tutti i 286 documenti Charlie Hebdo delle code al 10%, e 20 testi riproducibilmente estratti per ciascuno degli otto gruppi degli altri due casi. Sono {len(reviewed)} righe di lettura e {reviewed.id.nunique()} documenti unici, perché alcuni gruppi si sovrappongono. Al 25% sono analizzate le keyword complete, non tutti i testi in lettura individuale. Non è stata chiamata un’API esterna e non si tratta di una validazione da parte di un annotatore indipendente.\n',
'## Risultato centrale: Charlie Hebdo\n',
'Nel contesto `{05,11}`, la coda bassa dei favorite contiene più cronaca fattuale dell’attacco e dei suoi sviluppi. Quella alta contiene più partecipazione emotiva, solidarietà, commemorazione e prese di posizione sulla libertà di espressione. Il bit 02 acceso è associato soprattutto a vignette, tributi e argomentazioni sulla satira e sulla libertà di parola. È una componente espressiva ed emotivo-normativa, non una semplice opposizione tra sentiment positivo e negativo.\n',
'Al 10%, il bit 02 è attivo in 21/127 documenti della coda bassa (16,5%) e 99/159 della coda alta (62,3%). Al 25%, le percentuali sono 72/310 (23,2%) e 241/399 (60,4%). Il contrasto si accentua restringendo le code. Per Charlie Hebdo le soglie al 10% corrispondono a ≤24 favorite e ≥314 favorite; quelle al 25% a ≤42 e ≥169.\n']
rows=[]
for term in ['jesuischarlie','freedom','tribute','silence','dead','shooting','police']:
 z=k[(k.comparison=='F0511_10_tails_all')&(k.term==term)].iloc[0]
 v=k[(k.comparison=='F0511_25_tails_all')&(k.term==term)].iloc[0]
 rows.append([term,f'{z.right_count}/{z.right_docs} ({z.right_percent:.1f}%)',f'{z.left_count}/{z.left_docs} ({z.left_percent:.1f}%)',f'{v.difference_pp:+.1f}',f'{z.difference_pp:+.1f}'])
lines.append(table(['Termine','Coda bassa 10%','Coda alta 10%','Divario 25% (pp)','Divario 10% (pp)'],rows))
lines.extend([
'Le keyword sono indicatori lessicali osservabili, non una misura validata dell’emozione. La lettura completa sostiene una distinzione tra cronaca e partecipazione simbolica/emotiva, con sovrapposizioni: nella coda alta si trovano anche giudizi polemici e indignazione, e tributi compaiono anche con bit 02 spento. Non abbiamo assegnato etichette emozionali sistematiche ai singoli documenti o stimato una prevalenza tematica.\n',
'### Un confronto che aiuta a delimitare l’interpretazione\n',
'Due documenti nel contesto `{05,11}` con bit 02 acceso riportano lo stesso testo, tranne il link: “Can’t sleep tonight, thoughts with my French cartooning colleagues, their families and loved ones”. L’ID `552844593046097920` ha 37.983 favorite e 74.130 retweet, mentre l’ID `553149163181838336` ha 1 favorite e 307 retweet. I follower registrati sono rispettivamente 7.457 e 42. Questo esempio mostra che lo stesso contenuto può trovarsi nelle due code; non stima quanto della differenza sia causato dai follower, dall’autore, dai tempi o dalla diffusione.\n',
'Un altro esempio è il richiamo ai doppi standard sulla libertà di espressione: ID `553111157738057728` (1.948 favorite) e una ripubblicazione con prefisso RT, ID `553131557892804609` (zero favorite). Non sono quindi gruppi tematicamente impermeabili: il contenuto può essere associato alla distribuzione dell’engagement senza determinarlo.\n',
'## Lettura dei tre casi ed esempi verificabili\n'])
for obj in interpretations:
 lines += ['### '+obj['title']+'\n',obj['summary']+'\n',obj['boundary']+'\n']
 for group,ids in obj['examples'].items():
  lines.append('**'+group+'**\n')
  for id in ids:
   row=d[d.id.eq(id)].iloc[0]
   lines.append('- ID `'+id+'`: '+str(row.text).replace('\n',' ')+'\n')
lines += ['## Keyword degli altri due casi\n']
for case,terms in [('R06',['police','ferguson police','chief','robbery']),('F13',['thoughts','illridewithyou','prayers','breaking','cafe','hostages'])]:
 rows=[]
 for term in terms:
  z=k[(k.comparison==f'{case}_10_tails_all')&(k.term==term)].iloc[0]
  v=k[(k.comparison==f'{case}_25_tails_all')&(k.term==term)].iloc[0]
  rows.append([term,f'{z.right_count}/{z.right_docs} ({z.right_percent:.1f}%)',f'{z.left_count}/{z.left_docs} ({z.left_percent:.1f}%)',f'{v.difference_pp:+.1f}',f'{z.difference_pp:+.1f}'])
 lines.append('### '+case+'\n'+table(['Termine','Bassa 10%','Alta 10%','Divario 25% (pp)','Divario 10% (pp)'],rows))
lines += ['## Cosa possiamo riportare\n',
'Il caso Charlie Hebdo mostra che, senza fornire una tassonomia emotiva al modello, una combinazione di bit permette di individuare documenti con modalità espressive differenti e distribuzioni di favorite differenti. L’interpretazione successiva evidenzia una dimensione di partecipazione emotiva, simbolica e normativa associata alla coda alta. Questo è il risultato locale; non dimostra che l’emozione causi più engagement, che il bit 02 significhi “emozione” in ogni contesto, o che il risultato generalizzi al flusso non selezionato di Twitter/X.\n',
'I testi citati sono evidenze del contenuto dei tweet, non verifiche della verità delle notizie riportate. Non sono state aperte immagini o pagine collegate: i riferimenti a vignette e fotografie sono quelli leggibili nei testi. Nel caso di Banksy, alcuni tweet attribuiscono opere e altri correggono tale attribuzione; non adottiamo l’attribuzione come fatto.\n',
'## Materiali\n',
'- `keyword_contrasts.csv`: 120.324 righe di contrasti lessicali completi.\n- `keyword_document_frequencies.csv`: frequenze in tutti i 36 gruppi.\n- `*_documents.csv`: tutti i testi dei gruppi, con ID e conteggi.\n- `all_llm_reviewed_documents.csv`: elenco completo dei testi effettivamente letti dall’assistente.\n- `llm_reading_dossier.md`, `charlie_complete_reading_addendum.md`: dossier di lettura.\n- `llm_interpretation.json`, `interpreted_examples.csv`: interpretazioni ed esempi verificati.\n- `keyword_tail_contrasts.pdf` e `.png`: confronto lessicale al 10% e 25%.\n- `method.json`: definizioni, limiti e provenienza.\n\nNessuna modifica al paper.\n']
(P/'report.md').write_text('\n'.join(lines))
print('Report complete;',len(evidence),'example IDs checked;',len(reviewed),'LLM reading rows.')
