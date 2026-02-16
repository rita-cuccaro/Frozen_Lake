## PROGETTO ESAME AI ENVIRONMENT FROZEN LAKE

## 📌 Descrizione del progetto
Questo progetto analizza e confronta tre approcci differenti per la risoluzione dell’ambiente Frozen Lake:
1. Agente basato esclusivamente su Reinforcement Learning;
2. Agente guidato da un Large Language Model (LLM) come selettore di azioni;
3. Agente ibrido RL + LLM, che integra apprendimento per rinforzo e supporto semantico.

## 🎯 Obiettivi
● Dimostrazione dell'abilità degli agenti nel raggiungere il goal finale. 

● Ottimizzazione delle prestazioni attraverso l'addestramento iterativo, evidenziando miglioramenti nella strategia di scelta del percorso.

● Analisi delle prestazioni nei scenari 4x4 ed 8x8. 

● Confrontare i risultati ottenuti dal RL, dal RL con LLM e dal RL con le reward ottenute dal LLM. 

● Discutere le sfide affrontate durante l'implementazione e come sono state risolte. 

## 📊 Risultati principali
### Mappa 4×4
● RL apprende una policy efficace;

● L’agente ibrido mostra miglioramenti nella stabilità e nel success rate;

● LLM puro non migliora nel tempo;

### Mappa 8×8
● RL tabellare mostra forti limiti dovuti alla ricompensa sparsa;

● LLM puro fallisce quasi sistematicamente;

● L’approccio ibrido non scala efficacemente;

## 🛠️ Tecnologie utilizzate
## Linguaggio
● Python 3.11.9 per la gestione dell'esperimento

## Librerie principali
● NumPy

● Gymnasium

● Mathplotlib

● OpenAI

## Modello linguistico
● LM Studio per esecuzione locale del LLM

## 📈 Output
Il progetto produce:
● Grafici del success rate

● Heatmap delle Q-table

● Distribuzione del numero di passi
● Confronto tra approcci
