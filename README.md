# Ransomware Detection via LSTM

Classificazione binaria di processi Windows come ransomware o benigni,
a partire da sequenze di Windows API call tracciate con ProcMon.

Il modello si basa su *"Attention in Recurrent Neural Networks for Ransomware Detection"*
(Agrawal et al., ICASSP 2019).

---

## Struttura del progetto

```
cyber/
├── config.py       ← tutti i parametri (path, iperparametri)
├── tokenizer.py    ← costruzione vocabolario GLOBALE
├── parser.py       ← caricamento CSV e preprocessing
├── dataset.py      ← PyTorch Dataset + collate function
├── model.py        ← RansomwareLSTM e RansomwareARILSTM
├── train.py        ← training loop
├── data/
│   └── Logfile_labeled.CSV
├── checkpoints/    ← modelli salvati (auto-generata)
└── requirements.txt
```

---

## Setup

```bash
#per generare requirements.txt
pipreqs . --force 
#per installare i requirements
pip install -r requirements.txt
```


---

## Utilizzo

```bash
# LSTM standard
python train.py

# LSTM con meccanismo ARI (attenzione su input recenti, dal paper)
python train.py --ari
```

## Architettura

```
Input: sequenza di T eventi per processo
  ├── Operation  → EmbeddingBag (mode=mean) → vettore 16-dim
  ├── Result     → EmbeddingBag (mode=mean) → vettore  8-dim
  ├── Detail     → EmbeddingBag (mode=mean) → vettore 32-dim
  └── dt         → scalare (secondi dall'evento precedente)
        ↓
  Concatenazione → vettore 57-dim per timestep
        ↓
  LSTM (hidden=128, layers=2, dropout=0.2)
        ↓
  ultimo hidden state → Linear(128, 1) → logit
        ↓
  BCEWithLogitsLoss
```

### Variante ARI-LSTM (--ari)

Per ogni timestep, un modulo ARI calcola un vettore di attenzione `r_t`
sugli ultimi `L=5` input e lo somma all'input corrente prima di passarlo
alla LSTM. Questo cattura pattern ripetitivi brevi, tipici del comportamento
di cifratura del ransomware.

---

## Iperparametri principali (`config.py`)

| Parametro    | Default | Note                                     |
|--------------|---------|------------------------------------------|
| `SEQ_LEN`    | 50      | Lunghezza finestra API call              |
| `BATCH_SIZE` | 32      | Riduci a 8-16 con pochi dati             |
| `LR`         | 1e-3    | Adam                                     |
| `CLIP_GRAD`  | 1.0     | Gradient clipping                        |
| `ARI_L`      | 5       | Finestra di attenzione ARI               |
| `EPOCHS`     | 100     | Early stopping dopo 20 epoche senza migl.|
