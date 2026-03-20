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
pipreq . --force 
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

---

## Bug risolti rispetto alla versione originale

### 🔴 Bug critico: vocabolario per-PID

**Problema originale:**
```python
# parser.py — SBAGLIATO
for pid in datas:
    datas[pid] = parse_data(datas[pid])  # tokenize_column chiamato QUI
```
`tokenize_column` costruiva un dizionario separato per ogni processo.
Il token ID `5` per il PID 1404 significava `"CreateFile"`, ma per il PID 2000
poteva significare `"WriteFile"`. La LSTM imparava associazioni casuali.

**Fix:**
```python
# tokenizer.py — CORRETTO
vocab_op, vocab_res, vocab_det = build_global_vocabularies(df)  # su tutto il CSV
df["op_tokens"] = df["Operation"].apply(vocab_op.encode)         # poi si tokenizza
```
Il vocabolario viene costruito **una sola volta** sull'intero dataset, poi
applicato uniformemente a ogni PID.

---

### 🟡 Gradient clipping mancante

Le LSTM soffrono di *exploding gradients*: senza clipping il loss oscilla.

```python
# train.py
nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)  # max norm = 1.0
```

---

### 🟡 Dataset sbilanciato non gestito

Con pochi dati, il modello convergeva verso "predici sempre la classe maggioritaria".

```python
# train.py
pos_weight = n_negative / n_positive
criterion  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

---

### 🟡 Data leakage nel train/val split

L'originale calcolava l'accuracy sullo stesso DataLoader usato per il training.
Il refactor divide per **PID**, non per righe, così il validation set contiene
processi mai visti durante il training.

---

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
