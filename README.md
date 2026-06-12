# Ransomware Detection via LSTM

Questo progetto implementa una rete neurale ricorrente (**LSTM**) per la rilevazione di ransomware basata sull'analisi sequenziale delle *system call* (syscall). Il sistema è in grado di distinguere processi malevoli da processi legittimi analizzando il comportamento temporale delle chiamate API.

## Setup e Preparazione Dati

Assicurati di avere `uv` installato. Il workflow di elaborazione dati è diviso in due fasi:

1. **Etichettatura (`label_data.py`):** Assegna l'etichetta `1` (Ransomware) o `0` (Legittimo) ai file CSV grezzi basandosi sulla struttura delle directory.
2. **Estrazione (`extractor.py`):** Elabora i CSV etichettati per generare i file di feature pronti per il training.

```bash
# 1. Etichetta i file (modifica i path nel file prima di eseguire)
uv run label_data.py

# 2. Estrai le feature (modifica i path nel file prima di eseguire)
uv run extractor.py
```

## Utilizzo

Puoi addestrare il modello in modalità standard o con il meccanismo di attenzione ARI:

```bash
# Training LSTM Standard
uv run train.py

# Training LSTM con meccanismo ARI (Attention on Recent Inputs)
uv run train.py --ari
```

## Architettura del Modello

Il modello segue un approccio *Many-to-One*:

* **Embedding Layer:** Mappa le syscall discrete in uno spazio vettoriale continuo (dimensione `EMB_DIM_TOKEN`).
* **LSTM Layer:** Elabora la sequenza temporale per catturare le dipendenze a lungo termine.
* **Variante ARI-LSTM (`--ari`):** Aggiunge un modulo di attenzione che calcola un vettore `r_t` sugli ultimi `L` input. Questo modulo somma il contesto recente all'input corrente, eccellendo nel rilevare i *burst* di cifratura tipici dei ransomware.

## Configurazione (`config.py`)

Puoi modificare i seguenti iperparametri per il fine-tuning:

| Parametro | Default | Descrizione |
| --- | --- | --- |
| `WINDOW_SIZE_OPS` | 100 | Numero di syscall raggruppate per ogni finestra |
| `BATCH_SIZE` | 64 | Numero di sequenze per batch |
| `LR` | 1e-3 | Learning Rate (Adam Optimizer) |
| `CLIP_GRAD` | 1.0 | Previene l'esplosione del gradiente |
| `ARI_L` | 5 | Finestra di memoria breve termine (per ARI) |
| `EPOCHS` | 100 | Massime epoche di training |
| `PATIENCE` | 15 | Epoche per l'Early Stopping (plateau) |

## Struttura del Progetto

* `train.py`: Loop di addestramento e valutazione.
* `dataset.py`: Gestione dello streaming dati via `IterableDataset`.
* `models.py`: Definizioni delle classi `RansomwareLSTM` e `RansomwareARILSTM`.
* `config.py`: Gestione centralizzata dei path e degli iperparametri.
* `checkpoints/`: Directory dove verranno salvati i modelli migliori.

### Requisiti

* Python >= 3.12
* PyTorch (versione 2.2+)
* `polars`, `numpy`, `scikit-learn`

