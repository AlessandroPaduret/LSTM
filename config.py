"""
Configurazione centralizzata del progetto (Ransomware Detection).
Ottimizzata per l'architettura a singola feature (Token + Delta_Time).
Modifica qui i path e gli iperparametri per fare babysitting del modello.
"""

from pathlib import Path
import os

# ── Paths ────────────────────────────────────────────────────────────────────
# Cartella dove l'estrattore ha salvato i file CSV divisi per hash univoco
FEATURES_DIR = "./features"

# Il file JSON creato durante il primo parsing
VOCAB_FILE = "./vocab/vocabolario_global_syscalls.json"

# Cartella di output per il best_model.pt
MODEL_DIR = Path("checkpoints")
MODEL_DIR.mkdir(exist_ok=True)

# ── DataLoader & Finestre Temporali ──────────────────────────────────────────
# Strategia di split delle sequenze: "time-windows" o "n-operations"
WINDOW_STRATEGY = "n-operations"

# Se usi "time-windows": quanti secondi reali di esecuzione raggruppare insieme
TIME_WINDOW_SECS = 2.0

# Se usi "n-operations": quante syscall infilare in ogni batch
WINDOW_SIZE_OPS = 100

# ── Model ────────────────────────────────────────────────────────────────────
# L'unica dimensione di embedding rimasta (per il vocabolario globale)
EMB_DIM_TOKEN = 1

# Architettura LSTM base
HIDDEN_SIZE = 2  # ~4x meno parametri rispetto a 128
NUM_LAYERS = 1  # Meno overhead, annulla il rischio di exploding gradient
DROPOUT = 0.2

# Meccanismo di Attenzione (se usi --ari)
ARI_L = 5  # Memoria a breve termine per il blocco ARI

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
CLIP_GRAD = 1.0
VAL_RATIO = 0.2  # Sostituisce TRAIN_RATIO (80% Train / 20% Val)
PATIENCE = 15  # Epoche di attesa per l'Early Stopping (plateau)
SEED = 42

# ── CPU parallelism ──────────────────────────────────────────────────────────
# num_workers per DataLoader (0 = single process, >0 = multi-process prefetch)
NUM_WORKERS = min(4, os.cpu_count() or 1)

# thread pool per operazioni PyTorch (matmul, BLAS, ecc.)
NUM_THREADS = os.cpu_count() or 1
