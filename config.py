"""
Configurazione centralizzata del progetto.
Modifica qui i path e gli iperparametri.
"""
from pathlib import Path
import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/API")
LOG_FILE   = DATA_DIR / "Logfile_labeled.CSV"
MODEL_DIR  = Path("checkpoints")
MODEL_DIR.mkdir(exist_ok=True)

# ── Tokenizer ────────────────────────────────────────────────────────────────
TOP_N_OP   = 50    # vocabolario operazioni
TOP_N_RES  = 30    # vocabolario risultati
TOP_N_DET  = 100   # vocabolario dettagli

# ── Model ─────────────────────────────────────────────────────────────────────
# op e res usano nn.Embedding (scalare) — vocabolari piccoli (24/31 token)
# det usa EmbeddingBag (lista di token) — vocabolario più ricco
EMB_DIM_OP  = 16
EMB_DIM_RES = 8
EMB_DIM_DET = 32
HIDDEN_SIZE = 64    # ridotto da 128 → ~4x meno parametri nella LSTM
NUM_LAYERS  = 1     # ridotto da 2 → meno overhead, meno rischio exploding grad
DROPOUT     = 0.2
ARI_L       = 5

# ── Training ─────────────────────────────────────────────────────────────────
SEQ_LEN     = 50
# Stride della sliding window.
# stride=1        → ~4M campioni su 200k righe (esplode RAM)
# stride=SEQ_LEN//2 → ~160k campioni  (buon compromesso, default)
# stride=SEQ_LEN  → ~80k campioni    (nessun overlap, epoca in <1 min)
STRIDE      = SEQ_LEN #// 2
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-3
CLIP_GRAD   = 1.0
TRAIN_RATIO = 0.8
SEED        = 42

# ── CPU parallelism ───────────────────────────────────────────────────────────
# num_workers per DataLoader (0 = single process, >0 = multi-process prefetch)
NUM_WORKERS = min(4, os.cpu_count() or 1)
# thread pool per operazioni PyTorch (matmul, BLAS, ecc.)
NUM_THREADS = os.cpu_count() or 1