"""
Configurazione centralizzata del progetto.
Modifica qui i path e gli iperparametri.
"""
from pathlib import Path

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
EMB_DIM_OP  = 16
EMB_DIM_RES = 8
EMB_DIM_DET = 32
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
ARI_L       = 5    # lunghezza finestra di attenzione ARI (0 = LSTM standard)

# ── Training ─────────────────────────────────────────────────────────────────
SEQ_LEN     = 50
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1e-3
CLIP_GRAD   = 1.0   # gradient clipping (fondamentale per LSTM)
TRAIN_RATIO = 0.8
SEED        = 42
