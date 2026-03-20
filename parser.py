"""
Parsing e preprocessing del CSV di log.

Flusso corretto:
  1. Carica il CSV completo
  2. Costruisci vocabolari GLOBALI (una sola volta)
  3. Calcola dt per ogni PID
  4. Tokenizza con i vocabolari globali
  5. Suddividi per PID
"""
import pandas as pd
from typing import Dict, Tuple

from tokenizer import Vocabulary, build_global_vocabularies
import config


def load_and_prepare(
    file_path: str = str(config.LOG_FILE),
) -> Tuple[Dict[int, pd.DataFrame], Vocabulary, Vocabulary, Vocabulary]:
    """
    Carica il CSV, costruisce vocabolari globali e restituisce
    un dizionario {pid: dataframe} con colonne già tokenizzate.

    Returns
    -------
    pid_data   : dict {pid -> DataFrame con colonne dt, ops, res, det, label}
    vocab_op   : Vocabulary per Operation
    vocab_res  : Vocabulary per Result
    vocab_det  : Vocabulary per Detail
    """
    df = pd.read_csv(file_path)
    df.drop(columns=["Path"], inplace=True, errors="ignore")

    # ── 1. Vocabolari globali ────────────────────────────────────────────────
    vocab_op, vocab_res, vocab_det = build_global_vocabularies(
        df,
        top_n_op  = config.TOP_N_OP,
        top_n_res = config.TOP_N_RES,
        top_n_det = config.TOP_N_DET,
    )

    # ── 2. Tokenizza tutto il DataFrame in una volta ─────────────────────────
    df["ops"]  = df["Operation"].apply(vocab_op.encode)
    df["res"] = df["Result"].apply(vocab_res.encode)
    df["det"] = df["Detail"].apply(vocab_det.encode)
    df["label"]      = df["is_ransomware"].astype(int)

    # ── 3. Calcola dt per PID (differenza temporale intra-processo) ──────────
    df["time_parsed"] = pd.to_datetime(
        df["Time of Day"].str.strip(),
        format="%I:%M:%S.%f %p",
        errors="coerce",
    )

    pid_data: Dict[int, pd.DataFrame] = {}

    for pid, group in df.groupby("PID"):
        g = group.copy().reset_index(drop=True)

        # dt = differenza in secondi rispetto alla riga precedente dello stesso PID
        g["dt"] = g["time_parsed"].diff().dt.total_seconds().fillna(0.0)

        cols = ["dt", "ops", "res", "det", "label"]
        pid_data[pid] = g[cols]

    return pid_data, vocab_op, vocab_res, vocab_det
