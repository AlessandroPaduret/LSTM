"""
Tokenizer con vocabolario GLOBALE.

Soluzione: costruire il vocabolario UNA SOLA VOLTA su tutto il dataset,
poi applicarlo uniformemente.
"""

import re
from typing import List, Dict, Tuple
import pandas as pd


# ── Pulizia testo ─────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Normalizza una stringa rimuovendo valori numerici specifici e simboli."""
    if pd.isna(text) or str(text).strip() == "":
        return "NONE"

    s = str(text)
    s = re.sub(r"\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M", "", s)  # date/ore
    s = re.sub(r"0x[0-9a-fA-F]+", " hex ", s)  # indirizzi hex
    s = re.sub(r"([a-zA-Z]:\\|\/)[\\\/\w\s.\-]+", " path ", s)  # percorsi file
    s = re.sub(r"\b\w+\.\w+\b", " file ", s)  # nomi file
    s = re.sub(r"[^a-zA-Z\s]", " ", s)  # tutto il resto
    s = s.lower().strip()

    return s if s else "NONE"


# ── Costruzione vocabolario ────────────────────────────────────────────────────


class Vocabulary:
    """
    Vocabolario costruito su un'intera colonna (o più colonne concatenate).
    Token 0 = PAD / NONE / unknown.
    """

    def __init__(self, top_n: int = 100):
        self.top_n = top_n
        self.word2id: Dict[str, int] = {"NONE": 0}
        self.built = False

    def build(self, series: pd.Series) -> "Vocabulary":
        """Costruisce il vocabolario dalla serie. Chiamare UNA SOLA VOLTA sul dataset intero."""
        cleaned = series.fillna("NONE").astype(str).apply(clean_text)
        all_words = " ".join(cleaned).split()

        freq: Dict[str, int] = {}
        for w in all_words:
            freq[w] = freq.get(w, 0) + 1

        top_words = sorted(freq, key=freq.get, reverse=True)[: self.top_n]
        self.word2id = {"NONE": 0}
        for idx, word in enumerate(top_words, start=1):
            self.word2id[word] = idx

        self.built = True
        return self

    @property
    def size(self) -> int:
        return len(self.word2id)

    def encode(self, text: str) -> List[int]:
        """Trasforma una stringa in lista di token IDs. Ritorna [0] se nessun token noto."""
        tokens = [
            self.word2id[w] for w in clean_text(text).split() if w in self.word2id
        ]
        return tokens if tokens else [0]


# ── Costruzione vocabolari globali(old) ─────────────────────────────────────────────


def build_global_vocabularies(
    df: pd.DataFrame,
    top_n_op: int = 50,
    top_n_res: int = 30,
    top_n_det: int = 100,
) -> Tuple[Vocabulary, Vocabulary, Vocabulary]:
    """
    Costruisce tre vocabolari globali (Operation, Result, Detail)
    sull'intero dataframe PRIMA di suddividerlo per PID.
    Questo garantisce che gli stessi token ID abbiano lo stesso significato
    in ogni sequenza.
    """
    vocab_op = Vocabulary(top_n_op).build(df["Operation"])
    vocab_res = Vocabulary(top_n_res).build(df["Result"])
    vocab_det = Vocabulary(top_n_det).build(df["Detail"])
    return vocab_op, vocab_res, vocab_det


def syscall_vocabularies(
    df: pd.DataFrame,
    top_n_func: int = 100,
) -> Tuple[Vocabulary]:
    """Costruisce vocabolari globali per Function"""
    return Vocabulary(top_n_func).build(df["Function"])
