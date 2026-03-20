"""
Dataset PyTorch per sequenze di API call.

Ogni campione è una finestra scorrevole di lunghezza SEQ_LEN
estratta dalle API call di un singolo PID.
La label è determinata dal valore maggioritario nella finestra
(una finestra con anche una sola chiamata ransomware = positiva).
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

import config


class APIDataset(Dataset):
    def __init__(
        self,
        pid_data: Dict[int, pd.DataFrame],
        seq_len: int = config.SEQ_LEN,
    ):
        self.seq_len = seq_len
        self.samples: List[pd.DataFrame] = []

        for pid, df in pid_data.items():
            n = len(df)
            if n < seq_len:
                # Sequenza troppo corta: la usiamo tutta con padding implicito nel collate
                self.samples.append(df.copy())
            else:
                # Sliding window con stride = 1
                for start in range(0, n - seq_len + 1):
                    self.samples.append(df.iloc[start : start + seq_len].copy())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> pd.DataFrame:
        return self.samples[idx]


def collate_api(batch: List[pd.DataFrame]) -> Dict[str, torch.Tensor]:
    """
    Collate function per DataLoader.

    Ogni riga di ogni DataFrame nella batch contiene liste di token IDs
    (potenzialmente di lunghezze diverse). Usiamo EmbeddingBag che vuole
    un tensore flat + offsets.

    Returns un dict con:
      op  : (tokens_flat, offsets)   per EmbeddingBag
      res : (tokens_flat, offsets)
      det : (tokens_flat, offsets)
      dt  : Tensor (B, T, 1)
      label: Tensor (B,)  — 1 se la sequenza contiene almeno un evento ransomware
    """
    seq_len = len(batch[0])  # T (può variare solo se seq < SEQ_LEN)

    # Pre-alloca tensori per dt e raccogli token
    batch_size = len(batch)

    dt_list:   List[List[float]] = []
    op_tokens_flat:  List[int] = []
    res_tokens_flat: List[int] = []
    det_tokens_flat: List[int] = []

    op_offsets:  List[int] = [0]
    res_offsets: List[int] = [0]
    det_offsets: List[int] = [0]

    labels: List[int] = []

    for df in batch:
        dt_seq = []
        for _, row in df.iterrows():
            dt_seq.append(float(row["dt"]))

            op_toks  = row["op_tokens"]
            res_toks = row["res_tokens"]
            det_toks = row["det_tokens"]

            op_tokens_flat.extend(op_toks)
            res_tokens_flat.extend(res_toks)
            det_tokens_flat.extend(det_toks)

            op_offsets.append(op_offsets[-1]  + len(op_toks))
            res_offsets.append(res_offsets[-1] + len(res_toks))
            det_offsets.append(det_offsets[-1] + len(det_toks))

        dt_list.append(dt_seq)

        # Label: 1 se almeno un evento ransomware nella finestra
        labels.append(int(df["label"].max()))

    # Pad le sequenze dt alla lunghezza massima nella batch
    max_len = max(len(d) for d in dt_list)
    dt_padded = np.zeros((batch_size, max_len, 1), dtype=np.float32)
    for i, d in enumerate(dt_list):
        dt_padded[i, : len(d), 0] = d

    # EmbeddingBag: rimuove l'ultimo offset (non serve)
    n_steps = batch_size * max_len

    return {
        "op":    (torch.tensor(op_tokens_flat,  dtype=torch.long),
                  torch.tensor(op_offsets[:-1],  dtype=torch.long)),
        "res":   (torch.tensor(res_tokens_flat, dtype=torch.long),
                  torch.tensor(res_offsets[:-1], dtype=torch.long)),
        "det":   (torch.tensor(det_tokens_flat, dtype=torch.long),
                  torch.tensor(det_offsets[:-1], dtype=torch.long)),
        "dt":    torch.tensor(dt_padded,         dtype=torch.float32),
        "label": torch.tensor(labels,            dtype=torch.float32),
    }


def make_dataloaders(
    pid_data: Dict[int, pd.DataFrame],
    seq_len:     int   = config.SEQ_LEN,
    batch_size:  int   = config.BATCH_SIZE,
    train_ratio: float = config.TRAIN_RATIO,
    seed:        int   = config.SEED,
) -> Tuple[DataLoader, DataLoader]:
    """
    Crea train e validation DataLoader suddividendo i PID
    (non le singole righe) per evitare data leakage.
    """
    pids = list(pid_data.keys())
    rng  = np.random.default_rng(seed)
    rng.shuffle(pids)

    split      = int(len(pids) * train_ratio)
    train_pids = pids[:split]
    val_pids   = pids[split:]

    train_data = {p: pid_data[p] for p in train_pids}
    val_data   = {p: pid_data[p] for p in val_pids}

    train_ds = APIDataset(train_data, seq_len)
    val_ds   = APIDataset(val_data,   seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_api)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_api)

    return train_loader, val_loader
