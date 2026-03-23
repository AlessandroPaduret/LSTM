"""
Dataset PyTorch per sequenze di API call — LAZY VERSION.

op_token  (int scalare)  se parser.py è la versione aggiornata
op_tokens (lista int)    se parser.py è la versione originale
Il collate rileva automaticamente quale formato è presente.
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
        stride:  int = config.STRIDE,
    ):
        self.seq_len  = seq_len
        self.pid_data = pid_data
        self.index: List[Tuple[int, int]] = []

        for pid, df in pid_data.items():
            n = len(df)
            if n < seq_len:
                self.index.append((pid, 0))
            else:
                for start in range(0, n - seq_len + 1, stride):
                    self.index.append((pid, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> pd.DataFrame:
        pid, start = self.index[idx]
        df = self.pid_data[pid]
        return df.iloc[start : start + self.seq_len]

'''
Dataframe con colonne:
- dt  (float scalare)
- op  (int scalare)  se parser.py è la versione aggiornata
- res (int scalare)
- det (lista int)
- label (int scalare)
'''

def collate_api(batch: List[pd.DataFrame]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len    = max(len(df) for df in batch)

    op_idx  = np.zeros((batch_size, max_len), dtype=np.int64)
    res_idx = np.zeros((batch_size, max_len), dtype=np.int64)
    dt_arr  = np.zeros((batch_size, max_len, 1), dtype=np.float32)

    det_tokens_flat: List[int] = []
    det_offsets:     List[int] = [0]
    labels:          List[int] = []

    for b, df in enumerate(batch):
        for t, (_, row) in enumerate(df.iterrows()):
            
            op_toks  = row["op"]
            res_toks = row["res"]
            op_idx[b, t]  = int(op_toks[0])  if (isinstance(op_toks,  list) and op_toks)  else 0
            res_idx[b, t] = int(res_toks[0]) if (isinstance(res_toks, list) and res_toks) else 0

            dt_arr[b, t, 0] = float(row["dt"])

            det_toks = row["det"]
            det_tokens_flat.extend(det_toks)
            det_offsets.append(det_offsets[-1] + len(det_toks))

        # padding det per i timestep mancanti
        for _ in range(max_len - len(df)):
            det_tokens_flat.append(0)
            det_offsets.append(det_offsets[-1] + 1)

        labels.append(int(df["label"].max()))

    # ── Chiavi "op" e "res" (coerenti con _batch_to_device in train.py) ────────
    return {
        "op":  torch.from_numpy(op_idx),
        "res": torch.from_numpy(res_idx),       
        "det": (
            torch.tensor(det_tokens_flat, dtype=torch.long),
            torch.tensor(det_offsets[:-1], dtype=torch.long),
        ),
        "dt":    torch.from_numpy(dt_arr),         # (B, T, 1)
        "label": torch.tensor(labels, dtype=torch.float32),
    }


def make_dataloaders(
    pid_data:    Dict[int, pd.DataFrame],
    seq_len:     int   = config.SEQ_LEN,
    stride:      int   = config.STRIDE,
    batch_size:  int   = config.BATCH_SIZE,
    train_ratio: float = config.TRAIN_RATIO,
    seed:        int   = config.SEED,
    num_workers: int   = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader]:
    pids = list(pid_data.keys())
    rng  = np.random.default_rng(seed)
    rng.shuffle(pids)

    split      = int(len(pids) * train_ratio)
    train_pids = pids#[:split]
    val_pids   = pids#[split:]

    train_ds = APIDataset({p: pid_data[p] for p in train_pids}, seq_len, stride)
    val_ds   = APIDataset({p: pid_data[p] for p in val_pids},   seq_len, stride)

    print(f"  Campioni train   : {len(train_ds):,}")
    print(f"  Campioni val     : {len(val_ds):,}")

    worker_kwargs = dict(
        num_workers        = num_workers,
        persistent_workers = num_workers > 0,
        pin_memory         = False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_api, **worker_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_api, **worker_kwargs)

    return train_loader, val_loader