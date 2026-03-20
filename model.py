"""
Modello LSTM per ransomware detection.

Implementa due varianti:
  - RansomwareLSTM  : LSTM standard con 3 EmbeddingBag + dt
  - RansomwareARILSTM : LSTM con meccanismo ARI (Attended Recent Inputs)
                        dal paper "Attention in RNNs for Ransomware Detection"
                        (Agrawal et al., ICASSP 2019)

La differenza chiave rispetto all'implementazione originale è che i vocabolari
sono ora GLOBALI, quindi gli embedding imparano rappresentazioni coerenti.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

import config


# ── LSTM standard ──────────────────────────────────────────────────────────────

class RansomwareLSTM(nn.Module):
    """
    LSTM con tre EmbeddingBag (Operation, Result, Detail) e delta-time.
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        emb_dims:    Dict[str, int] = None,
        hidden_size: int  = config.HIDDEN_SIZE,
        num_layers:  int  = config.NUM_LAYERS,
        dropout:     float = config.DROPOUT,
    ):
        super().__init__()

        if emb_dims is None:
            emb_dims = {
                "op":  config.EMB_DIM_OP,
                "res": config.EMB_DIM_RES,
                "det": config.EMB_DIM_DET,
            }

        self.emb_op  = nn.EmbeddingBag(vocab_sizes["op"],  emb_dims["op"],  mode="mean", max_norm=1.0)
        self.emb_res = nn.EmbeddingBag(vocab_sizes["res"], emb_dims["res"], mode="mean", max_norm=1.0)
        self.emb_det = nn.EmbeddingBag(vocab_sizes["det"], emb_dims["det"], mode="mean", max_norm=1.0)

        input_dim = emb_dims["op"] + emb_dims["res"] + emb_dims["det"] + 1  # +1 per dt

        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def _embed(
        self,
        op_data:  Tuple[torch.Tensor, torch.Tensor],
        res_data: Tuple[torch.Tensor, torch.Tensor],
        det_data: Tuple[torch.Tensor, torch.Tensor],
        dt:       torch.Tensor,
    ) -> torch.Tensor:
        """Crea il tensore di input (B, T, input_dim) per la LSTM."""
        op_tokens,  op_off  = op_data
        res_tokens, res_off = res_data
        det_tokens, det_off = det_data

        v_op  = self.emb_op( op_tokens,  op_off)
        v_res = self.emb_res(res_tokens, res_off)
        v_det = self.emb_det(det_tokens, det_off)

        batch_size, seq_len, _ = dt.shape
        combined = torch.cat([v_op, v_res, v_det], dim=1)   # (B*T, sum_emb)
        combined = combined.view(batch_size, seq_len, -1)    # (B, T, sum_emb)
        return torch.cat([combined, dt], dim=2)              # (B, T, sum_emb+1)

    def forward(
        self,
        op_data:  Tuple[torch.Tensor, torch.Tensor],
        res_data: Tuple[torch.Tensor, torch.Tensor],
        det_data: Tuple[torch.Tensor, torch.Tensor],
        dt:       torch.Tensor,
    ) -> torch.Tensor:
        x, _ = self.lstm(self._embed(op_data, res_data, det_data, dt))
        return self.classifier(x[:, -1, :])  # logits, usare BCEWithLogitsLoss


# ── ARI cell + LSTM ────────────────────────────────────────────────────────────

class ARICell(nn.Module):
    """
    Attended Recent Inputs cell (Agrawal et al., ICASSP 2019).

    Per ogni timestep t, calcola un vettore di attenzione r_t sui precedenti L input
    e lo inietta nella LSTM insieme all'input corrente x_t.

    La finestra recente viene mantenuta come buffer FIFO.
    """

    def __init__(self, input_dim: int, ari_l: int = config.ARI_L):
        super().__init__()
        self.L       = ari_l
        self.W_dense = nn.Linear(input_dim, input_dim, bias=False)
        self.omega   = nn.Linear(input_dim, 1,         bias=False)

    def forward(
        self,
        x:      torch.Tensor,          # (B, input_dim)
        window: torch.Tensor,          # (B, L, input_dim)  ← storia recente
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        r_t    : (B, input_dim)  vettore di contesto ARI
        window : (B, L, input_dim)  finestra aggiornata (x viene inserita in coda)
        """
        # M_t = tanh(W_d * R_t)   →   (B, L, input_dim)
        M = torch.tanh(self.W_dense(window))

        # alpha_t = softmax(omega^T * M_t)  →  (B, L)
        alpha = F.softmax(self.omega(M).squeeze(-1), dim=1)

        # r_t = R_t^T * alpha_t  →  (B, input_dim)
        r = (window * alpha.unsqueeze(-1)).sum(dim=1)

        # Aggiorna finestra FIFO: rimuovi il più vecchio, aggiungi x in fondo
        window = torch.cat([window[:, 1:, :], x.unsqueeze(1)], dim=1)

        return r, window


class RansomwareARILSTM(RansomwareLSTM):
    """
    LSTM potenziata con il meccanismo ARI.
    Eredita l'embedding da RansomwareLSTM e sostituisce il forward LSTM.
    """

    def __init__(self, vocab_sizes: Dict[str, int], **kwargs):
        super().__init__(vocab_sizes, **kwargs)

        input_dim = (
            config.EMB_DIM_OP + config.EMB_DIM_RES + config.EMB_DIM_DET + 1
        )
        # Proiezione che riceve [x_t || r_t] → stesso input_dim originale
        # (come nell'equazione 4 del paper: Wr * r_t aggiunto ai gate)
        self.ari      = ARICell(input_dim, ari_l=config.ARI_L)
        self.W_r_proj = nn.Linear(input_dim, input_dim, bias=False)

    def forward(
        self,
        op_data:  Tuple[torch.Tensor, torch.Tensor],
        res_data: Tuple[torch.Tensor, torch.Tensor],
        det_data: Tuple[torch.Tensor, torch.Tensor],
        dt:       torch.Tensor,
    ) -> torch.Tensor:
        x_seq = self._embed(op_data, res_data, det_data, dt)  # (B, T, D)
        B, T, D = x_seq.shape

        window  = torch.zeros(B, self.ari.L, D, device=x_seq.device)
        h, c    = (torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x_seq.device),
                   torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x_seq.device))

        outputs = []
        for t in range(T):
            x_t = x_seq[:, t, :]          # (B, D)
            r_t, window = self.ari(x_t, window)

            # Combina input corrente con contesto ARI (addizione proiettata)
            x_aug = (x_t + self.W_r_proj(r_t)).unsqueeze(1)  # (B, 1, D)
            out, (h, c) = self.lstm(x_aug, (h, c))
            outputs.append(out)

        last = outputs[-1].squeeze(1)    # (B, hidden_size)
        return self.classifier(last)
