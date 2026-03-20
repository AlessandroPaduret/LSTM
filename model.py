"""
Modello LSTM per ransomware detection.

Cambiamenti rispetto alla versione precedente:
  - emb_op e emb_res: EmbeddingBag → nn.Embedding
      Operation e Result hanno vocabolari piccoli (24 e 31 token) e dopo
      clean_text producono quasi sempre un singolo token.
      nn.Embedding è più semplice, non richiede offset flat, ed è più veloce.
  - emb_det: rimane EmbeddingBag (Detail può avere più token per riga)
  - _embed: firma cambiata — op e res sono ora tensori (B, T) di interi
  - HIDDEN_SIZE / NUM_LAYERS ridotti in config per velocità su CPU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

import config


# ── LSTM standard ──────────────────────────────────────────────────────────────

class RansomwareLSTM(nn.Module):
    """
    LSTM con:
      - nn.Embedding per Operation e Result  (singolo token per timestep)
      - nn.EmbeddingBag per Detail           (multi-token per timestep)
      - delta-time scalare
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        emb_dims:    Dict[str, int] = None,
        hidden_size: int   = config.HIDDEN_SIZE,
        num_layers:  int   = config.NUM_LAYERS,
        dropout:     float = config.DROPOUT,
    ):
        super().__init__()

        if emb_dims is None:
            emb_dims = {
                "op":  config.EMB_DIM_OP,
                "res": config.EMB_DIM_RES,
                "det": config.EMB_DIM_DET,
            }

        # Embedding semplice: (B, T) → (B, T, emb_dim)
        self.emb_op  = nn.Embedding(vocab_sizes["op"],  emb_dims["op"],  max_norm=1.0)
        self.emb_res = nn.Embedding(vocab_sizes["res"], emb_dims["res"], max_norm=1.0)

        # EmbeddingBag: (flat_tokens, offsets) → (B*T, emb_dim)
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
        op_ids:   torch.Tensor,                          # (B, T)
        res_ids:  torch.Tensor,                          # (B, T)
        det_data: Tuple[torch.Tensor, torch.Tensor],     # (flat, offsets) per EmbeddingBag
        dt:       torch.Tensor,                          # (B, T, 1)
    ) -> torch.Tensor:
        """Restituisce il tensore di input (B, T, input_dim) per la LSTM."""
        v_op  = self.emb_op(op_ids)    # (B, T, emb_op)
        v_res = self.emb_res(res_ids)  # (B, T, emb_res)

        det_tokens, det_off = det_data
        v_det = self.emb_det(det_tokens, det_off)  # (B*T, emb_det)
        B, T, _ = dt.shape
        v_det = v_det.view(B, T, -1)               # (B, T, emb_det)

        return torch.cat([v_op, v_res, v_det, dt], dim=2)  # (B, T, input_dim)

    def forward(
        self,
        op_ids:   torch.Tensor,
        res_ids:  torch.Tensor,
        det_data: Tuple[torch.Tensor, torch.Tensor],
        dt:       torch.Tensor,
    ) -> torch.Tensor:
        x, _ = self.lstm(self._embed(op_ids, res_ids, det_data, dt))
        return self.classifier(x[:, -1, :])   # logit scalare per BCEWithLogitsLoss


# ── ARI cell + LSTM ────────────────────────────────────────────────────────────

class ARICell(nn.Module):
    """
    Attended Recent Inputs cell (Agrawal et al., ICASSP 2019).
    """

    def __init__(self, input_dim: int, ari_l: int = config.ARI_L):
        super().__init__()
        self.L       = ari_l
        self.W_dense = nn.Linear(input_dim, input_dim, bias=False)
        self.omega   = nn.Linear(input_dim, 1,         bias=False)

    def forward(
        self,
        x:      torch.Tensor,   # (B, input_dim)
        window: torch.Tensor,   # (B, L, input_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M     = torch.tanh(self.W_dense(window))            # (B, L, input_dim)
        alpha = F.softmax(self.omega(M).squeeze(-1), dim=1) # (B, L)
        r     = (window * alpha.unsqueeze(-1)).sum(dim=1)   # (B, input_dim)
        window = torch.cat([window[:, 1:, :], x.unsqueeze(1)], dim=1)
        return r, window


class RansomwareARILSTM(RansomwareLSTM):
    """LSTM con meccanismo ARI (eredita embedding da RansomwareLSTM)."""

    def __init__(self, vocab_sizes: Dict[str, int], **kwargs):
        super().__init__(vocab_sizes, **kwargs)
        input_dim = config.EMB_DIM_OP + config.EMB_DIM_RES + config.EMB_DIM_DET + 1
        self.ari      = ARICell(input_dim, ari_l=config.ARI_L)
        self.W_r_proj = nn.Linear(input_dim, input_dim, bias=False)

    def forward(
        self,
        op_ids:   torch.Tensor,
        res_ids:  torch.Tensor,
        det_data: Tuple[torch.Tensor, torch.Tensor],
        dt:       torch.Tensor,
    ) -> torch.Tensor:
        x_seq = self._embed(op_ids, res_ids, det_data, dt)  # (B, T, D)
        B, T, D = x_seq.shape

        window = torch.zeros(B, self.ari.L, D, device=x_seq.device)
        h = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x_seq.device)
        c = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x_seq.device)

        outputs = []
        for t in range(T):
            x_t = x_seq[:, t, :]
            r_t, window = self.ari(x_t, window)
            x_aug = (x_t + self.W_r_proj(r_t)).unsqueeze(1)   # (B, 1, D)
            out, (h, c) = self.lstm(x_aug, (h, c))
            outputs.append(out)

        last = outputs[-1].squeeze(1)
        return self.classifier(last)