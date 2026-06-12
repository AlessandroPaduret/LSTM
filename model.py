import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import config


# ── LSTM STANDARD (OTTIMIZZATA) ────────────────────────────────────────────────


class RansomwareLSTM(nn.Module):
    """
    LSTM ottimizzata per lavorare esclusivamente con 2 feature di Syscall:
      - nn.Embedding per il Token della Syscall (mappato dal vocabolario globale)
      - Delta_Time scalare (tempo intercorso dall'ultima syscall in microsecondi)
    """

    def __init__(
        self,
        vocab_size: int,  # Dimensione del vocabolario globale JSON
        emb_dim: int = None,
        hidden_size: int = config.HIDDEN_SIZE,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()

        # Se non passi una dimensione custom, peschiamo EMB_DIM_TOKEN dal tuo file config
        if emb_dim is None:
            emb_dim = getattr(config, "EMB_DIM_TOKEN", 64)

        # Unico embedding necessario: mappa l'ID della funzione in uno spazio continuo
        self.emb_token = nn.Embedding(vocab_size, emb_dim, max_norm=1.0)

        # La dimensione dell'input per la LSTM è: dimensione dell'embedding + 1 (il Delta_Time scalare)
        input_dim = emb_dim + 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),  # Logit singolo per la BCEWithLogitsLoss
        )

    def _embed(self, tokens: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Metodo modulare per mappare i token e concatenare il Delta_Time."""
        v_token = self.emb_token(tokens)  # Output shape: (B, T, emb_dim)
        return torch.cat([v_token, dt], dim=2)  # Output shape: (B, T, emb_dim + 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            features: Tensore proveniente dal nostro DataLoader con shape (B, T, 2)
                      dove:
                      - features[:, :, 0] -> Delta_Time
                      - features[:, :, 1] -> Token ID
        """
        # 1. Spacchettiamo il tensore in linea con la struttura del nostro DataLoader
        dt = features[:, :, 0].unsqueeze(-1)  # Estrae Delta_Time -> (B, T, 1)

        # Nel DataLoader abbiamo castato tutto a Float32 per unirlo nel file CSV.
        # L'Embedding di PyTorch richiede indici interi di tipo Long.
        tokens = features[:, :, 1].long()  # Estrae i Token -> (B, T)

        # 2. Generiamo l'input embedded concatenato tramite la funzione protetta
        x_seq = self._embed(tokens, dt)

        # 3. Passaggio nella LSTM
        out_seq, _ = self.lstm(x_seq)

        # 4. Many-to-One: Estraiamo l'output dell'ultimo timestep valido per il classificatore
        return self.classifier(out_seq[:, -1, :])


# ── ARI CELL + LSTM (OTTIMIZZATA) ──────────────────────────────────────────────


class ARICell(nn.Module):
    """
    Attended Recent Inputs cell (Agrawal et al., ICASSP 2019).
    Lavora sulla nuova dimensione input_dim ridotta, mantenendo intatta la logica di attenzione.
    """

    def __init__(self, input_dim: int, ari_l: int = config.ARI_L):
        super().__init__()
        self.L = ari_l
        self.W_dense = nn.Linear(input_dim, input_dim, bias=False)
        self.omega = nn.Linear(input_dim, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # (B, input_dim)
        window: torch.Tensor,  # (B, L, input_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M = torch.tanh(self.W_dense(window))  # (B, L, input_dim)
        alpha = F.softmax(self.omega(M).squeeze(-1), dim=1)  # (B, L)
        r = (window * alpha.unsqueeze(-1)).sum(dim=1)  # (B, input_dim)
        window = torch.cat([window[:, 1:, :], x.unsqueeze(1)], dim=1)
        return r, window


class RansomwareARILSTM(RansomwareLSTM):
    """LSTM potenziata con meccanismo di attenzione ARI (eredita embedding e logiche da RansomwareLSTM)."""

    def __init__(self, vocab_size: int, emb_dim: int = None, **kwargs):
        super().__init__(vocab_size, emb_dim, **kwargs)

        # Calcoliamo la dimensione dell'input basandoci sull'effettivo embedding creato
        actual_emb_dim = self.emb_token.embedding_dim
        input_dim = actual_emb_dim + 1

        self.ari = ARICell(input_dim, ari_l=config.ARI_L)
        self.W_r_proj = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Spacchettamento speculare rispetto alla classe madre
        dt = features[:, :, 0].unsqueeze(-1)  # (B, T, 1)
        tokens = features[:, :, 1].long()  # (B, T)

        x_seq = self._embed(tokens, dt)  # (B, T, D)
        B, T, D = x_seq.shape

        # Inizializzazione degli stati interni e della finestra ARI per l'iterazione temporale
        window = torch.zeros(B, self.ari.L, D, device=x_seq.device)
        h = torch.zeros(
            self.lstm.num_layers, B, self.lstm.hidden_size, device=x_seq.device
        )
        c = torch.zeros(
            self.lstm.num_layers, B, self.lstm.hidden_size, device=x_seq.device
        )

        outputs = []
        # Ciclo esplicito sui timestep (necessario per aggiornare la memoria di ARI passo dopo passo)
        for t in range(T):
            x_t = x_seq[:, t, :]
            r_t, window = self.ari(x_t, window)
            x_aug = (x_t + self.W_r_proj(r_t)).unsqueeze(1)  # (B, 1, D)
            out, (h, c) = self.lstm(x_aug, (h, c))
            outputs.append(out)

        last = outputs[-1].squeeze(1)
        return self.classifier(last)
