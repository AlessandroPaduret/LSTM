"""
Training loop per ransomware detection LSTM.

Caratteristiche rispetto all'originale:
  - Gradient clipping (essenziale per LSTM stabili)
  - Class weights automatici per dataset sbilanciati
  - Train / validation split per PID (no data leakage)
  - Early stopping
  - Salvataggio del miglior modello
  - Metriche complete (loss, accuracy, precision, recall, F1)
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import config
from parser  import load_and_prepare
from dataset import make_dataloaders
from model   import RansomwareLSTM, RansomwareARILSTM


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_class_weight(pid_data) -> float:
    """
    Calcola il peso della classe positiva per BCEWithLogitsLoss.
    pos_weight = n_negative / n_positive
    """
    total_pos = sum(df["label"].sum()   for df in pid_data.values())
    total_neg = sum((df["label"] == 0).sum() for df in pid_data.values())
    if total_pos == 0:
        return 1.0
    return float(total_neg) / float(total_pos)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        op    = (batch["op"][0].to(device),  batch["op"][1].to(device))
        res   = (batch["res"][0].to(device), batch["res"][1].to(device))
        det   = (batch["det"][0].to(device), batch["det"][1].to(device))
        dt    = batch["dt"].to(device)
        label = batch["label"].to(device)

        logits = model(op, res, det, dt).squeeze()
        loss   = criterion(logits, label)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > 0.5).cpu().int().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(label.cpu().int().numpy().tolist())

    avg_loss  = total_loss / max(len(loader), 1)
    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)

    return dict(loss=avg_loss, acc=accuracy, prec=precision, rec=recall, f1=f1)


# ── Main training loop ─────────────────────────────────────────────────────────

def train(use_ari: bool = False):
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dati ────────────────────────────────────────────────────────────────
    print("Caricamento e tokenizzazione globale del dataset...")
    pid_data, vocab_op, vocab_res, vocab_det = load_and_prepare()

    print(f"  PID trovati      : {len(pid_data)}")
    print(f"  Vocab Operation  : {vocab_op.size} token")
    print(f"  Vocab Result     : {vocab_res.size} token")
    print(f"  Vocab Detail     : {vocab_det.size} token")

    train_loader, val_loader = make_dataloaders(pid_data)
    print(f"  Train batches    : {len(train_loader)}")
    print(f"  Val   batches    : {len(val_loader)}")

    # ── Modello ─────────────────────────────────────────────────────────────
    vocab_sizes = {
        "op":  vocab_op.size,
        "res": vocab_res.size,
        "det": vocab_det.size,
    }

    ModelClass = RansomwareARILSTM if use_ari else RansomwareLSTM
    model = ModelClass(vocab_sizes).to(device)
    print(f"\nModello: {ModelClass.__name__}")
    print(f"  Parametri: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Loss con class weights ───────────────────────────────────────────────
    pos_weight = compute_class_weight(pid_data)
    print(f"pos_weight (neg/pos ratio): {pos_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    # ── Training ─────────────────────────────────────────────────────────────
    best_f1      = 0.0
    patience_cnt = 0
    patience_max = 20  # early stopping

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            op    = (batch["op"][0].to(device),  batch["op"][1].to(device))
            res   = (batch["res"][0].to(device), batch["res"][1].to(device))
            det   = (batch["det"][0].to(device), batch["det"][1].to(device))
            dt    = batch["dt"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(op, res, det, dt).squeeze()
            loss   = criterion(logits, label)
            loss.backward()

            # Gradient clipping — previene exploding gradients nelle LSTM
            nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)

            optimizer.step()
            total_loss += loss.item()

        train_metrics = evaluate(model, train_loader, criterion, device)
        
        # Stampa ogni 10 epoche o all'inizio
        if epoch % 10 == 0 or epoch <= 5:
            val_metrics = evaluate(model, val_loader, criterion, device) if len(val_loader) > 0 else {}
            val_str = (f"  val  → loss {val_metrics['loss']:.4f} | "
                       f"acc {val_metrics['acc']:.3f} | f1 {val_metrics['f1']:.3f}"
                       if val_metrics else "  (no val data)")
            print(
                f"Epoch {epoch:3d}/{config.EPOCHS} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"acc {train_metrics['acc']:.3f} | "
                f"f1 {train_metrics['f1']:.3f}"
            )
            print(val_str)

            # Scheduler e early stopping su val F1 (o train se non c'è val)
            monitor_f1 = val_metrics.get("f1", train_metrics["f1"]) if val_metrics else train_metrics["f1"]
            scheduler.step(monitor_f1)

            if monitor_f1 > best_f1:
                best_f1 = monitor_f1
                patience_cnt = 0
                torch.save(model.state_dict(), config.MODEL_DIR / "best_model.pt")
                print(f"  ✓ Miglior modello salvato (F1={best_f1:.3f})")
            else:
                patience_cnt += 1
                if patience_cnt >= patience_max:
                    print(f"\nEarly stopping a epoch {epoch} (best F1={best_f1:.3f})")
                    break

    # ── Risultati finali ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Risultati finali sul training set:")
    model.load_state_dict(torch.load(config.MODEL_DIR / "best_model.pt"))
    final = evaluate(model, train_loader, criterion, device)
    for k, v in final.items():
        print(f"  {k:10s}: {v:.4f}")

    if len(val_loader) > 0:
        print("\nRisultati finali sul validation set:")
        final_val = evaluate(model, val_loader, criterion, device)
        for k, v in final_val.items():
            print(f"  {k:10s}: {v:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ransomware LSTM training")
    parser.add_argument("--ari", action="store_true", help="Usa ARI-LSTM (attenzione locale)")
    args = parser.parse_args()

    train(use_ari=args.ari)
