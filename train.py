import os
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path

# Parallelismo intra-operazione: PyTorch userà tutti i core disponibili per CPU
torch.set_num_threads(os.cpu_count() or 4)

# Assumo che i file precedenti siano salvati come config.py, dataset.py e models.py
import config
from dataset import RansomwareSyscallDataLoader
from model import RansomwareLSTM, RansomwareARILSTM

# ── Helpers ────────────────────────────────────────────────────────────────────


def compute_class_weight(dataset) -> float:
    """
    Calcola il peso delle classi per bilanciare la loss function.
    Scorre velocemente il dataset per contare le etichette positive e negative.
    """
    total_pos = 0
    total_neg = 0

    for _, labels in dataset:
        # La label è la stessa per tutta la finestra, prendiamo la prima
        lbl = labels[0].item()
        if lbl == 1.0:
            total_pos += 1
        elif lbl == 0.0:
            total_neg += 1

    if total_pos == 0:
        return 1.0
    return float(total_neg) / float(total_pos)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    batch_count = 0  # <-- AGGIUNGI QUESTO

    for features, labels in loader:
        batch_count += 1  # <-- AGGIUNGI QUESTO
        features = features.to(device)
        seq_labels = labels[:, 0].to(device)

        logits = model(features).squeeze(-1)
        loss = criterion(logits, seq_labels)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > 0.5).cpu().int().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(seq_labels.cpu().int().numpy().tolist())

    # Usa il contatore invece di len(loader)
    avg_loss = total_loss / max(batch_count, 1)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return dict(loss=avg_loss, acc=accuracy, prec=precision, rec=recall, f1=f1)


# ── Main Training Loop ─────────────────────────────────────────────────────────


def train(use_ari: bool = False):
    torch.manual_seed(getattr(config, "SEED", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  torch threads: {torch.get_num_threads()}")

    # 1. Caricamento del Vocabolario Globale
    vocab_path = getattr(config, "VOCAB_FILE", "./vocabolario_global_syscalls.json")
    print(f"\nCaricamento vocabolario da: {vocab_path}...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"  Vocabolario globale: {vocab_size} token")

    # 2. Inizializzazione DataLoader
    features_dir = getattr(config, "FEATURES_DIR", "./features_per_processo")
    print(f"\nCreazione degli split Train/Val dalla cartella: {features_dir}...")

    train_loader, val_loader = RansomwareSyscallDataLoader.create_splits_from_folder(
        data_dir=features_dir,
        val_ratio=0.2,
        batch_size=getattr(config, "BATCH_SIZE", 32),
        strategy="time-windows",
        time_window_secs=10.0,  # O qualsiasi logica preferisci per le tue API
    )

    # 3. Inizializzazione Modello
    ModelClass = RansomwareARILSTM if use_ari else RansomwareLSTM
    model = ModelClass(vocab_size=vocab_size).to(device)

    print(f"\nModello selezionato: {ModelClass.__name__}")
    print(f"  Parametri totali: {sum(p.numel() for p in model.parameters()):,}\n")

    # 4. Bilanciamento delle Classi e Loss
    print("Calcolo pesi per la Loss (BCEWithLogitsLoss)...")
    pos_weight = compute_class_weight(train_loader.dataset)
    print(f"  pos_weight (neg/pos ratio): {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(config, "LR", 0.001))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    # Assicurati che la cartella dei modelli esista
    model_dir = Path(getattr(config, "MODEL_DIR", "./saved_models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_model.pt"

    best_f1 = 0.0
    patience_cnt = 0
    patience_max = getattr(config, "PATIENCE", 15)
    epochs = getattr(config, "EPOCHS", 50)

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()

        for features, labels in train_loader:
            features = features.to(device)
            # Estraiamo l'unica label necessaria per la Many-to-One
            seq_labels = labels[:, 0].to(device)

            optimizer.zero_grad()

            # Squeeze per portare (B, 1) a (B,)
            logits = model(features).squeeze(-1)
            loss = criterion(logits, seq_labels)

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), getattr(config, "CLIP_GRAD", 1.0)
            )
            optimizer.step()

        # Valutazione periodica (ogni 5 epoche o nelle prime 5)
        if epoch % 5 == 0 or epoch <= 5:
            train_metrics = evaluate(model, train_loader, criterion, device)
            has_val = len(val_loader.dataset.file_paths) > 0
            val_metrics = (
                evaluate(model, val_loader, criterion, device) if has_val else {}
            )

            val_str = (
                f"  val  → loss {val_metrics['loss']:.4f} | "
                f"acc {val_metrics['acc']:.3f} | f1 {val_metrics['f1']:.3f}"
                if val_metrics
                else "  (no val data)"
            )
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"acc {train_metrics['acc']:.3f} | "
                f"f1 {train_metrics['f1']:.3f}"
            )
            print(val_str)

            monitor_f1 = (
                val_metrics.get("f1", train_metrics["f1"])
                if val_metrics
                else train_metrics["f1"]
            )
            scheduler.step(monitor_f1)

            if monitor_f1 > best_f1:
                best_f1 = monitor_f1
                patience_cnt = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ Miglior modello salvato (F1={best_f1:.3f})")
            else:
                patience_cnt += 1
                if patience_cnt >= patience_max:
                    print(f"\nEarly stopping a epoch {epoch} (best F1={best_f1:.3f})")
                    break

    # 6. Report Finale
    print("\n" + "=" * 60)
    print("Risultati finali sul training set (Best Model):")
    model.load_state_dict(torch.load(best_model_path))
    final_train = evaluate(model, train_loader, criterion, device)
    for k, v in final_train.items():
        print(f"  {k:10s}: {v:.4f}")

    if len(val_loader.dataset.file_paths) > 0:
        print("\nRisultati finali sul validation set (Best Model):")
        final_val = evaluate(model, val_loader, criterion, device)
        for k, v in final_val.items():
            print(f"  {k:10s}: {v:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ransomware LSTM training")
    parser.add_argument(
        "--ari", action="store_true", help="Usa ARI-LSTM (Attention Mechanism)"
    )
    args = parser.parse_args()

    train(use_ari=args.ari)
