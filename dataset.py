import os
import io
import glob
import random
import tempfile
import torch
from typing import Literal
import polars as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ══════════════════════════════════════════════════════════════════
#  1. DATA PROCESSOR (Ottimizzato per le nuove Feature)
# ══════════════════════════════════════════════════════════════════


class SyscallLogProcessor:
    """
    Processore Polars per i file di feature estratti.
    Divide la sequenza di syscall in finestre (temporali o per numero di operazioni).
    """

    FEATURE_COLS = ["Delta_Time", "Token"]
    LABEL_COL = "Is_Ransomware"

    def __init__(self, source: str | io.StringIO):
        # Leggiamo il file CSV delle feature
        self.df = pl.read_csv(source)

    def create_windows(
        self,
        strategy: Literal["n-operations", "time-windows"],
        window_size: int = 10,
        time_window_secs: float = 5.0,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:

        if self.df.is_empty():
            return []

        # Assicuriamoci del tipo di dati corretto
        working_df = self.df.with_columns(
            [
                pl.col("Delta_Time").cast(pl.Float32),
                pl.col("Token").cast(
                    pl.Float32
                ),  # Cast a float per unirlo nel tensor delle feature
                pl.col(self.LABEL_COL).cast(pl.Float32),
            ]
        )

        # ── STRATEGIA 1: Finestre per Numero di Operazioni ──
        if strategy == "n-operations":
            working_df = working_df.with_columns(
                pl.int_range(0, pl.len()).alias("op_index")
            ).with_columns((pl.col("op_index") // window_size).alias("window_id"))

        # ── STRATEGIA 2: Finestre Temporali (Basate sul Delta_Time cumulativo) ──
        elif strategy == "time-windows":
            # Delta_Time è in microsecondi -> cum_sum() / 1e6 ci dà i secondi passati dall'inizio del log
            working_df = working_df.with_columns(
                (pl.col("Delta_Time").cum_sum() / 1e6).alias("elapsed_seconds")
            ).with_columns(
                (pl.col("elapsed_seconds") // time_window_secs).alias("window_id")
            )
        else:
            raise ValueError(f"Strategia sconosciuta: {strategy}")

        # Raggruppiamo per ID della finestra mantenendo rigorosamente l'ordine temporale
        grouped = working_df.group_by(["window_id"], maintain_order=True).agg(
            [pl.col(self.FEATURE_COLS), pl.col(self.LABEL_COL)]
        )

        prepared_windows = []
        for row in grouped.iter_rows(named=True):
            # Estraiamo le liste di Delta_Time e Token per questa finestra
            features_list = [row[f] for f in self.FEATURE_COLS]
            # Trasponiamo la matrice da (2, Lunghezza_Finestra) a (Lunghezza_Finestra, 2)
            features_transposed = list(zip(*features_list))

            if not features_transposed:
                continue

            # Creiamo i tensori PyTorch
            features_tensor = torch.tensor(features_transposed, dtype=torch.float32)
            labels_tensor = torch.tensor(row[self.LABEL_COL], dtype=torch.float32)

            prepared_windows.append((features_tensor, labels_tensor))

        return prepared_windows


# ══════════════════════════════════════════════════════════════════
#  2. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════


class RansomwareSyscallDataset(Dataset):
    """
    Dataset PyTorch che carica e aggrega le finestre di syscall da tutti i file di feature.
    """

    def __init__(
        self,
        file_paths: list[str],
        strategy: Literal["n-operations", "time-windows"],
        window_size: int = 10,
        time_window_secs: float = 5.0,
    ):
        self.windows = []
        for path in file_paths:
            processor = SyscallLogProcessor(path)
            file_windows = processor.create_windows(
                strategy=strategy,
                window_size=window_size,
                time_window_secs=time_window_secs,
            )
            self.windows.extend(file_windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


# ══════════════════════════════════════════════════════════════════
#  3. PYTORCH DATALOADER CUSTOM (Padding)
# ══════════════════════════════════════════════════════════════════


def pad_collate_fn(batch):
    """
    Effettua il padding a zero delle finestre che hanno lunghezze diverse all'interno del batch.
    Per le etichette usiamo -1.0 come valore di padding se si desidera ignorarle nel calcolo della loss,
    oppure 0.0 a seconda della configurazione della tua funzione di Loss.
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # padding_value=0.0 livella le feature (Delta_Time e Token) alla lunghezza massima del batch
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)

    # Per i target/labels usiamo -1.0 come valore di padding (comodo per CrossEntropyLoss(ignore_index=-1))
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1.0)

    return features_padded, labels_padded


class RansomwareSyscallDataLoader(DataLoader):
    """
    DataLoader pronto per l'addestramento LSTM / GRU.
    Dispone del metodo per dividere automaticamente i file in Train e Validation set.
    """

    def __init__(
        self,
        dataset: RansomwareSyscallDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=pad_collate_fn,
            **kwargs,
        )

    @classmethod
    def create_splits_from_folder(
        cls,
        data_dir: str,
        val_ratio: float = 0.2,
        batch_size: int = 32,
        seed: int = 42,
        strategy: Literal["n-operations", "time-windows"] = "n-operations",
        window_size: int = 10,
        time_window_secs: float = 5.0,
    ) -> tuple["RansomwareSyscallDataLoader", "RansomwareSyscallDataLoader"]:

        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        all_files.sort()

        if not all_files:
            raise ValueError(f"Nessun file .csv di feature trovato in {data_dir}")

        random.seed(seed)
        random.shuffle(all_files)

        split_idx = int(len(all_files) * (1 - val_ratio))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        print(
            f"[INFO] File allocati al Train: {len(train_files)} | Al Validation: {len(val_files)}"
        )

        train_dataset = RansomwareSyscallDataset(
            train_files, strategy, window_size, time_window_secs
        )
        val_dataset = RansomwareSyscallDataset(
            val_files, strategy, window_size, time_window_secs
        )

        train_loader = cls(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = cls(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════
#  4. MAIN TEST BLOCK
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("--- Avvio Test del Nuovo DataLoader di Feature ---\n")

    # Creazione di dati finti coerenti con l'estrattore precedente
    # Struttura: Delta_Time (in microsecondi), Token, Is_Ransomware
    # File 1: 3 syscall ravvicinate
    csv_dummy_1 = """Delta_Time,Token,Is_Ransomware
0.0,4,1
540.0,1,1
1200.0,4,1
"""
    # File 2: 5 syscall con un salto temporale netto (per testare le finestre temporali)
    csv_dummy_2 = """Delta_Time,Token,Is_Ransomware
0.0,2,0
150.0,3,0
4000000.0,8,0
200.0,5,0
150000.0,9,0
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "proc_hash_1.csv"), "w") as f:
            f.write(csv_dummy_1)
        with open(os.path.join(tmpdir, "proc_hash_2.csv"), "w") as f:
            f.write(csv_dummy_2)

        # Inizializziamo usando la strategia a finestre temporali da 2 secondi
        # (4000000.0 microsecondi = 4 secondi, quindi genererà più finestre)
        train_loader, val_loader = (
            RansomwareSyscallDataLoader.create_splits_from_folder(
                data_dir=tmpdir,
                val_ratio=0.5,
                batch_size=2,
                strategy="time-windows",
                time_window_secs=2.0,
            )
        )

        print(f"Finestre estratte nel Train Dataset: {len(train_loader.dataset)}")
        print(f"Finestre estratte nel Val Dataset:   {len(val_loader.dataset)}\n")

        print("--- Verifica Struttura Tensori del Primo Batch ---")
        for features, labels in train_loader:
            print(
                f"Shape delle Features: {list(features.shape)} -> (Batch, Lunghezza_Sequenza, Feature_Dim)"
            )
            print(
                f"Shape delle Labels:   {list(labels.shape)} -> (Batch, Lunghezza_Sequenza)"
            )
            print("\nPrimo elemento del Batch (Features):")
            print("Formato di ogni riga: [Delta_Time, Token]")
            print(features[0])
            print("\nCorrispettive Labels di questa sequenza (nota il padding a -1):")
            print(labels[0])
            break
