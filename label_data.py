import os
from pathlib import Path
import polars as pl


def etichetta_syscall_ransomware(cartella_input, cartella_output):
    path_input = Path(cartella_input)
    path_output = Path(cartella_output)

    # Crea la cartella di output principale se non esiste
    path_output.mkdir(parents=True, exist_ok=True)

    # Nome esatto del file da cercare in ogni sottocartella
    nome_file_target = "Syscalls_Syscalls_by_Process,_Function.csv"

    # Usa glob per trovare tutti i file target nelle sottocartelle
    # Evita di processare la cartella di output se inserita dentro quella di input
    file_da_elaborare = [
        p
        for p in path_input.glob(f"*/{nome_file_target}")
        if path_output not in p.parents
    ]

    print(f"Trovati {len(file_da_elaborare)} file da elaborare.\n")

    for csv_path in file_da_elaborare:
        # L'hash del ransomware corrisponde al nome della cartella che contiene il CSV
        ransomware_hash = csv_path.parent.name
        print(f"Elaborazione in corso per l'hash: {ransomware_hash}...")

        try:
            # 1. Legge il file CSV con Polars (molto veloce)
            df = pl.read_csv(csv_path)

            # 2. Aggiunge la colonna con l'etichetta (1 se contiene l'hash, 0 altrimenti)
            # Viene usato .str.contains() per intercettare stringhe come 'hash.exe (PID)'
            df = df.with_columns(
                pl.when(pl.col("Process").str.contains(ransomware_hash[:14]))
                .then(1)
                .otherwise(0)
                .alias("Is_Ransomware")
            )

            # 3. Prepara la struttura della cartella di output rispecchiando l'originale
            output_sottocartella = path_output / f"{ransomware_hash}.csv"
            # output_sottocartella.mkdir(parents=True, exist_ok=True)

            # 4. Salva il nuovo file CSV nella cartella di output
            output_file_path = output_sottocartella  # / nome_file_target
            df.write_csv(output_file_path)

            print(
                f"-> Completato! Salvato in: {output_file_path.relative_to(path_output)}"
            )

        except Exception as e:
            print(f"!! Errore durante l'elaborazione di {ransomware_hash}: {e}")

    print("\nProcesso terminato con successo!")


if __name__ == "__main__":
    # --- CONFIGURAZIONE PERCORSI ---
    # Sostituisci questi percorsi con quelli reali del tuo computer
    CARTELLA_ROOT_INPUT = "ransomware"
    CARTELLA_ROOT_OUTPUT = "data"

    etichetta_syscall_ransomware(CARTELLA_ROOT_INPUT, CARTELLA_ROOT_OUTPUT)
