import os
import json
import hashlib
from pathlib import Path
import polars as pl


def estrai_features_per_processo(cartella_input, cartella_output, file_vocabolario):
    path_input = Path(cartella_input)
    path_output = Path(cartella_output)

    # 1. Crea la cartella delle feature se non esiste
    path_output.mkdir(parents=True, exist_ok=True)

    # 2. Carica il vocabolario globale
    print("Caricamento del vocabolario...")
    with open(file_vocabolario, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabolario caricato! ({len(vocab)} token)")

    # Trova tutti i file .csv etichettati
    file_csv = list(path_input.glob("*.csv"))
    print(f"\nTrovati {len(file_csv)} file da cui estrarre le feature.\n")

    contatore_processi_totali = 0

    for csv_path in file_csv:
        print(f"Elaborazione file: {csv_path.name}...")
        try:
            # 3. Leggi il CSV
            df = pl.read_csv(csv_path)

            # 4. Mappatura Token
            # Sostituiamo il nome della funzione con il suo ID intero. Se non c'è, mettiamo 0 (NONE)
            df = df.with_columns(
                pl.col("Function")
                .replace(vocab, default=0)
                .cast(pl.Int32)
                .alias("Token")
            )

            # 5. Parsing del Tempo (converte la stringa in oggetto Datetime)
            # Gestisce il formato: 2026/04/21 14:23:22.1904238
            df = df.with_columns(
                pl.col("Start Time (UTC)")
                .str.replace(
                    r"(\d{4}/\d{2}/\d{2})(\d{2}:\d{2}:\d{2})", "$1 $2"
                )  # Aggiunge spazio se manca
                .str.to_datetime(format="%Y/%m/%d %H:%M:%S%.f", strict=False)
                .alias("Timestamp")
            )

            # Se dopo il parsing ci sono valori nulli, riempiamo con il timestamp precedente
            # (oppure potresti rimuoverli con .drop_nulls())
            df = df.with_columns(pl.col("Timestamp").forward_fill())

            # 6. Ordinamento vitale: prima per Processo, poi per Tempo
            df = df.sort(["Process", "Timestamp"])

            # 7. Calcolo del Delta Tempo
            # Calcola la differenza dal record precedente. .over("Process") assicura che
            # il calcolo non si accavalli tra processi diversi. Il primo valore sarà null, lo riempiamo con 0.
            df = df.with_columns(
                pl.col("Timestamp")
                .diff()
                .dt.total_microseconds()  # Usiamo i microsecondi per la precisione delle API
                .over("Process")
                .fill_null(0)
                .alias("Delta_Time")
            )

            # 8. Divisione in file separati per ogni processo
            # Raggruppiamo il dataframe per il nome del processo
            for (nome_processo,), gruppo_processo in df.group_by(["Process"]):
                # Creiamo una stringa univoca unendo il nome del CSV e il nome del processo
                # Es: e9dcda99bf..._explorer.exe (5540)
                identificatore_univoco = f"{csv_path.stem}_{nome_processo}"

                # Generiamo un hash breve e sicuro (MD5) per il nome del file
                hash_processo = hashlib.md5(
                    identificatore_univoco.encode("utf-8")
                ).hexdigest()

                nome_file_output = f"{hash_processo}.csv"
                percorso_output = path_output / nome_file_output

                # Selezioniamo solo le colonne utili per il Machine Learning
                df_finale = gruppo_processo.select(
                    ["Delta_Time", "Token", "Is_Ransomware"]
                )

                # Salvataggio
                df_finale.write_csv(percorso_output)
                contatore_processi_totali += 1

        except Exception as e:
            print(f"!! Errore durante l'elaborazione di {csv_path.name}: {e}")

    print("\n" + "=" * 50)
    print("Estrazione feature completata con successo!")
    print(f"Generati {contatore_processi_totali} file univoci in: {cartella_output}")
    print("=" * 50)


if __name__ == "__main__":
    # --- CONFIGURAZIONE PERCORSI ---
    CARTELLA_CON_I_CSV_ETICHETTATI = "./data"
    CARTELLA_OUTPUT_FEATURE = "./features_per_processo"
    FILE_VOCABOLARIO = "./vocab/vocabolario_global_syscalls.json"

    estrai_features_per_processo(
        CARTELLA_CON_I_CSV_ETICHETTATI, CARTELLA_OUTPUT_FEATURE, FILE_VOCABOLARIO
    )
