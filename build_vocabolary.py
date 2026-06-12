import json
from pathlib import Path
import polars as pl


def crea_vocabolario_funzioni(cartella_csv, file_output_json):
    path_cartella = Path(cartella_csv)

    # 1. Trova tutti i file .csv nella cartella
    file_csv = list(path_cartella.glob("*.csv"))

    if not file_csv:
        print(f"Nessun file .csv trovato in {cartella_csv}")
        return

    print(f"Trovati {len(file_csv)} file .csv da analizzare.")

    # Usiamo un set per raccogliere le funzioni uniche (evita duplicati)
    funzioni_uniche = set()

    # 2. Scansiona ogni file ed estrae i token unici dalla colonna 'Function'
    for csv_path in file_csv:
        print(f"Lettura di: {csv_path.name}...")
        try:
            # Leggiamo solo la colonna 'Function' per massima efficienza e velocità
            df_funzioni = pl.read_csv(csv_path, columns=["Function"])

            # Estraiamo i valori unici di questo file e li aggiungiamo al set globale
            funzioni_del_file = df_funzioni["Function"].unique().to_list()

            # Rimuoviamo eventuali valori nulli/NaN
            funzioni_pulite = {f for f in funzioni_del_file if f is not None}

            funzioni_uniche.update(funzioni_pulite)

        except Exception as e:
            print(f"!! Errore durante la lettura di {csv_path.name}: {e}")

    # 3. Ordiniamo alfabeticamente le funzioni trovate per avere un vocabolario deterministico
    lista_funzioni_ordinate = sorted(list(funzioni_uniche))

    # 4. Creiamo il vocabolario vero e proprio (Mapping stringa -> ID)
    # Assegniamo l'ID 0 a "NONE" o "PAD" come nel tuo vecchio script, e partiamo da 1 per le funzioni
    vocabolario = {"NONE": 0}
    for idx, funzione in enumerate(lista_funzioni_ordinate, start=1):
        vocabolario[funzione] = idx

    # 5. Salviamo il vocabolario in formato JSON
    with open(file_output_json, "w", encoding="utf-8") as f:
        json.dump(vocabolario, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 40)
    print(f"Vocabolario globale creato con successo!")
    print(f"Numero totale di funzioni (token) censite: {len(vocabolario)}")
    print(f"File salvato in: {file_output_json}")
    print("=" * 40)


if __name__ == "__main__":
    # --- CONFIGURAZIONE PERCORSI ---
    # La cartella dove hai salvato i file etichettati (es. e9dcda99bf1778.csv)
    CARTELLA_CON_I_CSV = "./data"

    # Il file JSON in cui salvare il vocabolario finale
    FILE_VOCABOLARIO_OUTPUT = "./vocab/vocabolario_global_syscalls.json"

    crea_vocabolario_funzioni(CARTELLA_CON_I_CSV, FILE_VOCABOLARIO_OUTPUT)
