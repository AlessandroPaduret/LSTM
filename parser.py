
import pandas as pd
import tokenizer as tk

'''
Questo modulo si occupa di leggere parda un dataframe e mette 
i token al posto del testo
e il dt al posto del tempo
- inuput: Dataframe
- output: dataframe con dt, Operazioni tokenizzate, Dettagli tokenizzate e infine i risultati dell'operazione
'''
def parse_data(data: pd.DataFrame) -> pd.DataFrame:

    data['dt'] = pd.to_datetime(
        data['Time of Day'].str.strip(),
        format='%I:%M:%S.%f %p',
        errors='coerce'
    ).diff().dt.total_seconds().fillna(0) # Calcola la differenza in secondi tra le righe e riempie i valori NaN con 0
    
    # Tokenizza le operazioni, i dettagli e i risultati
    ops = tk.tokenize_column(data['Operation'], top_n=100)
    data['Operation'] = data['Operation'].apply(lambda x: tk.get_multi_tokens(x, ops))
    det = tk.tokenize_column(data['Detail'], top_n=100)
    data['Detail'] = data['Detail'].apply(lambda x: tk.get_multi_tokens(x, det))
    res = tk.tokenize_column(data['Result'], top_n=100)
    data['Result'] = data['Result'].apply(lambda x: tk.get_multi_tokens(x, res))

    return data

'''
Una funzione che serve a dividere un grande datagrame in più piccoli, uno per ogni PID, in modo da poter addestrare un modello LSTM per ogni processo
- input: dataframe con tutte le API di tutti i processi
- output: dizionario con chiave il PID e valore il dataframe con le API di quel processo
'''
def split_by_pid(df: pd.DataFrame) -> dict:
    pid_dict = {}
    for pid, group in df.groupby('PID'):
        pid_dict[pid] = group.reset_index(drop=True)
    return pid_dict

'''
Esempio di utilizzo delle funzioni sopra, legge il file CSV, lo divide per PID e poi tokenizza ogni dataframe
'''
def test():
    file_path = "/home/arch_btw/cloud/GoogleDrive/unipd/2.2/Progetto Cyber/data/API/Logfile.CSV"
    csv = pd.read_csv(file_path)
    #rimuoviamo le colonne che non ci servono più
    csv.drop(columns=['Path'], inplace=True)
    datas = split_by_pid(csv)

    print("Logfile.CSV spezzato con successo.")
    print(datas)
    for pid in datas:
        parse_data(datas[pid])
        datas[pid].drop(columns=['Time of Day', 'PID', 'Process Name'], inplace=True)
    print("Il dataframe è stato diviso in più piccoli dataframe, uno per ogni PID.")
    print(datas)

# test()