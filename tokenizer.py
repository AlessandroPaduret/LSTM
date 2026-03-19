#!/home/arch_btw/python-env/bin/python
import re
from typing import List, Tuple
import pandas as pd

'''
Questo script contiene funzioni per pulire e normalizzare la colonna "Detail" del dataset
- Rimuove date e orari
- Rimuove file *.* e simili
- Rimuove indirizzi esadecimali (0x...)
- Rimuove i percorsi file (Windows style C:\... o Linux /home/...)
- Rimuove tutto ciò che non è una lettera
- Converte tutto in minuscolo
- Sostituisce stringhe vuote o NaN con "NONE"
'''

def clean_detail(detail_str):
    if pd.isna(detail_str) or detail_str == "":
        return "NONE"
    
    # 1. Rimuoviamo date e orari
    # Esempio: rimuove 7/31/2024 1:05:27 AM
    detail_str = re.sub(r'\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M', '', detail_str)
    
    # 2. Rimuovi file *.* e simili
    detail_str = re.sub(r'\b\w+\.\w+\b', ' file ', detail_str)

    # 3. Rimuovi date/ora e indirizzi esadecimali (0x...)
    detail_str = re.sub(r'0x[0-9a-fA-F]+', ' hex ', detail_str)
    
    # 4. Rimuovi i percorsi file (Windows style C:\... o Linux /home/...)
    # Sostituiamo con "path" per mantenere l'informazione che c'era un file
    detail_str = re.sub(r'([a-zA-Z]:\\|\/)[\\\/\w\s.-]+', ' path ', detail_str)

    # 5. Rimuovi TUTTO ciò che non è una lettera (rimuove *, ., \, /, numeri, ecc.)
    detail_str = re.sub(r'[^a-zA-Z\s]', ' ', detail_str)

    # 6. Converti tutto in minuscolo
    detail_str = detail_str.lower()
    
    return detail_str.strip()

'''
Funzione per estrarre le parole chiave più frequenti dalla colonna "Detail" dopo la pulizia
- Prende in input quello che viene fuori da csv["Detail"].apply(clean_detail)
- Restituisce una lista di tuple (parola, frequenza) ordinata per frequenza decrescente
'''
def extract_keywords(cleaned_details: pd.Series, top_n: int = 20) -> List[Tuple[str, int]]:
    # Uniamo tutte le stringhe in un unico testo
    all_text = ' '.join(cleaned_details)
    
    # Split per ottenere tutte le parole
    words = all_text.split()
    
    # Contiamo la frequenza di ogni parola
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
            
    # Ordiniamo le parole per frequenza decrescente e prendiamo le top_n
    sorted_keywords = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_keywords[:top_n]

'''
crea un dizionario. ad ogni parola associa un numero intero che rappresenta
il suo ID univoco. in più aggiunge "NONE" con ID 0 per rappresentare valori mancanti o stringhe vuote
'''
def create_dictionary(keywords_list: List[Tuple[str, int]]) -> dict:
    res = {word: idx+1 for idx, (word, _) in enumerate(keywords_list)}
    res["NONE"] = 0  # Aggiungiamo "NONE" con ID 0
    return res

def tokenize_column(column: pd.Series, top_n: int = 20) -> dict:
    column = column.fillna('NONE').astype(str)
    cleaned_column = column.apply(clean_detail)
    keywords = extract_keywords(cleaned_column, top_n=top_n)
    return create_dictionary(keywords)

# Funzione helper per trasformare il testo in un singolo token (per Operation e Result)
def get_single_token(text: str, dictionary: dict) -> int:
    cleaned = clean_detail(str(text)) # Usiamo la tua funzione di pulizia
    # Se l'operazione è composta da più parole, prendiamo la prima che esiste nel dizionario
    for word in cleaned.split():
        if word in dictionary:
            return dictionary[word]
    return 0 # Token per 'UNKNOWN'

# Funzione helper per trasformare i Detail in una LISTA di token
def get_multi_tokens(text: str, dictionary: dict) -> List[int]:
    cleaned = clean_detail(str(text))
    tokens = [dictionary[word] for word in cleaned.split() if word in dictionary]
    return tokens if tokens else [0]

# TO DO: tokenizer del file sistem, riconoscere file di sistema, utente, ecc. e tokenizzarli in modo diverso (es. file di sistema = 1, file utente = 2, ecc.)

'''
# Esempio di utilizzo:
pid = "/home/arch_btw/cloud/GoogleDrive/unipd/2.2/Progetto Cyber/data/API/{}.csv"

log_file = "/home/arch_btw/cloud/GoogleDrive/unipd/2.2/Progetto Cyber/data/API/Logfile.CSV"

# csv = pd.read_csv(pid.format(1404))
csv = pd.read_csv(log_file)

# detail_cleaned = csv["Result"].apply(clean_detail)

# keywords = extract_keywords(detail_cleaned, top_n=100)

keywords = tokenize_column(csv["Operation"], top_n=100)

print(keywords)
'''