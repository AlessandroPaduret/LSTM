#!/home/arch_btw/python-env/bin/python
import torch
from torch.utils.data import Dataset, DataLoader

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class APIDataset(Dataset):
    def __init__(self, pid_dict, seq_len=50):
        # Teniamo solo i PID che hanno abbastanza dati per una finestra
        self.pids = [pid for pid, df in pid_dict.items() if len(df) >= seq_len]
        self.pid_dict = pid_dict
        self.seq_len = seq_len

    def __len__(self):
        # La lunghezza del dataset è il numero di processi validi
        return len(self.pids)

    def _prepare_ebag(self, list_of_lists):
        flat = [t for sublist in list_of_lists for t in sublist]
        lengths = [len(sublist) for sublist in list_of_lists]
        offsets = [0] + torch.cumsum(torch.tensor(lengths), dim=0).tolist()[:-1]
        return torch.tensor(flat).long(), torch.tensor(offsets).long()

    def __getitem__(self, idx):
        # 1. Recuperiamo il dataframe del PID corrispondente
        pid = self.pids[idx]
        df = self.pid_dict[pid]
        
        # 2. SCEGLIAMO UN PUNTO DI INIZIO RANDOM
        # Il punto massimo di inizio è la lunghezza totale meno la finestra
        max_start = len(df) - self.seq_len
        start_idx = np.random.randint(0, max_start + 1)
        
        window = df.iloc[start_idx : start_idx + self.seq_len]
        
        # 3. Label (preso dal primo record della finestra o del PID)
        label = 1 if window['is_ransomware'].iloc[0] == 1 else 0
        
        # 4. Preparazione dati (stessa logica di prima)
        dt = torch.tensor(window['dt'].values).float().unsqueeze(-1)
        op_flat, op_off = self._prepare_ebag(window['Operation'].tolist())
        det_flat, det_off = self._prepare_ebag(window['Detail'].tolist())
        res_flat, res_off = self._prepare_ebag(window['Result'].tolist())
        
        return {
            'dt': dt,
            'op': (op_flat, op_off),
            'det': (det_flat, det_off),
            'res': (res_flat, res_off),
            'label': torch.tensor(label).float()
        }
    
def collate_api(batch):
    # Stack dei dati semplici
    dts = torch.stack([item['dt'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Funzione per unire le "borse" di token di tutto il batch in un unico grande flatten
    def merge_ebags(key):
        all_flats = []
        all_offsets = []
        current_total_tokens = 0
        
        for item in batch:
            flat, off = item[key]
            all_flats.append(flat)
            # Trasliamo gli offset locali rispetto al totale dei token già processati nel batch
            all_offsets.append(off + current_total_tokens)
            current_total_tokens += len(flat)
            
        return torch.cat(all_flats), torch.cat(all_offsets)

    return {
        'dt': dts,
        'op': merge_ebags('op'),
        'det': merge_ebags('det'),
        'res': merge_ebags('res'),
        'label': labels
    }

def test():
    from parser import split_by_pid, parse_data
    import pandas as pd
    
    file_path = "/home/arch_btw/cloud/GoogleDrive/unipd/2.2/Progetto Cyber/data/API/Logfile_labeled.CSV"
    csv = pd.read_csv(file_path)
    
    # ESEMPIO: Se non hai ancora la colonna, la creiamo per il test
    if 'is_ransomware' not in csv.columns:
        csv['is_ransomware'] = 0 

    csv.drop(columns=['Path'], inplace=True, errors='ignore')
    
    # Dividiamo per PID
    datas = split_by_pid(csv)

    for pid in datas:
        # Trasforma il testo in token e calcola il 'dt'
        datas[pid] = parse_data(datas[pid])
        # Rimuoviamo le colonne originali inutili per il training, mantenendo 'dt' e 'is_ransomware'
        cols_to_keep = ['dt', 'Operation', 'Detail', 'Result', 'is_ransomware']
        datas[pid] = datas[pid][cols_to_keep]

    dataset = APIDataset(datas, seq_len=50)
    # Shuffle=True è fondamentale per l'addestramento!
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_api)

    for batch in dataloader:
        print(f"Batch dt shape: {batch['dt'].shape}")         # [32, 50, 1]
        print(f"Batch labels shape: {batch['label'].shape}") # [32]
        print(f"Operation tokens: {batch['op'][0].shape}")    # Unico vettore gigante
        print(f"Operation offsets: {batch['op'][1].shape}")   # 32 * 50 = 1600 offsets
        break

if __name__ == "__main__":
    test()