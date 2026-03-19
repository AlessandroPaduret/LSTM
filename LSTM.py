
import torch
import torch.nn as nn

class RansomwareLSTM(nn.Module):
    def __init__(self, vocab_sizes, emb_dims):
        super(RansomwareLSTM, self).__init__()
        
        # 1. Definiamo i 3 EmbeddingBag con max_norm=1.0 per stabilità
        # Mode='mean' fa la media automatica dei token presenti in ogni riga
        self.emb_op = nn.EmbeddingBag(vocab_sizes['op'], emb_dims['op'], mode='mean', max_norm=1.0)
        self.emb_res = nn.EmbeddingBag(vocab_sizes['res'], emb_dims['res'], mode='mean', max_norm=1.0)
        self.emb_det = nn.EmbeddingBag(vocab_sizes['det'], emb_dims['det'], mode='mean', max_norm=1.0)
        
        # 2. Calcoliamo la dimensione totale dell'input per la LSTM
        # Somma delle dimensioni degli embedding + 1 (per il dt)
        self.input_dim = emb_dims['op'] + emb_dims['res'] + emb_dims['det'] + 1
        
        # 3. La LSTM riceve il vettore concatenato
        self.lstm = nn.LSTM(input_size=self.input_dim, 
                            hidden_size=128, 
                            num_layers=2, 
                            batch_first=True, 
                            dropout=0.2)
        
        # 4. Classificatore finale
        self.fc = nn.Linear(128, 1)

    def forward(self, op_data, res_data, det_data, dt):
        """
        Ogni data è una tupla: (tokens_flat, offsets)
        dt: Tensor di forma (Batch, Seq_Len, 1)
        """
        # Estraiamo i token e gli offsets
        op_tokens, op_off = op_data
        res_tokens, res_off = res_data
        det_tokens, det_off = det_data
        
        # Passiamo agli EmbeddingBag
        # Risultato: (Batch * Seq_Len, Emb_Dim)
        v_op = self.emb_op(op_tokens, op_off)
        v_res = self.emb_res(res_tokens, res_off)
        v_det = self.emb_det(det_tokens, det_off)
        
        # Concateniamo i 3 vettori di embedding
        # Risultato: (Batch * Seq_Len, sum_emb_dims)
        v_combined = torch.cat([v_op, v_res, v_det], dim=1)
        
        # Riportiamo alla forma (Batch, Seq_Len, sum_emb_dims) per la LSTM
        batch_size, seq_len, _ = dt.shape
        v_combined = v_combined.view(batch_size, seq_len, -1)
        
        # Concateniamo il dt (Batch, Seq_Len, sum_emb_dims + 1)
        x = torch.cat([v_combined, dt], dim=2)
        
        # Passiamo alla LSTM
        out, _ = self.lstm(x)
        
        # Prendiamo l'ultimo output della sequenza per la classificazione
        last_step = out[:, -1, :]
        return self.fc(last_step) # Restituisce i Logits (usa BCEWithLogitsLoss)
    
if __name__ == "__main__":
    # Esempio di test rapido
    vocab_sizes = {'op': 100, 'res': 100, 'det': 100}
    emb_dims = {'op': 16, 'res': 16, 'det': 16}
    model = RansomwareLSTM(vocab_sizes, emb_dims)
    print(model)

    # provamo un allenammento con dati di Logfile_labeled.CSV
    # importiamo il dataloader da Dataset.py
    from Dataset import APIDataset, DataLoader, collate_api
    import pandas as pd
    file_path = "/home/arch_btw/cloud/GoogleDrive/unipd/2.2/Progetto Cyber/data/API/Logfile_labeled.CSV"
    csv = pd.read_csv(file_path)
    csv.drop(columns=['Path'], inplace=True, errors='ignore')
    from parser import split_by_pid, parse_data
    datas = split_by_pid(csv)
    for pid in datas:
        datas[pid] = parse_data(datas[pid])
        cols_to_keep = ['dt', 'Operation', 'Detail', 'Result', 'is_ransomware']
        datas[pid] = datas[pid][cols_to_keep]
    
    dataset = APIDataset(datas, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_api)

    epochs = 50
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch['op'], batch['res'], batch['det'], batch['dt'])
            loss = criterion(outputs.squeeze(), batch['label'].float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # calcoliamo l'accuracy sullo stesso dataset (non è una buona pratica, ma è solo un test)
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloader:
            outputs = model(batch['op'], batch['res'], batch['det'], batch['dt'])
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).int()
            total += batch['label'].size(0)
            correct += (predicted == batch['label']).sum().item()
        print(f"Accuracy: {correct/total:.4f}")