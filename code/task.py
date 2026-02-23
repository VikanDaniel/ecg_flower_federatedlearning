import os
import ast
import wfdb
import pickle
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

class ECGDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32) 
        
        if x.shape[0] == 1000 and x.shape[1] == 12:
            x = x.transpose(0, 1)
            
        return x, y

def load_raw_data_ptbxl(df, path):

    if os.path.exists(path + 'raw100.npy'):
        data = np.load(path+'raw100.npy', allow_pickle=True)
    else:
        data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
        data = np.array([signal for signal, meta in data])
        pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    return data

def load_raw_data_ptbdb(df, sampling_rate, path):

    return None    

def aggregate_diagnostic(y_dic, diag_agg_df):
    tmp = []
    for key in y_dic.keys():
        if key in diag_agg_df.index:
            c = diag_agg_df.loc[key].diagnostic_class
            if str(c) != 'nan':
                tmp.append(c)
    return list(set(tmp))


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    print(f"Client {partition_id} downloading data")
    sampling_rate = 100
    path = None

    Y = None
    X = None
    
    if partition_id == 0: 
        # There is a csv file that contains what I need
        path = 'data/ptbxl/'
        Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        X = load_raw_data_ptbxl(Y, sampling_rate, path)
    else: 
        # I need to make my own csv file that contains what I need
        path = 'data/ptbdb/'
        Y = pd.read_csv(path + 'ptbdb_database.csv', index_col='ecg_id')
        X = load_raw_data_ptbdb(Y, sampling_rate, path)

    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    diag_agg_df = agg_df[agg_df.diagnostic == 1.0]
    Y['superdiagnostic'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(x, diag_agg_df))
    
    Y['superdiagnostic_len'] = Y.superdiagnostic.apply(len)
    mask = Y.superdiagnostic_len > 0
    X = X[mask]
    Y = Y[mask]

    mlb = MultiLabelBinarizer()
    mlb.fit(Y.superdiagnostic.values)
    y_multihot = mlb.transform(Y.superdiagnostic.values)

    total_samples = len(X)
    partition_size = total_samples // num_partitions
    
    start_idx = partition_id * partition_size

    if partition_id == num_partitions - 1:
        end_idx = total_samples
    else:
        end_idx = start_idx + partition_size
        
    client_X = X[start_idx:end_idx]
    client_Y = y_multihot[start_idx:end_idx]
    
    print(f"Client {partition_id} received {len(client_X)} ECG records.")
    
    client_dataset = ECGDataset(client_X, client_Y)
    
    train_len = int(len(client_dataset) * 0.8)
    val_len = len(client_dataset) - train_len
    train_dataset, val_dataset = random_split(
        client_dataset, [train_len, val_len], 
        generator=torch.Generator().manual_seed(42)
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



    return trainloader, testloader
