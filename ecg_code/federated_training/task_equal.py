import os
import ast
import pandas as pd
import numpy as np
import wfdb
import random
import scipy.signal
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Seed for reproducibility
SEED = 42

class InceptionModule1D(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels=32):
        super(InceptionModule1D, self).__init__()
        
        # Bottleneck implementation
        self.use_bottleneck = in_channels > 1
        
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_in_channels = bottleneck_channels
        else:
            conv_in_channels = in_channels
            
        self.conv1 = nn.Conv1d(conv_in_channels, out_channels, kernel_size=10, padding='same', bias=False)
        self.conv2 = nn.Conv1d(conv_in_channels, out_channels, kernel_size=20, padding='same', bias=False)
        self.conv3 = nn.Conv1d(conv_in_channels, out_channels, kernel_size=40, padding='same', bias=False)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels * 4)

    def forward(self, x):
        if self.use_bottleneck:
            x_bottleneck = self.bottleneck(x)
        else:
            x_bottleneck = x
            
        out1 = self.conv1(x_bottleneck)
        out2 = self.conv2(x_bottleneck)
        out3 = self.conv3(x_bottleneck)
        
        pool_out = self.pool_conv(self.maxpool(x))
        
        out = torch.cat([out1, out2, out3, pool_out], dim=1)
        return F.relu(self.bn(out))

class InceptionBlock(nn.Module):
    def __init__(self, in_channels=12, out_channels=32):
        super(InceptionBlock, self).__init__()
        
        self.inc1 = InceptionModule1D(in_channels, out_channels)
        self.inc2 = InceptionModule1D(out_channels * 4, out_channels)
        self.inc3 = InceptionModule1D(out_channels * 4, out_channels)
        
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * 4, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(out_channels * 4)
        )
        
    def forward(self, x):
        res = self.shortcut(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.inc3(x)
        x = x + res
        return F.relu(x)

# This is the model used for all experiments
# 3-layer InceptionTime CNN using 12-lead ECG data
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.block = InceptionBlock(in_channels=12, out_channels=32)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

def train(net, trainloader, optimizer, epochs, device, proximal_mu=0.0, global_params=None):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for local_weights, global_weights in zip(net.parameters(), global_params):
                    proximal_term += torch.square(torch.linalg.norm(local_weights - global_weights))
                loss += (proximal_mu / 2.0) * proximal_term
                
            loss.backward()
            optimizer.step()

def test(net, testloader, device, save_path=None):
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            
            probs = torch.softmax(outputs.data, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            
            preds = torch.max(outputs.data, 1)[1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
        
    if save_path is not None:
        labels_file = f"{save_path}_labels.npy"
        probs_file = f"{save_path}_probs.npy"
        if os.path.exists(labels_file) and os.path.exists(probs_file):
            exist_labels = np.load(labels_file)
            exist_probs = np.load(probs_file)
            np.save(labels_file, np.concatenate((exist_labels, np.array(all_labels))))
            np.save(probs_file, np.concatenate((exist_probs, np.array(all_probs))))
        else:
            np.save(labels_file, np.array(all_labels))
            np.save(probs_file, np.array(all_probs))
    
    return loss, accuracy, f1, auc

# PTB-XL
def parse_scp_codes(code_string):
    return ast.literal_eval(code_string)

def get_label_ptbxl(superclass_list):
    if 'MI' in superclass_list:
        return 1
    return 0

def load_and_filter_ptbxl_metadata(data_path):
    csv_path = os.path.join(data_path, 'ptbxl_database.csv')
    Y = pd.read_csv(csv_path, index_col='ecg_id')
    
    parsed_codes = []
    for code_string in Y.scp_codes:
        parsed_codes.append(parse_scp_codes(code_string))
    Y.scp_codes = parsed_codes

    statements_path = os.path.join(data_path, 'scp_statements.csv')
    agg_df = pd.read_csv(statements_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    superclasses = []
    for code_dict in Y.scp_codes:
        superclasses.append(aggregate_diagnostic(code_dict))
    Y['diagnostic_superclass'] = superclasses

    labels = []
    for superclass_list in Y['diagnostic_superclass']:
        labels.append(get_label_ptbxl(superclass_list))
    Y['label'] = labels
    
    # MI = 1, Normal = 0
    Y_mi = Y[Y['label'] == 1]
    Y_normal = Y[Y['label'] == 0]
    
    # Balance dataset
    if len(Y_normal) > len(Y_mi):
        Y_normal = Y_normal.sample(n=len(Y_mi), random_state=SEED)
    elif len(Y_mi) > len(Y_normal):
        Y_mi = Y_mi.sample(n=len(Y_normal), random_state=SEED)
        
    # Put together and shuffle
    Y_balanced = pd.concat([Y_mi, Y_normal]).sample(frac=1, random_state=SEED)
    
    print("\n!!! EXTREME ABLATION STUDY: Limiting PTB-XL to 362 patients to match PTB-DB !!!")
    # Stratified Sampling: Vi tvinger koden til å velge nøyaktig 181 syke og 181 friske!
    Y_mi_sample = Y_balanced[Y_balanced['label'] == 1].sample(n=181, random_state=SEED)
    Y_normal_sample = Y_balanced[Y_balanced['label'] == 0].sample(n=181, random_state=SEED)
    Y_balanced = pd.concat([Y_mi_sample, Y_normal_sample]).sample(frac=1, random_state=SEED)
        
    return Y_balanced

def load_raw_data_ptbxl(df, data_path):
    data = []
    for filename in df.filename_lr:
        full_path = os.path.join(data_path, filename)
        signal, meta = wfdb.rdsamp(full_path)
        
        # -- VOLTAGE HARMONIZATION --
        # Standardiserer (Z-score) alle spenninger til mean 0 og std 1 for å tvinge datasettene på samme skala
        signal = scipy.stats.zscore(signal, axis=0)
        signal = np.nan_to_num(signal)
        
        # Transpose so shape is (12, 1000) for PyTorch Conv1d
        data.append(signal.T)
    return np.array(data)

def load_ptbxl_dataset(data_path):
    print("Loading PTB-XL metadata...")
    Y_df = load_and_filter_ptbxl_metadata(data_path)
    
    print(f"Loading raw ECG signals for {len(Y_df)} records at 100 Hz...")
    X = load_raw_data_ptbxl(Y_df, data_path)
    y = Y_df['label'].values
    
    return X, y

# PTBDB
def load_and_filter_ptbdb_metadata(data_path):
    records_file = os.path.join(data_path, 'RECORDS')
    with open(records_file, 'r') as f:
        all_records = f.read().splitlines()
        
    filtered_records = []
    labels = []
    
    for record in all_records:
        if not record.strip():
            continue
        record_path = os.path.join(data_path, record)
        try:
            header = wfdb.rdheader(record_path)
            record_label = 0
            for comment in header.comments:
                if 'Reason for admission: Myocardial infarction' in comment:
                    record_label = 1
                    break
                    
            filtered_records.append(record)
            labels.append(record_label)
        except Exception as e:
            pass
            
    # Balance dataset
    records_mi = [r for r, l in zip(filtered_records, labels) if l == 1]
    records_normal = [r for r, l in zip(filtered_records, labels) if l == 0]
    
    random.seed(SEED)
    if len(records_normal) > len(records_mi):
        records_normal = random.sample(records_normal, len(records_mi))
    elif len(records_mi) > len(records_normal):
        records_mi = random.sample(records_mi, len(records_normal))
        
    balanced_records = records_mi + records_normal
    balanced_labels = [1]*len(records_mi) + [0]*len(records_normal)
    
    # Shuffle dataset
    combined = list(zip(balanced_records, balanced_labels))
    random.shuffle(combined)
    
    if len(combined) == 0:
        return [], []
        
    balanced_records, balanced_labels = zip(*combined)
    return list(balanced_records), list(balanced_labels)

def load_raw_data_ptbdb(records, data_path):
    data = []
    expected_length = 1000
    for record in records:
        full_path = os.path.join(data_path, record)
        signal, meta = wfdb.rdsamp(full_path)
        
        signal_12_leads = signal[:, :12]
        
        # -- FREQUENCY HARMONIZATION --
        fs = meta.get('fs', 1000)
        target_fs = 100
        if fs != target_fs:
            signal_100hz = scipy.signal.resample_poly(signal_12_leads, up=target_fs, down=fs, axis=0)
        else:
            signal_100hz = signal_12_leads
            
        if len(signal_100hz) >= expected_length:
            signal_final = signal_100hz[:expected_length, :]
        else:
            pad_width = expected_length - len(signal_100hz)
            signal_final = np.pad(signal_100hz, ((0, pad_width), (0, 0)), mode='constant')

        # -- VOLTAGE HARMONIZATION --
        # Z-score etter klipping for å garantere perfekt harmonisering (Mean 0, Std 1)
        signal_final = scipy.stats.zscore(signal_final, axis=0)
        signal_final = np.nan_to_num(signal_final)
        
        data.append(signal_final.T)
            
    return np.array(data)

def load_ptbdb_dataset(data_path):
    print("Loading PTBDB metadata...")
    records, y = load_and_filter_ptbdb_metadata(data_path)
    
    print(f"Loading raw ECG signals for {len(records)} records...")
    X = load_raw_data_ptbdb(records, data_path)
    
    return X, np.array(y)

# Utility
def load_data(partition_id):
    # Partition ID 0 = PTB-XL, Partition ID 1 = PTBDB
    #Limit to 0 or 1.
    partition_id = partition_id % 2
    
    if partition_id == 0:
        data_path = 'data/ptbxl'
        X, y = load_ptbxl_dataset(data_path)
    elif partition_id == 1:
        data_path = 'data/ptbdb'
        X, y = load_ptbdb_dataset(data_path)
    else:
        raise ValueError(f"Unknown partition_id: {partition_id}")
        
    fold_idx_str = os.environ.get("FOLD_IDX", None)
    
    # Split into train/validation
    if fold_idx_str is not None:
        fold_idx = int(fold_idx_str)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(skf.split(X, y))
        train_idx, test_idx = splits[fold_idx]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # Create PyTorch datasets
    trainset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    testset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    # DataLoaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    
    return trainloader, testloader, num_examples


