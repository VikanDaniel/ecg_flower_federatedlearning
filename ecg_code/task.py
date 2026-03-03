import os
import ast
import pandas as pd
import numpy as np
import wfdb
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ==========================================
# Neural Network Architecture
# ==========================================
class Net(nn.Module):
    """
    Simple 1D CNN for ECG 12-lead data.
    Input shape expected: (Batch, Channels=12, Length=1000)
    Output: 2 classes (0=Healthy/Norm, 1=MI)
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        # Length math:
        # L_in = 1000
        # conv1: (1000 - 5)/2 + 1 = 498
        # pool1: 498 / 2 = 249
        # conv2: (249 - 5)/2 + 1 = 123
        # pool2: 123 / 2 = 61
        self.fc1 = nn.Linear(64 * 61, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

# ==========================================
# PTB-XL
# ==========================================

def parse_scp_codes(code_string):
    return ast.literal_eval(code_string)

def get_label_ptbxl(superclass_list):
    if 'MI' in superclass_list:
        return 1
    return -1

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
    
    # Keep only MI
    Y_mi = Y[Y['label'] == 1]
    
    return Y_mi

def load_raw_data_ptbxl(df, data_path):
    data = []
    for filename in df.filename_lr:
        full_path = os.path.join(data_path, filename)
        signal, meta = wfdb.rdsamp(full_path)
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

# ==========================================
# PTBDB
# ==========================================

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
            record_label = -1
            for comment in header.comments:
                if 'Reason for admission: Myocardial infarction' in comment:
                    record_label = 1
                    break
                    
            if record_label == 1:
                filtered_records.append(record)
                labels.append(record_label)
        except Exception as e:
            pass
            
    return filtered_records, labels

def load_raw_data_ptbdb(records, data_path):
    data = []
    expected_length = 1000
    for record in records:
        full_path = os.path.join(data_path, record)
        signal, meta = wfdb.rdsamp(full_path)
        
        signal_12_leads = signal[:, :12]
        signal_100hz = signal_12_leads[::10, :]
        
        if len(signal_100hz) >= expected_length:
            signal_cropped = signal_100hz[:expected_length, :]
            # Transpose to (Channels, Length)
            data.append(signal_cropped.T)
        else:
            pad_width = expected_length - len(signal_100hz)
            signal_padded = np.pad(signal_100hz, ((0, pad_width), (0, 0)), mode='constant')
            data.append(signal_padded.T)
            
    return np.array(data)

def load_ptbdb_dataset(data_path):
    print("Loading PTBDB metadata...")
    records, y = load_and_filter_ptbdb_metadata(data_path)
    
    print(f"Loading raw ECG signals for {len(records)} records...")
    X = load_raw_data_ptbdb(records, data_path)
    
    return X, np.array(y)

# ==========================================
# FLOWER UTILITY
# ==========================================

def load_data(partition_id):
    """
    Flower uses partition_id to load specific client datasets.
    Client 0 -> PTB-XL
    Client 1 -> PTBDB
    """
    # Flower 1.13 spawns 10 supernodes by default. Limit to 0 or 1.
    partition_id = partition_id % 2
    
    if partition_id == 0:
        data_path = 'data/ptbxl'
        X, y = load_ptbxl_dataset(data_path)
    elif partition_id == 1:
        data_path = 'data/ptbdb'
        X, y = load_ptbdb_dataset(data_path)
    else:
        raise ValueError(f"Unknown partition_id: {partition_id}")
        
    # Split into train/validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create PyTorch datasets
    trainset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    testset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    # DataLoaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    
    return trainloader, testloader, num_examples

if __name__ == "__main__":
    # Test data loading for both clients locally
    print("Testing Client 0 (PTB-XL)")
    trainloader, testloader, num_examples = load_data(0)
    print(f"Shapes Client 0 - Train: {num_examples['trainset']}, Test: {num_examples['testset']}")
    
    print("\nTesting Client 1 (PTBDB)")
    trainloader, testloader, num_examples = load_data(1)
    print(f"Shapes Client 1 - Train: {num_examples['trainset']}, Test: {num_examples['testset']}")
