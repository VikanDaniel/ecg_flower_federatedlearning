# Federated vs. Centralized learning: A Comparative Analysis of CNN-Based Myocardial Infarction Classification

## Datasets
The datasets utilized for the experiments were provided by the [PhysioNet platform](https://physionet.org/).
* [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
* [PTB Diagnostics (PTB-DB)](https://physionet.org/content/ptbdb/1.0.0/)

## Flower framework structure
The Flower framework is based on client_app.py, server_app.py, and task.py.
* `task.py`: contains the model architecture and the training and testing functions.
* `server_app.py`: contains the server configuration and the federated learning strategy.
* `client_app.py`: contains the client configuration and the federated learning strategy.

## How to install dependencies and activate environment

### 1. Installation & Dependencies
```bash
python3 -m venv flwr-ecg-env
source flwr-ecg-env/bin/activate
pip install .
```

### 2. Activate Environment
```bash
source flwr-ecg-env/bin/activate
```

## How to run experiments

### 1. Run all experiments (Primary)
```bash
./run_all_experiments.sh
```

### 2. Run all experiments (Equal Ablation Study)
```bash
./run_equal_experiment.sh
```

## How to plot the results

### 1. Plot summary for primary experiment (PTB-XL)
```bash
python3 plot_ptbxl_methods.py
```

### 2. Plot summary for primary experiment (PTB-DB)
```bash
python3 plot_ptbdb_methods.py
```

### 3. Plot summary for equal experiment
```bash
python3 plot_summary_equal.py
```

## Calculate statistics (Bootstrapping)

### 1. Calculate for primary experiment
```bash
python3 compute_bootstrap.py
```

### 2. Calculate for equal experiment
```bash
python3 compute_bootstrap_equal.py
```

## Citation
This code was developed as part of a Master's thesis at the University of South-Eastern Norway (USN).
