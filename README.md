# Federated vs. Centralized learning: A Comparative Analysis of CNN-Based Myocardial Infarction Classification

## How to run the experiments

### 0. Installation & Dependencies
```bash
python3 -m venv flwr-ecg-env
source flwr-ecg-env/bin/activate
pip install .
```

### 1. Activate Environment
```bash
source flwr-ecg-env/bin/activate
```

### 2. Isolated Learning
```bash
python3 -m ecg_code.centralized_training.indcentralized
```

### 3. Federated Learning
```bash
flower-simulation --app . --num-supernodes 2 --backend-config '{"client-resources": {"num-cpus": 16, "num-gpus": 0}}'
```

### 4. Centralized Learning
```bash
python3 -m ecg_code.centralized_training.centralized
