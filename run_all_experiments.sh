#!/bin/bash

echo "==========================================="
echo " STARTER OVERNIGHT THESIS EXPERIMENT"
echo "==========================================="
echo "This will take 3-4 hours. Do not close the terminal!"
echo "Do not let the PC go into sleep mode."
echo ""

echo "-------------------------------------------"
echo " 1. Running Centralized AI (Absolute Ceiling)"
echo "-------------------------------------------"
python3 -m ecg_code.centralized_training.centralized
echo "✅ Centralized training completed!"
echo ""

echo "-------------------------------------------"
echo " 2. Running Isolated AI (Small Hospital Baseline)"
echo "-------------------------------------------"
python3 -m ecg_code.centralized_training.indcentralized
echo "✅ Isolated training completed!"
echo ""

echo "-------------------------------------------"
echo " 3. Running Federated AI (Flower Network)"
echo "-------------------------------------------"
# Flower requires these environment variables in WSL to prevent memory leaks (OOM)
RAY_enable_metrics_collection=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 flower-simulation --app . --num-supernodes 2 --backend-config '{"client-resources": {"num-cpus": 16, "num-gpus": 0}}'
echo "Federated training completed"
echo ""

echo "Completed all experiments"

