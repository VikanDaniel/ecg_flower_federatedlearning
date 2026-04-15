#!/bin/bash

echo "==========================================="
echo " STARTER OVERNIGHT THESIS EXPERIMENT WITH 5-FOLD CV"
echo "==========================================="
echo "This will take significantly longer due to 5-Fold CV. Do not close the terminal!"
echo "Do not let the PC go into sleep mode."
echo ""

echo "Cleaning up old prediction files to ensure fresh 5-Fold aggregation..."
rm -vf *_labels.npy *_probs.npy
echo ""

for FOLD_IDX in {0..4}; do
    echo "==========================================="
    echo " STARTING FOLD $FOLD_IDX / 4"
    echo "==========================================="
    export FOLD_IDX=$FOLD_IDX

    echo "-------------------------------------------"
    echo " 1. Running Centralized AI (Absolute Ceiling)"
    echo "-------------------------------------------"
    python3 -m ecg_code.centralized_training.centralized
    echo "✅ Centralized training for fold $FOLD_IDX completed!"
    echo ""

    echo "-------------------------------------------"
    echo " 2. Running Isolated AI (Small Hospital Baseline)"
    echo "-------------------------------------------"
    python3 -m ecg_code.centralized_training.indcentralized
    echo "✅ Isolated training for fold $FOLD_IDX completed!"
    echo ""

    echo "-------------------------------------------"
    echo " 3. Running Federated AI (Flower Network)"
    echo "-------------------------------------------"
    # Flower requires these environment variables in WSL to prevent memory leaks (OOM)
    RAY_enable_metrics_collection=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 flower-simulation --app . --num-supernodes 2 --backend-config '{"client-resources": {"num-cpus": 6, "num-gpus": 0.0}}'
    echo "✅ Federated training for fold $FOLD_IDX completed!"
    echo ""
done

echo "Completed all 5 folds for experiments!"
