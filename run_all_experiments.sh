#!/bin/bash

echo "Starting primary experiment..."
rm -vf *_labels.npy *_probs.npy
echo ""

for FOLD_IDX in {0..4}; do
    echo "Starting fold $FOLD_IDX / 4"

    echo "Running Centralized"
    python3 -m ecg_code.centralized_training.centralized
    echo "Centralized training for fold $FOLD_IDX completed."
    echo ""

    echo "Running Isolated"
    python3 -m ecg_code.centralized_training.indcentralized
    echo "Isolated training for fold $FOLD_IDX completed."
    echo ""

    echo "Running Federated"
    RAY_enable_metrics_collection=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 flower-simulation --app . --num-supernodes 2 --backend-config '{"client-resources": {"num-cpus": 16, "num-gpus": 0.0}}'
    echo "Federated training for fold $FOLD_IDX completed."
    echo ""
done

echo "Completed all 5 folds for the primary experiment."
