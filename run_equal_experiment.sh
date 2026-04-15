#!/bin/bash

echo "==========================================="
echo " STARTER EQUAL SIZE (362 vs 362) EXPERIMENT WITH 5-FOLD CV"
echo " This will force the massive PTB-XL to match the small hospital size"
echo "==========================================="

echo "Cleaning up old prediction files to ensure fresh 5-Fold aggregation..."
rm -vf *_labels.npy *_probs.npy
echo ""

# Switch to Equal-apps directly in pyproject temporarily
perl -pi -e 's/client_app:/client_app_equal:/g' pyproject.toml
perl -pi -e 's/server_app:/server_app_equal:/g' pyproject.toml

for FOLD_IDX in {0..4}; do
    echo "==========================================="
    echo " STARTING FOLD $FOLD_IDX / 4"
    echo "==========================================="
    export FOLD_IDX=$FOLD_IDX

    echo "-------------------------------------------"
    echo " 1. Running Centralized AI (Absolute Ceiling)"
    echo "-------------------------------------------"
    python3 -m ecg_code.centralized_training.centralized_equal

    echo "-------------------------------------------"
    echo " 2. Running Isolated AI (Small Hospital Baseline)"
    echo "-------------------------------------------"
    python3 -m ecg_code.centralized_training.indcentralized_equal

    echo "-------------------------------------------"
    echo " 3. Running Federated AI (Flower Network)"
    echo "-------------------------------------------"
    RAY_enable_metrics_collection=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 flower-simulation --app . --num-supernodes 2 --backend-config '{"client-resources": {"num-cpus": 6, "num-gpus": 0.0}}'
done

# Reset to original config
perl -pi -e 's/client_app_equal:/client_app:/g' pyproject.toml
perl -pi -e 's/server_app_equal:/server_app:/g' pyproject.toml

echo "==========================================="
echo " 🎉 ABLATION STUDY COMPLETED! 🎉"
echo " Run: python3 plot_equal_federated.py"
echo "==========================================="
