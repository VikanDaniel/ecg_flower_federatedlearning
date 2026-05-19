#!/bin/bash

echo "Starting equal experiment..."

echo "Cleaning up old prediction files..."
rm -vf *_labels.npy *_probs.npy
echo ""

# Switch to Equal-apps directly in pyproject temporarily
perl -pi -e 's/client_app:/client_app_equal:/g' pyproject.toml
perl -pi -e 's/server_app:/server_app_equal:/g' pyproject.toml

for FOLD_IDX in {0..4}; do
    echo "Starting fold $FOLD_IDX / 4"
    export FOLD_IDX=$FOLD_IDX

    echo "Running Centralized"
    python3 -m ecg_code.centralized_training.centralized_equal

    echo "Running Isolated"
    python3 -m ecg_code.centralized_training.indcentralized_equal

    echo "Running Federated"
    RAY_enable_metrics_collection=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 flower-simulation --app . --num-supernodes 2 --backend-config '{"client-resources": {"num-cpus": 6, "num-gpus": 0.0}}'
done

# Reset to original config
perl -pi -e 's/client_app_equal:/client_app:/g' pyproject.toml
perl -pi -e 's/server_app_equal:/server_app:/g' pyproject.toml

echo "Completed all 5 folds for the equal experiment."
