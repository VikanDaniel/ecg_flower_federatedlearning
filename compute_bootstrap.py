import numpy as np
from sklearn.metrics import roc_auc_score
# This file was created by Gemini AI with user guidance to analyze the results of the primary experiment.

# Reusing the same seeding as the training
np.random.seed(42)

# Load data
def load(prefix):
    return np.load(prefix+'_labels.npy'), np.load(prefix+'_probs.npy')

def bootstrap_auroc(labels, probs, n=10000):
    aucs = []
    n_samples = len(labels)
    for _ in range(n):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    aucs = np.array(aucs)
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
    return aucs, ci_low, ci_high

def bootstrap_pvalue(labels, probs_a, probs_b, n=10000):
    observed_diff = roc_auc_score(labels, probs_a) - roc_auc_score(labels, probs_b)
    n_samples = len(labels)
    diffs = []
    
    # Collecting 10 000 bootstrap differences
    for _ in range(n):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        d = roc_auc_score(labels[idx], probs_a[idx]) - roc_auc_score(labels[idx], probs_b[idx])
        diffs.append(d)
        
    diffs = np.array(diffs)
    
    # Shifting the distribution to simulate the Null Hypothesis (mean = 0)
    shifted_diffs = diffs - np.mean(diffs)
    
    # Count how many are more extreme than the observed difference
    extreme_count = np.sum(np.abs(shifted_diffs) >= np.abs(observed_diff))
    
    return extreme_count / len(diffs)

# Main
if __name__ == "__main__":
    print("\nLoading data...")
    l0, p_iso0  = load('iso_client0')
    _,  p_cent0 = load('cent_client0')
    _,  p_fl0   = load('fl_client0')
    
    l1, p_iso1  = load('iso_client1')
    _,  p_cent1 = load('cent_client1')
    _,  p_fl1   = load('fl_client1')

    print("\nComputing bootstrap CIs and p-values (10,000 iterations)")
    print("This will take 1-2 minutes. Please wait...\n")

    datasets = [
        ("PTB-XL", l0, [("Isolated", p_iso0), ("Centralized", p_cent0), ("Federated", p_fl0)]),
        ("PTB-DB", l1, [("Isolated", p_iso1), ("Centralized", p_cent1), ("Federated", p_fl1)]),
    ]

    print("Dataset    Method          AUROC   95% CI              p-value vs FL")
    print("-" * 72)

    for dataset, labels, methods in datasets:
        fl_probs = methods[2][1]  # The Federated probs are always at index 2
        for method, probs in methods:
            auroc = roc_auc_score(labels, probs)
            _, ci_low, ci_high = bootstrap_auroc(labels, probs)
            
            if method == "Federated":
                p_str = "  (reference)"
            else:
                p = bootstrap_pvalue(labels, fl_probs, probs)
                p_str = f"  {p:.4f}  {'YES *' if p < 0.05 else 'no'}"
                
            print(dataset.ljust(10) + method.ljust(16) +
                  str(round(auroc,4)).ljust(8) +
                  f"[{ci_low:.3f}-{ci_high:.3f}]".ljust(20) + p_str)
        print()

    print("* p < 0.05 = statistically significant difference")
    print("Method: Empirical Bootstrapping | Iterations: 10,000 | Seed: 42")
