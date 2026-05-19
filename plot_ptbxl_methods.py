import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import os
# This file was created by Gemini AI with user guidance to create a plot for the primary experiment (PTB-XL).

def load_data(prefix):
    file_labels = f"{prefix}_labels.npy"
    file_probs = f"{prefix}_probs.npy"
    if not os.path.exists(file_labels) or not os.path.exists(file_probs):
        print(f"WARNING: Data not found for {prefix}!")
        return None, None
    return np.load(file_labels), np.load(file_probs)

def main():
    print("Generating graph for PTB-XL (Client 0)...")
    
    experiments = [
        {"name": "Centralized Training", "prefix": "cent_client0", "color": "#1f77b4", "linestyle": "--"},
        {"name": "Federated Learning", "prefix": "fl_client0", "color": "#d62728", "linestyle": "-"},
        {"name": "Isolated Training", "prefix": "iso_client0", "color": "#000000", "linestyle": ":"}
    ]
    
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    
    for exp in experiments:
        labels, probs = load_data(exp["prefix"])
        if labels is None:
            continue
            
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
            
        # ROC Curve
        fpr, tpr, _ = roc_curve(labels, probs)
        ax_roc.plot(fpr, tpr, color=exp["color"], linestyle=exp["linestyle"], 
                    label=f'{exp["name"]} (AUROC: {auroc:.3f})', linewidth=2.5, alpha=0.9)
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(labels, probs)
        ax_pr.plot(recall, precision, color=exp["color"], linestyle=exp["linestyle"],
                   label=f'{exp["name"]} (AUPRC: {auprc:.3f})', linewidth=2.5, alpha=0.9)

    # Styling ROC
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='lightgray', linewidth=1)
    ax_roc.set_title("ROC Curve: PTB-XL Hospital", fontsize=16, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.legend(fontsize=10, loc='lower right', frameon=True)
    ax_roc.grid(True, alpha=0.3)
    
    # Styling PR
    ax_pr.set_title("Precision-Recall Curve: PTB-XL Hospital", fontsize=16, fontweight='bold')
    ax_pr.set_xlabel("Recall", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)
    ax_pr.legend(fontsize=10, loc='lower left', frameon=True)
    ax_pr.grid(True, alpha=0.3)
    
    plt.suptitle("Performance on PTB-XL Dataset Across Training Methods", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    filename = "Graph_PTBXL_Methods.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    print(f"Saved graph to {filename}")

if __name__ == "__main__":
    main()
