import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import os

# This file was created by Gemini AI with user guidance to create a plot for the equal dataset experiment.

def load_data(prefix):
    file_labels = f"{prefix}_labels.npy"
    file_probs = f"{prefix}_probs.npy"
    if not os.path.exists(file_labels) or not os.path.exists(file_probs):
        print(f"WARNING: Data not found for {prefix}!")
        return None, None
    return np.load(file_labels), np.load(file_probs)

def draw_graph(experiments, filename):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 8))
    
    for exp in experiments:
        labels, probs = load_data(exp["prefix"])
        if labels is None: continue
            
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
            
        fpr, tpr, _ = roc_curve(labels, probs)
        ax_roc.plot(fpr, tpr, color=exp["color"], linestyle=exp.get("linestyle", "-"), 
                    label=f'{exp["name"]} (AUROC: {auroc:.3f})', linewidth=2.5, alpha=0.9)
        
        precision, recall, _ = precision_recall_curve(labels, probs)
        ax_pr.plot(recall, precision, color=exp["color"], linestyle=exp.get("linestyle", "-"),
                   label=f'{exp["name"]} (AUPRC: {auprc:.3f})', linewidth=2.5, alpha=0.9)

    # Styling ROC
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='lightgray', linewidth=1)
    ax_roc.set_title("ROC Curve: Equal-XL & PTB-DB", fontsize=16, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.legend(fontsize=10, loc='lower right', frameon=True)
    ax_roc.grid(True, alpha=0.3)
    
    # Styling PR
    ax_pr.set_title("Precision-Recall Curve: Equal-XL & PTB-DB", fontsize=16, fontweight='bold')
    ax_pr.set_xlabel("Recall", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)
    ax_pr.legend(fontsize=10, loc='lower left', frameon=True)
    ax_pr.grid(True, alpha=0.3)
    
    plt.suptitle("Performance on Equal-XL and PTB-DB Across Training Methods", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight', transparent=True)
    print(f"SAVED TO: '{filename}'")

def main():
    print("Generating graphs for the final research (Equal Size)...")
    
    summary_impact = [
        {"name": "EQUAL-XL: Centralized", "prefix": "equal_cent_client0", "color": "#1f77b4", "linestyle": "-"},
        {"name": "EQUAL-XL: Federated", "prefix": "equal_fl_client0", "color": "black", "linestyle": "--"},
        {"name": "EQUAL-XL: Isolated", "prefix": "equal_iso_client0", "color": "#1f77b4", "linestyle": ":"},
        {"name": "DB: Federated", "prefix": "equal_fl_client1", "color": "#d62728", "linestyle": "-"},
        {"name": "DB: Centralized", "prefix": "equal_cent_client1", "color": "#ff7f0e", "linestyle": "-"},
        {"name": "DB: Isolated", "prefix": "equal_iso_client1", "color": "#d62728", "linestyle": ":"}
    ]
    
    draw_graph(summary_impact, "Graph_Equal_Methods.png")

if __name__ == "__main__":
    main()
