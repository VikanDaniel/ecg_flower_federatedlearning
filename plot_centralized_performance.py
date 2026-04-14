import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import os

def load_data(prefix):
    file_labels = f"{prefix}_labels.npy"
    file_probs = f"{prefix}_probs.npy"
    if not os.path.exists(file_labels) or not os.path.exists(file_probs):
        print(f"WARNING: Data not found for {prefix}!")
        return None, None
    return np.load(file_labels), np.load(file_probs)

def draw_graph(experiments, title, filename):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    
    for exp in experiments:
        labels, probs = load_data(exp["prefix"])
        if labels is None: continue
            
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
            
        fpr, tpr, _ = roc_curve(labels, probs)
        ax_roc.plot(fpr, tpr, color=exp["color"], label=f'{exp["name"]} (AUROC: {auroc:.3f})', linewidth=3, alpha=0.9)
        
        precision, recall, _ = precision_recall_curve(labels, probs)
        ax_pr.plot(recall, precision, color=exp["color"], label=f'{exp["name"]} (AUPRC: {auprc:.3f})', linewidth=3, alpha=0.9)

    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    ax_roc.set_title(f"ROC Curve ({title})", fontsize=16, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate", fontsize=14)
    ax_roc.set_ylabel("True Positive Rate", fontsize=14)
    ax_roc.legend(fontsize=12)
    ax_roc.grid(True, alpha=0.4, linewidth=1.5)
    
    ax_pr.set_title(f"Precision-Recall ({title})", fontsize=16, fontweight='bold')
    ax_pr.set_xlabel("Recall", fontsize=14)
    ax_pr.set_ylabel("Precision", fontsize=14)
    ax_pr.legend(fontsize=12)
    ax_pr.grid(True, alpha=0.4, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    print(f"✅ Graph saved as '{filename}'")

def main():
    print("==================================================")
    print(" Generating Centralized (97/3) Performance Graphs...")
    print("==================================================")
    
    # We want to see how the Centralized model performs on XL vs DB individually
    exp_centralized = [
        {"name": "Centralized Model (Tested on PTB-XL)", "prefix": "cent_client0", "color": "#1f77b4"},
        {"name": "Centralized Model (Tested on PTB-DB)", "prefix": "cent_client1", "color": "#d62728"}
    ]
    draw_graph(exp_centralized, "Centralized 97/3 Imbalance Impact", "Graph_Centralized_Local_Performance.png")

if __name__ == "__main__":
    main()
