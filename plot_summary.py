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
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Sort experiments by AUROC for cleaner legend
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
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax_roc.set_title(f"Summary ROC: {title}", fontsize=18, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate", fontsize=14)
    ax_roc.set_ylabel("True Positive Rate", fontsize=14)
    ax_roc.legend(fontsize=9, loc='lower right', frameon=True, shadow=True)
    ax_roc.grid(True, alpha=0.25)
    
    # Styling PR
    ax_pr.set_title(f"Summary PR: {title}", fontsize=18, fontweight='bold')
    ax_pr.set_xlabel("Recall", fontsize=14)
    ax_pr.set_ylabel("Precision", fontsize=14)
    ax_pr.legend(fontsize=9, loc='lower left', frameon=True, shadow=True)
    ax_pr.grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    print(f"🚀 EXECUTIVE SUMMARY SAVED TO: '{filename}'")

def main():
    print("==================================================")
    print(" GENERATING FINAL RESEARCH SUMMARY GRAPH...")
    print("==================================================")
    
    # The 6 Most Important Findings (The Story of the Thesis)
    summary_impact = [
        # --- PTB-XL (Massive Hospital) ---
        {"name": "XL: Centralized Ceiling", "prefix": "cent_client0", "color": "#1f77b4", "linestyle": "-"},
        {"name": "XL: Federated Collaboration", "prefix": "fl_client0", "color": "black", "linestyle": "--"},
        {"name": "XL: Isolated Silo", "prefix": "iso_client0", "color": "#1f77b4", "linestyle": ":"},
        
        # --- PTB-DB (Small Hospital) ---
        {"name": "DB: Federated Rescue", "prefix": "fl_client1", "color": "#d62728", "linestyle": "-"},
        {"name": "DB: Centralized Bias", "prefix": "cent_client1", "color": "#ff7f0e", "linestyle": "-"},
        {"name": "DB: Isolated Silo", "prefix": "iso_client1", "color": "#d62728", "linestyle": ":"}
    ]
    
    draw_graph(summary_impact, "Federated vs. Centralized vs. Isolated", "Executive_Summary_Benchmark.png")

if __name__ == "__main__":
    main()
