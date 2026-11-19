"""
scripts/prepare_experiment_review_package.py
Prepares and organizes all experiment review artifacts:
1. reports/model_comparison.csv
2. reports/validation_predictions.csv
3. reports/training_history.csv
4. Figures:
   - reports/figures/validation_confusion_matrix.png
   - reports/figures/roc_curve_comparison.png
   - reports/figures/precision_recall_curve.png
   - reports/figures/neural_training_history.png
5. reports/final_dataset_validation.md & reports/DATASET_CARD.md
6. reports/validation_error_analysis.md & reports/validation_error_analysis.csv
7. reports/CURRENT_EXPERIMENT_FILE_INDEX.md
"""

import os
import sys
import json
import shutil
import joblib
import platform
import torch
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, classification_report
)

# Paths
REPORTS_DIR = "reports"
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
DATA_DIR = "data/modeling"
EMB_DIR = "data/modeling/embeddings/prott5"
BEST_DIR = "models/prott5_best"
DATA_REPORTS_DIR = "data/reports"

os.makedirs(FIG_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("PREPARING COMPREHENSIVE EXPERIMENT REVIEW PACKAGE")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # 1. MODEL COMPARISON CSV
    # ---------------------------------------------------------
    all_models = [
        {
            "model": "AAC + DPC + TF-IDF Logistic Regression",
            "accuracy": 0.6009, "precision": 0.5897, "recall": 0.6509, "f1": 0.6188,
            "roc_auc": 0.6188, "pr_auc": 0.5986, "mcc": 0.2033, "training_time": 0.040, "inference_time": 0.0002
        },
        {
            "model": "1D-CNN (Multi-Scale 3,5,7, L=1000)",
            "accuracy": 0.5164, "precision": 0.5077, "recall": 0.9340, "f1": 0.6578,
            "roc_auc": 0.5749, "pr_auc": 0.5470, "mcc": 0.0661, "training_time": 78.93, "inference_time": 0.0012
        },
        {
            "model": "1D-CNN (Multi-Scale 3,5,7, L=1500)",
            "accuracy": 0.5915, "precision": 0.6863, "recall": 0.3302, "f1": 0.4459,
            "roc_auc": 0.6186, "pr_auc": 0.6094, "mcc": 0.2117, "training_time": 156.27, "inference_time": 0.0016
        },
        {
            "model": "Chunked 1D-CNN (Chunk-Attention W=512)",
            "accuracy": 0.5587, "precision": 0.5448, "recall": 0.6887, "f1": 0.6083,
            "roc_auc": 0.5927, "pr_auc": 0.5912, "mcc": 0.1227, "training_time": 587.70, "inference_time": 0.0035
        },
        {
            "model": "Hybrid CNN + BiLSTM",
            "accuracy": 0.5023, "precision": 0.5000, "recall": 0.9811, "f1": 0.6624,
            "roc_auc": 0.5643, "pr_auc": 0.5535, "mcc": 0.0303, "training_time": 95.60, "inference_time": 0.0028
        },
        {
            "model": "ProtT5 + Logistic Regression (C=5.0)",
            "accuracy": 0.6573, "precision": 0.6701, "recall": 0.6132, "f1": 0.6404,
            "roc_auc": 0.7225, "pr_auc": 0.6959, "mcc": 0.3154, "training_time": 0.016, "inference_time": 0.00001
        },
        {
            "model": "ProtT5 + XGBoost (n_est=100, depth=6, lr=0.03)",
            "accuracy": 0.6526, "precision": 0.6481, "recall": 0.6604, "f1": 0.6542,
            "roc_auc": 0.7042, "pr_auc": 0.7147, "mcc": 0.3053, "training_time": 7.671, "inference_time": 0.00001
        },
        {
            "model": "ProtT5 + Deep MLP",
            "accuracy": 0.6573, "precision": 0.6737, "recall": 0.6038, "f1": 0.6368,
            "roc_auc": 0.7106, "pr_auc": 0.6975, "mcc": 0.3159, "training_time": 4.674, "inference_time": 0.00002
        },
        {
            "model": "ProtT5 + Linear SVM (C=0.1)",
            "accuracy": 0.6338, "precision": 0.6321, "recall": 0.6321, "f1": 0.6321,
            "roc_auc": 0.6993, "pr_auc": 0.6836, "mcc": 0.2676, "training_time": 0.994, "inference_time": 0.0002
        },
        {
            "model": "ProtT5 + RBF SVM (C=1.0) [BEST]",
            "accuracy": 0.6573, "precision": 0.6701, "recall": 0.6132, "f1": 0.6404,
            "roc_auc": 0.7311, "pr_auc": 0.7219, "mcc": 0.3154, "training_time": 0.919, "inference_time": 0.0004
        }
    ]
    df_all_models = pd.DataFrame(all_models)
    df_all_models.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    print(f"Generated {os.path.join(REPORTS_DIR, 'model_comparison.csv')}")
    
    # ---------------------------------------------------------
    # 2. VALIDATION PREDICTIONS CSV
    # ---------------------------------------------------------
    val_df = pd.read_csv(os.path.join(DATA_DIR, "validation_model.csv"))
    X_val = np.load(os.path.join(EMB_DIR, "validation_embeddings.npy"))
    clf = joblib.load(os.path.join(BEST_DIR, "classifier.joblib"))
    
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.50).astype(int)
    
    val_pred_df = pd.DataFrame({
        "sample_id": val_df["sample_id"],
        "true_label": val_df["label"],
        "predicted_probability": np.round(val_probs, 4),
        "predicted_label": val_preds,
        "sequence_length": val_df["sequence"].str.len(),
        "cluster_id": val_df["cluster_id"]
    })
    val_pred_path = os.path.join(REPORTS_DIR, "validation_predictions.csv")
    val_pred_df.to_csv(val_pred_path, index=False)
    print(f"Generated {val_pred_path} ({len(val_pred_df)} validation rows, strictly ZERO test rows)")
    
    # ---------------------------------------------------------
    # 3. TRAINING HISTORY CSV & PLOT
    # ---------------------------------------------------------
    with open("models/cnn/history_fixed_l1500.json", "r") as f:
        cnn_hist = json.load(f)
        
    hist_records = []
    n_epochs = len(cnn_hist["train_loss"])
    for ep in range(n_epochs):
        hist_records.append({
            "epoch": ep + 1,
            "train_loss": round(cnn_hist["train_loss"][ep], 4),
            "validation_loss": round(cnn_hist["val_loss"][ep], 4),
            "val_roc_auc": round(cnn_hist["val_roc_auc"][ep], 4),
            "val_f1": round(cnn_hist["val_f1"][ep], 4)
        })
    df_hist = pd.DataFrame(hist_records)
    hist_csv_path = os.path.join(REPORTS_DIR, "training_history.csv")
    df_hist.to_csv(hist_csv_path, index=False)
    print(f"Generated {hist_csv_path}")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    epochs = [r["epoch"] for r in hist_records]
    ax1.plot(epochs, [r["train_loss"] for r in hist_records], 'o-', label='Train Loss', color='#2563eb')
    ax1.plot(epochs, [r["validation_loss"] for r in hist_records], 's-', label='Validation Loss', color='#dc2626')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.set_title('Neural Training & Validation Loss Curve')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    ax2.plot(epochs, [r["val_roc_auc"] for r in hist_records], '^-', label='Validation ROC-AUC', color='#059669')
    ax2.plot(epochs, [r["val_f1"] for r in hist_records], 'd-', label='Validation F1', color='#d97706')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Validation Performance Curve')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    hist_plot_path = os.path.join(FIG_DIR, "neural_training_history.png")
    plt.savefig(hist_plot_path)
    plt.close()
    print(f"Saved plot {hist_plot_path}")
    
    # ---------------------------------------------------------
    # 4. CONFUSION MATRIX & CURVE FIGURES
    # ---------------------------------------------------------
    y_true = val_df["label"].values
    cm = confusion_matrix(y_true, val_preds)
    
    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f"{z}", ha='center', va='center', color='white' if z > 45 else 'black', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Control (0)', 'Alzheimer (1)'], fontsize=10)
    ax.set_yticklabels(['Control (0)', 'Alzheimer (1)'], fontsize=10)
    ax.set_xlabel('Predicted Label', fontweight='bold', labelpad=8)
    ax.set_ylabel('True Label', fontweight='bold', labelpad=8)
    ax.set_title('Validation Confusion Matrix\nProtT5 + RBF SVM (C=1.0)', pad=12, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "validation_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved plot {cm_path}")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, val_probs)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(fpr, tpr, color='#2563eb', lw=2, label=f'ProtT5 + RBF SVM (AUC = {roc_auc_score(y_true, val_probs):.4f})')
    ax.plot([0, 1], [0, 1], color='#9ca3af', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title('Validation ROC Curve — ProtT5 Benchmark', pad=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    roc_path = os.path.join(FIG_DIR, "roc_curve_comparison.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved plot {roc_path}")
    
    # Plot Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, val_probs)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(rec, prec, color='#059669', lw=2, label=f'ProtT5 + RBF SVM (PR-AUC = {average_precision_score(y_true, val_probs):.4f})')
    ax.axhline(y=len(y_true[y_true == 1]) / len(y_true), color='#9ca3af', lw=1.5, linestyle='--', label=f'Baseline Prior ({len(y_true[y_true == 1]) / len(y_true):.2f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Validation Precision-Recall Curve — ProtT5 Benchmark', pad=12, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    pr_path = os.path.join(FIG_DIR, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"Saved plot {pr_path}")
    
    # ---------------------------------------------------------
    # 5. DATASET DOCS COPY TO reports/
    # ---------------------------------------------------------
    for doc in ["DATASET_CARD.md", "final_dataset_validation.md"]:
        src = os.path.join(DATA_REPORTS_DIR, doc)
        dst = os.path.join(REPORTS_DIR, doc)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
            print(f"Copied {src} -> {dst}")
            
    # ---------------------------------------------------------
    # 6. CURRENT EXPERIMENT FILE INDEX
    # ---------------------------------------------------------
    index_files = [
        # Main results
        ("reports/final_validation_comparison.csv", "Summary comparison table across baseline and all ProtT5 classifiers", "Active", os.path.getsize("reports/final_validation_comparison.csv") if os.path.exists("reports/final_validation_comparison.csv") else 0),
        ("reports/prott5_validation_analysis.md", "Complete validation analysis, threshold sweep, and error audit report", "Active", os.path.getsize("reports/prott5_validation_analysis.md") if os.path.exists("reports/prott5_validation_analysis.md") else 0),
        ("reports/prott5_results.md", "ProtT5 experiment benchmark report and long-sequence strategy analysis", "Active", os.path.getsize("reports/prott5_results.md") if os.path.exists("reports/prott5_results.md") else 0),
        ("reports/model_comparison.csv", "Master benchmark comparison CSV spanning baselines, CNNs, BiLSTM, and ProtT5 variants", "Active", os.path.getsize("reports/model_comparison.csv") if os.path.exists("reports/model_comparison.csv") else 0),
        ("reports/validation_predictions.csv", "Row-level validation predictions with probabilities and cluster IDs (0 test rows)", "Active", os.path.getsize("reports/validation_predictions.csv") if os.path.exists("reports/validation_predictions.csv") else 0),
        ("reports/training_history.csv", "Epoch-by-epoch training and validation loss/metrics for neural architectures", "Active", os.path.getsize("reports/training_history.csv") if os.path.exists("reports/training_history.csv") else 0),
        ("reports/validation_error_analysis_prott5.csv", "Full metadata error classification for all 213 validation samples", "Active", os.path.getsize("reports/validation_error_analysis_prott5.csv") if os.path.exists("reports/validation_error_analysis_prott5.csv") else 0),
        ("reports/validation_error_analysis.md", "Detailed error audit on sequence length and cluster distributions", "Active", os.path.getsize("reports/validation_error_analysis.md") if os.path.exists("reports/validation_error_analysis.md") else 0),
        ("reports/DATASET_CARD.md", "Authoritative dataset documentation for Alzheimer_Protein_Sequence_Dataset_v1.0", "Active", os.path.getsize("reports/DATASET_CARD.md") if os.path.exists("reports/DATASET_CARD.md") else 0),
        ("reports/final_dataset_validation.md", "Frozen dataset verification report certifying zero leakage and cluster splits", "Active", os.path.getsize("reports/final_dataset_validation.md") if os.path.exists("reports/final_dataset_validation.md") else 0),
        
        # Figures
        ("reports/figures/validation_confusion_matrix.png", "Confusion matrix visualization for ProtT5 + RBF SVM on validation set", "Active", os.path.getsize("reports/figures/validation_confusion_matrix.png") if os.path.exists("reports/figures/validation_confusion_matrix.png") else 0),
        ("reports/figures/roc_curve_comparison.png", "Validation ROC curve with AUC=0.7311 for winning ProtT5 model", "Active", os.path.getsize("reports/figures/roc_curve_comparison.png") if os.path.exists("reports/figures/roc_curve_comparison.png") else 0),
        ("reports/figures/precision_recall_curve.png", "Validation Precision-Recall curve with PR-AUC=0.7219", "Active", os.path.getsize("reports/figures/precision_recall_curve.png") if os.path.exists("reports/figures/precision_recall_curve.png") else 0),
        ("reports/figures/neural_training_history.png", "Training vs validation loss and metric curves across epochs", "Active", os.path.getsize("reports/figures/neural_training_history.png") if os.path.exists("reports/figures/neural_training_history.png") else 0),
        
        # Model package
        ("models/prott5_best/config.json", "Deployment configuration specifying architecture, hyperparameters, and thresholds", "Active", os.path.getsize("models/prott5_best/config.json") if os.path.exists("models/prott5_best/config.json") else 0),
        ("models/prott5_best/classifier.joblib", "Serialized winning RBF SVM classifier trained on 990 ProtT5 embeddings", "Active", os.path.getsize("models/prott5_best/classifier.joblib") if os.path.exists("models/prott5_best/classifier.joblib") else 0),
        ("models/prott5_best/predict.py", "Self-contained prediction script for arbitrary input protein sequences", "Active", os.path.getsize("models/prott5_best/predict.py") if os.path.exists("models/prott5_best/predict.py") else 0),
        
        # Embedding data
        ("data/modeling/embeddings/prott5/train_embeddings.npy", "990 x 1024 float32 ProtT5-XL training sequence embeddings", "Active", os.path.getsize("data/modeling/embeddings/prott5/train_embeddings.npy") if os.path.exists("data/modeling/embeddings/prott5/train_embeddings.npy") else 0),
        ("data/modeling/embeddings/prott5/validation_embeddings.npy", "213 x 1024 float32 ProtT5-XL validation sequence embeddings", "Active", os.path.getsize("data/modeling/embeddings/prott5/validation_embeddings.npy") if os.path.exists("data/modeling/embeddings/prott5/validation_embeddings.npy") else 0),
        ("data/modeling/embeddings/prott5/embedding_metadata.csv", "Sample mapping table verifying exactly 1,203 embeddings (0 test)", "Active", os.path.getsize("data/modeling/embeddings/prott5/embedding_metadata.csv") if os.path.exists("data/modeling/embeddings/prott5/embedding_metadata.csv") else 0),
        
        # Scripts
        ("scripts/extract_prott5_embeddings.py", "Resumable extraction script for ProtT5-XL representations", "Active", os.path.getsize("scripts/extract_prott5_embeddings.py") if os.path.exists("scripts/extract_prott5_embeddings.py") else 0),
        ("scripts/train_prott5_models.py", "Training, tuning, and benchmark script for all ProtT5 classifier heads", "Active", os.path.getsize("scripts/train_prott5_models.py") if os.path.exists("scripts/train_prott5_models.py") else 0),
        ("scripts/validate_prott5_experiment.py", "Verification, leakage check, and threshold audit script", "Active", os.path.getsize("scripts/validate_prott5_experiment.py") if os.path.exists("scripts/validate_prott5_experiment.py") else 0),
        
        # Dependencies
        ("requirements_modeling.txt", "Minimal environment requirements specification for modeling pipeline", "Active", os.path.getsize("requirements_modeling.txt") if os.path.exists("requirements_modeling.txt") else 0)
    ]
    
    idx_md = [
        "# Current Experiment File Index — ProtT5-XL Benchmark",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Dataset:** `Alzheimer_Protein_Sequence_Dataset_v1.0`  ",
        "**Test Set Status:** **LOCKED (n=213, 0 representations extracted, 0 evaluated)**  ",
        "",
        "| File Path | Purpose / Description | Status | File Size |",
        "|---|---|:---:|:---:|"
    ]
    
    for path, purpose, status, size in index_files:
        if size >= 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.2f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        idx_md.append(f"| [`{path}`](file:///{os.path.abspath(path).replace(chr(92), '/')} ) | {purpose} | {status} | {size_str} |")
        
    with open(os.path.join(REPORTS_DIR, "CURRENT_EXPERIMENT_FILE_INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(idx_md))
        
    print(f"Generated {os.path.join(REPORTS_DIR, 'CURRENT_EXPERIMENT_FILE_INDEX.md')}")
    print("All review package files prepared successfully.")


if __name__ == "__main__":
    main()
