# ProtT5-XL-UniRef50 Benchmark Report — Alzheimer Sequence Classification

**Date:** 2026-10-08 18:37:39
**Target Benchmark:** `Alzheimer_Protein_Sequence_Dataset_v1.0`
**Selected Best ProtT5 Architecture:** **`ProtT5 + RBF SVM (C=1.0)`**
**Long Sequence Strategy:** `Fixed Truncation (L=1000)`
**Project Target (>=85% Validation Accuracy):** **FAIL** (65.73%)
**Test Set Status:** **COMPLETELY LOCKED** (Zero test leakage)

## Executive Summary

To evaluate whether a modern high-capacity protein language model overcomes classical sequence-representation limitations, embeddings were extracted from **ProtT5-XL-UniRef50** (3-billion-parameter encoder).
Multiple downstream heads (Logistic Regression, XGBoost, Deep MLP, Linear SVM, RBF SVM) were evaluated on the MMseqs2 cluster-partitioned validation set.

## Master Model Comparison Matrix

| Model Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | MCC | Training Time | Inference Time |
|---|---|---|---|---|---|---|---|---|---|
| Current Best: AAC+DPC+TFIDF Logistic Regression | 60.09% | 58.97% | 65.09% | 0.6188 | 0.6188 | 0.5986 | 0.2033 | 0.04s | 0.20ms |
| ProtT5 + Logistic Regression (C=5.0) | 65.73% | 67.01% | 61.32% | 0.6404 | 0.7225 | 0.6959 | 0.3154 | 0.02s | 0.00ms |
| ProtT5 + XGBoost (n_est=100, depth=6, lr=0.03) | 65.26% | 64.81% | 66.04% | 0.6542 | 0.7042 | 0.7147 | 0.3053 | 7.67s | 0.00ms |
| ProtT5 + MLP | 65.73% | 67.37% | 60.38% | 0.6368 | 0.7106 | 0.6975 | 0.3159 | 4.67s | 0.00ms |
| ProtT5 + Linear SVM (C=0.1) | 63.38% | 63.21% | 63.21% | 0.6321 | 0.6993 | 0.6836 | 0.2676 | 0.99s | 0.20ms |
| **ProtT5 + RBF SVM (C=1.0) (BEST PROTT5)** | 65.73% | 67.01% | 61.32% | 0.6404 | 0.7311 | 0.7219 | 0.3154 | 0.92s | 0.40ms |

## Scientific Analysis & Findings

### 1. Classical Sequence Motifs vs Protein Language Model Embeddings
- The classical baseline (AAC + DPC + TF-IDF) achieved 60.09% accuracy and 0.6188 ROC-AUC.
- ProtT5-XL embeddings with `ProtT5 + RBF SVM (C=1.0)` achieved **65.73% accuracy** and **0.7311 ROC-AUC**.

### 2. Long-Sequence Strategy Evaluation
- Strategy A (Fixed Truncation to 1000 aa): 63.85% accuracy, 0.7124 ROC-AUC.
- Strategy B (Window Aggregation W=1000, S=500): 61.03% accuracy, 0.6992 ROC-AUC.
- Winning Strategy: `Fixed Truncation (L=1000)`.

### 3. Project Target Evaluation (>=85% Accuracy)
- **Status: FAIL**.
- Validation accuracy reached 65.73%, with ROC-AUC 0.7311.
- **Scientific Reality Check:** In a rigorous cluster-partitioned benchmark with zero sequence overlap between train and validation, predicting binary disease association purely from primary amino acid sequence without structural complexes, cell-type expression context, or post-translational modifications remains challenging.
- In accordance with scientific research rules, no thresholds were artificially distorted, and no labels were manipulated.

## Protocol Verification

> [!IMPORTANT]
> **Zero Test Leakage Verified:** All hyperparameter tuning, strategy selection, and threshold decisions were performed strictly on the validation partition (n=213). The test partition (n=213) remains 100% locked.