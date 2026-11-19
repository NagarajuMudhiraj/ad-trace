# Classical Sequence-Only Baseline Benchmark Report

**Date:** 2026-10-07 21:46:44
**Pipeline Phase:** Phase 4 (Establish a Simple Baseline)
**Training Set:** `data/modeling/train_model.csv` (n=990)
**Evaluation Set:** `data/modeling/validation_model.csv` (n=213, cluster-separated)
**Final Test Set Status:** **LOCKED** (Strictly withheld)

## Executive Summary

Before applying deep sequence architectures, a rigorous classical machine-learning baseline was established.
All features were derived **exclusively from primary amino-acid sequences** without using any clinical, genetic, or biological metadata.

### Feature Representations Evaluated
1. **Amino-Acid Composition (AAC):** Normalized frequency of 20 standard amino acids.
2. **Dipeptide Composition (DPC):** Normalized frequency of all 400 possible 2-mer transitions.
3. **Character (2,3)-mer TF-IDF:** Sublinear term-frequency inverse-document-frequency representation capturing local sequence motifs (2,500 dimensions).
**Total Feature Space:** 2,920 continuous sequence-derived features.

## Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | MCC | Training Time |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression (AAC)** | 0.5164 | 0.5133 | 0.5472 | 0.5297 | 0.5205 | 0.5292 | 0.0332 | 0.01s |
| **Logistic Regression (Comb)** | 0.6009 | 0.5897 | 0.6509 | 0.6188 | 0.6188 | 0.5986 | 0.2033 | 0.04s |
| **Linear SVM (Comb)** | 0.5869 | 0.5804 | 0.6132 | 0.5963 | 0.6064 | 0.5813 | 0.1742 | 0.07s |
| **Random Forest (Comb)** | 0.5540 | 0.5470 | 0.6038 | 0.5740 | 0.5715 | 0.5693 | 0.1090 | 0.63s |
| **XGBoost (Comb)** | 0.5446 | 0.5372 | 0.6132 | 0.5727 | 0.5744 | 0.5948 | 0.0907 | 8.87s |

## Confusion Matrix Breakdown

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|---|---|---|---|---|
| Logistic Regression (AAC) | 52 | 55 | 48 | 58 |
| Logistic Regression (Comb) | 59 | 48 | 37 | 69 |
| Linear SVM (Comb) | 60 | 47 | 41 | 65 |
| Random Forest (Comb) | 54 | 53 | 42 | 64 |
| XGBoost (Comb) | 51 | 56 | 41 | 65 |

## Analysis & Observations

1. **Top Classical Performer:** `Logistic Regression (Comb)` achieved the highest F1 score (0.6188), ROC-AUC (0.6188), and MCC (0.2033).
2. **Value of Motif Information:** Moving from pure AAC (20 features, ROC-AUC 0.5205) to combined k-mer TF-IDF + DPC (2,920 features) yielded an immediate ~10% gain in ROC-AUC (0.6189), confirming that short sequence motifs contain discriminative Alzheimer association signals.
3. **Linear vs Non-Linear Models:** High-dimensional linear classifiers (Logistic Regression and Linear SVM) outperformed tree ensembles (Random Forest and XGBoost) on sparse n-gram features, as tree models struggle to partition high-dimensional k-mer spaces with small sample sizes.

## Next Steps

These classical metrics establish the benchmark threshold. Neural architectures (1D CNNs, BiLSTMs) must demonstrate equal or superior feature extraction and contextual representation to justify complexity.