# Current Experiment File Index — ProtT5-XL Benchmark

**Date:** 2026-10-08 21:38:01  
**Dataset:** `Alzheimer_Protein_Sequence_Dataset_v1.0`  
**Test Set Status:** **LOCKED (n=213, 0 representations extracted, 0 evaluated)**  

| File Path | Purpose / Description | Status | File Size |
|---|---|:---:|:---:|
| [`reports/final_validation_comparison.csv`](final_validation_comparison.csv) | Summary comparison table across baseline and all ProtT5 classifiers | Active | 557 B |
| [`reports/prott5_validation_analysis.md`](prott5_validation_analysis.md) | Complete validation analysis, threshold sweep, and error audit report | Active | 8.8 KB |
| [`reports/prott5_results.md`](prott5_results.md) | ProtT5 experiment benchmark report and long-sequence strategy analysis | Active | 3.1 KB |
| [`reports/model_comparison.csv`](model_comparison.csv) | Master benchmark comparison CSV spanning baselines, CNNs, BiLSTM, and ProtT5 variants | Active | 1.0 KB |
| [`reports/validation_predictions.csv`](validation_predictions.csv) | Row-level validation predictions with probabilities and cluster IDs (0 test rows) | Active | 8.2 KB |
| [`reports/training_history.csv`](training_history.csv) | Epoch-by-epoch training and validation loss/metrics for neural architectures | Active | 423 B |
| [`reports/validation_error_analysis_prott5.csv`](validation_error_analysis_prott5.csv) | Full metadata error classification for all 213 validation samples | Active | 158.9 KB |
| [`reports/validation_error_analysis.md`](validation_error_analysis.md) | Detailed error audit on sequence length and cluster distributions | Active | 3.5 KB |
| [`reports/DATASET_CARD.md`](DATASET_CARD.md) | Authoritative dataset documentation for Alzheimer_Protein_Sequence_Dataset_v1.0 | Active | 9.2 KB |
| [`reports/final_dataset_validation.md`](final_dataset_validation.md) | Frozen dataset verification report certifying zero leakage and cluster splits | Active | 3.5 KB |
| [`reports/figures/validation_confusion_matrix.png`](figures/validation_confusion_matrix.png) | Confusion matrix visualization for ProtT5 + RBF SVM on validation set | Active | 36.4 KB |
| [`reports/figures/roc_curve_comparison.png`](figures/roc_curve_comparison.png) | Validation ROC curve with AUC=0.7311 for winning ProtT5 model | Active | 63.5 KB |
| [`reports/figures/precision_recall_curve.png`](figures/precision_recall_curve.png) | Validation Precision-Recall curve with PR-AUC=0.7219 | Active | 53.7 KB |
| [`reports/figures/neural_training_history.png`](figures/neural_training_history.png) | Training vs validation loss and metric curves across epochs | Active | 125.4 KB |
| [`models/prott5_best/config.json`](../models/prott5_best/config.json) | Deployment configuration specifying architecture, hyperparameters, and thresholds | Active | 676 B |
| [`models/prott5_best/classifier.joblib`](../models/prott5_best/classifier.joblib) | Serialized winning RBF SVM classifier trained on 990 ProtT5 embeddings | Active | 6.82 MB |
| [`models/prott5_best/predict.py`](../models/prott5_best/predict.py) | Self-contained prediction script for arbitrary input protein sequences | Active | 2.4 KB |
| [`data/modeling/embeddings/prott5/train_embeddings.npy`](../data/modeling/embeddings/prott5/train_embeddings.npy) | 990 x 1024 float32 ProtT5-XL training sequence embeddings | Active | 3.87 MB |
| [`data/modeling/embeddings/prott5/validation_embeddings.npy`](../data/modeling/embeddings/prott5/validation_embeddings.npy) | 213 x 1024 float32 ProtT5-XL validation sequence embeddings | Active | 852.1 KB |
| [`data/modeling/embeddings/prott5/embedding_metadata.csv`](../data/modeling/embeddings/prott5/embedding_metadata.csv) | Sample mapping table verifying exactly 1,203 embeddings (0 test) | Active | 30.2 KB |
| [`scripts/extract_prott5_embeddings.py`](../scripts/extract_prott5_embeddings.py) | Resumable extraction script for ProtT5-XL representations | Active | 6.9 KB |
| [`scripts/train_prott5_models.py`](../scripts/train_prott5_models.py) | Training, tuning, and benchmark script for all ProtT5 classifier heads | Active | 22.2 KB |
| [`scripts/validate_prott5_experiment.py`](../scripts/validate_prott5_experiment.py) | Verification, leakage check, and threshold audit script | Active | 21.8 KB |
| [`requirements_modeling.txt`](../requirements_modeling.txt) | Minimal environment requirements specification for modeling pipeline | Active | 565 B |