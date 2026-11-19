# AD-TRACE — Alzheimer's Disease Protein Sequence Classification & Multi-Modal Benchmark

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg)](https://pytorch.org/)
[![ProtT5-XL](https://img.shields.io/badge/PLM-ProtT5--XL--UniRef50-darkgreen.svg)](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A rigorous, zero-leakage computational biology platform and machine learning benchmark for classifying Alzheimer's Disease (MONDO_0004975) associated proteins directly from sequence representations, molecular biophysics, and interactome networks.**

---

## Table of Contents

- [Overview](#overview)
- [Quick Start: Essential Running Commands](#quick-start-essential-running-commands)
- [Key Results & Milestone Evolution](#key-results--milestone-evolution)
- [Zero-Leakage Scientific Methodology](#zero-leakage-scientific-methodology)
- [Repository Architecture](#repository-architecture)
- [System Requirements & Installation](#system-requirements--installation)
- [End-to-End Pipeline Reproduction](#end-to-end-pipeline-reproduction)
  - [1. Data Engineering & QC Suite](#1-data-engineering--qc-suite)
  - [2. Model Training & Benchmarking](#2-model-training--benchmarking)
  - [3. Visualization & Publication Package](#3-visualization--publication-package)
- [Command-Line Inference (CLI)](#command-line-inference-cli)
- [AD-TRACE Live Web Platform & API Server](#ad-trace-live-web-platform--api-server)
  - [Starting the Server](#starting-the-server)
  - [API Endpoints & cURL Examples](#api-endpoints--curl-examples)
  - [Interactive Web UI Capabilities](#interactive-web-ui-capabilities)
- [Dataset Specifications & Schema](#dataset-specifications--schema)
- [Hardware & Reproducibility Specifications](#hardware--reproducibility-specifications)
- [Citation & License](#citation--license)

---

## Overview

Predicting disease-associated targets directly from primary sequences is a fundamental challenge in computational proteogenomics. **AD-TRACE** establishes a systematic benchmark on human proteins associated with Alzheimer's Disease (MONDO_0004975) mapped from Open Targets and UniProtKB/Swiss-Prot (`KW-0026`).

This repository implements:
1. **Deterministic Data Engineering Pipeline**: Cleans, verifies standard 20 amino acids, detects label conflicts, samples length-stratified non-AD controls, and clusters sequences via **MMseqs2** (40% identity, 80% coverage) to prevent homology leakage.
2. **Comprehensive Model Benchmark**:
   - Classical sequence baselines (AAC, DPC, k-mer TF-IDF with Logistic Regression, SVM, Random Forest, XGBoost)
   - Deep learning architectures (Multi-scale 1D-CNNs with kernels 3, 5, 7 and Chunk-Attention Pooling)
   - Recurrent hybrids (CNN + BiLSTM)
   - High-capacity Protein Language Models (**ProtT5-XL-UniRef50**, 3B parameters)
   - Multi-Modal Sequence Model v2.0 (ProtT5-XL + 24 biophysical & codon composition descriptors)
   - High-Accuracy Multi-Modal Pipeline v3.0 (**STRING Interactome Hub Proximity + DNA Codon Usage + ProtT5-XL**) achieving **93.44% Accuracy** and **0.9468 ROC-AUC**.
3. **AD-TRACE Web Platform**: Live interactive web application (FastAPI backend + 3Dmol.js viewer) providing real-time sequence classification, biophysical property profiling, GTEx brain-region RNA expression lookup, STRING functional interaction graphs, and AlphaFold 3D structure visualization.

---

## Quick Start: Essential Running Commands

A quick reference for the most common commands to set up, launch, predict, benchmark, and evaluate AD-TRACE:

### 1. Installation & Environment Setup
```bash
# Clone the repository
git clone https://github.com/NagarajuMudhiraj/ad-trace.git
cd ad-trace

# Create and activate virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
# On Linux / macOS: source venv/bin/activate

# Install all dependencies (Data Engineering, Modeling, API & Web App)
pip install -r requirements.txt
pip install -r requirements_modeling.txt
pip install fastapi uvicorn pydantic
```

### 2. Launch the Web Platform & Interactive UI
Runs the FastAPI server with live multi-omics query capabilities and 3D structural viewer:
```bash
python frontend/server.py
```
* **Interactive Web UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 3. Fast Command-Line Inference (CLI)
Predict disease association for any protein sequence directly from terminal:
```bash
# Sub-millisecond sequence-based inference (Human APP fragment example):
python models/best_validation_model/predict.py "MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQYCQEVYPELQITNVVEANQPVTIQNWCKRGRKQCKTHPHFVIPYRCLVGEFVSDALLVPDKCKFLHQERMDVCETHLHWHTVAKETCSEKSTNLHDYGMLLPCGIDKFRGVEFVCCPLAEESDNVDSADAEEDDSDVWWGGADTDYADGSEDKVVEVAEEEEVAEVEEEEADDDEDDEDGDEVEEEAEEPYEEATERTTSIATTTTTTTESVEEVVREVCSEQAETGPCRAMISRWYFDVTEGKCAPFFYGGCGGNRNNFDTEEYCMAVCGSAIPTTAASTPDAVDKYLETPGDENEHAHFQKAKERLEAKHRERMSQVMREWEEAERQAKNLPKADKKAVIQHFQEKVESLEQEAANERQQLVETHMARVEAMLNDRRRLALENYITALQAVPPRPRHVFNMLKKYVRAEQKDRQHTLKHFEHVRMVDPKKAAQIRSQVMTHLRVIYERMNQSLSLLYNVPAVAEEIQDEVDELLQKEQNYSDDVLANMISEPRISYGNDALMPSLTETKTTVELLPVNGEFSLDDLQPWHSFGADSVPANTENEVEPVDARPAADRGLTTRPGSGLTNIKTEEISEVKMDAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIATVIVITLVMLKKKQYTSIHHGVVEVDAAVTPEERHLSKMQQNGYENPTYKFFEQMQN"

# ProtT5-XL Foundation Model + RBF SVM inference:
python models/prott5_best/predict.py "MVLLRVLILLLSWAAGMGGQYGNPLNKYIRHYEGLSYNVDSLHQK"
```

### 4. Molecular Evidence Stress-Testing Suite (MEST)
Evaluate the 5 scientific stress dimensions (Tier Monotonicity, Modality Fragility, Hub Degree Bias, Tissue Selectivity, and PACI Protective Counterfactuals):
```bash
python scripts/stress_test_molecular_evidence.py
```
* **Output Composite Figure:** `reports/figures/8_molecular_evidence_stress_test.png`
* **Comprehensive Audit Report:** `reports/MOLECULAR_EVIDENCE_STRESS_TEST.md`

### 5. Reproduce Model Training & Benchmarks
```bash
# Train SOTA Multi-Modal Pipeline (93.44% Accuracy, 0.9468 ROC-AUC):
python scripts/train_interactome_dna_model.py

# Train Classical Sequence Baselines (LR, SVM, RF, XGBoost):
python scripts/train_baseline.py

# Train ProtT5-XL Classifier Heads:
python scripts/train_prott5_models.py

# Compare all models and generate milestone curves:
python scripts/compare_models.py
```

### 6. Dataset Validation & Quality Control (QC)
```bash
# Execute 12-point zero-leakage dataset validation:
python scripts/validate_dataset.py

# Run Open Targets & Swiss-Prot source data audit:
python scripts/audit_sources.py
```

### 7. Generate All Publication Figures (300 DPI)
```bash
python scripts/generate_benchmark_visualizations.py
```

---

## Key Results & Milestone Evolution

Across multiple architectural iterations evaluated on cluster-partitioned validation data (zero sequence overlap with training), model performance evolved as follows:

| Milestone | Architecture | Modalities & Features | Accuracy | Balanced Acc | F1-Score | ROC-AUC | PR-AUC | Status |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **v1.0 Baseline** | Logistic Regression (`EXP_02`) | AAC + DPC + TF-IDF (2,920-d) | 60.09% | 60.09% | 0.6188 | 0.6188 | 0.5986 | Baseline Winner |
| **v1.0 Deep Net** | 1D-CNN Multi-Scale (L=1500) | 1D Conv Blocks (k=3,5,7) | 59.15% | 58.74% | 0.4459 | 0.6186 | 0.6094 | Evaluated |
| **v1.0 Deep Net** | Chunked 1D-CNN | Chunk-Attention (W=512) | 55.87% | 55.79% | 0.6083 | 0.5927 | 0.5912 | Evaluated |
| **v1.0 Recurrent**| Hybrid CNN + BiLSTM | Conv1D + BiLSTM (dim=64) | 50.23% | 50.11% | 0.6624 | 0.5643 | 0.5535 | Evaluated |
| **v1.0 Foundation PLM** | ProtT5 + RBF SVM ($C=1.0$) | ProtT5-XL Embeddings (1,024-d) | 65.73% | 65.73% | 0.6404 | 0.7311 | 0.7219 | PLM Winner |
| **v2.0 Curated Benchmark** | ProtT5 + Seq Descriptors | ProtT5-XL + Biophysical Descriptors | **85.90%** | 85.00% | 0.8197 | 0.8868 | 0.8559 | **PASS ($\ge 85\%$)** |
| **v3.0 Multi-Modal SOTA** | **Tuned XGBoost (STRING+DNA+ProtT5)** | **STRING Seed Hubs + DNA + ProtT5-XL** | **93.44%** | **90.73%** | **0.8571** | **0.9468** | **0.8820** | **OUTPERFORMS ($\ge 90\%$)** |

### Confusion Matrix of SOTA Model (v3.0 Validation Cohort, $n=61$):
- **True Negatives (TN):** $45 / 47$ non-neuro controls correctly identified (**95.74% Specificity**)
- **True Positives (TP):** $12 / 14$ causal Alzheimer targets correctly identified (**85.71% Sensitivity**)
- **False Positives (FP):** $2$
- **False Negatives (FN):** $2$
- **Optimal Decision Threshold:** $0.21$

### Molecular Evidence Stress-Testing Suite (MEST) Results:

| Model Architecture | Accuracy | ROC-AUC | Tier Monotonicity (TMR) | Hub Resistance (BRR) | PACI Risk Delta | **Molecular Stress Score (MSTS)** | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (LR Comb)** | 60.09% | 0.6188 | 0.50 | 0.25 | +0.0020 (Blind) | **40.43 / 100** | Fragile |
| **ProtT5-XL (RBF SVM)** | 65.73% | 0.7311 | 0.00 | 0.45 | +0.0150 (Low) | **34.58 / 100** | Noise-Prone |
| **Multi-Modal SOTA (XGBoost)** | **93.44%** | **0.9468** | **1.00** | **0.84** | **+0.3217 (Rescued)** | **62.29 / 100** | **SOTA Robust** |

- **Tier Monotonicity (TMR = 1.00):** SOTA correctly yields highest confidence on Tier 1 causal targets ($P=0.613$) and lower on Tier 3 screening bystanders ($P=0.291$), unlike pure sequence PLMs that suffer from high-throughput screen label noise.
- **Protective-Allele Inversion (PACI):** SOTA correctly differentiates pathogenic drivers (APP Swedish, APOE4) from life-saving protective counterfactual alleles (APP Icelandic A673T, APOE2) with a strong risk reduction gradient ($\Delta_{\text{PACI}} > +0.32$).

---

## Zero-Leakage Scientific Methodology

To guarantee complete scientific integrity and prevent benchmark inflation:

1. **Homology-Aware Splitting**: Partitions were generated using **MMseqs2** set-cover clustering at **40% sequence identity** and **80% coverage**. Sequences belonging to the same cluster are placed entirely within one split.
2. **Strictly Locked Test Set**: The 213 test samples in `data/modeling/test_model.csv` were **never accessed, extracted, or evaluated** during model tuning or threshold selection.
3. **Pure Sequence Separation**: In v1.0 and v2.0 models, exactly 0 database scores, external annotations, or clinical values were fed to the classifier. All features were extracted purely from amino-acid sequences.
4. **12-Point Comprehensive Data QC**: Verified zero missing sequences, zero non-standard amino acids, zero exact duplicates, and zero conflicting labels.

---

## Repository Architecture

```
ad-trace/
├── data/
│   ├── raw/                              # Original source data (unmodified)
│   │   ├── OT-MONDO_0004975-associated-targets-10_7_2026-v26_09.tsv
│   │   ├── uniprot_sprot.fasta
│   │   └── uniprotkb_keyword_KW_0026_AND_reviewed_2026_10_07.fasta
│   ├── interim/                          # Intermediate QC, mappings, and clusters
│   ├── splits/                           # Zero-leakage cluster partitions
│   │   ├── train.csv                     # 990 samples (495 AD, 495 control)
│   │   ├── validation.csv                # 213 samples (106 AD, 107 control)
│   │   └── test.csv                      # 213 samples (LOCKED)
│   ├── processed/                        # Final dataset & molecular properties
│   │   ├── Alzheimer_Protein_Sequence_Dataset_v1.0.csv
│   │   └── protein_properties.csv
│   └── modeling/                         # Tokenized representations & embeddings
│       ├── train_model.csv
│       ├── validation_model.csv
│       ├── test_model.csv                # LOCKED
│       └── embeddings/prott5/            # ProtT5-XL 1024-d float32 embeddings
├── experiments/                          # Master model benchmark comparison tables
├── frontend/                             # Live AD-TRACE Web Application
│   ├── index.html                        # Multi-panel research UI
│   ├── server.py                         # FastAPI backend & inference server
│   ├── css/style.css                     # Glassmorphic responsive styling
│   └── js/app.js                         # State management & live API integration
├── models/
│   ├── baseline/                         # Classical baseline model artifacts
│   ├── cnn/                              # 1D-CNN checkpoints & training histories
│   ├── advanced/                         # Hybrid CNN + BiLSTM weights
│   ├── best_validation_model/            # Packaged v1.0 Winner (LR Comb)
│   │   ├── model_weights.joblib
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── config.json
│   │   └── predict.py                    # Standalone CLI inference script
│   ├── prott5_best/                      # Packaged ProtT5 PLM Winner (RBF SVM)
│   │   ├── classifier.joblib
│   │   ├── config.json
│   │   └── predict.py                    # Standalone CLI inference script
│   ├── multimodal_v2/                    # Packaged Multi-Modal v2 (85.90% Acc)
│   │   ├── multimodal_svc_best.joblib
│   │   └── sequence_feature_scaler.joblib
│   └── interactome_dna_v3/               # Packaged SOTA Interactome Pipeline (93.44% Acc)
│       ├── interactome_dna_xgb_best.joblib
│       ├── scaler_dna.joblib
│       └── scaler_string.joblib
├── reports/                              # Detailed audit logs & scientific reports
│   ├── figures/                          # 7 publication-grade figures (300 DPI)
│   ├── DATASET_CARD.md                   # Full dataset documentation
│   ├── final_dataset_validation.md       # 12-point integrity report
│   ├── model_selection.md                # Classical & CNN selection analysis
│   ├── prott5_results.md                 # ProtT5 benchmark report
│   ├── multimodal_v2_results.md          # Multi-modal v2 report
│   └── interactome_dna_model_results.md  # SOTA interactome pipeline report
├── scripts/                              # Complete executable pipeline scripts
├── tools/mmseqs2/                        # Bundled native MMseqs2 binaries
├── requirements.txt                      # Data engineering & QC dependencies
├── requirements_modeling.txt             # PyTorch, Transformers, XGBoost dependencies
├── README_DATASET.md                     # Dataset reproduction documentation
└── README.md                             # Authoritative repository documentation
```

---

## System Requirements & Installation

### Environment Requirements
- **Operating System:** Windows 10/11 (AMD64) or Linux (Ubuntu 20.04+) or macOS
- **Python:** Python `3.11.x` recommended (3.10+ supported)
- **Hardware:** Multi-core CPU (12 logical cores utilized for vectorized execution) or NVIDIA CUDA-enabled GPU (optional)
- **Disk Space:** ~8 GB (accounting for ProtT5-XL encoder cache and embeddings)

### Step 1: Clone the Repository & Create Virtual Environment

```bash
# Clone the repository
git clone https://github.com/NagarajuMudhiraj/ad-trace.git
cd ad-trace

# Create a virtual environment
python -m venv venv

# Activate virtual environment:
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install Core Data Engineering & QC Dependencies
pip install -r requirements.txt

# Install Deep Learning, Modeling & PLM Dependencies
pip install -r requirements_modeling.txt

# Install Web Application & API Dependencies
pip install fastapi uvicorn pydantic
```

### Step 3: Verify Installation

```bash
python -c "import torch, transformers, xgboost, sklearn, fastapi, uvicorn; print('All core modules imported successfully! PyTorch version:', torch.__version__)"
```

---

## End-to-End Pipeline Reproduction

All pipeline phases are deterministic (fixed master seed `42`). Run the following commands in sequence to reproduce all datasets, models, evaluations, and figures from scratch.

### 1. Data Engineering & QC Suite

```bash
# Phase 1: Audit raw data sources (Open Targets, Swiss-Prot, KW-0026)
python scripts/audit_sources.py

# Phase 2: Map Open Targets ENSG targets to canonical Swiss-Prot reviewed accessions
python scripts/map_open_targets_to_uniprot.py

# Phase 3: Perform sequence cleaning and quality control (standard 20 AA filtering)
python scripts/clean_sequences.py

# Phase 4 & 5: Evaluate score thresholds and construct Alzheimer positive cohort
python scripts/build_labels.py

# Phase 6: Sample length-stratified non-AD negative control proteins (1:1 ratio)
python scripts/build_control_set.py

# Phase 7: Deduplicate sequences and verify zero conflicting labels
python scripts/remove_duplicates.py

# Phase 8: Cluster sequences by homology using MMseqs2 (40% identity, 80% coverage)
python scripts/cluster_sequences.py

# Phase 9 & 10: Partition into cluster-aware train/validation/test splits (Zero Leakage)
python scripts/create_splits.py

# Phase 11 & 12: Calculate ProtParam molecular properties and assemble final CSV
python scripts/calculate_protein_properties.py

# Phase 13: Execute the authoritative 12-point dataset validation suite
python scripts/validate_dataset.py

# Phase 14: Execute Dataset & Label Quality Audit (identifies evidence tiers and label noise)
python scripts/audit_dataset_and_labels.py
```

---

### 2. Model Training & Benchmarking

#### Phase A: Classical Sequence-Only Baselines
Extracts 2,920 biophysical features (AAC + DPC + TF-IDF 2/3-mers) and benchmarks Logistic Regression, Linear SVM, Random Forest, and XGBoost:
```bash
python scripts/train_baseline.py
```

#### Phase B: 1D Convolutional Neural Networks
Trains tokenized embedding + multi-scale 1D-CNNs (kernels 3, 5, 7) evaluating fixed truncation ($L=1000$, $L=1500$) versus Chunk-Attention Pooling ($W=512$):
```bash
python scripts/train_cnn.py
```

#### Phase C: Hybrid CNN + BiLSTM Architecture
Trains a 1D Conv local motif extractor followed by bidirectional LSTM global context scanning:
```bash
python scripts/train_advanced.py
```

#### Phase D: Benchmark Classical & Neural Models
Compares classical and neural models, plots training/validation curves, and packages the winning v1.0 baseline model:
```bash
python scripts/compare_models.py
```

#### Phase E: ProtT5-XL Protein Language Model Pipeline
Extracts 1,024-dimensional residue-level masked mean-pooled representations from `Rostlab/prot_t5_xl_half_uniref50-enc` and trains downstream classifier heads:
```bash
# 1. Extract 1024-d embeddings for train (n=990) and validation (n=213) [resumable]
python scripts/extract_prott5_embeddings.py

# 2. Train and tune ProtT5 classifier heads (Logistic Regression, XGBoost, MLP, Linear/RBF SVM)
python scripts/train_prott5_models.py

# 3. Perform ProtT5 validation audit, threshold sweep, and error analysis
python scripts/validate_prott5_experiment.py
```

#### Phase F: Multi-Modal v2.0 Pipeline (85.90% Accuracy)
Combines ProtT5-XL embeddings with 24-dimensional sequence & DNA codon descriptors on the curated benchmark:
```bash
python scripts/train_multimodal_v2.py
```

#### Phase G: SOTA Multi-Modal Interactome Pipeline v3.0 (93.44% Accuracy)
Integrates STRING interactome hub proximity (confidence to core AD seeds: *APP*, *PSEN1*, *PSEN2*, *MAPT*, *APOE*) + DNA codon properties + ProtT5-XL:
```bash
python scripts/train_interactome_dna_model.py
```

---

### 3. Visualization & Publication Package

Generate the 7 publication-grade figures (300 DPI) and the master experiment review package:

```bash
# Generate all 7 benchmark figures in reports/figures/
python scripts/generate_benchmark_visualizations.py

# Perform in-depth validation error audit
python scripts/evaluate_validation.py

# Assemble master experiment review package
python scripts/prepare_experiment_review_package.py

# Execute the Molecular Evidence Stress-Testing (MEST) suite
python scripts/stress_test_molecular_evidence.py
```

Generated publication figures in `reports/figures/`:
1. `1_benchmark_accuracy_comparison.png` — Accuracy & ROC-AUC across all architectural milestones
2. `2_modality_contribution_comparison.png` — Step-by-step ablation & incremental performance gain
3. `3_confusion_matrices_comparison.png` — Side-by-side confusion matrices (Baseline vs PLM vs SOTA)
4. `4_roc_curves_comparison.png` — ROC curve comparison across architectures
5. `5_precision_recall_curves.png` — Precision-Recall curves & PR-AUC
6. `6_impact_of_label_noise.png` — Accuracy stratified across Open Targets evidence tiers
7. `7_per_class_metrics_breakdown.png` — Precision, Recall, and F1 per class
8. `8_molecular_evidence_stress_test.png` — Evidence Tier Monotonicity, Modality Fragility Decay, PACI Inversion, and MSTS Radar Score

---

## Command-Line Inference (CLI)

You can run fast, standalone predictions directly from the command line for any arbitrary protein sequence using the packaged models.

### 1. Classical Sequence Model (`models/best_validation_model/predict.py`)
Processes full-length sequences via 2,920-d AAC + DPC + TF-IDF with sub-millisecond latency:

```bash
# Test on Human Amyloid Beta Precursor Protein (APP) fragment
python models/best_validation_model/predict.py "MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQYCQEVYPELQITNVVEANQPVTIQNWCKRGRKQCKTHPHFVIPYRCLVGEFVSDALLVPDKCKFLHQERMDVCETHLHWHTVAKETCSEKSTNLHDYGMLLPCGIDKFRGVEFVCCPLAEESDNVDSADAEEDDSDVWWGGADTDYADGSEDKVVEVAEEEEVAEVEEEEADDDEDDEDGDEVEEEAEEPYEEATERTTSIATTTTTTTESVEEVVREVCSEQAETGPCRAMISRWYFDVTEGKCAPFFYGGCGGNRNNFDTEEYCMAVCGSAIPTTAASTPDAVDKYLETPGDENEHAHFQKAKERLEAKHRERMSQVMREWEEAERQAKNLPKADKKAVIQHFQEKVESLEQEAANERQQLVETHMARVEAMLNDRRRLALENYITALQAVPPRPRHVFNMLKKYVRAEQKDRQHTLKHFEHVRMVDPKKAAQIRSQVMTHLRVIYERMNQSLSLLYNVPAVAEEIQDEVDELLQKEQNYSDDVLANMISEPRISYGNDALMPSLTETKTTVELLPVNGEFSLDDLQPWHSFGADSVPANTENEVEPVDARPAADRGLTTRPGSGLTNIKTEEISEVKMDAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIATVIVITLVMLKKKQYTSIHHGVVEVDAAVTPEERHLSKMQQNGYENPTYKFFEQMQN"
```

**Example JSON Output:**
```json
{
  "sequence_length": 770,
  "prediction": "Alzheimer-associated",
  "probability": 0.9427,
  "confidence": 0.8854
}
```

### 2. ProtT5-XL + RBF SVM Model (`models/prott5_best/predict.py`)
Encodes sequence through ProtT5-XL and evaluates via serialized RBF SVM:

```bash
python models/prott5_best/predict.py "MVLLRVLILLLSWAAGMGGQYGNPLNKYIRHYEGLSYNVDSLHQK"
```

**Example JSON Output:**
```json
{
  "sequence_length": 46,
  "prediction": "Alzheimer-associated",
  "probability": 0.8712,
  "confidence": 0.7424
}
```

---

## AD-TRACE Live Web Platform & API Server

AD-TRACE includes a full-featured web application and FastAPI backend server that integrates sequence inference, identity matching, GTEx brain expression, STRING interactomes, and AlphaFold 3D structures.

### Starting the Server

```bash
# Option 1: Direct Python execution
python frontend/server.py

# Option 2: Using Uvicorn CLI
uvicorn frontend.server:app --host 127.0.0.1 --port 8000 --reload
```

Once started, the platform is available at:
- **Interactive Web UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Interactive Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### API Endpoints & cURL Examples

#### 1. System Health & Performance Benchmarks (`GET /api/health`)
Checks backend operational status and returns current model accuracies:

```bash
curl -X GET "http://127.0.0.1:8000/api/health"
```

**Response:**
```json
{
  "status": "healthy",
  "app_name": "AD-TRACE",
  "model_loaded": true,
  "tfidf_loaded": true,
  "dataset_loaded": true,
  "dataset_size": 1416,
  "current_live_inference": "Logistic Regression Comb (Zero-Leakage Sequence Inference)",
  "sota_model_accuracy": 0.9344,
  "sota_model_roc_auc": 0.9468,
  "sota_model_balanced_acc": 0.9073,
  "sota_architecture": "Tuned XGBoost (STRING Interactome + DNA Codons + ProtT5-XL)",
  "multimodal_v2_accuracy": 0.8590,
  "multimodal_v2_roc_auc": 0.8868,
  "plm_v1_accuracy": 0.6573,
  "plm_v1_roc_auc": 0.7311,
  "baseline_v1_accuracy": 0.6009,
  "baseline_v1_roc_auc": 0.6188,
  "protocol": "Zero-Leakage Benchmark (Strictly Locked Test Set n=213)"
}
```

---

#### 2. Comprehensive Multi-Omics Protein Analysis (`POST /api/analyze-protein`)
Executes sequence validation, biophysical calculations, model inference, protein identity resolution, GTEx brain-region median RNA expression query, STRING protein-protein interaction network query, and AlphaFold DB 3D metadata retrieval:

```bash
curl -X POST "http://127.0.0.1:8000/api/analyze-protein" \
     -H "Content-Type: application/json" \
     -d '{
       "sequence": ">sp|P05067|A4_HUMAN Amyloid-beta precursor protein\nMLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQYCQEVYPELQITNVVEANQPVTIQNWCKRGRKQCKTHPHFVIPYRCLVGEFVSDALLVPDKCKFLHQERMDVCETHLHWHTVAKETCSEKSTNLHDYGMLLPCGIDKFRGVEFVCCPLAEESDNVDSADAEEDDSDVWWGGADTDYADGSEDKVVEVAEEEEVAEVEEEEADDDEDDEDGDEVEEEAEEPYEEATERTTSIATTTTTTTESVEEVVREVCSEQAETGPCRAMISRWYFDVTEGKCAPFFYGGCGGNRNNFDTEEYCMAVCGSAIPTTAASTPDAVDKYLETPGDENEHAHFQKAKERLEAKHRERMSQVMREWEEAERQAKNLPKADKKAVIQHFQEKVESLEQEAANERQQLVETHMARVEAMLNDRRRLALENYITALQAVPPRPRHVFNMLKKYVRAEQKDRQHTLKHFEHVRMVDPKKAAQIRSQVMTHLRVIYERMNQSLSLLYNVPAVAEEIQDEVDELLQKEQNYSDDVLANMISEPRISYGNDALMPSLTETKTTVELLPVNGEFSLDDLQPWHSFGADSVPANTENEVEPVDARPAADRGLTTRPGSGLTNIKTEEISEVKMDAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIATVIVITLVMLKKKQYTSIHHGVVEVDAAVTPEERHLSKMQQNGYENPTYKFFEQMQN"
     }'
```

**Returned Payload Structure:**
- `input_summary`: Sequence length, FASTA header, validation status.
- `prediction`: Disease association probability, binary class call, confidence score, feature attribution.
- `identity`: Matched UniProt accession, HGNC gene symbol, organism, identity percentage, sequence coverage.
- `sequence_properties`: Molecular weight (Da), theoretical isoelectric point (pI), GRAVY hydropathicity, aromaticity, instability index, charged residue count.
- `brain_expression`: GTEx median RNA TPM across 13 human brain sub-regions (Cortex, Hippocampus, Amygdala, Cerebellum, Substantia Nigra, etc.).
- `interactions`: Physical & functional partners with confidence scores from STRING DB.
- `structure`: AlphaFold DB 3D model status, pLDDT per-residue confidence summary, direct PDB streaming link.

---

#### 3. Preset Example Proteins (`GET /api/examples`)
Returns pre-configured benchmark proteins for instant testing in the UI:

```bash
curl -X GET "http://127.0.0.1:8000/api/examples"
```

Available presets include:
- `APP` (Amyloid-beta Precursor Protein, `P05067`) — Positive
- `MAPT` (Microtubule-Associated Protein Tau, `P10636`) — Positive
- `PSEN1` (Presenilin-1, `P49768`) — Positive
- `APOE` (Apolipoprotein E, `P02649`) — Positive
- `GAPDH` (Glyceraldehyde-3-phosphate dehydrogenase, `P04406`) — Negative Control
- `ALB` (Serum Albumin, `P02768`) — Negative Control

---

#### 4. AlphaFold PDB Proxy (`GET /api/proxy-pdb/{accession}`)
Streams 3D coordinates for the specified UniProt accession directly through the backend to avoid browser CORS restrictions:

```bash
curl -X GET "http://127.0.0.1:8000/api/proxy-pdb/P05067" -o P05067.pdb
```

---

### Interactive Web UI Capabilities

The AD-TRACE frontend provides 5 dedicated views:
1. **Analyze Protein:** Interactive sequence input supporting raw amino acids or FASTA headers, real-time residue validation, gauge probability visualization, radar charts of biophysical properties, GTEx brain expression bar charts, interactive STRING network graphs, and a 3D macromolecular structure viewer powered by **3Dmol.js** with cartoon/surface representations and pLDDT coloring.
2. **Preset Examples:** One-click loading of canonical Alzheimer targets (*APP*, *MAPT*, *PSEN1*, *APOE*) and verified negative controls (*GAPDH*, *ALB*).
3. **Benchmark Matrices:** Full multi-metric comparison table across all 11 architectures, alongside interactive viewing of all 7 publication-grade figures.
4. **Scientific Methodology:** Detailed exposition of zero-leakage protocols, MMseqs2 set-cover clustering, evidence scoring, and cross-validation guarantees.
5. **API & Research Documentation:** Complete schema reference for integration into bioinformatic workflows.

---

## Dataset Specifications & Schema

The frozen benchmark dataset is located at:
`data/processed/Alzheimer_Protein_Sequence_Dataset_v1.0.csv`

### Feature Columns:

| Column Name | Data Type | Description |
|---|---|---|
| `sample_id` | String | Unique record identifier (`ALZ_PROT_0001` - `ALZ_PROT_1416`) |
| `uniprot_accession` | String | Canonical UniProtKB/Swiss-Prot accession |
| `gene` | String | Official HGNC gene symbol |
| `protein_name` | String | Curated protein descriptive name |
| `sequence` | String | Validated primary amino acid sequence (standard 20 AA) |
| `sequence_length` | Integer | Sequence length in amino acids |
| `molecular_weight` | Float | Molecular weight in Daltons |
| `open_targets_globalScore` | Float | Target association score for MONDO_0004975 (Metadata only) |
| `evidence_sources` | String | Documented evidence streams (genetics, literature, clinical trials) |
| `uniprot_alzheimer_annotation` | Boolean | True if present in UniProt KW-0026 gold-standard reference set |
| `cluster_id` | String | MMseqs2 homology cluster identifier |
| `label` | Integer | Binary classification label (`1` = Alzheimer-associated, `0` = Control) |
| `split` | String | Partition assignment (`train`, `validation`, `test`) |

> **IMPORTANT SCIENTIFIC PROTOCOL:** `open_targets_globalScore` and `uniprot_alzheimer_annotation` are strictly metadata columns and MUST NOT be fed to sequence classifiers as features.

---

## Hardware & Reproducibility Specifications

- **Python Version:** `3.11.9` (64-bit AMD64)
- **Host OS:** Windows 11 Enterprise (AMD64)
- **CPU Cores:** 12 Logical Cores (Vectorized batch execution)
- **Master Reproducibility Random Seed:** `42`
- **PyTorch Execution:** CPU vectorized inference (`torch.set_num_threads(12)`)
- **ProtT5 Model:** `Rostlab/prot_t5_xl_half_uniref50-enc` (3B parameters)
- **Clustering Binary:** Native MMseqs2 (`tools/mmseqs2/mmseqs/bin/mmseqs.exe`)

---

## Citation & License

This project is licensed under the [MIT License](LICENSE).
