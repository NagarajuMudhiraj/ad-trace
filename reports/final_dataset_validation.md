# Alzheimer Protein Sequence Dataset v1.0 — Final Validation Report

**Date:** 2026-10-07 21:35:59
**Target Artifact:** `data/processed/Alzheimer_Protein_Sequence_Dataset_v1.0.csv`
**Overall Validation Status:** **PASS**

## Executive Summary

A rigorous, 12-point quality and integrity audit was executed against `Alzheimer_Protein_Sequence_Dataset_v1.0`.
All 12 validation checks have completed with **100% compliance**.

## 12-Point Data Validation Matrix

| Check ID | Requirement Description | Status | Verification Findings |
|---|---|---|---|
| **CHECK 1** | No missing sequences | **PASS** | Null sequences: 0, Empty sequences: 0, Zero length: 0 across 1416 records. |
| **CHECK 2** | No invalid amino-acid sequences | **PASS** | Sequences with non-canonical/ambiguous residues: 0. 100% of sequences contain strictly standard 20 amino acids. |
| **CHECK 3** | No exact duplicate sequences | **PASS** | Duplicate sequences found: 0. Unique sequences: 1416/1416. |
| **CHECK 4** | No conflicting labels | **PASS** | Sequences mapped to multiple opposing labels: 0. |
| **CHECK 5** | No sequence overlap across splits | **PASS** | Train-Val sequence overlap: 0, Train-Test: 0, Val-Test: 0. |
| **CHECK 6** | No homologous cluster overlap across splits | **PASS** | Train-Val cluster overlap: 0, Train-Test: 0, Val-Test: 0 (MMseqs2 40% identity clusters). |
| **CHECK 7** | No label-generation feature leakage into model input | **PASS** | open_targets_globalScore and uniprot_alzheimer_annotation are strictly marked as provenance metadata. No model training scripts or weights exist. |
| **CHECK 8** | All positive labels have documented evidence | **PASS** | Positive samples with missing evidence: 0/708. 100% trace to Open Targets globalScore >= 0.3 or UniProt KW-0026 gold-standard. |
| **CHECK 9** | All negative/control labels satisfy control-selection rules | **PASS** | Controls with KW-0026 annotation: 0, Controls with OT score >= 0.1: 0. All 708 controls satisfy strict non-AD criteria. |
| **CHECK 10** | All accessions traceable to original source | **PASS** | Untraceable accessions: 0/1416. 100% of accessions match canonical reviewed Swiss-Prot database. |
| **CHECK 11** | Dataset class balance documented | **PASS** | Positive: 708, Control: 708. Class ratio exactly 1:1.00 (50.0% / 50.0%). Documented in class_distribution.md. |
| **CHECK 12** | Dataset statistics are reproducible | **PASS** | Split file sample counts: {'train': 990, 'validation': 213, 'test': 213}. Exact match with pipeline specifications (Train: 990, Val: 213, Test: 213). |

## Data Leakage & Homology Control Audit

- **Sequence Identity Clustering:** MMseqs2 at 40% sequence identity (`--min-seq-id 0.40`) and 80% coverage (`-c 0.80`).
- **Cluster Overlap:** Zero clusters shared across any split pair.
- **Sequence Overlap:** Zero exact amino-acid sequences shared across any split pair.
- **Class Leakage:** No metadata features (`open_targets_globalScore`, `uniprot_alzheimer_annotation`) are exposed as model input features.

## Dataset Certification

```
============================================================
DATASET VALIDATION CERTIFICATE
Dataset: Alzheimer_Protein_Sequence_Dataset_v1.0.csv
Validation Result: PASS
Data Leakage Checks: PASSED (0 cluster overlap, 0 sequence overlap)
Class Balance: PASSED (708 positive : 708 control, 1:1)
Provenance & Traceability: PASSED (100% Swiss-Prot reviewed human)
Ready for Supervised Model Training: YES
============================================================
```