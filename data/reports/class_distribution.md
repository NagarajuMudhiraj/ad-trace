# Dataset Class Balance and Distribution Report

**Date:** 2026-10-07 21:32:28
**Pipeline Phase:** Phase 9 (Splits) & Phase 10 (Balance Audit)
**Total Cohort Size:** 1,416 protein sequences

## Executive Summary

- **Total Samples:** 1,416
- **Positive Samples:** 708 (50.0%)
- **Control Samples:** 708 (50.0%)
- **Class Ratio:** Exactly 1:1 balanced across positive and length-matched controls
- **Sequence Homology Clustering:** MMseqs2 (40% sequence identity, 80% coverage)
- **Total Sequence Clusters:** 1,302
- **Cross-Split Overlap:** **0 clusters, 0 sequences** (Strict zero leakage)

> [!IMPORTANT]
> **Methodological Integrity Notice:** Zero synthetic sequences were generated. No SMOTE or synthetic oversampling techniques were applied to protein sequences. Pre-split oversampling is strictly prohibited to preserve biological sequence integrity.

## Split Breakdown

| Split | Total Samples | % of Cohort | Positive (1) | Control (0) | Class Ratio | Sequence Clusters | Cluster % |
|---|---|---|---|---|---|---|---|
| **Train** | 990 | 69.9% | 495 | 495 | 495:495 (1:1.00) | 913 | 70.1% |
| **Validation** | 213 | 15.0% | 106 | 107 | 106:107 (1:1.01) | 194 | 14.9% |
| **Test** | 213 | 15.0% | 107 | 106 | 107:106 (1:0.99) | 195 | 15.0% |
| **Total** | 1,416 | 100.0% | 708 | 708 | 708:708 (1:1.00) | 1,302 | 100.0% |

## Sequence Length Distribution (Amino Acids)

| Split / Class | Minimum | 25th %ile | Median | Mean | 75th %ile | Maximum | Standard Deviation |
|---|---|---|---|---|---|---|---|
| Complete Cohort (All) | 31 | 333.0 | 526.5 | 756.8 | 874.2 | 34,350 | 1179.8 |
| Positive Class (Label=1) | 63 | 333.0 | 514.0 | 775.9 | 870.2 | 34,350 | 1451.6 |
| Control Class (Label=0) | 31 | 331.8 | 538.0 | 737.7 | 876.8 | 14,507 | 822.2 |
| Training Split | 31 | 337.2 | 539.5 | 794.8 | 899.0 | 34,350 | 1353.2 |
| Validation Split | 61 | 310.0 | 466.0 | 628.0 | 779.0 | 3,056 | 508.1 |
| Test Split | 78 | 339.0 | 560.0 | 708.7 | 840.0 | 5,795 | 677.0 |

## Leakage Verification Matrix

| Check Description | Formula | Observed Value | Validation Status |
|---|---|---|---|
| Train-Val Cluster Intersection | `Train Clusters ∩ Val Clusters` | 0 | **PASS** |
| Train-Test Cluster Intersection | `Train Clusters ∩ Test Clusters` | 0 | **PASS** |
| Val-Test Cluster Intersection | `Val Clusters ∩ Test Clusters` | 0 | **PASS** |
| Train-Val Sequence Intersection | `Train Seqs ∩ Val Seqs` | 0 | **PASS** |
| Train-Test Sequence Intersection | `Train Seqs ∩ Test Seqs` | 0 | **PASS** |
| Val-Test Sequence Intersection | `Val Seqs ∩ Test Seqs` | 0 | **PASS** |

## Audit Conclusion

**STATUS: PASSED.**
The cluster-aware dataset splitting guarantees zero data leakage between training, validation, and evaluation cohorts while preserving exact 1:1 balance and harmonious length distributions.