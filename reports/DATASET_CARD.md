# Dataset Card: Alzheimer_Protein_Sequence_Dataset_v1.0

## 1. Dataset Name
`Alzheimer_Protein_Sequence_Dataset_v1.0`

## 2. Version
`1.0.0` (Frozen Release — October 7, 2026)

## 3. Purpose
This benchmark dataset is engineered for supervised machine-learning and deep-learning protein sequence classification tasks in Alzheimer's disease (AD) research. Its primary objective is to enable models to discriminate between genuine Alzheimer's disease-associated human proteins and biological control proteins directly from primary amino-acid sequences, free from data leakage, length confounding, and label noise.

## 4. Source Databases
1. **UniProtKB/Swiss-Prot (Reviewed Canonical Database):** Global repository of expert-curated, non-redundant protein sequences.
2. **UniProtKB Keyword `KW-0026` (*Alzheimer disease*):** Gold-standard reviewed human proteins causally or mechanistically tied to Alzheimer's pathology.
3. **Open Targets Platform (Release v26.09, Disease ID: `MONDO_0004975`):** Comprehensive multi-omic target-disease association database integrating genetic evidence, somatic mutations, clinical trials, and curated literature.

## 5. Download Dates
- UniProtKB KW-0026 Reviewed Fasta: `2026-10-07`
- UniProtKB/Swiss-Prot Canonical Fasta: `2026-10-07`
- Open Targets Platform Target Association Dataset: `2026-10-07` (Release v26.09)

## 6. Number of Source Records
- **Open Targets Associated Targets:** 13,006 raw rows (12,689 unique gene symbols).
- **UniProt Swiss-Prot Database:** 575,748 canonical reviewed protein sequences (20,432 *Homo sapiens*).
- **UniProt KW-0026 Reviewed Alzheimer Cohort:** 19 gold-standard human proteins.

## 7. Number of Final Sequences
- **Total Final Sequences:** **1,416** unique human protein sequences.
- **Positive Class (`label = 1`):** **708** sequences.
- **Control Class (`label = 0`):** **708** sequences.
- **Class Ratio:** Exactly **1:1.00** (50.0% positive, 50.0% control).

## 8. Positive Class Definition
The positive class is constructed using a scientifically justified multi-tier framework:
1. **Open Targets Multi-Evidence Threshold:** Gene symbols with an Open Targets `globalScore >= 0.3` for Alzheimer disease (`MONDO_0004975`), providing high biological convergence (258 genetic loci, 118 clinical trial targets, and 579 literature-supported targets).
2. **Canonical Swiss-Prot Mapping:** Deterministically mapped to canonical reviewed *Homo sapiens* Swiss-Prot entries using lowest Protein Existence level (`PE`), longest sequence length, and UniProt provenance.
3. **UniProt KW-0026 Anchor Cohort:** All 19 reviewed UniProt Alzheimer reference proteins (e.g., APP, PSEN1, PSEN2, APOE, MAPT, TREM2, ABCA7, SORL1, ADAM10, GSK3B) are included and flagged with `uniprot_alzheimer_annotation = True`.

## 9. Control Class Definition
Control proteins (`label = 0`) are sampled from reviewed *Homo sapiens* Swiss-Prot proteins meeting strict negative criteria:
1. Must not appear in the Alzheimer positive set (by accession, gene symbol, or sequence).
2. Must not possess the UniProt `KW-0026` (*Alzheimer disease*) annotation.
3. Must not possess borderline Open Targets Alzheimer association (`globalScore >= 0.1` excluded).
4. Must consist exclusively of standard 20 amino acids.
5. **Length-Stratified Matching:** Controls are sampled from 10 sequence-length bins matching the exact distribution of the positive cohort, eliminating length as an artificial classification shortcut.

## 10. Sequence Preprocessing
1. Converted to uppercase.
2. Stripped of all whitespace, carriage returns, and line feeds.
3. Validated against the 20 standard IUPAC amino acids (`ACDEFGHIKLMNPQRSTVWY`).
4. Non-standard residues (U, O) and ambiguous characters (B, Z, J, X) logged in `sequence_qc.csv` and excluded from model cohorts.
5. Full traceability preserved without silent record deletion.

## 11. Duplicate Handling
1. **Cross-Class Conflicting Labels:** Monitored; 0 conflicting sequences detected.
2. **Within-Class Duplicate Sequences:** Deduplicated by prioritizing higher Open Targets scores and primary canonical accession provenance.

## 12. Homology Handling & Clustering
- **Tool:** MMseqs2 (Release: 8cc5ce367b5638c4306c2d7cfc652dd099a4643f, win64).
- **Parameters:** Cascaded Set-Cover clustering at `--min-seq-id 0.40` (40% sequence identity) and `-c 0.80` (80% bidirectional coverage).
- **Clustering Outcome:** 1,416 proteins partitioned into **1,302 independent sequence clusters** (1,226 singletons, 76 multi-protein families).
- **Cluster Integrity:** No homologous cluster spans multiple splits.

## 13. Split Strategy
- **Partitioning Algorithm:** Multi-objective deficit-minimizing cluster allocation.
- **Split Proportions:**
  - **Training Set:** 990 samples (69.9%) | 495 Positive, 495 Control | 913 clusters
  - **Validation Set:** 213 samples (15.0%) | 106 Positive, 107 Control | 194 clusters
  - **Test Set:** 213 samples (15.0%) | 107 Positive, 106 Control | 195 clusters
- **Leakage Safeguards:**
  - `Train Clusters ∩ Val Clusters = 0`
  - `Train Clusters ∩ Test Clusters = 0`
  - `Val Clusters ∩ Test Clusters = 0`
  - Exact sequence overlap between all splits is **0**.

## 14. Class Distribution
| Split | Total Samples | Positive (1) | Control (0) | Class Ratio | Clusters |
|---|---|---|---|---|---|
| **Train** | 990 | 495 | 495 | 1:1.00 | 913 |
| **Validation** | 213 | 106 | 107 | 1:1.01 | 194 |
| **Test** | 213 | 107 | 106 | 1:0.99 | 195 |
| **Complete Dataset** | **1,416** | **708** | **708** | **1:1.00** | **1,302** |

## 15. Molecular Property Statistics
Physicochemical properties derived via `Bio.SeqUtils.ProtParam`:

| Property | All Samples (Mean ± SD) | Positive (Mean) | Control (Mean) | Units |
|---|---|---|---|---|
| **Sequence Length** | 756.77 ± 1180.21 | 775.86 | 737.68 | Amino acids |
| **Molecular Weight** | 84,198.31 ± 129,835.61 | 86,442.99 | 81,953.63 | Daltons |
| **Isoelectric Point (pI)** | 7.21 ± 1.69 | 7.20 | 7.23 | pH |
| **Aromaticity** | 0.080 ± 0.028 | 0.081 | 0.078 | Fraction |
| **Instability Index** | 48.92 ± 11.50 | 47.83 | 50.00 | Score |
| **GRAVY Hydrophobicity** | -0.356 ± 0.377 | -0.355 | -0.357 | Hydropathy index |

*Note: Properties are virtually identical across classes, confirming that length-stratified sampling eliminated simple physical confounding.*

## 16. Potential Biases
1. **Research Publication Bias:** High-scoring targets in Open Targets inherently possess more literature mentions.
2. **Gene Family Clustering:** Certain multi-protein families (e.g., GPCRs, kinases) have multiple members represented; MMseqs2 cluster-splitting prevents intra-family leakage.
3. **Canonical Isoform Selection:** Sequences reflect canonical Swiss-Prot isoforms; alternative splicing variants are omitted from this canonical benchmark.

## 17. Known Limitations
1. Does not model post-translational modifications (PTMs, such as tau phosphorylation or APP proteolytic cleavage products).
2. Does not incorporate 3D quaternary complexes or membrane lipid environment directly in primary sequence strings.

## 18. Data Leakage Controls
1. **Homology Leakage:** Prevented via MMseqs2 cluster-aware splitting (zero cluster overlap across train, validation, and test).
2. **Identity Leakage:** Exact sequence deduplication across all splits (zero shared sequences).
3. **Feature Leakage:** `open_targets_globalScore` and `uniprot_alzheimer_annotation` are restricted to metadata provenance and are explicitly prohibited from model feature inputs.
4. **Oversampling Leakage:** No synthetic oversampling (SMOTE) or sequence generation prior to splitting.

## 19. Reproducibility Instructions
To regenerate the complete dataset from raw source files:
```bash
python scripts/audit_sources.py
python scripts/map_open_targets_to_uniprot.py
python scripts/clean_sequences.py
python scripts/build_labels.py
python scripts/build_control_set.py
python scripts/remove_duplicates.py
python scripts/cluster_sequences.py
python scripts/create_splits.py
python scripts/calculate_protein_properties.py
python scripts/validate_dataset.py
```

## 20. Exact Scripts Used
- [`scripts/audit_sources.py`](../scripts/audit_sources.py): Comprehensive raw data audit.
- [`scripts/clean_sequences.py`](../scripts/clean_sequences.py): Sequence cleaning and quality control.
- [`scripts/map_open_targets_to_uniprot.py`](../scripts/map_open_targets_to_uniprot.py): Deterministic Swiss-Prot canonical mapping.
- [`scripts/build_labels.py`](../scripts/build_labels.py): Scientific threshold justification and positive set construction.
- [`scripts/build_control_set.py`](../scripts/build_control_set.py): Length-stratified negative control selection.
- [`scripts/remove_duplicates.py`](../scripts/remove_duplicates.py): Deduplication and conflicting label resolution.
- [`scripts/cluster_sequences.py`](../scripts/cluster_sequences.py): MMseqs2 homology clustering at 40% sequence identity.
- [`scripts/create_splits.py`](../scripts/create_splits.py): Cluster-aware train/val/test splitting.
- [`scripts/calculate_protein_properties.py`](../scripts/calculate_protein_properties.py): Molecular property extraction and final dataset assembly.
- [`scripts/validate_dataset.py`](../scripts/validate_dataset.py): 12-point dataset validation suite.
