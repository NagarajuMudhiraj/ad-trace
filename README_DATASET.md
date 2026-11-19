# Alzheimer's Disease Protein Sequence Classification Project
## Dataset v1.0 Reproduction Guide

This repository contains the complete, reproducible **Data Engineering + Quality Control + Homology Clustering + Validation Pipeline** for `Alzheimer_Protein_Sequence_Dataset_v1.0`.

---

## Directory Architecture

```
ad-trace/
├── data/
│   ├── raw/                           # Original staging files (unmodified)
│   │   ├── OT-MONDO_0004975-associated-targets-10_7_2026-v26_09.tsv
│   │   ├── uniprot_sprot.fasta
│   │   └── uniprotkb_keyword_KW_0026_AND_reviewed_2026_10_07.fasta
│   ├── interim/                       # Intermediate QC, mappings, and clusters
│   │   ├── open_targets_uniprot_mapping.csv
│   │   ├── sequence_qc.csv
│   │   ├── alzheimer_positive_candidates.csv
│   │   ├── control_candidates.csv
│   │   ├── conflicting_labels.csv
│   │   ├── deduplicated_candidates.csv
│   │   └── sequence_clusters.tsv
│   ├── splits/                        # Cluster-aware data partitions (Zero Leakage)
│   │   ├── train.csv                  # 990 samples (495 pos, 495 ctrl)
│   │   ├── validation.csv             # 213 samples (106 pos, 107 ctrl)
│   │   └── test.csv                   # 213 samples (107 pos, 106 ctrl)
│   ├── processed/                     # Frozen final benchmark datasets
│   │   ├── Alzheimer_Protein_Sequence_Dataset_v1.0.csv
│   │   └── protein_properties.csv
│   ├── reports/                       # Formal audit, QC, validation, and dataset card reports
│   │   ├── uniprot_alzheimer_audit.md
│   │   ├── open_targets_audit.md
│   │   ├── swissprot_audit.md
│   │   ├── sequence_qc_report.md
│   │   ├── label_definition.md
│   │   ├── class_distribution.csv
│   │   ├── class_distribution.md
│   │   ├── final_dataset_validation.md
│   │   └── DATASET_CARD.md
│   └── logs/
├── scripts/
│   ├── audit_sources.py               # Phase 1: Source file audit
│   ├── map_open_targets_to_uniprot.py # Phase 3: Deterministic canonical Swiss-Prot mapping
│   ├── clean_sequences.py             # Phase 2: Sequence QC & character validation
│   ├── build_labels.py                # Phase 4 & 5: Threshold analysis & positive set construction
│   ├── build_control_set.py           # Phase 6: Length-stratified negative control sampling
│   ├── remove_duplicates.py           # Phase 7: Deduplication & conflict detection
│   ├── cluster_sequences.py           # Phase 8: MMseqs2 homology clustering (40% identity)
│   ├── create_splits.py               # Phase 9 & 10: Cluster-aware splitting & balance report
│   ├── calculate_protein_properties.py# Phase 11 & 12: ProtParam properties & final dataset
│   └── validate_dataset.py            # Phase 13: 12-point integrity and leakage validation
├── tools/
│   └── mmseqs2/                       # MMseqs2 native clustering suite
├── requirements.txt
└── README_DATASET.md
```

---

## Prerequisites & Installation

Python 3.11+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

Clustering uses MMseqs2 (pre-configured in `tools/mmseqs2/mmseqs/bin`).

---

## End-to-End Pipeline Execution

To reproduce `Alzheimer_Protein_Sequence_Dataset_v1.0` deterministically, run the scripts in sequence:

```bash
# 1. Audit raw data sources
python scripts/audit_sources.py

# 2. Map Open Targets targets to canonical Swiss-Prot reviewed proteins
python scripts/map_open_targets_to_uniprot.py

# 3. Perform protein sequence cleaning and quality control
python scripts/clean_sequences.py

# 4. Evaluate score thresholds and build the Alzheimer-associated positive set
python scripts/build_labels.py

# 5. Sample length-stratified non-AD control proteins (1:1 ratio)
python scripts/build_control_set.py

# 6. Deduplicate sequences and verify zero conflicting labels
python scripts/remove_duplicates.py

# 7. Cluster sequences by homology using MMseqs2 (40% identity, 80% coverage)
python scripts/cluster_sequences.py

# 8. Create cluster-aware train / validation / test splits (no leakage)
python scripts/create_splits.py

# 9. Calculate physicochemical molecular properties and assemble final dataset
python scripts/calculate_protein_properties.py

# 10. Execute the comprehensive 12-point dataset validation suite
python scripts/validate_dataset.py
```

---

## Dataset Schema

The frozen dataset `data/processed/Alzheimer_Protein_Sequence_Dataset_v1.0.csv` provides the following columns:

| Column | Type | Description |
|---|---|---|
| `sample_id` | String | Unique sample identifier (`ALZ_PROT_0001` - `ALZ_PROT_1416`) |
| `uniprot_accession` | String | Canonical UniProtKB/Swiss-Prot accession |
| `gene` | String | Official HGNC gene symbol |
| `protein_name` | String | Curated protein descriptive name |
| `sequence` | String | Validated primary amino acid sequence (standard 20 AA) |
| `sequence_length` | Integer | Sequence length in amino acids |
| `molecular_weight` | Float | Molecular weight in Daltons |
| `open_targets_globalScore`| Float | Target association score for MONDO_0004975 (Provenance metadata only) |
| `evidence_sources` | String | Active evidence streams from genetics, clinical trials, and literature |
| `uniprot_alzheimer_annotation` | Boolean | True if in UniProt KW-0026 gold-standard reference set |
| `cluster_id` | String | MMseqs2 homology cluster identifier |
| `label` | Integer | Binary classification label (`1` = Alzheimer-associated, `0` = Control) |
| `split` | String | Partition assignment (`train`, `validation`, `test`) |

> **IMPORTANT:** In downstream model training, `open_targets_globalScore` and `uniprot_alzheimer_annotation` MUST NOT be used as model input features. They are strictly provenance metadata.
