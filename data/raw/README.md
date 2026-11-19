# Raw Data Staging Directory

This directory stores the primary unmodified inputs used to construct **AD-TRACE** (`Alzheimer_Protein_Sequence_Dataset_v1.0`).

## Included Datasets

1. **`OT-MONDO_0004975-associated-targets-10_7_2026-v26_09.tsv`**
   - Source: Open Targets Platform (Release 26.09)
   - Scope: Target associations for Alzheimer's disease (`MONDO_0004975`)
   - Included directly in the repository (~4.8 MB).

2. **`uniprotkb_keyword_KW_0026_AND_reviewed_2026_10_07.fasta`**
   - Source: UniProtKB / Swiss-Prot
   - Scope: Gold-standard reviewed human proteins annotated with Keyword `KW-0026` (*Alzheimer disease*)
   - Included directly in the repository (~15 KB).

## Large Database Files

3. **`uniprot_sprot.fasta`** (Canonical Swiss-Prot reviewed proteins, ~288 MB uncompressed)
   - *Note:* Exceeds GitHub's 100 MB single-file limit and is omitted from git tracking. All pre-processed splits, final datasets, and model representations are already fully computed and included under `data/splits/`, `data/processed/`, and `data/modeling/`.
   - If you wish to re-run raw preprocessing from scratch (`scripts/build_control_set.py` and `scripts/audit_sources.py`), download the UniProt release:
   ```bash
   # Download into data/raw/
   curl -o data/raw/uniprot_sprot.fasta.gz https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
   gzip -d data/raw/uniprot_sprot.fasta.gz
   ```
