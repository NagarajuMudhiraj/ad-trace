# Protein Sequence Quality Control Report

**Date:** 2026-10-07 21:24:31
**Pipeline Phase:** Phase 2 (Protein Sequence Quality Control)
**Output Artifact:** `data/interim/sequence_qc.csv`

## Executive Summary

- **Total Entities Evaluated:** 12,689
- **Sequences Passing All QC Checks:** 12,209 (96.22%)
- **Unmapped / Missing Sequences:** 461
- **Empty Sequences:** 0
- **Sequences with Non-Standard Residues:** 19
- **Duplicate Sequence Instances Among Valid:** 4 identical sequence groups (9 accessions)

## QC Decision Log & Removal Rationale

In strict adherence to project guidelines, **zero sequences were silently deleted**. All decisions and removal reasons are systematically logged:

| Removal Reason / Status | Record Count | Description & Handling Decision |
|---|---|---|
| `NONE_VALID` | 12,209 | Passed all checks: non-empty, standard 20 amino acids, valid length, canonical Swiss-Prot. |
| `unmapped_gene_no_swiss_prot_protein` | 461 | Gene symbol in Open Targets does not correspond to a reviewed Swiss-Prot human protein (e.g., non-coding RNA, pseudogene). |
| `selenocysteine_U` | 19 | Contains Selenocysteine (U, 21st amino acid). Biologically genuine, but flagged to prevent invalid computation in downstream standard matrix and physicochemical feature tools. |

## Non-Standard Amino Acid Breakdown

| Character | Description | Count Detected in Candidates | Scientific Treatment |
|---|---|---|---|
| `U` | Selenocysteine (Sec) | 19 | Documented as genuine 21st amino acid; excluded from standard 20-AA downstream sequence models. |

## Duplicate Sequence Analysis

Across the valid candidate pool, 4 distinct sequences are shared by more than one gene/accession (comprising 9 accessions).
These include conserved paralogs, identical histone isoforms, and ubiquitin precursor proteins.
As mandated by Phase 7 and Phase 8, exact duplicates and homologous sequences are strictly deduplicated and clustered before data splitting to prevent cross-split leakage.

## Conclusion

**STATUS: PASSED.**
Full data provenance has been maintained. Every single candidate record has an explicit QC validation flag and documented rationale in `data/interim/sequence_qc.csv`.