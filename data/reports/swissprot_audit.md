# UniProt Swiss-Prot Database Audit Report

**Date:** 2026-10-07 21:20:34
**Source File:** `data/raw/uniprot_sprot.fasta`
**Release:** UniProtKB/Swiss-Prot (Reviewed Canonical)

## Executive Summary

- **Total Sequences:** 575,748
- **Unique Protein Accessions:** 575,748
- **Human (`Homo sapiens`) Proteins:** 20,431 (3.55%)
- **Human Proteins with Documented Gene Symbol (`GN=`):** 20,306
- **Sequences with Non-Standard Characters:** 2,685 (0.466%)
- **Duplicate Sequence Instances:** 88,272 (identical sequences across distinct accessions / orthologs)
- **Processing Method:** Chunked streaming parse without in-memory full database duplication (Runtime: 5.96s)

## Sequence Length Statistics (Full Swiss-Prot)

| Metric | Value |
|---|---|
| Minimum Length | 2 aa |
| Maximum Length | 35,213 aa |
| Mean Length | 363.04 aa |
| Median Length | 296.00 aa |
| Standard Deviation | 345.04 aa |

## Top 10 Model Organisms in Swiss-Prot

| Rank | Organism | Protein Count | Percentage |
|---|---|---|---|
| 1 | *Homo sapiens* | 20,431 | 3.55% |
| 2 | *Mus musculus* | 17,283 | 3.00% |
| 3 | *Arabidopsis thaliana* | 16,421 | 2.85% |
| 4 | *Rattus norvegicus* | 8,233 | 1.43% |
| 5 | *Saccharomyces cerevisiae (strain ATCC 204508 / S288c)* | 6,733 | 1.17% |
| 6 | *Bos taurus* | 6,054 | 1.05% |
| 7 | *Schizosaccharomyces pombe (strain 972 / ATCC 24843)* | 5,129 | 0.89% |
| 8 | *Escherichia coli (strain K12)* | 4,531 | 0.79% |
| 9 | *Caenorhabditis elegans* | 4,489 | 0.78% |
| 10 | *Oryza sativa subsp. japonica* | 4,197 | 0.73% |

## Non-Standard Amino Acid Analysis

The following non-canonical or ambiguous residues were detected across Swiss-Prot:

| Residue Code | Meaning | Occurrences across Database | Classification |
|---|---|---|---|
| `X` | Unknown / Unspecified | 2,287 | Ambiguous sequencing residue |
| `U` | Selenocysteine | 256 | Unusual but 21st genetically encoded amino acid |
| `B` | Asparagine or Aspartate (Asx) | 113 | Ambiguous mass-spectrometry/sequencing residue |
| `Z` | Glutamine or Glutamate (Glx) | 87 | Ambiguous mass-spectrometry/sequencing residue |
| `O` | Pyrrolysine | 29 | Unusual but 22nd genetically encoded amino acid |

## Audit Conclusion

**STATUS: PASSED.**
The Swiss-Prot database contains 20,431 reviewed human proteins, providing an extensive pool from which to draw length-matched, biologically valid control proteins that exclude Alzheimer-associated targets.