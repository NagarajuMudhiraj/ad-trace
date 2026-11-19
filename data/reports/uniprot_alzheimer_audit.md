# UniProt Alzheimer Disease Dataset Audit Report

**Date:** 2026-10-07 21:20:27
**Source File:** `data/raw/uniprotkb_keyword_KW_0026_AND_reviewed_2026_10_07.fasta`
**Target Annotation:** UniProt Keyword `KW-0026` (Alzheimer disease), Reviewed (Swiss-Prot)

## Executive Summary

- **Total Records:** 19
- **Unique Accessions:** 19
- **Unique Gene Symbols:** 19
- **Unique Protein Sequences:** 19
- **Reviewed (Swiss-Prot) Status:** 19/19 (100% Swiss-Prot reviewed)
- **Organism Distribution:** All 19 records are `Homo sapiens` (TaxID 9606)
- **Sequence Availability:** 19/19 (100% full sequences present)
- **Duplicate Records (Accession):** 0
- **Duplicate Sequences:** 0
- **Sequences with Invalid Characters:** 0 (all contain exclusively the 20 standard amino acids)

## Sequence Length Statistics

| Metric | Value |
|---|---|
| Minimum Length | 140 aa |
| Maximum Length | 2214 aa |
| Mean Length | 677.89 aa |
| Median Length | 483.00 aa |
| Standard Deviation | 551.19 aa |

## Protein Inventory (19 Gold-Standard Reviewed Reference Proteins)

| Accession | Gene | Entry Name | Protein Name | Length (aa) | Valid AA |
|---|---|---|---|---|---|
| `O14672` | **ADAM10** | `ADA10_HUMAN` | Disintegrin and metalloproteinase domain-containing protein 10 | 748 | Yes |
| `O95185` | **UNC5C** | `UNC5C_HUMAN` | Netrin receptor UNC5C | 931 | Yes |
| `P02649` | **APOE** | `APOE_HUMAN` | Apolipoprotein E | 317 | Yes |
| `P03886` | **MT-ND1** | `NU1M_HUMAN` | NADH-ubiquinone oxidoreductase chain 1 | 318 | Yes |
| `P03891` | **MT-ND2** | `NU2M_HUMAN` | NADH-ubiquinone oxidoreductase chain 2 | 347 | Yes |
| `P05067` | **APP** | `A4_HUMAN` | Amyloid-beta precursor protein | 770 | Yes |
| `P07339` | **CTSD** | `CATD_HUMAN` | Cathepsin D | 412 | Yes |
| `P10636` | **MAPT** | `TAU_HUMAN` | Microtubule-associated protein tau | 758 | Yes |
| `P37840` | **SNCA** | `SYUA_HUMAN` | Alpha-synuclein | 140 | Yes |
| `P49768` | **PSEN1** | `PSN1_HUMAN` | Presenilin-1 | 467 | Yes |
| `P49810` | **PSEN2** | `PSN2_HUMAN` | Presenilin-2 | 448 | Yes |
| `P49840` | **GSK3A** | `GSK3A_HUMAN` | Glycogen synthase kinase-3 alpha | 483 | Yes |
| `P49841` | **GSK3B** | `GSK3B_HUMAN` | Glycogen synthase kinase-3 beta | 420 | Yes |
| `P53041` | **PPP5C** | `PPP5_HUMAN` | Serine/threonine-protein phosphatase 5 | 499 | Yes |
| `Q16643` | **DBN1** | `DREB_HUMAN` | Drebrin | 649 | Yes |
| `Q8IZY2` | **ABCA7** | `ABCA7_HUMAN` | Phospholipid-transporting ATPase ABCA7 | 2146 | Yes |
| `Q92673` | **SORL1** | `SORL_HUMAN` | Sortilin-related receptor | 2214 | Yes |
| `Q9NZC2` | **TREM2** | `TREM2_HUMAN` | Triggering receptor expressed on myeloid cells 2 | 230 | Yes |
| `Q86X27` | **RALGPS2** | `RGPS2_HUMAN` | Ras-specific guanine nucleotide-releasing factor RalGPS2 | 583 | Yes |

## Biological Validation & KW-0026 Keyword Confirmation

All 19 records were verified against the UniProtKB reference database for keyword `KW-0026` (*Alzheimer disease*).
This canonical cohort comprises primary causal Alzheimer proteins (e.g., APP, PSEN1, PSEN2), major genetic risk factors (e.g., APOE, TREM2, ABCA7, SORL1), tau pathology elements (MAPT), alpha-synuclein (SNCA), and key enzymatic/pathway regulators (ADAM10, GSK3A, GSK3B, CTSD, PPP5C, DBN1, UNC5C, RALGPS2, MT-ND1, MT-ND2).

## Audit Conclusion

**STATUS: PASSED.**
The UniProt Alzheimer dataset provides a gold-standard, high-confidence benchmark cohort with zero missing sequences, zero corrupted amino acids, and 100% reviewed human annotations.