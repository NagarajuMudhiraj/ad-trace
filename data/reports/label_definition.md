# Scientific Label Definition and Positive Threshold Selection Report

**Date:** 2026-10-07 21:26:15
**Pipeline Phase:** Phase 4 (Define Positive Data Scientifically)
**Selected Positive Threshold:** `Open Targets globalScore >= 0.3` + All 19 UniProt KW-0026 Gold-Standard Reference Proteins

## Executive Summary

In supervised sequence classification of complex disease proteins, the positive class cannot be defined through naive binary thresholding of large associative databases.
The Open Targets Alzheimer's dataset (`MONDO_0004975`) contains 13,006 targets, of which >75% have negligible association scores (< 0.062).
Labeling all 13,006 targets as `Alzheimer = 1` would corrupt the ground truth with severe false-positive noise.

To establish an authoritative, research-grade positive class, we systematically audited candidate thresholds against:
1. **Sample Size & Machine Learning Feasibility**
2. **Evidence Strength (Genetics, Clinical Trials, Literature)**
3. **Biological Validity & Known Alzheimer Pathophysiology**
4. **Alignment with the 19 Reviewed UniProt KW-0026 Reference Proteins**
5. **Downstream Class Balance with Length-Matched Controls**

## Threshold Comparison Analysis

| Threshold (`globalScore`) | Gene Count | Valid Proteins | Unique Sequences | KW-0026 Included | Genetic Evidence | Clinical Trials | Assessment & ML Feasibility |
|---|---|---|---|---|---|---|---|
| `>= 0.9` | 0 | 0 | 0 | 0/19 | 0 (0.0%) | 0 (0.0%) | Empty cohort (max score is 0.874 for APP). Not viable. |
| `>= 0.8` | 3 | 3 | 3 | 3/19 | 3 (100.0%) | 3 (100.0%) | Severe under-sampling (n=3: APP, PSEN1, PSEN2). Infeasible for ML. |
| `>= 0.7` | 7 | 7 | 7 | 6/19 | 6 (85.7%) | 4 (57.1%) | Critically small (n=7). Excludes APOE risk variants and TREM2. |
| `>= 0.6` | 25 | 25 | 25 | 7/19 | 14 (56.0%) | 15 (60.0%) | Extremely small (n=25). High confidence, but statistically underpowered for generalization. |
| `>= 0.5` | 93 | 93 | 93 | 9/19 | 47 (50.5%) | 28 (30.1%) | Small cohort (n=93). High precision, but excludes Tau (MAPT) and PSENEN from OT. |
| `>= 0.4` | 272 | 272 | 272 | 9/19 | 99 (36.4%) | 59 (21.7%) | Moderate cohort (n=272). Solid precision, but excludes key pathway members (MAPT, PSENEN, TSPO). |
| `>= 0.3` | 700 | 700 | 700 | 11/19 | 257 (36.7%) | 118 (16.9%) | **Optimal Scientific Sweet Spot (n=700).** High biological coverage, 258 genetic loci, captures MAPT & PSENEN, ideal ML power. |

## Scientific Rationale for Selecting Threshold >= 0.3

### 1. Retention of Core Alzheimer Biology
At thresholds `>= 0.4` or `>= 0.5`, pivotal Alzheimer disease proteins are excluded from Open Targets:
- **MAPT (Microtubule-associated protein tau):** Open Targets globalScore = `0.3499`. Under a 0.4 threshold, tau pathology targets would be omitted from OT evidence.
- **PSENEN (Presenilin enhancer 2):** Open Targets globalScore = `0.3644`. Essential component of the gamma-secretase complex required for amyloid-beta cleavage.
- **TSPO (Translocator protein):** Open Targets globalScore = `0.3512`. Premier in vivo neuroinflammation and microglial activation biomarker.
- **MT-ND1 (NADH-ubiquinone oxidoreductase):** Open Targets globalScore = `0.3398`. Canonical mitochondrial complex I component associated with neuronal metabolic decline.

Selecting `globalScore >= 0.3` retains these critical mechanistic players directly from the multi-evidence Open Targets database.

### 2. High Evidence Convergence
At `globalScore >= 0.3`:
- **36.7% (257 genes)** possess rigorous causal/risk genetic evidence (GWAS credible sets, ClinVar/EVA, Genomics England, Gene Burden).
- **16.8% (118 genes)** have active or past clinical trial precedence in ChEMBL.
- **82.4% (577 genes)** have corroborated biomedical literature evidence in Europe PMC.

### 3. Integration of the 19 Gold-Standard UniProt KW-0026 Proteins
The 19 reviewed human proteins in UniProt annotated with keyword `KW-0026` (*Alzheimer disease*) serve as an immutable reference anchor.
Eleven of these 19 exceed `globalScore >= 0.3` directly. The remaining 8 reviewed proteins (ADAM10, UNC5C, CTSD, SNCA, GSK3A, PPP5C, DBN1, RALGPS2) are preserved via explicit gold-standard provenance.

### 4. Machine Learning Feasibility & Statistical Power
The threshold yields 700 Open Targets proteins + 8 unique KW-0026 reference proteins = **708 total positive proteins**.
When paired 1:1 with length-matched negative controls, this creates a rigorous, balanced benchmark of **1,416 proteins**.
This sample size provides robust statistical support for homology-aware cluster splitting (approx. 990 train, 213 val, 213 test) without data sparsity in the validation and test splits.

## Conclusion

**STATUS: APPROVED.**
The positive cohort definition is frozen at: `Open Targets globalScore >= 0.3` + All 19 reviewed UniProt KW-0026 reference proteins.