"""
scripts/build_control_set.py
Phase 6: Construct Length-Matched Control / Negative Set.

Scientific methodology:
1. Candidate pool: All reviewed Homo sapiens proteins from Swiss-Prot.
2. Exclusion criteria:
   - Must NOT be in the Alzheimer positive candidate set (accession, gene, or sequence).
   - Must NOT be annotated with UniProt KW-0026 (Alzheimer disease).
   - Must NOT have borderline Open Targets Alzheimer association (globalScore >= 0.1).
   - Must have strictly valid protein sequence (20 standard amino acids, non-empty).
3. Stratified length matching:
   - Divide sequences into 10 length bins matching the positive set distribution.
   - For each bin, sample exactly N_pos controls (1:1 ratio) with fixed seed (seed=42).
4. Output: data/interim/control_candidates.csv
"""

import os
import re
import sys
import time
import numpy as np
import pandas as pd


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def build_control_candidates(
    positive_csv_path,
    sp_fasta_path,
    kw0026_fasta_path,
    mapping_csv_path,
    output_control_csv_path,
    random_seed=42
):
    print("=" * 60)
    print("PHASE 6: CONSTRUCTING CONTROL / NEGATIVE SET")
    print("=" * 60)
    
    # 1. Load positive set
    df_pos = pd.read_csv(positive_csv_path)
    pos_accs = set(df_pos["accession"].dropna())
    pos_genes = set(df_pos["gene"].dropna())
    pos_seqs = set(df_pos["sequence"].dropna())
    total_pos = len(df_pos)
    print(f"Loaded {total_pos} positive proteins.")
    
    # 2. Load KW-0026 accessions
    kw26_accs = set()
    with open(kw0026_fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                m = re.match(r">sp\|([A-Z0-9]+)\|", line)
                if m:
                    kw26_accs.add(m.group(1))
                    
    # 3. Load Open Targets borderline genes (globalScore >= 0.1)
    df_map = pd.read_csv(mapping_csv_path)
    ot_borderline_genes = set(
        df_map[df_map["open_targets_score"] >= 0.1]["gene_symbol"].dropna()
    )
    print(f"Excluded {len(ot_borderline_genes)} borderline Open Targets genes (score >= 0.1) from control pool.")
    
    # 4. Stream Swiss-Prot human proteins to find eligible controls
    eligible_controls = []
    with open(sp_fasta_path, "r", encoding="utf-8") as f:
        current_acc = None
        current_gn = None
        current_pname = None
        current_header = None
        current_seq_chunks = []
        
        def evaluate_entry(acc, gn, pname, header, seq_str):
            if "OS=Homo sapiens" not in header:
                return
            if not acc or not seq_str:
                return
            # Exclusion 1: positive set
            if acc in pos_accs or gn in pos_genes or seq_str in pos_seqs:
                return
            # Exclusion 2: KW-0026
            if acc in kw26_accs:
                return
            # Exclusion 3: Open Targets borderline
            if gn in ot_borderline_genes:
                return
            # Exclusion 4: Sequence validity (20 standard AA only)
            if not set(seq_str).issubset(STANDARD_AA):
                return
                
            eligible_controls.append({
                "accession": acc,
                "gene": gn if gn else f"LOC_{acc}",
                "protein_name": pname,
                "sequence": seq_str,
                "sequence_length": len(seq_str)
            })

        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header:
                    evaluate_entry(
                        current_acc, current_gn, current_pname,
                        current_header, "".join(current_seq_chunks).upper()
                    )
                m_acc = re.match(r">sp\|([A-Z0-9]+)\|", line)
                m_gn = re.search(r"GN=(\S+)", line)
                m_pn = re.search(r"^sp\|[A-Z0-9]+\|\S+\s+(.*?)\s+OS=", line)
                current_acc = m_acc.group(1) if m_acc else ""
                current_gn = m_gn.group(1) if m_gn else ""
                current_pname = m_pn.group(1).strip() if m_pn else ""
                current_header = line
                current_seq_chunks = []
            else:
                current_seq_chunks.append(line)
                
        if current_header:
            evaluate_entry(
                current_acc, current_gn, current_pname,
                current_header, "".join(current_seq_chunks).upper()
            )
            
    df_eligible = pd.DataFrame(eligible_controls)
    # Deduplicate eligible controls by sequence (keep first)
    df_eligible = df_eligible.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    print(f"Eligible unique human control candidates: {len(df_eligible):,}")
    
    # 5. Define sequence length bins and perform stratified sampling
    bins = [0, 150, 250, 350, 450, 600, 800, 1100, 1600, 2500, 100000]
    bin_labels = [f"bin_{bins[i]}_{bins[i+1]}" for i in range(len(bins)-1)]
    
    df_pos["len_bin"] = pd.cut(df_pos["sequence_length"], bins=bins, labels=bin_labels)
    df_eligible["len_bin"] = pd.cut(df_eligible["sequence_length"], bins=bins, labels=bin_labels)
    
    pos_bin_counts = df_pos["len_bin"].value_counts().sort_index()
    print("\nLength Bin Stratification Plan:")
    
    selected_controls = []
    rng = np.random.RandomState(random_seed)
    
    for b_label, n_needed in pos_bin_counts.items():
        pool_in_bin = df_eligible[df_eligible["len_bin"] == b_label]
        n_avail = len(pool_in_bin)
        print(f"  {b_label:<15}: Needed={n_needed:<4} | Available={n_avail:<5}")
        
        if n_avail < n_needed:
            raise ValueError(f"Insufficient controls in {b_label}: needed {n_needed}, got {n_avail}")
            
        sampled = pool_in_bin.sample(n=n_needed, random_state=rng, replace=False)
        selected_controls.append(sampled)
        
    df_ctrl_sampled = pd.concat(selected_controls, ignore_index=True)
    
    # Format columns according to specification
    df_ctrl_final = pd.DataFrame({
        "accession": df_ctrl_sampled["accession"],
        "gene": df_ctrl_sampled["gene"],
        "protein_name": df_ctrl_sampled["protein_name"],
        "sequence": df_ctrl_sampled["sequence"],
        "sequence_length": df_ctrl_sampled["sequence_length"],
        "open_targets_globalScore": np.nan,
        "evidence_sources": "Swiss-Prot Reviewed Homo sapiens Non-AD Control",
        "uniprot_alzheimer_annotation": False,
        "label": 0
    })
    
    os.makedirs(os.path.dirname(output_control_csv_path), exist_ok=True)
    df_ctrl_final.to_csv(output_control_csv_path, index=False)
    print(f"\nSaved {len(df_ctrl_final)} control candidates to {output_control_csv_path}")
    print(f"Control class ratio: {len(df_pos)} Positive : {len(df_ctrl_final)} Control (1:1 exact length match)")
    print("Phase 6 completed successfully.\n")
    return df_ctrl_final


def main():
    pos_csv = "data/interim/alzheimer_positive_candidates.csv"
    sp_fasta = "data/raw/uniprot_sprot.fasta"
    kw26_fasta = "data/raw/uniprotkb_keyword_KW_0026_AND_reviewed_2026_10_07.fasta"
    mapping_csv = "data/interim/open_targets_uniprot_mapping.csv"
    out_ctrl = "data/interim/control_candidates.csv"
    
    build_control_candidates(pos_csv, sp_fasta, kw26_fasta, mapping_csv, out_ctrl)


if __name__ == "__main__":
    main()
