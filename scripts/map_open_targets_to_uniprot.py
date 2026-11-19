"""
scripts/map_open_targets_to_uniprot.py
Phase 3: Map Open Targets Alzheimer-associated genes to UniProt Swiss-Prot reviewed accessions.

Methodology:
1. Extract all canonical reviewed human proteins (OS=Homo sapiens) from Swiss-Prot with gene symbols (GN=...).
2. Cross-reference with the 19 gold-standard UniProt KW-0026 Alzheimer proteins.
3. Group and aggregate duplicate Open Targets records per gene symbol:
   - Retain maximum globalScore
   - Aggregate active evidence streams
4. Map gene symbols to Swiss-Prot:
   - For multi-isoform/multi-accession mappings, apply deterministic canonical selection:
     1. Match to KW-0026 reference accession if present
     2. Lowest Protein Existence (PE) level (experimental evidence)
     3. Longest canonical sequence length
     4. Lexicographical tie-breaker on accession
5. Output data/interim/open_targets_uniprot_mapping.csv
"""

import os
import re
import sys
import time
import numpy as np
import pandas as pd


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_gold_standard_kw0026(kw0026_fasta_path):
    kw_proteins = {}
    with open(kw0026_fasta_path, "r", encoding="utf-8") as f:
        current_acc = None
        current_gn = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                m_acc = re.match(r">sp\|([A-Z0-9]+)\|", line)
                m_gn = re.search(r"GN=(\S+)", line)
                if m_acc:
                    acc = m_acc.group(1)
                    gn = m_gn.group(1) if m_gn else None
                    kw_proteins[acc] = gn
    return kw_proteins


def load_swissprot_human(sp_fasta_path, kw26_accs):
    print("Loading reviewed human proteins from Swiss-Prot...")
    sp_human_by_gene = {} # gene -> list of dicts
    total_human = 0
    
    with open(sp_fasta_path, "r", encoding="utf-8") as f:
        current_header = None
        current_seq_chunks = []
        
        def process_entry(header, seq_str):
            nonlocal total_human
            if "OS=Homo sapiens" not in header:
                return
            total_human += 1
            m_acc = re.match(r">sp\|([A-Z0-9]+)\|(\S+)\s+(.*?)\s+OS=Homo sapiens", header)
            m_gn = re.search(r"GN=(\S+)", header)
            m_pe = re.search(r"PE=(\d+)", header)
            
            if not m_acc or not m_gn:
                return
                
            acc = m_acc.group(1)
            entry_name = m_acc.group(2)
            pname = m_acc.group(3)
            gene = m_gn.group(1)
            pe = int(m_pe.group(1)) if m_pe else 99
            
            sp_human_by_gene.setdefault(gene, []).append({
                "accession": acc,
                "entry_name": entry_name,
                "protein_name": pname,
                "gene": gene,
                "pe": pe,
                "sequence": seq_str,
                "sequence_length": len(seq_str),
                "is_kw0026": acc in kw26_accs
            })

        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header:
                    process_entry(current_header, "".join(current_seq_chunks).upper())
                current_header = line
                current_seq_chunks = []
            else:
                current_seq_chunks.append(line)
                
        if current_header:
            process_entry(current_header, "".join(current_seq_chunks).upper())
            
    print(f"Loaded {total_human} human Swiss-Prot entries across {len(sp_human_by_gene)} unique genes.")
    
    # Resolve canonical protein for each gene
    canonical_by_gene = {}
    for gene, candidates in sp_human_by_gene.items():
        if len(candidates) == 1:
            canonical_by_gene[gene] = candidates[0]
        else:
            # Multi-accession canonical selection rule:
            # 1. KW-0026 membership (True before False)
            # 2. Lowest PE (1 before 2, 3, etc.)
            # 3. Longest sequence length (descending)
            # 4. Accession string (ascending)
            sorted_cands = sorted(
                candidates,
                key=lambda x: (not x["is_kw0026"], x["pe"], -x["sequence_length"], x["accession"])
            )
            canonical_by_gene[gene] = sorted_cands[0]
            
    return canonical_by_gene


def map_open_targets(ot_tsv_path, canonical_by_gene, output_csv_path):
    print("Reading and parsing Open Targets dataset...")
    ot_df = pd.read_csv(ot_tsv_path, sep="\t", low_memory=False)
    
    evidence_columns = [
        "clinicalPrecedence", "gwasCredibleSets", "geneBurden", "eva",
        "genomicsEngland", "gene2Phenotype", "uniprotLiterature", "uniprotVariants",
        "orphanet", "clingen", "cancerGeneCensus", "intogen", "evaSomatic",
        "cancerBiomarkers", "crisprScreen", "crispr", "reactome", "europepmc",
        "expressionAtlas", "impc"
    ]
    
    def extract_evidence_summary(row):
        active = []
        for col in evidence_columns:
            if col in row:
                val = row[col]
                if pd.notna(val):
                    s = str(val).strip()
                    if s not in ["No data", "0", "0.0", "", "nan", "None"]:
                        try:
                            num = float(s)
                            if num > 0:
                                active.append(f"{col}:{num:.3f}")
                        except ValueError:
                            active.append(f"{col}:{s}")
        return "; ".join(active) if active else "None"
        
    ot_df["globalScore_num"] = pd.to_numeric(ot_df["globalScore"], errors="coerce")
    ot_df["evidence_summary"] = ot_df.apply(extract_evidence_summary, axis=1)
    
    # Group by symbol to deduplicate
    print("Deduplicating Open Targets by gene symbol...")
    grouped = []
    for symbol, group in ot_df.groupby("symbol"):
        max_idx = group["globalScore_num"].idxmax()
        best_row = group.loc[max_idx]
        max_score = best_row["globalScore_num"]
        
        # Combine evidence summaries across any duplicate rows for the same gene
        all_ev = set()
        for ev_str in group["evidence_summary"]:
            if ev_str != "None":
                for item in ev_str.split("; "):
                    all_ev.add(item)
        combined_ev = "; ".join(sorted(all_ev)) if all_ev else "None"
        
        grouped.append({
            "gene_symbol": symbol,
            "open_targets_score": max_score,
            "evidence_information": combined_ev
        })
        
    unique_ot_df = pd.DataFrame(grouped)
    total_ot_genes = len(unique_ot_df)
    
    print(f"Mapping {total_ot_genes} unique Open Targets gene symbols to Swiss-Prot...")
    
    mapped_rows = []
    mapped_count = 0
    valid_seq_count = 0
    unique_seqs = set()
    
    for _, row in unique_ot_df.iterrows():
        gene = row["gene_symbol"]
        score = row["open_targets_score"]
        ev = row["evidence_information"]
        
        if gene in canonical_by_gene:
            sp_info = canonical_by_gene[gene]
            acc = sp_info["accession"]
            pname = sp_info["protein_name"]
            seq = sp_info["sequence"]
            seq_len = sp_info["sequence_length"]
            status = "mapped_sprot_canonical"
            mapped_count += 1
            
            is_valid_aa = len(seq) > 0 and set(seq).issubset(STANDARD_AA)
            if is_valid_aa:
                valid_seq_count += 1
                unique_seqs.add(seq)
        else:
            acc = ""
            pname = ""
            seq = ""
            seq_len = 0
            status = "unmapped_no_sprot_human"
            
        mapped_rows.append({
            "gene_symbol": gene,
            "open_targets_score": score,
            "evidence_information": ev,
            "uniprot_accession": acc,
            "protein_name": pname,
            "sequence_length": seq_len,
            "sequence": seq,
            "mapping_status": status
        })
        
    out_df = pd.DataFrame(mapped_rows)
    # Sort descending by score
    out_df = out_df.sort_values("open_targets_score", ascending=False).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    print(f"Saved mapping table to {output_csv_path}")
    
    # Print the attrition funnel
    print("\n" + "=" * 50)
    print("OPEN TARGETS TO UNIPROT MAPPING ATTRITION FUNNEL")
    print("=" * 50)
    print(f"Open Targets genes:             {total_ot_genes:,}")
    print("        |")
    print("        v")
    print(f"Successfully mapped genes:      {mapped_count:,} ({mapped_count/total_ot_genes*100:.2f}%)")
    print("        |")
    print("        v")
    print(f"Mapped UniProt proteins:        {mapped_count:,}")
    print("        |")
    print("        v")
    print(f"Proteins with valid sequences:  {valid_seq_count:,}")
    print("        |")
    print("        v")
    print(f"Unique protein sequences:       {len(unique_seqs):,}")
    print("=" * 50 + "\n")
    
    return out_df


def main():
    kw0026_fasta = "data/raw/uniprotkb_keyword_KW_0026_AND_reviewed_2026_10_07.fasta"
    sp_fasta = "data/raw/uniprot_sprot.fasta"
    ot_tsv = "data/raw/OT-MONDO_0004975-associated-targets-10_7_2026-v26_09.tsv"
    out_csv = "data/interim/open_targets_uniprot_mapping.csv"
    
    kw26_accs = load_gold_standard_kw0026(kw0026_fasta)
    canonical_by_gene = load_swissprot_human(sp_fasta, set(kw26_accs.keys()))
    map_open_targets(ot_tsv, canonical_by_gene, out_csv)


if __name__ == "__main__":
    main()
