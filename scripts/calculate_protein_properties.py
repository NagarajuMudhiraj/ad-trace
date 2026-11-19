"""
scripts/calculate_protein_properties.py
Phase 11 & Phase 12: Molecular Feature Extraction and Final Dataset Generation.

Methodology:
1. For every protein sequence in the final cohort (1,416 proteins across train, validation, and test):
   - Computes sequence length
   - Computes molecular weight (Daltons) via Bio.SeqUtils.ProtParam
   - Computes isoelectric point (pI)
   - Computes aromaticity fraction
   - Computes instability index (Guruprasad method)
   - Computes GRAVY (Grand Average of Hydropathicity / Kyte-Doolittle hydrophobicity)
   - Computes amino acid composition frequencies across all 20 standard amino acids
2. Exports detailed physicochemical features to:
   - data/processed/protein_properties.csv
3. Assembles and freezes the final primary dataset:
   - data/processed/Alzheimer_Protein_Sequence_Dataset_v1.0.csv
   With strictly required columns:
   sample_id, uniprot_accession, gene, protein_name, sequence,
   sequence_length, molecular_weight, open_targets_globalScore,
   evidence_sources, uniprot_alzheimer_annotation, cluster_id, label, split
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis


STANDARD_20_AA = sorted(list("ACDEFGHIKLMNPQRSTVWY"))


def calculate_molecular_properties(
    splits_dir="data/splits",
    output_properties_csv="data/processed/protein_properties.csv",
    output_final_dataset_csv="data/processed/Alzheimer_Protein_Sequence_Dataset_v1.0.csv"
):
    print("=" * 60)
    print("PHASE 11: MOLECULAR PROPERTY & PHYSICOCHEMICAL FEATURE EXTRACTION")
    print("=" * 60)
    
    # Load splits
    split_dfs = []
    for s_name in ["train", "validation", "test"]:
        s_path = os.path.join(splits_dir, f"{s_name}.csv")
        s_df = pd.read_csv(s_path)
        s_df["split"] = s_name
        split_dfs.append(s_df)
        
    df_all = pd.concat(split_dfs, ignore_index=True)
    print(f"Loaded {len(df_all)} proteins across splits.")
    
    # Sort deterministically by split, label desc, accession asc
    split_order = {"train": 0, "validation": 1, "test": 2}
    df_all["split_rank"] = df_all["split"].map(split_order)
    df_all = df_all.sort_values(
        by=["split_rank", "label", "accession"],
        ascending=[True, False, True]
    ).reset_index(drop=True)
    df_all = df_all.drop(columns=["split_rank"])
    
    # Generate stable sample_id
    df_all["sample_id"] = [f"ALZ_PROT_{i+1:04d}" for i in range(len(df_all))]
    
    print("Computing ProtParam physicochemical properties for all sequences...")
    prop_records = []
    
    for _, row in df_all.iterrows():
        sid = row["sample_id"]
        acc = row["accession"]
        gene = row["gene"]
        seq = row["sequence"]
        seq_len = len(seq)
        
        pa = ProteinAnalysis(seq)
        mw = pa.molecular_weight()
        pi = pa.isoelectric_point()
        aroma = pa.aromaticity()
        instab = pa.instability_index()
        gravy = pa.gravy()
        aa_percent = pa.amino_acids_percent
        
        rec = {
            "sample_id": sid,
            "accession": acc,
            "gene": gene,
            "sequence_length": seq_len,
            "molecular_weight": round(mw, 2),
            "isoelectric_point": round(pi, 4),
            "aromaticity": round(aroma, 6),
            "instability_index": round(instab, 4),
            "hydrophobicity_gravy": round(gravy, 4)
        }
        for aa in STANDARD_20_AA:
            rec[f"aa_comp_{aa}"] = round(aa_percent.get(aa, 0.0), 6)
            
        prop_records.append(rec)
        
    df_props = pd.DataFrame(prop_records)
    
    os.makedirs(os.path.dirname(output_properties_csv), exist_ok=True)
    df_props.to_csv(output_properties_csv, index=False)
    print(f"Saved {len(df_props)} protein property records to {output_properties_csv}")
    
    # ============================================================
    # PHASE 12: ASSEMBLE FINAL FROZEN DATASET
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 12: ASSEMBLING FINAL DATASET: Alzheimer_Protein_Sequence_Dataset_v1.0")
    print("=" * 60)
    
    # Merge molecular_weight into df_all
    mw_map = dict(zip(df_props["sample_id"], df_props["molecular_weight"]))
    df_all["molecular_weight"] = df_all["sample_id"].map(mw_map)
    
    # Format and enforce required schema
    required_columns = [
        "sample_id",
        "uniprot_accession",
        "gene",
        "protein_name",
        "sequence",
        "sequence_length",
        "molecular_weight",
        "open_targets_globalScore",
        "evidence_sources",
        "uniprot_alzheimer_annotation",
        "cluster_id",
        "label",
        "split"
    ]
    
    df_final = pd.DataFrame({
        "sample_id": df_all["sample_id"],
        "uniprot_accession": df_all["accession"],
        "gene": df_all["gene"],
        "protein_name": df_all["protein_name"],
        "sequence": df_all["sequence"],
        "sequence_length": df_all["sequence_length"].astype(int),
        "molecular_weight": df_all["molecular_weight"].astype(float),
        "open_targets_globalScore": df_all["open_targets_globalScore"],
        "evidence_sources": df_all["evidence_sources"],
        "uniprot_alzheimer_annotation": df_all["uniprot_alzheimer_annotation"].astype(bool),
        "cluster_id": df_all["cluster_id"],
        "label": df_all["label"].astype(int),
        "split": df_all["split"]
    })
    
    # Verify column alignment
    assert list(df_final.columns) == required_columns, "Schema mismatch in final dataset!"
    
    os.makedirs(os.path.dirname(output_final_dataset_csv), exist_ok=True)
    df_final.to_csv(output_final_dataset_csv, index=False)
    print(f"Saved frozen final dataset ({len(df_final)} records) to {output_final_dataset_csv}")
    
    print("\nFinal Dataset Overview:")
    print(f"  Total records:         {len(df_final):,}")
    print(f"  Positive samples:      {(df_final['label'] == 1).sum():,}")
    print(f"  Control samples:       {(df_final['label'] == 0).sum():,}")
    print(f"  Training split:        {(df_final['split'] == 'train').sum():,}")
    print(f"  Validation split:      {(df_final['split'] == 'validation').sum():,}")
    print(f"  Test split:            {(df_final['split'] == 'test').sum():,}")
    print(f"  Unique clusters:       {df_final['cluster_id'].nunique():,}")
    print(f"  Mean length:           {df_final['sequence_length'].mean():.2f} aa")
    print(f"  Mean molecular weight: {df_final['molecular_weight'].mean():.2f} Da")
    print("Phase 11 & Phase 12 completed successfully.\n")
    return df_final


def main():
    splits_dir = "data/splits"
    out_props = "data/processed/protein_properties.csv"
    out_final = "data/processed/Alzheimer_Protein_Sequence_Dataset_v1.0.csv"
    calculate_molecular_properties(splits_dir, out_props, out_final)


if __name__ == "__main__":
    main()
