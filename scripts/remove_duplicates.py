"""
scripts/remove_duplicates.py
Phase 7: Sequence Deduplication and Conflicting Label Resolution.

Methodology:
1. Load positive candidates (label=1) and control candidates (label=0).
2. Detect conflicting sequence labels (sequence appearing as both positive and control):
   - Conflicting sequences are quarantined into data/interim/conflicting_labels.csv.
   - They are NOT assigned arbitrary labels and are removed from the candidate pool.
3. Detect duplicate sequences within the same class:
   - Positive class: retain record with highest open_targets_globalScore / KW-0026 provenance.
   - Control class: retain primary canonical accession.
4. Output data/interim/conflicting_labels.csv and data/interim/deduplicated_candidates.csv.
"""

import os
import sys
import time
import numpy as np
import pandas as pd


def remove_duplicates_and_resolve_conflicts(
    positive_csv_path,
    control_csv_path,
    conflicts_csv_path,
    output_dedup_csv_path
):
    print("=" * 60)
    print("PHASE 7: DUPLICATE REMOVAL & CONFLICTING LABEL RESOLUTION")
    print("=" * 60)
    
    df_pos = pd.read_csv(positive_csv_path)
    df_ctrl = pd.read_csv(control_csv_path)
    
    df_pos["label"] = 1
    df_ctrl["label"] = 0
    
    print(f"Loaded {len(df_pos)} positive candidates and {len(df_ctrl)} control candidates.")
    
    # 1. Combine pools
    df_combined = pd.concat([df_pos, df_ctrl], ignore_index=True)
    
    # 2. Check for conflicting labels across classes
    # Group by sequence and check if labels set has >1 unique values
    seq_groups = df_combined.groupby("sequence")
    
    conflicting_seqs = []
    conflict_rows = []
    
    for seq, group in seq_groups:
        labels_present = group["label"].unique()
        if len(labels_present) > 1:
            conflicting_seqs.append(seq)
            conflict_rows.append(group)
            
    os.makedirs(os.path.dirname(conflicts_csv_path), exist_ok=True)
    if conflict_rows:
        df_conflicts = pd.concat(conflict_rows, ignore_index=True)
        df_conflicts.to_csv(conflicts_csv_path, index=False)
        print(f"CRITICAL WARNING: Found {len(conflicting_seqs)} sequences with CONFLICTING labels!")
        print(f"Exported {len(df_conflicts)} conflicting records to {conflicts_csv_path}")
    else:
        # Save empty schema
        cols = list(df_combined.columns) + ["conflict_reason"]
        df_empty = pd.DataFrame(columns=cols)
        df_empty.to_csv(conflicts_csv_path, index=False)
        print(f"Zero conflicting sequence labels detected. Saved clean schema to {conflicts_csv_path}")
        
    # Exclude any conflicting sequences from further processing
    if conflicting_seqs:
        df_filtered = df_combined[~df_combined["sequence"].isin(conflicting_seqs)].copy()
    else:
        df_filtered = df_combined.copy()
        
    # 3. Check for within-class duplicate sequences
    total_before = len(df_filtered)
    # Sort to prioritize KW-0026, highest score, longest length
    df_filtered["is_kw26_int"] = df_filtered["uniprot_alzheimer_annotation"].astype(int)
    df_filtered["score_filled"] = df_filtered["open_targets_globalScore"].fillna(-1.0)
    
    df_sorted = df_filtered.sort_values(
        by=["label", "is_kw26_int", "score_filled", "sequence_length"],
        ascending=[True, False, False, False]
    )
    
    # Drop duplicate sequences, keeping first
    df_dedup = df_sorted.drop_duplicates(subset=["sequence"], keep="first").copy()
    df_dedup = df_dedup.drop(columns=["is_kw26_int", "score_filled"]).reset_index(drop=True)
    
    exact_duplicates_removed = total_before - len(df_dedup)
    print(f"Exact within-class duplicates removed: {exact_duplicates_removed}")
    
    pos_count = (df_dedup["label"] == 1).sum()
    ctrl_count = (df_dedup["label"] == 0).sum()
    print(f"Deduplicated dataset: {len(df_dedup)} total ({pos_count} Positive, {ctrl_count} Control)")
    
    os.makedirs(os.path.dirname(output_dedup_csv_path), exist_ok=True)
    df_dedup.to_csv(output_dedup_csv_path, index=False)
    print(f"Saved deduplicated dataset to {output_dedup_csv_path}")
    print("Phase 7 completed successfully.\n")
    return df_dedup, exact_duplicates_removed


def main():
    pos_csv = "data/interim/alzheimer_positive_candidates.csv"
    ctrl_csv = "data/interim/control_candidates.csv"
    conflicts_csv = "data/interim/conflicting_labels.csv"
    dedup_csv = "data/interim/deduplicated_candidates.csv"
    
    remove_duplicates_and_resolve_conflicts(
        pos_csv, ctrl_csv, conflicts_csv, dedup_csv
    )


if __name__ == "__main__":
    main()
