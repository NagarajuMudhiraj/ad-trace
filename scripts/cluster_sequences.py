"""
scripts/cluster_sequences.py
Phase 8: Homology and Sequence Redundancy Control via MMseqs2.

Methodology:
1. Export all 1,416 candidate protein sequences (708 Positive, 708 Control) to FASTA.
2. Run MMseqs2 cascaded clustering:
   - Tool: MMseqs2 (Release: 8cc5ce367b5638c4306c2d7cfc652dd099a4643f, win64)
   - Sequence Identity Threshold: --min-seq-id 0.40 (40% identity)
   - Coverage Threshold: -c 0.80 (80% bidirectional coverage, --cov-mode 0)
3. Parse and format cluster assignments.
4. Clean temporary database files.
5. Save data/interim/sequence_clusters.tsv
"""

import os
import shutil
import subprocess
import sys
import time
from collections import Counter
import numpy as np
import pandas as pd


def cluster_protein_sequences(
    candidates_csv_path,
    output_clusters_tsv_path,
    mmseqs_bin_dir="tools/mmseqs2/mmseqs/bin",
    min_seq_id=0.4,
    coverage=0.8
):
    print("=" * 60)
    print("PHASE 8: HOMOLOGY & SEQUENCE REDUNDANCY CLUSTERING")
    print("=" * 60)
    
    # 1. Load candidates
    df = pd.read_csv(candidates_csv_path)
    print(f"Loaded {len(df)} candidate proteins for clustering.")
    
    interim_dir = os.path.dirname(output_clusters_tsv_path)
    os.makedirs(interim_dir, exist_ok=True)
    
    fasta_path = os.path.join(interim_dir, "candidate_sequences.fasta")
    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f">{row['accession']}\n{row['sequence']}\n")
    print(f"Exported FASTA to {fasta_path}")
    
    # 2. Locate MMseqs2 executable
    mmseqs_exe = os.path.join(mmseqs_bin_dir, "mmseqs.exe")
    if not os.path.exists(mmseqs_exe):
        # Check system PATH
        which_mmseqs = shutil.which("mmseqs")
        if which_mmseqs:
            mmseqs_exe = which_mmseqs
        else:
            raise FileNotFoundError(f"MMseqs2 binary not found at {mmseqs_exe} or on PATH.")
            
    # Add mmseqs bin dir to PATH for helper binaries (sh, busybox)
    env = os.environ.copy()
    env["PATH"] = os.path.abspath(mmseqs_bin_dir) + os.pathsep + env.get("PATH", "")
    
    # Query MMseqs version
    ver_proc = subprocess.run([mmseqs_exe, "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    mmseqs_version = ver_proc.stdout.strip()
    print(f"MMseqs2 Version: {mmseqs_version}")
    
    # Paths for MMseqs databases
    seq_db = os.path.join(interim_dir, "seqDB")
    clu_db = os.path.join(interim_dir, "cluDB")
    tmp_clu = os.path.join(interim_dir, "tmp_clu")
    raw_clu_tsv = os.path.join(interim_dir, "sequence_clusters_raw.tsv")
    
    # Clean previous run artifacts
    for fname in os.listdir(interim_dir):
        if fname.startswith("seqDB") or fname.startswith("cluDB") or fname.startswith("tmp_clu") or fname.startswith("sequence_clusters_raw"):
            fpath = os.path.join(interim_dir, fname)
            if os.path.isdir(fpath):
                shutil.rmtree(fpath, ignore_errors=True)
            else:
                try:
                    os.remove(fpath)
                except OSError:
                    pass
                    
    # Step A: createdb
    print("Running mmseqs createdb...")
    cmd_createdb = [mmseqs_exe, "createdb", fasta_path, seq_db, "-v", "1"]
    subprocess.run(cmd_createdb, check=True, env=env)
    
    # Step B: cluster
    print(f"Running mmseqs cluster (min-seq-id={min_seq_id}, coverage={coverage})...")
    cmd_cluster = [
        mmseqs_exe, "cluster", seq_db, clu_db, tmp_clu,
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "0",
        "-v", "1"
    ]
    subprocess.run(cmd_cluster, check=True, env=env)
    
    # Step C: createtsv
    print("Running mmseqs createtsv...")
    cmd_createtsv = [
        mmseqs_exe, "createtsv", seq_db, seq_db, clu_db, raw_clu_tsv,
        "-v", "1"
    ]
    subprocess.run(cmd_createtsv, check=True, env=env)
    
    # 3. Parse cluster TSV
    df_raw = pd.read_csv(raw_clu_tsv, sep="\t", header=None, names=["representative", "member"])
    
    # Calculate cluster sizes
    rep_counts = df_raw["representative"].value_counts().to_dict()
    
    # Assign deterministic cluster IDs (sorted by cluster size desc, representative asc)
    sorted_reps = sorted(rep_counts.keys(), key=lambda r: (-rep_counts[r], r))
    rep_to_cid = {r: f"CLU_{i+1:04d}" for i, r in enumerate(sorted_reps)}
    
    df_clusters = pd.DataFrame({
        "cluster_id": [rep_to_cid[r] for r in df_raw["representative"]],
        "representative_accession": df_raw["representative"],
        "member_accession": df_raw["member"],
        "cluster_size": [rep_counts[r] for r in df_raw["representative"]]
    })
    
    # Sort by cluster_id, member_accession
    df_clusters = df_clusters.sort_values(by=["cluster_id", "member_accession"]).reset_index(drop=True)
    df_clusters.to_csv(output_clusters_tsv_path, sep="\t", index=False)
    print(f"Saved cluster assignments to {output_clusters_tsv_path}")
    
    # Cluster statistics
    total_seqs = len(df_clusters)
    num_clusters = len(sorted_reps)
    cluster_sizes = list(rep_counts.values())
    
    print("\n" + "=" * 50)
    print("SEQUENCE CLUSTERING METRICS")
    print("=" * 50)
    print(f"Tool:                 MMseqs2")
    print(f"Version:              {mmseqs_version}")
    print(f"Parameters:           --min-seq-id {min_seq_id} -c {coverage} --cov-mode 0")
    print(f"Similarity Threshold: {min_seq_id * 100:.0f}% sequence identity")
    print(f"Coverage Threshold:   {coverage * 100:.0f}% bidirectional alignment coverage")
    print(f"Total Sequences:      {total_seqs:,}")
    print(f"Total Clusters:       {num_clusters:,}")
    print(f"Singleton Clusters:   {cluster_sizes.count(1):,} ({cluster_sizes.count(1)/num_clusters*100:.1f}%)")
    print(f"Multi-Member Clusters:{num_clusters - cluster_sizes.count(1):,}")
    print(f"Max Cluster Size:     {max(cluster_sizes)}")
    print(f"Mean Cluster Size:    {np.mean(cluster_sizes):.2f}")
    print(f"Median Cluster Size:  {np.median(cluster_sizes):.1f}")
    print("=" * 50 + "\n")
    
    # Cleanup heavy temp files
    for p in [seq_db, clu_db, tmp_clu, raw_clu_tsv]:
        if os.path.exists(p):
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass
    # Also clean ancillary db files created by mmseqs (seqDB*, cluDB*)
    for f_name in os.listdir(interim_dir):
        if f_name.startswith("seqDB") or f_name.startswith("cluDB"):
            try:
                os.remove(os.path.join(interim_dir, f_name))
            except OSError:
                pass
                
    return df_clusters


def main():
    cand_csv = "data/interim/deduplicated_candidates.csv"
    out_tsv = "data/interim/sequence_clusters.tsv"
    cluster_protein_sequences(cand_csv, out_tsv)


if __name__ == "__main__":
    main()
