"""
scripts/extract_prott5_embeddings.py
Generates ProtT5-XL-UniRef50 embeddings for Alzheimer protein sequence classification.

Pipeline:
1. Load train_model.csv (n=990) and validation_model.csv (n=213).
   TEST SET IS NEVER LOADED OR ACCESSED.
2. Tokenize with ProtT5 tokenizer (space-separated uppercase characters, rare [UZOB] -> X).
3. Encode with ProtT5-XL encoder (float32, CPU optimized with 12 threads).
4. Residue-level masked mean pooling -> 1024-dimensional representation.
5. Extract both Long Sequence Strategies:
   - Strategy A: Fixed Truncation (first 1000 residues)
   - Strategy B: Overlapping Window Aggregation (window=1000, stride=500, mean pooled across windows)
6. Checkpoint periodically so execution is 100% resumable.
7. Save embeddings to data/modeling/embeddings/prott5/.
"""

import os
import sys
import re
import time
import json
import torch
import numpy as np
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer

MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
OUT_DIR = "data/modeling/embeddings/prott5"
WINDOW_SIZE = 1000
STRIDE = 500
MAX_WINDOWS = 4

torch.set_num_threads(12)


def clean_sequence(seq):
    seq = seq.strip().upper()
    seq = re.sub(r"[UZOB]", "X", seq)
    return seq


def embed_window(model, tokenizer, window_seq):
    spaced = " ".join(list(window_seq))
    inputs = tokenizer(spaced, return_tensors="pt")
    out = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1)
    rep = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return rep.squeeze(0).cpu().numpy().astype(np.float32)


def process_dataset(df, split_name, model, tokenizer):
    n_total = len(df)
    fixed_file = os.path.join(OUT_DIR, f"{split_name}_embeddings_fixed.npy")
    window_file = os.path.join(OUT_DIR, f"{split_name}_embeddings_window.npy")
    meta_file = os.path.join(OUT_DIR, f"{split_name}_progress.json")
    
    # Check for existing partial progress
    start_idx = 0
    if os.path.exists(fixed_file) and os.path.exists(window_file) and os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            prog = json.load(f)
        start_idx = prog.get("completed_idx", 0)
        if start_idx >= n_total:
            print(f"[{split_name.upper()}] Already fully computed ({n_total} sequences). Loading...")
            fixed_embs = np.load(fixed_file)
            window_embs = np.load(window_file)
            return fixed_embs, window_embs
        print(f"[{split_name.upper()}] Resuming from index {start_idx}/{n_total}...")
        fixed_embs = np.load(fixed_file)
        window_embs = np.load(window_file)
    else:
        fixed_embs = np.zeros((n_total, 1024), dtype=np.float32)
        window_embs = np.zeros((n_total, 1024), dtype=np.float32)
        
    print(f"\n[{split_name.upper()}] Starting embedding extraction for {n_total - start_idx} sequences...")
    t_start = time.time()
    
    with torch.inference_mode():
        for i in range(start_idx, n_total):
            seq = clean_sequence(df.iloc[i]["sequence"])
            seq_len = len(seq)
            
            # Window 1 (first 1000 residues) -> Fixed Truncation representation
            win1_seq = seq[:WINDOW_SIZE]
            w1_emb = embed_window(model, tokenizer, win1_seq)
            fixed_embs[i] = w1_emb
            
            # Window Aggregation
            if seq_len <= WINDOW_SIZE:
                window_embs[i] = w1_emb
            else:
                window_reps = [w1_emb]
                start = STRIDE
                win_count = 1
                while start < seq_len and win_count < MAX_WINDOWS:
                    end = min(start + WINDOW_SIZE, seq_len)
                    sub_seq = seq[start:end]
                    sub_emb = embed_window(model, tokenizer, sub_seq)
                    window_reps.append(sub_emb)
                    start += STRIDE
                    win_count += 1
                window_embs[i] = np.mean(window_reps, axis=0)
                
            # Periodically save checkpoints every 20 sequences or on completion
            if (i + 1) % 20 == 0 or (i + 1) == n_total:
                elapsed = time.time() - t_start
                rate = (i + 1 - start_idx) / max(1e-5, elapsed)
                rem = (n_total - (i + 1)) / max(1e-5, rate)
                print(f"[{split_name.upper()}] {i + 1}/{n_total} | {rate:.2f} seq/s | Elapsed: {elapsed/60:.1f}m | ETA: {rem/60:.1f}m", flush=True)
                np.save(fixed_file, fixed_embs)
                np.save(window_file, window_embs)
                with open(meta_file, "w") as f:
                    json.dump({"completed_idx": i + 1, "total": n_total}, f)
                    
    print(f"[{split_name.upper()}] Completed in {(time.time() - t_start)/60:.1f} minutes.")
    return fixed_embs, window_embs


def main():
    print("=" * 60)
    print("PROTT5-XL EMBEDDING EXTRACTION PIPELINE")
    print("=" * 60)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Inspect sequence length limits
    train_df = pd.read_csv("data/modeling/train_model.csv")
    val_df = pd.read_csv("data/modeling/validation_model.csv")
    
    train_lens = train_df["sequence"].str.len()
    val_lens = val_df["sequence"].str.len()
    
    train_exceed = (train_lens > WINDOW_SIZE).sum()
    val_exceed = (val_lens > WINDOW_SIZE).sum()
    
    print(f"\nSequence Length Analysis against ProtT5 Limit ({WINDOW_SIZE} aa):")
    print(f"  Training:   {len(train_df)} samples, {train_exceed} exceed {WINDOW_SIZE} aa ({train_exceed/len(train_df)*100:.2f}%)")
    print(f"  Validation: {len(val_df)} samples, {val_exceed} exceed {WINDOW_SIZE} aa ({val_exceed/len(val_df)*100:.2f}%)")
    print(f"  Long sequence strategy to evaluate: Fixed Truncation vs Overlapping Window Aggregation (W={WINDOW_SIZE}, S={STRIDE})")
    
    # 2. Load Model & Tokenizer
    print(f"\nLoading ProtT5 Model ({MODEL_NAME})...")
    t0 = time.time()
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    print(f"Model and tokenizer ready in {time.time() - t0:.2f}s.")
    
    # 3. Process Validation first (smaller cohort, fast verification)
    val_fixed, val_window = process_dataset(val_df, "validation", model, tokenizer)
    
    # 4. Process Training
    train_fixed, train_window = process_dataset(train_df, "train", model, tokenizer)
    
    # 5. Build Metadata Table
    meta_rows = []
    for idx, sid in enumerate(train_df["sample_id"]):
        meta_rows.append({"sample_id": sid, "split": "train", "embedding_index": idx})
    for idx, sid in enumerate(val_df["sample_id"]):
        meta_rows.append({"sample_id": sid, "split": "validation", "embedding_index": idx})
    pd.DataFrame(meta_rows).to_csv(os.path.join(OUT_DIR, "embedding_metadata.csv"), index=False)
    print(f"\nSaved metadata index to {os.path.join(OUT_DIR, 'embedding_metadata.csv')}")
    print("Embedding extraction complete.")


if __name__ == "__main__":
    main()
