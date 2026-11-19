"""
scripts/train_cnn.py
Phase 5 & Phase 6: Sequence Deep Learning Model & Long-Sequence Handling.

Architecture:
Amino-Acid Tokenization -> Embedding Layer -> Multi-Scale 1D CNN Blocks -> Global Pooling -> Dense Classification Head.

Long-Sequence Handling Comparison:
- Approach A1: Fixed Length Truncation/Padding (L=1000)
- Approach A2: Fixed Length Truncation/Padding (L=1500)
- Approach B:  Window/Chunk-Based Representation (Chunk Size W=512, Attention Pooling over Chunks)

Trained ONLY on data/modeling/train_model.csv.
Monitored with Early Stopping on data/modeling/validation_model.csv.
Test set remains completely LOCKED.

Outputs:
- models/cnn/
- Training history and comparison logs
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix
)


STANDARD_20_AA = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(STANDARD_20_AA)} # 0 is PAD
VOCAB_SIZE = len(STANDARD_20_AA) + 1 # 21


# ============================================================
# DATASETS & TOKENIZATION
# ============================================================

class FixedLengthProteinDataset(Dataset):
    def __init__(self, sequences, labels, max_len=1000):
        self.max_len = max_len
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
        tokenized = []
        for seq in sequences:
            tokens = [AA_TO_IDX.get(aa, 0) for aa in seq[:max_len]]
            if len(tokens) < max_len:
                tokens = tokens + [0] * (max_len - len(tokens))
            tokenized.append(tokens)
        self.tokens = torch.tensor(tokenized, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


class ChunkedProteinDataset(Dataset):
    def __init__(self, sequences, labels, chunk_size=512, max_chunks=8):
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
        all_chunks = []
        all_masks = []
        
        for seq in sequences:
            # Divide into chunks of chunk_size
            chunks = []
            for i in range(0, len(seq), chunk_size):
                sub = seq[i:i+chunk_size]
                toks = [AA_TO_IDX.get(aa, 0) for aa in sub]
                if len(toks) < chunk_size:
                    toks = toks + [0] * (chunk_size - len(toks))
                chunks.append(toks)
                if len(chunks) == max_chunks:
                    break
                    
            n_chunks = len(chunks)
            mask = [True] * n_chunks + [False] * (max_chunks - n_chunks)
            # Pad empty chunks
            while len(chunks) < max_chunks:
                chunks.append([0] * chunk_size)
                
            all_chunks.append(chunks)
            all_masks.append(mask)
            
        self.tokens = torch.tensor(all_chunks, dtype=torch.long) # (N, max_chunks, chunk_size)
        self.masks = torch.tensor(all_masks, dtype=torch.bool)   # (N, max_chunks)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.tokens[idx], self.masks[idx], self.labels[idx]


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class MultiScale1DCNN(nn.Module):
    """Approach A: Fixed Length Multi-Scale 1D CNN"""
    def __init__(self, vocab_size=21, embed_dim=64, num_filters=64, kernel_sizes=[3, 5, 7], dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, k, padding=k//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ) for k in kernel_sizes
        ])
        in_dim = len(kernel_sizes) * num_filters * 2 # Max + Avg pooling per kernel
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x: (B, L)
        emb = self.embedding(x).transpose(1, 2) # (B, embed_dim, L)
        pooled_reps = []
        for conv in self.conv_blocks:
            c = conv(emb) # (B, num_filters, L)
            max_p = torch.max(c, dim=2)[0]
            avg_p = torch.mean(c, dim=2)
            pooled_reps.extend([max_p, avg_p])
        rep = torch.cat(pooled_reps, dim=1) # (B, in_dim)
        return self.head(rep).squeeze(-1)


class ChunkedMultiScale1DCNN(nn.Module):
    """Approach B: Window / Chunk-Based Multi-Scale 1D CNN with Attention Aggregation"""
    def __init__(self, vocab_size=21, embed_dim=64, num_filters=64, kernel_sizes=[3, 5, 7], dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, k, padding=k//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ) for k in kernel_sizes
        ])
        in_dim = len(kernel_sizes) * num_filters * 2
        
        # Chunk-level attention pooling
        self.chunk_attention = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, chunk_mask):
        # x: (B, N_c, W), chunk_mask: (B, N_c)
        B, Nc, W = x.shape
        x_flat = x.view(B * Nc, W)
        emb = self.embedding(x_flat).transpose(1, 2) # (B*Nc, embed_dim, W)
        
        pooled_reps = []
        for conv in self.conv_blocks:
            c = conv(emb)
            max_p = torch.max(c, dim=2)[0]
            avg_p = torch.mean(c, dim=2)
            pooled_reps.extend([max_p, avg_p])
        flat_rep = torch.cat(pooled_reps, dim=1) # (B*Nc, in_dim)
        
        chunk_rep = flat_rep.view(B, Nc, -1) # (B, Nc, in_dim)
        
        # Attention across chunks
        attn_logits = self.chunk_attention(chunk_rep).squeeze(-1) # (B, Nc)
        attn_logits = attn_logits.masked_fill(~chunk_mask, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1) # (B, Nc, 1)
        
        prot_rep = torch.sum(chunk_rep * attn_weights, dim=1) # (B, in_dim)
        return self.head(prot_rep).squeeze(-1)


# ============================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================

def train_fixed_cnn(
    train_df, val_df, max_len=1000, embed_dim=64, num_filters=64,
    lr=0.001, batch_size=32, epochs=30, patience=8, seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_ds = FixedLengthProteinDataset(train_df["sequence"], train_df["label"].values, max_len=max_len)
    val_ds = FixedLengthProteinDataset(val_df["sequence"], val_df["label"].values, max_len=max_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = MultiScale1DCNN(embed_dim=embed_dim, num_filters=num_filters, dropout=0.3)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    history = {"train_loss": [], "val_loss": [], "val_roc_auc": [], "val_f1": []}
    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    patience_cnt = 0
    
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
        train_loss = total_loss / len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_probs = []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
        val_loss /= len(val_ds)
        val_probs = np.array(val_probs)
        y_val = val_df["label"].values
        val_pred = (val_probs >= 0.5).astype(int)
        
        val_auc = roc_auc_score(y_val, val_probs)
        val_f1 = f1_score(y_val, val_pred)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_roc_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break
                
    train_time = time.time() - t0
    
    # Load best model for final evaluation
    model.load_state_dict(best_state)
    model.eval()
    val_probs = []
    with torch.no_grad():
        for bx, by in val_loader:
            probs = torch.sigmoid(model(bx)).cpu().numpy()
            val_probs.extend(probs)
    val_probs = np.array(val_probs)
    val_pred = (val_probs >= 0.5).astype(int)
    
    metrics = {
        "accuracy": round(accuracy_score(y_val, val_pred), 4),
        "precision": round(precision_score(y_val, val_pred), 4),
        "recall": round(recall_score(y_val, val_pred), 4),
        "f1": round(f1_score(y_val, val_pred), 4),
        "roc_auc": round(roc_auc_score(y_val, val_probs), 4),
        "pr_auc": round(average_precision_score(y_val, val_probs), 4),
        "mcc": round(matthews_corrcoef(y_val, val_pred), 4),
        "training_time": round(train_time, 2),
        "best_epoch": best_epoch,
        "total_epochs": len(history["train_loss"])
    }
    return model, history, metrics, val_probs


def train_chunked_cnn(
    train_df, val_df, chunk_size=512, max_chunks=8, embed_dim=64, num_filters=64,
    lr=0.001, batch_size=16, epochs=30, patience=8, seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_ds = ChunkedProteinDataset(train_df["sequence"], train_df["label"].values, chunk_size=chunk_size, max_chunks=max_chunks)
    val_ds = ChunkedProteinDataset(val_df["sequence"], val_df["label"].values, chunk_size=chunk_size, max_chunks=max_chunks)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = ChunkedMultiScale1DCNN(embed_dim=embed_dim, num_filters=num_filters, dropout=0.3)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    history = {"train_loss": [], "val_loss": [], "val_roc_auc": [], "val_f1": []}
    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    patience_cnt = 0
    
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, bm, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx, bm)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
        train_loss = total_loss / len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_probs = []
        with torch.no_grad():
            for bx, bm, by in val_loader:
                logits = model(bx, bm)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
        val_loss /= len(val_ds)
        val_probs = np.array(val_probs)
        y_val = val_df["label"].values
        val_pred = (val_probs >= 0.5).astype(int)
        
        val_auc = roc_auc_score(y_val, val_probs)
        val_f1 = f1_score(y_val, val_pred)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_roc_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break
                
    train_time = time.time() - t0
    
    model.load_state_dict(best_state)
    model.eval()
    val_probs = []
    with torch.no_grad():
        for bx, bm, by in val_loader:
            probs = torch.sigmoid(model(bx, bm)).cpu().numpy()
            val_probs.extend(probs)
    val_probs = np.array(val_probs)
    val_pred = (val_probs >= 0.5).astype(int)
    
    metrics = {
        "accuracy": round(accuracy_score(y_val, val_pred), 4),
        "precision": round(precision_score(y_val, val_pred), 4),
        "recall": round(recall_score(y_val, val_pred), 4),
        "f1": round(f1_score(y_val, val_pred), 4),
        "roc_auc": round(roc_auc_score(y_val, val_probs), 4),
        "pr_auc": round(average_precision_score(y_val, val_probs), 4),
        "mcc": round(matthews_corrcoef(y_val, val_pred), 4),
        "training_time": round(train_time, 2),
        "best_epoch": best_epoch,
        "total_epochs": len(history["train_loss"])
    }
    return model, history, metrics, val_probs


def main():
    print("=" * 60)
    print("PHASE 5 & 6: TRAINING 1D CNN MODELS WITH LONG SEQUENCE STRATEGIES")
    print("=" * 60)
    
    train_df = pd.read_csv("data/modeling/train_model.csv")
    val_df = pd.read_csv("data/modeling/validation_model.csv")
    os.makedirs("models/cnn", exist_ok=True)
    
    results = []
    
    # 1. Approach A1: Fixed L=1000
    print("\n--- Training Approach A1: 1D CNN Fixed (L=1000) ---")
    model_a1, hist_a1, metrics_a1, probs_a1 = train_fixed_cnn(
        train_df, val_df, max_len=1000, embed_dim=64, num_filters=64, lr=0.001, batch_size=32
    )
    torch.save(model_a1.state_dict(), "models/cnn/cnn_fixed_l1000.pt")
    with open("models/cnn/history_fixed_l1000.json", "w") as f:
        json.dump(hist_a1, f)
    metrics_a1["strategy"] = "Fixed Length (L=1000)"
    metrics_a1["model"] = "1D-CNN (Multi-Scale 3,5,7)"
    results.append(metrics_a1)
    print("Metrics A1 (L=1000):", metrics_a1)
    
    # 2. Approach A2: Fixed L=1500
    print("\n--- Training Approach A2: 1D CNN Fixed (L=1500) ---")
    model_a2, hist_a2, metrics_a2, probs_a2 = train_fixed_cnn(
        train_df, val_df, max_len=1500, embed_dim=64, num_filters=64, lr=0.001, batch_size=32
    )
    torch.save(model_a2.state_dict(), "models/cnn/cnn_fixed_l1500.pt")
    with open("models/cnn/history_fixed_l1500.json", "w") as f:
        json.dump(hist_a2, f)
    metrics_a2["strategy"] = "Fixed Length (L=1500)"
    metrics_a2["model"] = "1D-CNN (Multi-Scale 3,5,7)"
    results.append(metrics_a2)
    print("Metrics A2 (L=1500):", metrics_a2)
    
    # 3. Approach B: Chunked W=512, Max Chunks=8
    print("\n--- Training Approach B: 1D CNN Chunked (W=512, Chunks<=8) ---")
    model_b, hist_b, metrics_b, probs_b = train_chunked_cnn(
        train_df, val_df, chunk_size=512, max_chunks=8, embed_dim=64, num_filters=64, lr=0.001, batch_size=16
    )
    torch.save(model_b.state_dict(), "models/cnn/cnn_chunked_w512.pt")
    with open("models/cnn/history_chunked_w512.json", "w") as f:
        json.dump(hist_b, f)
    metrics_b["strategy"] = "Chunk-Attention (W=512, Nc<=8)"
    metrics_b["model"] = "Chunked 1D-CNN"
    results.append(metrics_b)
    print("Metrics B (Chunked):", metrics_b)
    
    # Save comparison table
    df_cnn = pd.DataFrame(results)
    df_cnn.to_csv("models/cnn/cnn_strategy_comparison.csv", index=False)
    print("\nLong Sequence Strategy Comparison:\n", df_cnn.to_string())


if __name__ == "__main__":
    main()
