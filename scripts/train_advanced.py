"""
scripts/train_advanced.py
Phase 7: Advanced Sequence Architecture — Hybrid CNN + BiLSTM.

Architecture:
1. Tokenization: Primary sequence character embedding (dim=64).
2. Local Motif Extraction: 1D Conv (kernel=5, filters=64) + BatchNorm + ReLU + MaxPool1d(2).
3. Global Context Scanning: Bidirectional LSTM (hidden=64, bidirectional=True).
4. Representation Pooling: Concatenated Global Max & Mean Pooling across LSTM states.
5. Classification Head: Dense MLP with Dropout & BatchNorm.

Evaluated strictly on data/modeling/validation_model.csv.
Test set remains completely LOCKED.

Outputs:
- models/advanced/
- Advanced model weights and training history
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
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(STANDARD_20_AA)}
VOCAB_SIZE = len(STANDARD_20_AA) + 1


class ProteinSequenceDataset(Dataset):
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


class HybridCNNBiLSTM(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=64, conv_filters=64, lstm_hidden=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, conv_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5)
        )
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        in_dim = lstm_hidden * 2 * 2 # 2 directions * 2 pooling types (max, mean)
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
        c = self.conv(emb).transpose(1, 2)      # (B, L/2, conv_filters)
        lstm_out, _ = self.lstm(c)              # (B, L/2, 2*lstm_hidden)
        max_p = torch.max(lstm_out, dim=1)[0]
        avg_p = torch.mean(lstm_out, dim=1)
        rep = torch.cat([max_p, avg_p], dim=1) # (B, in_dim)
        return self.head(rep).squeeze(-1)


def train_cnn_bilstm(
    train_path="data/modeling/train_model.csv",
    val_path="data/modeling/validation_model.csv",
    output_dir="models/advanced",
    max_len=1000,
    embed_dim=64,
    conv_filters=64,
    lstm_hidden=64,
    dropout=0.3,
    lr=0.001,
    batch_size=32,
    epochs=30,
    patience=8,
    seed=42
):
    print("=" * 60)
    print("PHASE 7: TRAINING HYBRID CNN + BiLSTM SEQUENCE MODEL")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    y_val = val_df["label"].values
    
    train_ds = ProteinSequenceDataset(train_df["sequence"], train_df["label"].values, max_len=max_len)
    val_ds = ProteinSequenceDataset(val_df["sequence"], val_df["label"].values, max_len=max_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = HybridCNNBiLSTM(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        conv_filters=conv_filters,
        lstm_hidden=lstm_hidden,
        dropout=dropout
    )
    
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
    
    # Final evaluation on best state
    model.load_state_dict(best_state)
    model.eval()
    val_probs = []
    with torch.no_grad():
        for bx, by in val_loader:
            probs = torch.sigmoid(model(bx)).cpu().numpy()
            val_probs.extend(probs)
    val_probs = np.array(val_probs)
    val_pred = (val_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_val, val_pred)
    
    metrics = {
        "model": "Hybrid CNN + BiLSTM",
        "parameters": f"embed={embed_dim}, conv={conv_filters}, lstm={lstm_hidden}, max_len={max_len}, lr={lr}",
        "training_time": round(train_time, 2),
        "accuracy": round(accuracy_score(y_val, val_pred), 4),
        "precision": round(precision_score(y_val, val_pred), 4),
        "recall": round(recall_score(y_val, val_pred), 4),
        "f1": round(f1_score(y_val, val_pred), 4),
        "roc_auc": round(roc_auc_score(y_val, val_probs), 4),
        "pr_auc": round(average_precision_score(y_val, val_probs), 4),
        "mcc": round(matthews_corrcoef(y_val, val_pred), 4),
        "confusion_matrix": f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}",
        "best_epoch": best_epoch,
        "total_epochs": len(history["train_loss"])
    }
    
    print("\nCNN + BiLSTM Validation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    torch.save(model.state_dict(), os.path.join(output_dir, "cnn_bilstm_best.pt"))
    with open(os.path.join(output_dir, "history_cnn_bilstm.json"), "w") as f:
        json.dump(history, f)
    with open(os.path.join(output_dir, "metrics_cnn_bilstm.json"), "w") as f:
        json.dump(metrics, f)
        
    print(f"\nSaved advanced model artifacts to {output_dir}")
    return model, history, metrics, val_probs


def main():
    train_cnn_bilstm()


if __name__ == "__main__":
    main()
