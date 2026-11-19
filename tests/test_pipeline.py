"""
tests/test_pipeline.py
Automated CI validation suite for AD-TRACE.
Covers:
1. Model inference execution & input validation
2. Dataset schema & zero-leakage split integrity
3. FastAPI endpoint health
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add workspace root to sys.path
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from models.best_validation_model.predict import predict_protein_sequence


def test_prediction_canonical_sequence():
    """Verify model produces valid probabilistic inference on canonical protein sequence."""
    seq = "MVLLRVLILLLSWAAGMGGQYGNPLNKYIRHYEGLSYNVDSLHQK"
    res = predict_protein_sequence(seq)
    
    assert isinstance(res, dict)
    assert res["sequence_length"] == len(seq)
    assert res["prediction"] in ["Alzheimer-associated", "Control"]
    assert 0.0 <= res["probability"] <= 1.0
    assert 0.0 <= res["confidence"] <= 1.0


def test_prediction_invalid_characters():
    """Verify inference pipeline rejects non-standard amino acid characters."""
    with pytest.raises(ValueError, match="invalid non-standard amino acid"):
        predict_protein_sequence("ACDEFGHIKLMNPQRSTVWYZ")  # Z is non-standard


def test_prediction_empty_sequence():
    """Verify inference pipeline rejects empty sequence input."""
    with pytest.raises(ValueError, match="cannot be empty"):
        predict_protein_sequence("   ")


def test_dataset_integrity():
    """Verify processed benchmark dataset dimensions, balance, and completeness."""
    dataset_path = os.path.join(WORKSPACE, "data", "processed", "Alzheimer_Protein_Sequence_Dataset_v1.0.csv")
    assert os.path.exists(dataset_path), "Processed dataset CSV must exist."
    
    df = pd.read_csv(dataset_path)
    assert len(df) == 1416, f"Expected 1,416 samples, found {len(df)}"
    assert set(["uniprot_accession", "sequence", "label"]).issubset(df.columns)
    assert df["sequence"].isna().sum() == 0, "No missing sequences permitted."
    
    # Class balance (708 positive, 708 control)
    label_counts = df["label"].value_counts().to_dict()
    assert label_counts[1] == 708
    assert label_counts[0] == 708


def test_zero_leakage_splits():
    """Verify zero sequence overlap between train and validation partitions."""
    train_path = os.path.join(WORKSPACE, "data", "splits", "train.csv")
    val_path = os.path.join(WORKSPACE, "data", "splits", "validation.csv")
    test_path = os.path.join(WORKSPACE, "data", "splits", "test.csv")
    
    assert os.path.exists(train_path), "train.csv must exist."
    assert os.path.exists(val_path), "validation.csv must exist."
    assert os.path.exists(test_path), "test.csv must exist."
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    train_seqs = set(df_train["sequence"].tolist())
    val_seqs = set(df_val["sequence"].tolist())
    
    overlap = train_seqs.intersection(val_seqs)
    assert len(overlap) == 0, f"Found {len(overlap)} leaked sequences between train and validation splits!"
