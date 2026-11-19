import os
import sys
import re
import json
import joblib
import torch
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(MODEL_DIR, "config.json")) as f:
    CONFIG = json.load(f)

STANDARD_20_AA = set("ACDEFGHIKLMNPQRSTVWY")

def predict(sequence: str):
    clean_seq = sequence.strip().upper()
    diff = set(clean_seq) - STANDARD_20_AA
    if diff:
        clean_seq = re.sub(r"[UZOB]", "X", clean_seq)
        
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["embedding_model"], do_lower_case=False)
    encoder = T5EncoderModel.from_pretrained(CONFIG["embedding_model"], torch_dtype=torch.float32)
    encoder.eval()
    
    # Embed
    win_size = CONFIG["window_size"]
    stride = CONFIG["stride"]
    seq_len = len(clean_seq)
    
    with torch.inference_mode():
        if seq_len <= win_size or "Fixed" in CONFIG["long_sequence_strategy"]:
            spaced = " ".join(list(clean_seq[:win_size]))
            inp = tokenizer(spaced, return_tensors="pt")
            out = encoder(**inp)
            m = inp["attention_mask"].unsqueeze(-1)
            emb = (out.last_hidden_state * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
            feat = emb.numpy()
        else:
            reps = []
            start = 0
            while start < seq_len and len(reps) < 4:
                sub = clean_seq[start:min(start + win_size, seq_len)]
                spaced = " ".join(list(sub))
                inp = tokenizer(spaced, return_tensors="pt")
                out = encoder(**inp)
                m = inp["attention_mask"].unsqueeze(-1)
                e = (out.last_hidden_state * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
                reps.append(e.numpy())
                start += stride
            feat = np.mean(reps, axis=0)
            
    clf = joblib.load(os.path.join(MODEL_DIR, "classifier.joblib"))
    prob = float(clf.predict_proba(feat)[0, 1])
    pred = "Alzheimer-associated" if prob >= 0.5 else "Control"
    
    return {
        "sequence_length": seq_len,
        "prediction": pred,
        "probability": round(prob, 4),
        "confidence": round(abs(prob - 0.5) * 2.0, 4)
    }

if __name__ == "__main__":
    test_seq = sys.argv[1] if len(sys.argv) > 1 else "MVLLRVLILLLSWAAGMGGQYGNPLNKYIRHYEGLSYNVDSLHQK"
    print(json.dumps(predict(test_seq), indent=2))
