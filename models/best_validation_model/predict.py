import os
import sys
import json
import joblib
import numpy as np

STANDARD_20_AA = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
DIPEPTIDES = [a1 + a2 for a1 in STANDARD_20_AA for a2 in STANDARD_20_AA]

def predict_protein_sequence(sequence: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_seq = sequence.strip().upper()
    
    # Validate characters
    diff = set(clean_seq) - set(STANDARD_20_AA)
    if diff:
        raise ValueError(f"Sequence contains invalid non-standard amino acid characters: {diff}")
    if len(clean_seq) == 0:
        raise ValueError("Sequence cannot be empty.")
        
    # Load model and vectorizer
    model = joblib.load(os.path.join(base_dir, "model_weights.joblib"))
    tfidf = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.joblib"))
    
    # 1. AAC
    l = len(clean_seq)
    aac = np.array([[clean_seq.count(aa) / l for aa in STANDARD_20_AA]], dtype=np.float32)
    
    # 2. DPC
    d_counts = {dp: 0 for dp in DIPEPTIDES}
    for i in range(l - 1):
        dp = clean_seq[i:i+2]
        if dp in d_counts:
            d_counts[dp] += 1
    dpc = np.array([[d_counts[dp] / max(1, l - 1) for dp in DIPEPTIDES]], dtype=np.float32)
    
    # 3. TF-IDF
    tf = tfidf.transform([clean_seq]).toarray().astype(np.float32)
    
    X = np.hstack([aac, dpc, tf])
    prob = float(model.predict_proba(X)[0, 1])
    pred = "Alzheimer-associated" if prob >= 0.5 else "Control"
    
    return {
        "sequence_length": l,
        "prediction": pred,
        "probability": round(prob, 4),
        "confidence": round(abs(prob - 0.5) * 2.0, 4)
    }

if __name__ == "__main__":
    test_seq = sys.argv[1] if len(sys.argv) > 1 else "MVLLRVLILLLSWAAGMGGQYGNPLNKYIRHYEGLSYNVDSLHQKHQRAKRAVSHEDQFLRLDFHAHGRHFNLRMKRDTSLFSDEFKVETSNKVLDYDTSHIYTGHIYGEEGSFSHGSVIDGRFEGFIQTRGGTFYVEPAERYIKDRTLPFHSVIYHEDDINYPHKYGPQGGCADHSVFERMRKYQMTGVEEVTQIPQEEHAANGPELLRKKRTTSAEKNTCQLYIQTDHLFFKYYGTREAVIAQISSHVKAIDTIYQTTDFSGIRNISFMVKRIRINTTADEKDPTNPFRFPNIGVEKFLELNSEQNHDDYCLAYVFTDRDFDDGVLGLAWVGAPSGSSGGICEKSKLYSDGKKKSLNTGIITVQNYGSHVPPKVSHITFAHEVGHNFGSPHDSGTECTPGESKNLGQKENGNYIMYARATSGDKLNNNKFSLCSIRNISQVLEKKRNNCFVESGQPICGNGMVEQGEECDCGYSDQCKDECCFDANQPEGRKCKLKPGKQCSPSQGPCCTAQCAFKSKSEKCRDDSDCAREGICNGFTALCPASDPKPNFTDCNRHTQVCINGQCAGSICEKYGLEECTCASSDGKDDKELCHVCCMKKMDPSTCASTGSVQWSRHFSGRTITLQPGSPCNDFRGYCDVFMRCRLVDADGPLARLKKAIFSPELYENIAEWIVAHWWAVLLMGIALIMLMAGFIKICSVHTPSSNPKLPPPKPLPGTLKRRRPPQPIQQPQRQRPRESYQMGHMRR"
    res = predict_protein_sequence(test_seq)
    print(json.dumps(res, indent=2))
