# Contributing to AD-TRACE

Thank you for your interest in contributing to **AD-TRACE**! We welcome contributions from computational biologists, bioinformaticians, and machine learning researchers.

---

## Areas Where You Can Contribute

1. **New Model Architectures & Benchmark Submissions:**
   - Pretrained Protein Language Models (e.g., ESM-2, ESM-1b, Ankh, ProtBERT).
   - Graph Neural Networks (GNNs) on STRING protein-protein interactome topologies.
   - Novel multi-modal fusion heads.
2. **Additional Biological Modalities:**
   - 3D structural embeddings (AlphaFold predicted pLDDT, surface electrostatics).
   - Single-cell brain transcriptomics or temporal post-mortem stage data.
3. **Bug Reports & Performance Optimizations:**
   - Vectorized data processing improvements.
   - Inference latency optimizations in `frontend/server.py`.

---

## Zero-Leakage Scientific Integrity Principles

All model submissions must adhere to the **AD-TRACE Zero-Leakage Benchmark Protocol**:
1. **Never train on the test set:** The 213 test samples in `data/modeling/test_model.csv` and `data/splits/test.csv` are strictly held out.
2. **Respect MMseqs2 homology clusters:** Evaluate models using the cluster partitions in `data/splits/` (40% identity, 80% coverage).
3. **No circular database contamination:** Do not feed target disease annotations or clinical association scores as model input features.

---

## Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/ad-trace.git
cd ad-trace

# 2. Set up virtual environment
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1
# Linux/macOS: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements_modeling.txt
pip install pytest

# 4. Verify local test suite passes
pytest tests/ -v
```

---

## Submitting Pull Requests

1. Create a feature branch: `git checkout -b feature/your-model-name`.
2. Ensure all tests pass: `pytest tests/ -v`.
3. If proposing a new model, document its validation metrics (`Accuracy`, `Balanced Accuracy`, `ROC-AUC`, `PR-AUC`, and MEST stress score).
4. Submit a Pull Request targeting the `main` branch with a clear description of the methodology.

---

## Code of Conduct

Be welcoming, respectful, and constructive in all issue discussions and pull request reviews.
