# DeepLearning-Project

One-line summary
---
A compact research codebase for training and evaluating sequential models (RNNs) on lyric/text generation tasks. The repository contains training and generation scripts, model implementations, preprocessing utilities, and example datasets to reproduce baseline experiments.

Table of contents
---
- Highlights
- Architecture (components & responsibilities)
- Pipeline (data → model → output)
- Quickstart
- Repository layout
- Reproducibility & experiments
- Recommendations & next steps
- Contact

Highlights
---
- Clear, modular baseline implementations of RNN architectures suitable for quick experimentation and comparison.
- Separate training and generation workflows to keep experiments reproducible and composable.
- Utilities for text and MIDI feature handling (integration-ready for multimodal experiments).

Architecture
---
The following table summarizes the main components, their location in the repo and their responsibilities.

| Component | Location | Responsibility |
|---|---:|---|
| Data sets | `data/sets/` | Raw CSV datasets for training and testing (lyrics). |
| Preprocessing utils | `utils/text_utils.py` | Tokenization, vocabulary creation, batching helpers. |
| Feature extraction | `utils/midi_features.py` | Extract and normalize MIDI-derived features (if used). |
| Model implementations | `models/` | `RNN_baseline.py`, `RNN_baseline_V1.py`, `RNN_baseline_V2.py` — core architectures and forward logic. |
| Training loop | `train.py` | Loads data, trains model, handles checkpoints and logging. |
| Generation / sampling | `generate.py` | Loads a trained checkpoint and generates text from a prompt/seed. |

Pipeline (high-level)
---
This project follows a linear pipeline designed for experiment reproducibility and easy swapping of components.

1. Data collection: add or edit CSVs under `data/sets/`.
2. Preprocessing: `utils/text_utils.py` tokenizes text, builds or loads vocab, and produces batches.
3. Model definition: pick or modify an RNN in `models/`.
4. Training: `train.py` runs epochs, logs metrics, and saves checkpoints.
5. Generation: `generate.py` loads a checkpoint and produces output samples.
6. Evaluation: compare outputs against test set and qualitative inspection.

ASCII flow diagram
---

Data (CSV) --> Preprocessing --> Batches --> Trainer --> Checkpoints
						    |
						    v
					    Generation

Key pipeline notes
---
- Checkpointing: `train.py` should save model + optimizer state so generation can resume from any saved state.
- Configs: keep hyperparameters in a single place (recommended: `config.yaml` or CLI args) to reproduce experiments.
- Determinism: log random seeds and library versions for exact reproduction.

Quickstart (Windows example)
---
1. Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. (Optional) Install dependencies (create `requirements.txt` if absent):

```bash
pip install torch pandas numpy
```

3. Train a baseline model (defaults in `train.py`):

```bash
python train.py
```

4. Generate samples from a saved checkpoint:

```bash
python generate.py --checkpoint path/to/checkpoint.pt --seed "start prompt"
```

Repository layout (concise)
---

| Path | Purpose |
|---|---|
| `train.py` | Training loop and CLI for experiments |
| `generate.py` | Generation / sampling script |
| `models/` | RNN implementations and model definitions |
| `data/sets/` | Example datasets (lyrics_train_set.csv, lyrics_test_set.csv) |
| `utils/` | Preprocessing and feature utilities (`text_utils.py`, `midi_features.py`) |

Reproducibility & experiments
---
- Save: model checkpoints, training logs (loss/metrics), and a copy of the config used for each run.
- Document: random seeds, Python and library versions, and the exact dataset files used.
- Evaluate: include both quantitative metrics and qualitative samples in any report.

Recommendations & next steps
---
- Add a `requirements.txt` or `pyproject.toml` to lock dependencies.
- Add a `config.yaml` or expand CLI arg parsing to centralize hyperparameters and dataset paths.
- Add automated example notebooks or a `scripts/` folder with reproducible experiment runners.
- Consider adding unit tests for preprocessing utilities to ensure stable pipelines.

Contact
---
For questions or collaboration, open an issue in this repository or contact the maintainer directly via the repository profile.

If you want, I can:
- produce a concise English one-paragraph pitch for sharing on LinkedIn/GitHub profile,
- create a short CV-friendly bullet summary,
- or add `requirements.txt` and a minimal `config.yaml` to the repo now.
