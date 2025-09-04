# ETF-LLM-for-DBQA
This repository is created for research published at the International Conference on Information and Communication Technologies (ICIST 2025) and later included in the Springer book series Lecture Notes in Networks and Systems: 
```
DOI1
DOI1
```

LLM-powered Database Question Answering (DBQA) for relational data: ask a question in natural language, get an answer grounded in your database.

This repository provides a starting point for:
- executing queries and post-processing results,
- evaluating answers (currently includes a cosine-similarity utility)

## Repo Structure

```bash
ETF-LLM-for-DBQA/
├─ src/                      # (to be populated) DBQA pipeline modules, model wrappers, runners
├─ pdfs/                     # paper figures / PDFs for the accompanying write-up
├─ cosine_sim.py             # cosine similarity helper for evaluation
├─ requirements.txt          # default dependencies (Linux/WSL friendly)
├─ requirements_win.txt      # Windows-focused dependencies
├─ requirements-mac.txt      # macOS-focused dependencies
└─ .gitignore
```

```cosine_sim.py```: Utility for cosine similarity (useful for text-based answer evaluation or embedding comparisons).

Platform-specific requirements: pick the file that matches your OS to reduce dependency friction.

## Quickstart

### Environment

Choose one of the requirements files:
```bash
# Linux / WSL
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# macOS (Apple Silicon or Intel)
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements-mac.txt

# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements_win.txt
```

Note: If you run into compiler/system deps (e.g., audio/BLAS libs), install the platform’s recommended build tools first.

### Configuration
Create a .env at the repo root (or use environment variables directly):
```
MODEL=text-embedding-3-large
```

## Citation 
If you use this codebase or ideas from the accompanying write-up, please use the following citations:
```

```
with the following BibTeX codes:
```

```

## Licence
This project is licensed under the GNU General Public License v3.0.
