# Zepto Data & AI Platform — Capstone

This repository implements all three capstone modules from the supplied brief:

- `/data_pipeline` — requests + BeautifulSoup scraping, cleaning, fixed GBP→INR conversion, normalized SQLite, SQL/pandas queries.
- `/analytics` — Titanic profiling, EDA, preprocessing/modeling, imbalance comparison, Random Forest tuning, regression, and persisted pipeline.
- `/support_assistant` — local embeddings + ChromaDB, LangGraph routing/RAG, deterministic `MOCK_LLM` baseline, FastAPI and Docker.

## Setup

Use Python 3.10+ (3.11/3.12 recommended). Each module has its own `requirements.txt`; install the requirements for the module you run.

```bash
cd data_pipeline && pip install -r requirements.txt
cd ../analytics && pip install -r requirements.txt
cd ../support_assistant && pip install -r requirements.txt
```

No paid service is required. The support assistant defaults to `MOCK_LLM=1` and therefore needs no LLM API key.

## Module 1 — data pipeline

```bash
cd data_pipeline
python -m src.pipeline
```

The scraper uses `requests` and `BeautifulSoup`, crawls the first five pages of the all-products catalogue (100 books), cleans the fields, converts `price_gbp` using the required fixed rate **1 GBP = 105.50 INR**, creates `catalogue.db`, executes the required SQL queries, and writes their outputs to `output/`.

If the target site is unavailable, the script fails with a clear network error rather than silently fabricating scraped data.

## Module 2 — analytics

The required source load is `sns.load_dataset('titanic')`, exactly once. The script immediately saves the raw DataFrame to `analytics/titanic.csv`, then uses that same DataFrame/CSV for all later work.

```bash
cd analytics
python run_pipeline.py
```

The committed `titanic.csv` is an offline fallback with the same 891-row Titanic source and the Seaborn dataset's analytical columns. When network access is available, `run_pipeline.py` refreshes it from `sns.load_dataset('titanic')` exactly once.

Outputs include missingness, EDA charts, correlation heatmap, standardized age/fare check, model metrics, confusion matrices, ROC/AUC, decision-tree plot, imbalance comparison, GridSearchCV results, regression residual plot, and `best_pipeline.joblib`.

## Module 3 — support assistant

```bash
cd support_assistant
pip install -r requirements.txt
python build_index.py
uvicorn main:app --reload --port 7860
```

Then:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"What is the delivery fee below INR 149?"}'
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"Tell me a joke."}'
```

Leave `MOCK_LLM` unset (or set it to `1`) for the graded deterministic baseline. `MOCK_LLM=0` is an optional real-LLM extension and requires the corresponding provider credentials.

## Architecture

### Data pipeline

`requests → BeautifulSoup → clean/parse → GBP→INR fixed-rate enrichment → normalized SQLite → SQL + pandas validation`

### Analytics

`Seaborn Titanic load (once) → profile/missingness → threshold-based cleaning → EDA → train/test split → train-only ColumnTransformer → 3 classifiers → evaluation → imbalance comparison → RF GridSearchCV → regression → persisted full pipeline`

### Support assistant

`8 policy .txt files → chunking → all-MiniLM-L6-v2 embeddings → ChromaDB → LangGraph intent router → top-3 retrieval for policy questions → structured answer → FastAPI`

Only generation/classification stages branch on `MOCK_LLM`. Embedding and ChromaDB retrieval are real in both modes.

## Git workflow requirement

The project should be committed to one repository. To satisfy the brief's history criterion, create a feature branch, make at least two commits on it, and merge it back into `main`, for example:

```bash
git checkout -b feature/capstone-implementation
git add . && git commit -m "feat: implement capstone modules"
git add . && git commit -m "test: add outputs and documentation"
git checkout main
git merge --no-ff feature/capstone-implementation -m "merge: capstone feature"
git log --graph --all --oneline
```

## Source brief

All task requirements in this implementation are based on the supplied `Capstone project.pdf`. The project brief explicitly requires one repository containing the three root modules and a root README, with no separate PDF/slides submission.
