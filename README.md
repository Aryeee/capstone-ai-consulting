# AI-Powered Strategy Consulting Assistant

> **Capstone Project** — Deep Representation Learning + Case-Based Reasoning
> for Evidence-Based Strategy Recommendations

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Streamlit UI  (app.py)                  │
│  User: "describe a business problem"                         │
│         ↓                                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │  Embedding &  │──>│  Prediction  │──>│  Explainability  │ │
│  │  Retrieval    │   │  (Weighted   │   │  (similarity,    │ │
│  │  (SentTrans + │   │   kNN vote)  │   │   shared feats,  │ │
│  │   FAISS)      │   │              │   │   drivers)       │ │
│  └──────┬───────┘   └──────────────┘   └──────────────────┘ │
│         │                                                    │
│  ┌──────▼───────┐                                            │
│  │ Case Database │  28 structured business cases (JSON)      │
│  └──────────────┘                                            │
└──────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `data/cases.json` | 28 realistic business cases with structured + text fields |
| `src/schema.py` | `Case` and `OutcomeMetrics` dataclasses |
| `src/case_db.py` | Sentence-transformer embeddings, FAISS index build & query |
| `src/predictor.py` | Similarity-weighted kNN vote for strategy + outcome prediction |
| `src/explainer.py` | Generates human-readable explanations of retrieval & prediction |
| `app.py` | Streamlit web UI tying everything together |
| `run.py` | One-command setup + launch script |

## Quick Start

```bash
cd capstone-ai-consulting

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Build index + launch UI
python run.py
```

Or step-by-step:
```bash
python -c "from src.case_db import build_all; build_all()"
streamlit run app.py
```

## Data Schema

Each case has:

| Field | Type | Purpose |
|-------|------|---------|
| `case_id` | string | Unique identifier |
| `title` | string | Human-readable name |
| `industry` | string | Industry category |
| `company_size` | enum | Small / Medium / Large |
| `region` | string | Geographic region |
| `year` | int | Year of engagement |
| `problem_description` | text | Free-text business problem (main embedding input) |
| `key_challenges` | list[str] | Concise challenge bullets |
| `actions_taken` | text | What was done |
| `strategy_type` | string | Strategy category label |
| `outcome` | enum | Positive / Mixed / Negative |
| `outcome_description` | text | What happened |
| `outcome_metrics` | object | `revenue_change_pct`, `time_to_result_months`, `roi_estimate` |
| `tags` | list[str] | Free-form topic tags |

## How It Works

### 1. Representation Learning
Each case's `problem_description`, `key_challenges`, `actions_taken`, and metadata
are concatenated and encoded using the `all-MiniLM-L6-v2` sentence-transformer
into a 384-dimensional dense vector, L2-normalized for cosine similarity.

### 2. Semantic Retrieval
The user's problem description is encoded with the same model. FAISS
`IndexFlatIP` (inner product on normalized vectors = cosine similarity)
retrieves the top-k most similar cases.

### 3. Strategy Prediction
A similarity-weighted kNN vote over the retrieved cases' `strategy_type`
labels produces the recommended strategy. The same approach predicts the
likely outcome bucket (Positive/Mixed/Negative) and estimates an ROI range
(weighted mean ± 1 std of retrieved case ROIs).

### 4. Explainability
For each retrieved case the system reports:
- **Similarity score** (cosine)
- **Key overlapping phrases** between query and case
- **Shared vs. differing structured features** (industry, size, region)
- **Strategy drivers** — which cases voted for which strategy and with what weight
- **Outcome evidence** — historical outcomes of similar cases
- **Confidence assessment** — based on average similarity score

## Evaluation Plan

See the "Suggested Evaluation" section below for metrics to include in
your project report.
