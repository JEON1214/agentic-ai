# Search Quality Report

## Improvements Implemented

- Query Expansion for short queries (uses Gemini to expand short queries into paragraph form).
- Hybrid Search: dense similarity re-ranked with keyword overlap (alpha=0.7).

## Results Table

| Query | Expected Contains? | Baseline Hit | Improved Hit | Improvement? |
|-------|-------------------:|:------------:|:------------:|:------------:|
| What distance metric does Qdrant use? | "Cosine" |  |  |  |
| What is the travel budget for flights? | "$2000" |  |  |  |
| What is ReAct? | "ReAct stands for Reasoning" |  |  |  |

## Analysis

- For queries missed by the baseline but found by improved search, likely causes include embedding asymmetry (short queries) and embedding dilution (long documents).

## Trade-offs

- Extra LLM calls for query expansion and potential re-ranking increases latency and API costs.
- Hybrid re-ranking adds CPU cost locally but improves top-1 precision.

## How to run

1. Start Qdrant with Docker Compose:

```bash
cd activity14
docker compose up -d
```

2. Seed the collection (optional):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python seed_qdrant.py
```

3. Run comparison:

```bash
python search_comparison.py
```

