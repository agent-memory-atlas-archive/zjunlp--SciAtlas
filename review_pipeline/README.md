# Review Pipeline

This folder contains the automated review pipeline built around top-level `pipeline.py`.

Main entry point:
- `pipeline.py`

Recommended repository entry point:

```bash
python run_sciatlas.py idea-evaluate --idea "LLM-based idea evaluation" --workflow flash
python run_sciatlas.py idea-evaluate --idea "LLM-based idea evaluation" --workflow full
```

`flash` is the default interactive path: it uses smaller KG/S2/manifest budgets, fewer reviewer and evidence branches, and compact reporting. `full` keeps the broader reviewer, rubric, grounding, review, and report path.

Rubric-only mode (no KG / S2 / GROBID required), from the repository root:

```bash
pip install -r requirements-rubric.txt
export HF_ENDPOINT=https://hf-mirror.com   # mainland-China mirror; omit when huggingface.co is reachable
huggingface-cli download BAAI/bge-large-en-v1.5 --local-dir models/bge-large-en-v1.5
huggingface-cli download BAAI/bge-reranker-large --local-dir models/bge-reranker-large
python scripts/download_rubric_assets.py     # ~160MB precomputed NC-paper artifacts + FAISS index
python scripts/check_rubric_setup.py         # pre-flight check; prints fixes for anything missing
python run_rubric.py --idea "LLM-based idea evaluation"
```

Models under `<repo>/models/` and the asset pack under `<repo>/assets/rubric` are
auto-detected — no env vars needed for the standard layout (pin custom locations via
`RUBRIC_EMBED_MODEL_PATH` / `RUBRIC_RERANK_MODEL_PATH` etc., absolute paths). With the
precomputed asset pack (`RUBRIC_PRECOMPUTED_ROOT`, default `<repo>/assets/rubric`),
per-paper summary/dimension LLM calls, Europe PMC fetches, and review-PDF extraction
are all skipped.

KG search backend for the full pipeline is switched by `--kg-backend` (`auto`/`hosted`/`local`/`none`)
in `search/merge_search.py`: `local` uses a Neo4j instance on this machine, `hosted` uses the
SciAtlas API (`SCIATLAS_API_KEY`).

Main local modules:
- `review/`
- `workers/`
- `reviewer-evaluate/`

Shared retrieval dependencies:
- `kg_search/`
- `search/merge_search.py`
- `search/pdf_xml_pipeline.py`
- `s2api/`
