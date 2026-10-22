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

Rubric-only mode (no S2 / GROBID required), from the repository root. Retrieval goes through the hosted SciAtlas KG (`SCIATLAS_API_KEY`) and reranking is done by the configured LLM — no local embedding/reranker models are needed:

```bash
python run_rubric.py --idea "LLM-based idea evaluation"
```

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
