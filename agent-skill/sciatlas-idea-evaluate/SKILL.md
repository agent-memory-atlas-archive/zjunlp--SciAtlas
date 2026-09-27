---
name: sciatlas-idea-evaluate
description: Use only the current SciAtlas automated review workflow (`review_pipeline`) to take a novice user from zero setup to a final automated review of a research idea or paper, including setup, registration guidance, workflow configuration, retrieval, artifact reading, and synthesis. Trigger when the user asks whether an idea is worth pursuing, wants automatic review, meta-review, rubric-based critique, novelty/feasibility/soundness assessment, or literature-backed evaluation.
---

# SciAtlas Idea Evaluate

Use this skill to run the repository automated review workflow. The workflow builds idea context, searches KG/S2 evidence, creates a manifest, grounds claims to paper paragraphs, builds rubric evidence, samples reviewer backgrounds, generates reviewer reports, and synthesizes a final report.

## Operating Contract

- Run only `sciatlas idea-evaluate` or `python run_sciatlas.py idea-evaluate` for this skill.
- Own the end-to-end novice flow: install or locate the CLI, guide registration, configure `.env` or shell variables, run the workflow, inspect artifacts, and synthesize the final review result.
- Ask the user only for human-only values: missing idea/PDF path, email, verification code, SciAtlas token, LLM/S2/KG credentials that are not already configured, or one necessary scope clarification.
- Do not ask the user to run shell commands when tool access is available.
- Use `--workflow flash` by default.
- Use `--workflow full` for a broader reviewer/rubric/evidence pass or when the user wants a more comprehensive review.
- Never disclose full API keys or tokens.
- Read saved artifacts before answering.

## Zero-Start Bootstrap

1. Check whether the repository command works:

```bash
python run_sciatlas.py idea-evaluate -h
```

If needed, fall back to `sciatlas idea-evaluate -h` after installing the full checkout.

2. This dedicated workflow requires a full SciAtlas checkout. If it is missing, clone the repository, change into it, then run `python -m pip install -e ./sciatlas` and `python -m pip install -r requirements-workflows.txt`. Do not use the GitHub `#subdirectory=sciatlas` package-only installation for this workflow.
3. Run `python scripts/check_rubric_setup.py`. If it reports missing models, assets, packages, or keys, fix them yourself following [Local Model Setup](#local-model-setup) before proceeding — the workflow cannot run without the local embedding/rerank models.
4. Check current environment and `.env` for `SCIATLAS_API_KEY`, LLM settings, S2 settings, and KG settings before asking the user.
5. If no SciAtlas token is configured, guide the user to `http://sciatlas.openkg.cn/register`; ask for email, verification code, and returned `sciatlas_xxx` token only when needed.
6. If LLM/S2/KG credentials are required and missing, ask only for the missing values. Use the user's provider values without printing them back.
7. Configure the current shell or `.env` yourself, then run the workflow.

Configure workflow credentials in `.env` or the shell:

```bash
SCIATLAS_API_BASE_URL=http://sciatlas.openkg.cn
SCIATLAS_API_KEY=<sciatlas-token>
NEO4J_URI=<local-or-hosted-neo4j-uri>
NEO4J_USER=<neo4j-user>
NEO4J_PASSWORD=<neo4j-password>
S2_API_KEY=<semantic-scholar-key>
OPENAI_API_KEY=<llm-key>
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```

The workflow also accepts `DMX-API-KEY`, `DMX_API_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_API_URL`, `SEARCH_LLM_API_KEY`, `SEARCH_LLM_API_URL`, and `SEARCH_LLM_MODEL`.

## Local Model Setup

The rubric and grounding stages run on two local models that you must download once — do this yourself with tool access, never ask the user to run download commands, and never substitute a remote embeddings service.

First check what is missing (the script prints the exact fix commands):

```bash
python scripts/check_rubric_setup.py
```

Download both models (~1.3GB each) into `<repo>/models/`; the workflow auto-detects them there, so no environment variables are required:

```bash
pip install -r requirements-rubric.txt
export HF_ENDPOINT=https://hf-mirror.com   # mainland-China mirror; omit when huggingface.co is reachable
huggingface-cli download BAAI/bge-large-en-v1.5 --local-dir <repo>/models/bge-large-en-v1.5
huggingface-cli download BAAI/bge-reranker-large --local-dir <repo>/models/bge-reranker-large
python scripts/download_rubric_assets.py   # FAISS index + precomputed NC-paper artifacts (~160MB)
```

ModelScope alternative (mainland China, no mirror needed):

```bash
pip install modelscope
modelscope download --model AI-ModelScope/bge-large-en-v1.5 --local_dir <repo>/models/bge-large-en-v1.5
modelscope download --model BAAI/bge-reranker-large --local_dir <repo>/models/bge-reranker-large
```

Verification checklist before running the workflow:

- `python scripts/check_rubric_setup.py` exits 0 — re-run it after every download or config change.
- Directory names are exactly `bge-large-en-v1.5` / `bge-reranker-large` under `<repo>/models/` (auto-detection is name-based). Models kept elsewhere must be pinned with **absolute paths**: `RUBRIC_EMBED_MODEL_PATH` / `RUBRIC_RERANK_MODEL_PATH` (rubric retrieval), `INNOEVAL_EMBEDDING_MODEL_PATH` / `INNOEVAL_RERANKER_MODEL_PATH` (KG search / author profiling), `SCIATLAS_EMBEDDING_MODEL_PATH` / `SCIATLAS_RERANKER_MODEL_PATH` (grounding).
- Each model directory contains `config.json` plus weights (`model.safetensors` or `pytorch_model.bin`).
- The embedding model must be exactly `bge-large-en-v1.5` (1024 dims) — the rubric FAISS index was built with it; other embedding models will not match.
- On shared GPU machines set `INNOEVAL_CUDA_DEVICES` to free GPU ids; CPU also works (slower).
- The asset pack is always required: it provides the FAISS index, `nc_meta.json`, and per-paper precomputed artifacts (skipping Europe PMC fetches and per-paper LLM calls). When default dataset paths do not exist, index and metadata are auto-detected from the pack.

## Run Plan

Flash path for an idea:

```bash
python run_sciatlas.py idea-evaluate \
  --idea "<research idea>" \
  --workflow flash
```

Full path:

```bash
python run_sciatlas.py idea-evaluate \
  --idea "<research idea>" \
  --workflow full
```

For a paper PDF:

```bash
python run_sciatlas.py idea-evaluate \
  --pdf path/to/paper.pdf \
  --workflow full
```

Useful overrides:

- `--top-k N`, `--kg-top-k N`, `--s2-top-k N`, `--manifest-top-k N` tune evidence breadth.
- `--max-reviewers N` controls reviewer fan-out.
- `--grounding-final-top-k N` controls paragraph evidence depth.
- `--short-report` requests compact meta-review output in full mode.
- `--disable-review-llm` uses deterministic fallback when LLM review is unavailable.
- `--smoke` runs a stubbed pipeline for structural validation.

## Workflow Modes

`flash` compresses nonessential stages:

- smaller KG/S2/manifest budgets;
- one reviewer by default;
- compact evidence-card budgets;
- grounding refinement disabled;
- short meta-review enabled.

`full` keeps the comprehensive path:

- broader KG/S2/manifest budgets;
- reviewer fan-out and reviewer-background branch;
- rubric-source and rubric-LLM branches;
- paragraph grounding, reviewer reports, and full report synthesis.

## Artifacts To Read

Read the run directory:

- `summary.json`: wrapper status, workflow mode, subprocess command, and artifact pointers.
- `result.json`: review pipeline final payload and stage statuses.
- `report.md`: user-facing final report or compact meta-review.
- `idea_context/idea_context.json`: normalized idea/PDF context and structured idea extraction.
- `search/result.json`: KG/S2 search evidence.
- `manifest/manifest.json`: papers selected for paragraph extraction.
- `grounding/result.json`: paragraph grounding and experiment grounding.
- `rubric_sources/result.json`, `rubric_llm/result.json`: rubric evidence when available.
- `review/reviewer_reviews.index.json`: reviewer-level evaluations.
- `report/final_report.md`, `report/final_report.json`: native final outputs.
- `logs/*.log` and `logs/review_pipeline.*.txt`: stage logs when anything is partial or failed.

Treat `partial_error` as usable when the final report exists, but disclose which branch failed or was skipped.

## Deliverable

Return:

- exact command used, with credentials omitted;
- workflow mode and artifact paths;
- overall go/revise/no-go recommendation;
- novelty, feasibility, soundness, and differentiation assessment;
- closest-prior-art risk;
- strongest supporting evidence and biggest missing evidence;
- concrete revision advice and next SciAtlas query.

Keep judgments tied to retrieved papers, grounding snippets, reviewer reports, or rubric artifacts.
