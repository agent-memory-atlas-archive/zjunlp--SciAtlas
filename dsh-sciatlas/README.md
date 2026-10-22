# dsh-sciatlas

English · [中文](README.zh.md)

SciAtlas research tools for the [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness). Version 0.2 provides five retrieval tools and six research workflows in TypeScript. Workflows use DSH models, credentials, filesystem input and owned background jobs. Python, PyTorch, FAISS and local embedding models are not runtime dependencies.

| Tool | Behavior |
|---|---|
| `sciatlas_search_papers` | Graph retrieval, keyword/title/reference anchors, strict field/date filters, bias and ranking |
| `sciatlas_paper_search` | Low-level `/v1/papers/search`, independent mode/field controls; default topK 5 |
| `sciatlas_related_authors` | Topic-related researchers, metrics and optional supporting papers |
| `sciatlas_author_papers` | Publications by name/OpenAlex id; merge, dedupe, abstract and embedding controls |
| `sciatlas_support_papers` | Topic-supporting publications per candidate author |
| `sciatlas_idea_grounding` | Precision retrieval and a prior-art comparison guide |
| `sciatlas_trend_report` | Impact-oriented retrieval and publication timeline |
| `sciatlas_researcher_review` | Author-paper seeds followed by title-anchored hybrid graph retrieval |
| `sciatlas_literature_review` | Multiround retrieval, clustering, coverage gaps, bibliography/outline/packs/full review |
| `sciatlas_idea_evaluate` | Idea/PDF context, passage evidence, historical rubric, simulated reviewers and meta-review |
| `sciatlas_idea_generate` | Seeds, research graph, trend, RSS, inspirations, analogy, selection, ideas and novelty feedback |

## Install and credentials

From npm, into the `web` profile:

```sh
dsh plugin --profile web add dsh-sciatlas
dsh web
```

For development, use the sibling DSH checkout for linked development dependencies:

```sh
cd dsh-sciatlas
pnpm install
pnpm run build
```

Development overlay, from `deepseek-harness/`:

```sh
pnpm dsh web --patch ../dsh-sciatlas/cordis.dev.yml
```

Persistent bundle installation from a local checkout, after building:

```sh
dsh plugin --profile web add /absolute/path/to/dsh-sciatlas
dsh web
```

Register a SciAtlas token at <http://sciatlas.openkg.cn/register>. Set `SCIATLAS_API_KEY` before launching DSH or store that named credential in `$DSH_HOME/.credentials.yaml`. The credential store wins over the launching environment. Do not pass secrets as tool arguments.

The three long workflows require a registered DSH LLM adapter. They inherit the provider/model from the caller's logged request header. To choose another registered route, set both `research.llmProvider` and `research.llmModel`. Background execution requires DSH jobs and its generic control tools. PDF/text-file input requires a DSH filesystem provider; PDF parsing also requires GROBID.

## Configuration

Under the plugin row's `config`:

```yaml
baseUrl: http://sciatlas.openkg.cn
apiKeyEnv: SCIATLAS_API_KEY
timeoutMs: 360000
defaultTopK: 10
maxTopK: 50
abstractMaxChars: 4000
research:
  runsDir: runs/sciatlas-dsh
  background: true
  # Leave both blank to inherit the logged caller route.
  llmProvider: ''
  llmModel: ''
  grobidBaseUrl: '' # e.g. http://127.0.0.1:8070
  semanticScholarEnabled: true
  semanticScholarKeyEnv: S2_API_KEY
  maxConcurrentRuns: 2
  maxStageConcurrency: 3
  maxModelCalls: 150
  maxSearchCalls: 100
  maxPapers: 400
  maxInputChars: 200000
  maxOutputTokens: 8192
  llmTimeoutMs: 180000
  workflowTimeoutMs: 7200000
  maxPdfBytes: 20971520
  pdfTimeoutMs: 180000
```

Paths are resolved relative to the DSH process working directory. Optional Semantic Scholar search/reference retrieval uses its named credential when available; external failures are retained as limitations. Set `semanticScholarEnabled: false` for SciAtlas-only acquisition. [Implementation and parameter coverage](docs/implementation.md) lists every configuration default and migration difference.

## Calls and results

Use `query` for topic/idea text, `idea` as its alias, or `queryFile` for UTF-8 input. Defaults are `workflow: "flash"` and background execution for the three long workflows.

```json
{"query":"molecular graph generalization","workflow":"flash"}
```

For `sciatlas_literature_review`, flash stops at evidence packs; full writes a review. Set `options.reportStopAfter` to `bibliography`, `outline`, `packs` or `full` explicitly:

```json
{"query":"molecular graph generalization","workflow":"full","options":{"subjectDomain":"chemistry","reportStopAfter":"full"}}
```

For `sciatlas_idea_evaluate`, choose idea text or exactly one PDF:

```json
{"pdf":"papers/proposal.pdf","workflow":"flash","options":{"disableReviewLlm":true}}
```

`disableReviewLlm` creates an evidence inventory instead of simulated reviewer prose; other planning/rubric stages still use the model. Generation accepts a topic and optional `pdfs`:

```json
{"query":"causal molecular graph models","pdfs":["papers/seed.pdf"],"workflow":"full","options":{"ideaCount":2,"graphBudgetMin":12,"graphBudgetMax":30}}
```

A background call returns `kind`, `jobId`, `runId`, `status`, `artifactDir`, `reportPath` and `resultPath`. Use the installed DSH job control tools to read, wait or cancel by `jobId`. Set `background:false` for a foreground call. Artifacts may not exist until their stages complete; a returned path is not a completion claim.

Every run persists `manifest.json`, validated stage checkpoints, `papers.json`, `result.json` and `report.md`; failures retain `error.json` and successful checkpoints. Raw requests/responses and TEI are saved by default. `options.saveIntermediate:false` suppresses raw diagnostic copies while keeping checkpoints and scientific output needed for resume. Model input/chunks remain in the DSH session log.

Resume with the same research tool in the same DSH session:

```json
{"resumeRunId":"research-..."}
```

Resume validates owner, pipeline/prompt versions, effective inputs/options, file byte digests, model route and configuration. It reuses successful checkpoints and retries failed stages under cumulative budgets. Changed inputs require a new run. Native job ids do not survive restart; plugin checkpoints do, provided DSH restores the same session identity. Resume is explicit.

## Evidence and migration scope

GROBID `processFulltextDocument` output supplies section/paragraph evidence. Abstracts remain explicitly `abstract`, unreadable scans are rejected, and page numbers are not invented. Generated citation ids must resolve to acquired papers/passages; bibliography links come from acquired metadata. Reviewers are simulated perspectives, not opinions attributed to real people. Research graph output separates retrieved, inferred and independently acquired bibliographic citation edges.

This is a TypeScript migration of scientific behavior, not byte-identical Python output. Free-text CLI flag inference is replaced by explicit tool fields. Local Neo4j, embedding-space expansion, auxiliary CLI commands (`health/config/build-plan/make-report/skill`) and arbitrary Python-run-directory resume are outside this release.

## Verification

```sh
pnpm test
pnpm run typecheck
pnpm run build
```

Offline tests use actual mock HTTP servers, fixed model tapes and real Cordis/DSH model, session and job services. Explicit network checks are separate:

```sh
SCIATLAS_LIVE_ENV=../SciAtlas/.env pnpm run test:live
# Or export SCIATLAS_API_KEY without a file.
```

Live tests make small calls to all five retrieval endpoints and persist only timing/count summaries in `runs/verification/live-retrieval.json`. Normal tests skip them. They do not prove live model/PDF scientific quality. The approved plan is [here](../docs/plans/2026-10-11-dsh-sciatlas-enhancement.md); migration source hashes are in [migration-baseline.json](docs/migration-baseline.json). Development-only prompt export: `python3 scripts/sync-prompts.py`.
