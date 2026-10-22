/** Research workflow inputs and services, independent of DSH presentation. */
import type { NormalizedPaper, NormalizedAuthor } from '../normalize.ts'
import type { OptionsInput, PlanInput, SciAtlasToolDeps } from '../tools/shared.ts'
import type { Json, JsonObject } from './json.ts'

export type ResearchCommand = 'idea-grounding' | 'trend-report' | 'researcher-review' | 'literature-review' | 'idea-evaluate' | 'idea-generate'
export type WorkflowMode = 'flash' | 'full'

/** A stable paper identity attached to every evidence/citation reference. */
export interface Paper extends NormalizedPaper {
  paperId: string
  sources: string[]
}

/** An acquired passage; abstract evidence is never represented as full text. */
export interface Evidence {
  evidenceId: string
  paperId: string
  section: string[]
  paragraphIndex: number
  text: string
  level: 'abstract' | 'fulltext'
  source: string
}

/** Structured stage requests sent through the caller's DSH model service. */
export interface ModelRequest {
  stage: string
  system: string
  input: Json
  signal: AbortSignal
  timeoutMs?: number
}

export interface ModelRoute { provider: string; model: string }

/** Plugin deployment limits, separate from per-run research options. */
export interface ResearchConfig {
  runsDir: string
  llmProvider: string
  llmModel: string
  llmTimeoutMs: number
  maxOutputTokens: number
  maxInputChars: number
  workflowTimeoutMs: number
  maxConcurrentRuns: number
  maxStageConcurrency: number
  maxModelCalls: number
  maxSearchCalls: number
  maxPapers: number
  maxPdfBytes: number
  grobidBaseUrl: string
  pdfTimeoutMs: number
  background: boolean
  semanticScholarEnabled: boolean
  semanticScholarBaseUrl: string
  semanticScholarKeyEnv: string
}

/** Model-controlled options accepted at the tool boundary. */
export interface ResearchOptions {
  probeTopK?: number
  roundTopK?: number
  round1ActionLimit?: number
  round2ActionLimit?: number
  llmPaperLimit?: number
  subjectDomain?: 'general' | 'chemistry' | 'biology'
  reportStopAfter?: 'bibliography' | 'outline' | 'packs' | 'full'
  kgTopK?: number
  s2TopK?: number
  manifestTopK?: number
  groundingFinalTopK?: number
  groundingCandidateK?: number
  groundingRerankTopN?: number
  maxParagraphsPerPaperPerQuery?: number
  rerankBatchSize?: number
  maxReviewers?: number
  reviewerSelectionTopK?: number
  reviewerMaxWorkers?: number
  reviewMaxWorkers?: number
  maxEvidenceCardsPerSection?: number
  maxEvidenceCardsPerQuery?: number
  maxExperimentCardsPerSection?: number
  maxEvidenceCharsPerSection?: number
  shortReport?: boolean
  disableLlmRanking?: boolean
  disableReviewLlm?: boolean
  seedRecentYears?: number
  graphBudgetMax?: number
  graphBudgetMin?: number
  kStep1?: number
  seedNumQueries?: number
  seedPerQuery?: number
  anchorTopK?: number
  ideaCount?: number
  maxNoveltyFeedbackRounds?: number
  llmTimeoutMs?: number
  outlineTimeoutMs?: number
  integrationTimeoutMs?: number
  reviewTimeoutMs?: number
  rubricTimeoutMs?: number
  saveIntermediate?: boolean
}

export interface ResearchInput extends PlanInput, OptionsInput {
  command: ResearchCommand
  workflow: WorkflowMode
  pdfs: string[]
  author?: string
  limit?: number
  mergeSameNameAuthors?: boolean
  dedupePapers?: boolean
  includeAbstract?: boolean
  includeEmbeddings?: boolean
  options: ResearchOptions
  smoke: boolean
}

/** One run's external capabilities; all operations belong to its cancellation signal. */
export interface ResearchServices {
  config: ResearchConfig
  api: SciAtlasToolDeps
  route: ModelRoute | null
  model(request: ModelRequest): Promise<string>
  readFile(path: string, signal: AbortSignal): Promise<Uint8Array>
  /** Fetch a public PDF without SciAtlas or model credentials. */
  fetchPdf(url: string, signal: AbortSignal): Promise<Uint8Array>
  semanticSearch(query: string, limit: number, signal: AbortSignal): Promise<SearchResult>
  referenceSearch(paper: Paper, limit: number, signal: AbortSignal): Promise<SearchResult>
}

/** A serializable workflow result; complete research data stays in its artifacts. */
export interface ResearchResult {
  status: 'completed' | 'partial' | 'failed' | 'cancelled'
  summary: string
  papers: Paper[]
  warnings: string[]
  report: string
  data: JsonObject
}

export interface SearchResult { papers: Paper[]; request: JsonObject; response: Json; authors: NormalizedAuthor[] }
