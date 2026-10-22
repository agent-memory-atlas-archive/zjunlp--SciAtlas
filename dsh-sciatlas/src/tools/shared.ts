/**
 * Shared pieces of the four SciAtlas tools: dependency bag passed down from
 * the plugin entry, search-plan/options assembly mirroring the SciAtlas CLI's
 * expert-mode behavior, the paper value schema reused across tool outputs,
 * and plain-text renderers for model-facing results.
 *
 * @module dsh-sciatlas/tools/shared
 */

import type { Relevance } from '../relevance.ts'
import {
  KEYWORD_SCORE_BY_RELEVANCE,
  TITLE_CONFIDENCE_BY_RELEVANCE,
  referenceAnchorConfidence,
} from '../relevance.ts'
import type { NormalizedAuthor, NormalizedPaper, NormalizedSupportAuthor } from '../normalize.ts'

/** Services and configuration handed to each tool factory by the plugin entry. */
export interface SciAtlasToolDeps {
  /** Normalized SciAtlas API base URL. */
  baseUrl: string
  /** Per-request timeout in milliseconds. */
  timeoutMs: number
  /** Default result count when a call carries no explicit topK/limit. */
  defaultTopK: number
  /** Hard cap applied to every result count. */
  maxTopK: number
  /** Default retrieval mode for search calls. */
  defaultRetrievalMode: 'keyword' | 'semantic' | 'title' | 'hybrid'
  /** Truncation bound for abstracts in normalized papers. */
  abstractMaxChars: number
  /** Resolve the SciAtlas API key at call time; throws an actionable error when unconfigured. */
  resolveApiKey: () => Promise<string>
}

/** One keyword anchor argument. */
export interface AnchorKeyword {
  text: string
  relevance?: Relevance | undefined
}

/** One title anchor argument. */
export interface AnchorTitle {
  title: string
  relevance?: Relevance | undefined
}

/** One reference-title anchor argument. */
export interface AnchorReference {
  title: string
  relevance?: Relevance | undefined
}

/** Date window argument (inclusive bounds, `YYYY-MM-DD`). */
export interface TimeRangeInput {
  after?: string | undefined
  before?: string | undefined
}

/** Graph-walk bias argument; every key maps to one `retrieval_bias` field. */
export interface BiasInput {
  keyword?: Relevance | undefined
  nonSeedKeyword?: Relevance | undefined
  citation?: Relevance | undefined
  related?: Relevance | undefined
  authorship?: Relevance | undefined
  coauthorship?: Relevance | undefined
  cooccurrence?: Relevance | undefined
  exploration?: Relevance | undefined
}

const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/

/**
 * Read a required, non-empty string argument. The parameter schema enforces
 * presence and type; this enforces the non-blank constraint the DSL cannot
 * express.
 *
 * @param value - the schema-validated string.
 * @param field - parameter name for the error message.
 * @returns the trimmed string.
 */
export function requireNonBlank(value: string, field: string): string {
  const trimmed = value.trim()
  if (!trimmed) throw new Error(`Invalid arguments: "${field}" must be a non-empty string.`)
  return trimmed
}

/**
 * Validate a `YYYY-MM-DD` date bound.
 *
 * @param value - the configured date string.
 * @param field - parameter name for the error message.
 * @returns the validated string.
 */
export function requireDate(value: string, field: string): string {
  const trimmed = value.trim()
  if (!DATE_PATTERN.test(trimmed) || Number.isNaN(Date.parse(trimmed)) || new Date(trimmed).toISOString().slice(0, 10) !== trimmed) {
    throw new Error(`Invalid arguments: "${field}" must be a YYYY-MM-DD date, got "${value}".`)
  }
  return trimmed
}

/**
 * Clamp a requested result count into `[1, maxTopK]`.
 *
 * @param requested - the caller's requested count, or `undefined` for the default.
 * @param defaultTopK - fallback when `requested` is absent.
 * @param maxTopK - hard cap.
 * @returns the effective count.
 */
export function clampTopK(requested: number | undefined, defaultTopK: number, maxTopK: number): number {
  const value = requested ?? defaultTopK
  return Math.max(1, Math.min(maxTopK, Math.trunc(value)))
}

/** Plan assembly input shared by paper search and related-author retrieval. */
export interface PlanInput {
  query: string
  domain?: string | undefined
  keywords?: readonly AnchorKeyword[] | undefined
  titles?: readonly AnchorTitle[] | undefined
  referenceTitles?: readonly AnchorReference[] | undefined
}

/** The `plan` object sent to `/v1/search` and `/v1/authors/related`. */
export interface SearchPlan {
  query_text: string
  source_type: 'idea_text'
  source_title: null
  keywords: { text: string; score: number }[]
  titles: { title: string; confidence: number }[]
  reference_titles: string[]
}

/**
 * Assemble a SciAtlas search plan from structured anchors. Domain is appended
 * to the query text as a soft hint (CLI `append_soft_domain_to_query`).
 * Keyword dedupe is upsert-by-text with the later entry winning; title dedupe
 * keeps the highest confidence. High/middle reference anchors also become
 * title anchors (CLI `apply_relevance_items_to_plan`). With no anchors at
 * all, the query itself becomes the single keyword so the backend's
 * keywords-or-titles requirement is satisfied.
 *
 * @param input - the structured plan input.
 * @returns the plan payload.
 */
export function buildSearchPlan(input: PlanInput): SearchPlan {
  const queryText = input.domain?.trim()
    ? `${input.query.trim()} (Domain: ${input.domain.trim()})`
    : input.query.trim()

  const keywords = new Map<string, { text: string; score: number }>()
  for (const anchor of input.keywords ?? []) {
    const text = anchor.text.trim().slice(0, 120)
    if (!text) continue
    keywords.set(text.toLowerCase(), {
      text,
      score: KEYWORD_SCORE_BY_RELEVANCE[anchor.relevance ?? 'high'],
    })
  }

  const titles = new Map<string, { title: string; confidence: number }>()
  const upsertTitle = (title: string, confidence: number): void => {
    const key = title.toLowerCase()
    const existing = titles.get(key)
    if (existing === undefined || confidence > existing.confidence) {
      titles.set(key, { title, confidence })
    }
  }
  for (const anchor of input.titles ?? []) {
    const title = anchor.title.trim()
    if (!title) continue
    upsertTitle(title, TITLE_CONFIDENCE_BY_RELEVANCE[anchor.relevance ?? 'high'])
  }

  const referenceTitles: string[] = []
  const seenReferences = new Set<string>()
  for (const anchor of input.referenceTitles ?? []) {
    const title = anchor.title.trim()
    if (!title) continue
    const key = title.toLowerCase()
    if (!seenReferences.has(key)) {
      seenReferences.add(key)
      referenceTitles.push(title)
    }
    const confidence = referenceAnchorConfidence(anchor.relevance ?? 'high')
    if (confidence !== undefined) upsertTitle(title, confidence)
  }

  if (keywords.size === 0 && titles.size === 0) {
    keywords.set(queryText.toLowerCase(), { text: queryText.slice(0, 120), score: 8 })
  }

  return {
    query_text: queryText,
    source_type: 'idea_text',
    source_title: null,
    keywords: [...keywords.values()],
    titles: [...titles.values()],
    reference_titles: referenceTitles,
  }
}

/** Mapping from bias argument keys to `retrieval_bias` payload fields. */
const BIAS_FIELD_BY_KEY = {
  keyword: 'keyword_association',
  nonSeedKeyword: 'non_seed_keyword_expansion',
  citation: 'citation',
  related: 'paper_relatedness',
  authorship: 'authorship',
  coauthorship: 'coauthorship',
  cooccurrence: 'keyword_cooccurrence',
  exploration: 'graph_exploration',
} as const satisfies Record<keyof BiasInput, string>

/** Options assembly input shared by paper search and related-author retrieval. */
export interface OptionsInput {
  targetField?: string | undefined
  topK?: number | undefined
  retrievalMode?: 'keyword' | 'semantic' | 'title' | 'hybrid' | undefined
  timeRange?: TimeRangeInput | undefined
  rankingProfile?: 'precision' | 'balanced' | 'discovery' | 'impact' | undefined
  bias?: BiasInput | undefined
}

/**
 * Assemble the `options` object for `/v1/search` and `/v1/authors/related`.
 * The ranking profile travels inside `retrieval_bias`, matching the CLI's
 * `build_retrieval_bias_from_args`.
 *
 * @param input - the structured options input.
 * @param deps - plugin configuration for defaults and caps.
 * @returns the options payload.
 */
export function buildSearchOptions(input: OptionsInput, deps: Pick<SciAtlasToolDeps, 'defaultTopK' | 'maxTopK' | 'defaultRetrievalMode'>): Record<string, unknown> {
  const options: Record<string, unknown> = {
    top_k: clampTopK(input.topK, deps.defaultTopK, deps.maxTopK),
    retrieval_mode: input.retrievalMode ?? deps.defaultRetrievalMode,
  }
  if (input.timeRange?.after !== undefined) options['after'] = requireDate(input.timeRange.after, 'timeRange.after')
  if (input.timeRange?.before !== undefined) options['before'] = requireDate(input.timeRange.before, 'timeRange.before')
  if (input.timeRange?.after && input.timeRange.before && input.timeRange.after > input.timeRange.before) {
    throw new Error('Invalid arguments: timeRange.after must not be later than timeRange.before.')
  }
  if (input.targetField !== undefined) options['target_field'] = requireNonBlank(input.targetField, 'targetField')

  const retrievalBias: Record<string, unknown> = {}
  if (input.bias !== undefined) {
    for (const key of Object.keys(BIAS_FIELD_BY_KEY) as (keyof BiasInput)[]) {
      const level = input.bias[key]
      if (level !== undefined) retrievalBias[BIAS_FIELD_BY_KEY[key]] = level
    }
  }
  if (input.rankingProfile !== undefined) retrievalBias['ranking_profile'] = input.rankingProfile
  if (Object.keys(retrievalBias).length > 0) options['retrieval_bias'] = retrievalBias

  return options
}

/** Truncate a text for one-line display. */
export function truncateText(value: string, maxChars: number): string {
  const compacted = value.replace(/\s+/g, ' ').trim()
  if (compacted.length <= maxChars) return compacted
  return `${compacted.slice(0, maxChars - 1)}…`
}

/** Per-paper abstract bound inside model-facing renderings. */
const RENDER_ABSTRACT_CHARS = 400

/** Keep a capped request budget visible in native model result text. */
export function renderBudgetNotice(value: { requestedCount?: number; effectiveCount?: number; truncated?: boolean }): string {
  return value.truncated ? `Result budget capped: requested ${value.requestedCount}, effective ${value.effectiveCount}.\n\n` : ''
}

/** Render rank-ordered papers, or guidance when acquisition returned none. */
export function renderPaperList(papers: readonly NormalizedPaper[]): string {
  if (papers.length === 0) {
    return 'No papers found. Suggest widening the time range, lowering keyword bias, or removing overly specific title anchors.'
  }
  const lines = papers.map((paper, index) => {
    const head = `${index + 1}. ${paper.title}${paper.year === undefined ? '' : ` (${paper.year})`}`
    const facts = [
      paper.venue,
      paper.citationCount === undefined ? undefined : `citations: ${paper.citationCount}`,
      paper.score === undefined ? undefined : `score: ${paper.score.toFixed(3)}`,
    ].filter((part): part is string => part !== undefined)
    const links = [
      paper.url === undefined ? undefined : `link: ${paper.url}`,
      paper.pdfUrl === undefined ? undefined : `PDF: ${paper.pdfUrl}`,
      paper.doi === undefined ? undefined : `DOI: ${paper.doi}`,
    ].filter((part): part is string => part !== undefined)
    const rows = [head]
    if (facts.length > 0) rows.push(`   ${facts.join(' | ')}`)
    if (paper.authors !== undefined && paper.authors.length > 0) rows.push(`   Authors: ${paper.authors.join(', ')}`)
    if (links.length > 0) rows.push(`   ${links.join(' | ')}`)
    if (paper.abstract !== undefined) rows.push(`   Abstract: ${truncateText(paper.abstract, RENDER_ABSTRACT_CHARS)}`)
    return rows.join('\n')
  })
  return lines.join('\n\n')
}

/**
 * Render normalized authors as a numbered plain-text list for the model.
 *
 * @param authors - the normalized authors, already rank-ordered.
 * @returns the model-facing list text, or a guidance line when empty.
 */
export function renderAuthorList(authors: readonly NormalizedAuthor[]): string {
  if (authors.length === 0) {
    return 'No authors found. Suggest widening the time range or lowering keyword bias.'
  }
  return authors
    .map((author, index) => {
      const facts = [
        author.id,
        author.worksCount === undefined ? undefined : `works: ${author.worksCount}`,
        author.citationCount === undefined ? undefined : `citations: ${author.citationCount}`,
        author.hIndex === undefined ? undefined : `h-index: ${author.hIndex}`,
        author.score === undefined ? undefined : `score: ${author.score.toFixed(3)}`,
      ].filter((part): part is string => part !== undefined)
      return facts.length > 0 ? `${index + 1}. ${author.name} (${facts.join(' | ')})` : `${index + 1}. ${author.name}`
    })
    .join('\n')
}

/**
 * Render support-paper entries as sections per author for the model.
 *
 * @param entries - the normalized support-author entries.
 * @returns the model-facing text, or a guidance line when empty.
 */
export function renderSupportAuthors(entries: readonly NormalizedSupportAuthor[]): string {
  if (entries.length === 0) {
    return 'No support papers found for the given authors and topic.'
  }
  return entries
    .map((entry, index) => {
      const label = entry.name ?? entry.id ?? `author ${index + 1}`
      const header = `### ${index + 1}. ${label}${entry.id === undefined ? '' : ` (${entry.id})`}`
      return `${header}\n${renderPaperList(entry.supportPapers)}`
    })
    .join('\n\n')
}

/** The paper value schema shared by every tool output that lists papers. */
export const PAPER_VALUE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    rank: { type: 'integer', required: true, description: '1-based rank in backend order.' },
    title: { type: 'string', required: true },
    id: { type: 'string' },
    titleEmbedding: { type: 'array', items: { type: 'number' } },
    abstractEmbedding: { type: 'array', items: { type: 'number' } },
    year: { type: 'integer' },
    venue: { type: 'string' },
    authors: { type: 'array', items: { type: 'string' } },
    citationCount: { type: 'number' },
    score: { type: 'number' },
    doi: { type: 'string' },
    url: { type: 'string', description: 'Landing page (DOI resolver, OpenAlex, publisher) — never a PDF.' },
    pdfUrl: { type: 'string' },
    abstract: { type: 'string' },
  },
} as const

/** The relevance enum schema shared by anchor inputs. */
export const RELEVANCE_VALUE_SCHEMA = {
  type: 'string',
  enum: ['low', 'middle', 'high'],
  description: 'Anchor strength: high keeps the term central, middle marks a supporting concept, low is loose context.',
} as const

/** Add 1-based ranks to normalized records for the output schema. */
export function withRanks<T>(items: readonly T[]): (T & { rank: number })[] {
  return items.map((item, index) => ({ ...item, rank: index + 1 }))
}
