/**
 * `sciatlas_related_authors`: retrieve authors related to a research
 * direction from the hosted SciAtlas API (`POST /v1/authors/related`).
 *
 * @module dsh-sciatlas/tools/related-authors
 */

import { defineTool } from '@deepseek-ai/dsh-tools'
import { postJson } from '../client.ts'
import { collectAuthors, collectPapers } from '../normalize.ts'
import type { NormalizedAuthor, NormalizedPaper } from '../normalize.ts'
import {
  PAPER_VALUE_SCHEMA,
  RELEVANCE_VALUE_SCHEMA,
  buildSearchOptions,
  buildSearchPlan,
  clampTopK,
  renderBudgetNotice,
  renderAuthorList,
  renderPaperList,
  requireNonBlank,
  truncateText,
  withRanks,
} from './shared.ts'
import type { SciAtlasToolDeps } from './shared.ts'
import type { OptionsInput, PlanInput } from './shared.ts'

/** Output value of one `sciatlas_related_authors` call. */
export interface RelatedAuthorsOutput {
  query: string
  total: number
  elapsedSeconds: number
  requestedCount?: number
  effectiveCount?: number
  truncated?: boolean
  authors: (NormalizedAuthor & { rank: number })[]
  supportingPapers?: (NormalizedPaper & { rank: number })[]
}

/** Cap on supporting papers surfaced alongside the author list. */
const MAX_SUPPORTING_PAPERS = 10

/**
 * Create the `sciatlas_related_authors` tool definition.
 *
 * @param deps - plugin configuration and the call-time API key resolver.
 * @returns the registry-ready tool definition.
 */
export function createRelatedAuthorsTool(deps: SciAtlasToolDeps) {
  return defineTool({
    name: 'sciatlas_related_authors',
    description:
      'Find researchers working on a topic via the hosted SciAtlas knowledge graph. Use for "who works on X" questions: finding potential collaborators, reviewers, or authors to profile further with sciatlas_author_papers. Returns ranked authors with name, id, and metrics (works, citations, h-index) plus a few supporting papers when available. Requires a SciAtlas API key.',
    parameters: {
      targetField: { type: 'string', description: 'Strict backend field filter.' },
      retrievalMode: { type: 'string', enum: ['keyword', 'semantic', 'title', 'hybrid'] },
      rankingProfile: { type: 'string', enum: ['precision', 'balanced', 'discovery', 'impact'] },
      titles: { type: 'array', items: { type: 'object', additionalProperties: false, properties: { title: { type: 'string', required: true }, relevance: RELEVANCE_VALUE_SCHEMA } } },
      referenceTitles: { type: 'array', items: { type: 'object', additionalProperties: false, properties: { title: { type: 'string', required: true }, relevance: RELEVANCE_VALUE_SCHEMA } } },
      query: {
        type: 'string',
        required: true,
        description: 'Research direction, preferably in English.',
      },
      keywords: {
        type: 'array',
        description: 'Keyword anchors steering graph retrieval.',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            text: { type: 'string', required: true },
            relevance: RELEVANCE_VALUE_SCHEMA,
          },
        },
      },
      domain: {
        type: 'string',
        description: 'Soft domain hint appended to the query text.',
      },
      timeRange: {
        type: 'object',
        additionalProperties: false,
        description: 'Publication-date window for the underlying paper evidence; inclusive YYYY-MM-DD bounds.',
        properties: {
          after: { type: 'string', description: 'Start date, YYYY-MM-DD.' },
          before: { type: 'string', description: 'End date, YYYY-MM-DD.' },
        },
      },
      topK: {
        type: 'integer',
        description: `Number of authors to return (1–${deps.maxTopK}; default 5).`,
      },
      bias: {
        type: 'object',
        additionalProperties: false,
        description: 'Graph-walk strengths; omit to use backend defaults.',
        properties: {
          keyword: { ...RELEVANCE_VALUE_SCHEMA, description: 'Keyword-association strength.' },
          nonSeedKeyword: RELEVANCE_VALUE_SCHEMA,
          related: RELEVANCE_VALUE_SCHEMA,
          cooccurrence: RELEVANCE_VALUE_SCHEMA,
          citation: { ...RELEVANCE_VALUE_SCHEMA, description: 'Citation-edge following strength.' },
          authorship: { ...RELEVANCE_VALUE_SCHEMA, description: 'Author–paper edge strength.' },
          coauthorship: { ...RELEVANCE_VALUE_SCHEMA, description: 'Co-author network strength.' },
          exploration: { ...RELEVANCE_VALUE_SCHEMA, description: 'Graph exploration breadth.' },
        },
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          query: { type: 'string', required: true },
          total: { type: 'integer', required: true },
          elapsedSeconds: { type: 'number', required: true },
          requestedCount: { type: 'integer' }, effectiveCount: { type: 'integer' }, truncated: { type: 'boolean' },
          authors: {
            type: 'array',
            required: true,
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                rank: { type: 'integer', required: true },
                name: { type: 'string', required: true },
                id: { type: 'string' },
                score: { type: 'number' },
                worksCount: { type: 'integer' },
                citationCount: { type: 'number' },
                hIndex: { type: 'number' },
              },
            },
          },
          supportingPapers: { type: 'array', items: PAPER_VALUE_SCHEMA },
        },
      },
      render: (_args, value) => {
        const sections = [`SciAtlas related authors for "${truncateText(value.query, 120)}" — ${value.total} author(s):\n\n${renderBudgetNotice(value)}${renderAuthorList(value.authors)}`]
        if (value.supportingPapers !== undefined && value.supportingPapers.length > 0) {
          sections.push(`Supporting papers:\n\n${renderPaperList(value.supportingPapers)}`)
        }
        return [{ type: 'text', text: sections.join('\n\n') }]
      },
      presentationMeta: (_args, value) => ({
        query: truncateText(value.query, 120),
        count: value.total,
        topAuthors: value.authors.slice(0, 3).map(author => author.name),
      }),
    },
    timeoutMs: deps.timeoutMs + 30_000,
    isConcurrencySafe: () => true,
    async execute(args, exec): Promise<RelatedAuthorsOutput> {
      const query = requireNonBlank(args.query, 'query')
      const planInput: PlanInput = {
        query,
        ...(args.titles !== undefined ? { titles: args.titles } : {}),
        ...(args.referenceTitles !== undefined ? { referenceTitles: args.referenceTitles } : {}),
        ...(args.domain !== undefined ? { domain: args.domain } : {}),
        ...(args.keywords !== undefined ? { keywords: args.keywords } : {}),
      }
      const optionsInput: OptionsInput = {
        ...(args.targetField !== undefined ? { targetField: args.targetField } : {}),
        ...(args.retrievalMode !== undefined ? { retrievalMode: args.retrievalMode } : {}),
        ...(args.rankingProfile !== undefined ? { rankingProfile: args.rankingProfile } : {}),
        topK: clampTopK(args.topK, 5, deps.maxTopK),
        ...(args.timeRange !== undefined ? { timeRange: args.timeRange } : {}),
        ...(args.bias !== undefined ? { bias: args.bias } : {}),
      }
      const plan = buildSearchPlan(planInput)
      const options = buildSearchOptions(optionsInput, deps)
      const apiKey = await deps.resolveApiKey()
      const started = performance.now()
      const body = await postJson('/v1/authors/related', { plan, options }, {
        baseUrl: deps.baseUrl,
        apiKey,
        timeoutMs: deps.timeoutMs,
        signal: exec.signal,
      })
      const authors = withRanks(collectAuthors(body, clampTopK(args.topK, 5, deps.maxTopK)))
      const supportingPapers = withRanks(collectPapers(body, {
        maxItems: MAX_SUPPORTING_PAPERS,
        abstractMaxChars: deps.abstractMaxChars,
      }))
      return {
        query,
        total: authors.length,
        requestedCount: args.topK ?? 5, effectiveCount: clampTopK(args.topK, 5, deps.maxTopK), truncated: (args.topK ?? 5) > deps.maxTopK,
        elapsedSeconds: Math.round(((performance.now() - started) / 1000) * 1000) / 1000,
        authors,
        ...supportingPapers.length > 0 ? { supportingPapers } : {},
      }
    },
    presentCall: args => ({
      card: 'generic',
      kind: 'search',
      title: `SciAtlas authors: ${truncateText(args.query, 80)}`,
    }),
    presentResult: (_args, result) => {
      const meta = result.meta
      if (result.isError || typeof meta !== 'object' || meta === null || Array.isArray(meta)) return undefined
      const count = (meta as Record<string, unknown>)['count']
      return typeof count === 'number' ? { card: 'generic', title: `SciAtlas: ${count} author(s)` } : undefined
    },
  })
}
