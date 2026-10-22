/**
 * `sciatlas_search_papers`: graph-aware paper search against the hosted
 * SciAtlas API (`POST /v1/search`). The model supplies structured anchors
 * (keywords, titles, references) directly; no natural-language plan parsing
 * is involved.
 *
 * @module dsh-sciatlas/tools/search-papers
 */

import { defineTool } from '@deepseek-ai/dsh-tools'
import { postJson } from '../client.ts'
import { collectPapers } from '../normalize.ts'
import type { NormalizedPaper } from '../normalize.ts'
import {
  PAPER_VALUE_SCHEMA,
  RELEVANCE_VALUE_SCHEMA,
  buildSearchOptions,
  buildSearchPlan,
  clampTopK,
  renderBudgetNotice,
  renderPaperList,
  requireNonBlank,
  truncateText,
  withRanks,
} from './shared.ts'
import type { SciAtlasToolDeps } from './shared.ts'
import type { OptionsInput, PlanInput } from './shared.ts'

/** Output value of one `sciatlas_search_papers` call. */
export interface SearchPapersOutput {
  requestedCount?: number
  effectiveCount?: number
  truncated?: boolean
  query: string
  total: number
  elapsedSeconds: number
  papers: (NormalizedPaper & { rank: number })[]
}

/**
 * Create the `sciatlas_search_papers` tool definition.
 *
 * @param deps - plugin configuration and the call-time API key resolver.
 * @returns the registry-ready tool definition.
 */
export function createSearchPapersTool(deps: SciAtlasToolDeps) {
  return defineTool({
    name: 'sciatlas_search_papers',
    description:
      'Search scientific papers from the hosted SciAtlas knowledge graph, which links papers to authors, venues, citations, and a four-level research taxonomy. Use for literature evidence: related work, prior-art checks, or a topic overview. Prefer explicit keyword/title anchors over a bare query when the user names concrete concepts or papers. Returns ranked papers with title, year, venue, authors, citation count, landing/DOI/PDF links, and abstract. Requires a SciAtlas API key (register at http://sciatlas.openkg.cn/register).',
    parameters: {
      targetField: { type: 'string', description: 'Strict backend field filter; prefer domain for a soft hint.' },
      query: {
        type: 'string',
        required: true,
        description: 'Research topic or question, preferably in English.',
      },
      keywords: {
        type: 'array',
        description: 'Keyword anchors steering graph retrieval; repeat for each concept the results must stay close to.',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            text: { type: 'string', required: true, description: 'Keyword phrase, e.g. "retrieval augmented generation".' },
            relevance: RELEVANCE_VALUE_SCHEMA,
          },
        },
      },
      titles: {
        type: 'array',
        description: 'Exact paper title anchors; use for papers the user explicitly names.',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            title: { type: 'string', required: true },
            relevance: RELEVANCE_VALUE_SCHEMA,
          },
        },
      },
      referenceTitles: {
        type: 'array',
        description: 'Titles of cited works to anchor on; high/middle relevance also adds them as title anchors.',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            title: { type: 'string', required: true },
            relevance: RELEVANCE_VALUE_SCHEMA,
          },
        },
      },
      domain: {
        type: 'string',
        description: 'Soft domain hint appended to the query text, e.g. "artificial intelligence".',
      },
      timeRange: {
        type: 'object',
        additionalProperties: false,
        description: 'Publication-date window; bounds are inclusive YYYY-MM-DD dates.',
        properties: {
          after: { type: 'string', description: 'Start date, YYYY-MM-DD.' },
          before: { type: 'string', description: 'End date, YYYY-MM-DD.' },
        },
      },
      topK: {
        type: 'integer',
        description: `Number of papers to return (1–${deps.maxTopK}; default ${deps.defaultTopK}).`,
      },
      retrievalMode: {
        type: 'string',
        enum: ['keyword', 'semantic', 'title', 'hybrid'],
        description: `Retrieval path mix; hybrid combines all paths (default ${deps.defaultRetrievalMode}).`,
      },
      rankingProfile: {
        type: 'string',
        enum: ['precision', 'balanced', 'discovery', 'impact'],
        description: 'Final ranking taste: precision for closest matches, discovery for breadth, impact for highly cited work.',
      },
      bias: {
        type: 'object',
        additionalProperties: false,
        description: 'Graph-walk strengths; omit to use backend defaults.',
        properties: {
          keyword: { ...RELEVANCE_VALUE_SCHEMA, description: 'Keyword-association strength.' },
          nonSeedKeyword: { ...RELEVANCE_VALUE_SCHEMA, description: 'Expansion into non-seed keywords.' },
          citation: { ...RELEVANCE_VALUE_SCHEMA, description: 'Citation-edge following strength.' },
          related: { ...RELEVANCE_VALUE_SCHEMA, description: 'RELATED_TO paper-relation strength.' },
          authorship: { ...RELEVANCE_VALUE_SCHEMA, description: 'Author–paper edge strength.' },
          coauthorship: { ...RELEVANCE_VALUE_SCHEMA, description: 'Co-author network strength.' },
          cooccurrence: { ...RELEVANCE_VALUE_SCHEMA, description: 'Keyword co-occurrence network strength.' },
          exploration: { ...RELEVANCE_VALUE_SCHEMA, description: 'Graph exploration breadth; high wanders further from the seed anchors.' },
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
          requestedCount: { type: 'integer' },
          effectiveCount: { type: 'integer' },
          truncated: { type: 'boolean' },
          papers: { type: 'array', required: true, items: PAPER_VALUE_SCHEMA },
        },
      },
      render: (_args, value) => [
        {
          type: 'text',
          text: `SciAtlas paper search for "${truncateText(value.query, 120)}" — ${value.total} paper(s):\n\n${renderBudgetNotice(value)}${renderPaperList(value.papers)}`,
        },
      ],
      presentationMeta: (_args, value) => ({
        query: truncateText(value.query, 120),
        count: value.total,
        topTitles: value.papers.slice(0, 3).map(paper => paper.title),
      }),
    },
    timeoutMs: deps.timeoutMs + 30_000,
    isConcurrencySafe: () => true,
    async execute(args, exec): Promise<SearchPapersOutput> {
      const query = requireNonBlank(args.query, 'query')
      const planInput: PlanInput = {
        query,
        ...(args.domain !== undefined ? { domain: args.domain } : {}),
        ...(args.keywords !== undefined ? { keywords: args.keywords } : {}),
        ...(args.titles !== undefined ? { titles: args.titles } : {}),
        ...(args.referenceTitles !== undefined ? { referenceTitles: args.referenceTitles } : {}),
      }
      const optionsInput: OptionsInput = {
        ...(args.targetField !== undefined ? { targetField: args.targetField } : {}),
        ...(args.topK !== undefined ? { topK: args.topK } : {}),
        ...(args.retrievalMode !== undefined ? { retrievalMode: args.retrievalMode } : {}),
        ...(args.timeRange !== undefined ? { timeRange: args.timeRange } : {}),
        ...(args.rankingProfile !== undefined ? { rankingProfile: args.rankingProfile } : {}),
        ...(args.bias !== undefined ? { bias: args.bias } : {}),
      }
      const plan = buildSearchPlan(planInput)
      const options = buildSearchOptions(optionsInput, deps)
      const apiKey = await deps.resolveApiKey()
      const started = performance.now()
      const body = await postJson('/v1/search', { plan, options }, {
        baseUrl: deps.baseUrl,
        apiKey,
        timeoutMs: deps.timeoutMs,
        signal: exec.signal,
      })
      const papers = withRanks(collectPapers(body, {
        maxItems: clampTopK(args.topK, deps.defaultTopK, deps.maxTopK),
        abstractMaxChars: deps.abstractMaxChars,
      }))
      return {
        requestedCount: args.topK ?? deps.defaultTopK,
        effectiveCount: clampTopK(args.topK, deps.defaultTopK, deps.maxTopK),
        truncated: (args.topK ?? deps.defaultTopK) > deps.maxTopK,
        query,
        total: papers.length,
        elapsedSeconds: Math.round(((performance.now() - started) / 1000) * 1000) / 1000,
        papers,
      }
    },
    presentCall: args => ({
      card: 'generic',
      kind: 'search',
      title: `SciAtlas papers: ${truncateText(args.query, 80)}`,
    }),
    presentResult: (_args, result) => {
      const meta = result.meta
      if (result.isError || typeof meta !== 'object' || meta === null || Array.isArray(meta)) return undefined
      const count = (meta as Record<string, unknown>)['count']
      return typeof count === 'number' ? { card: 'generic', title: `SciAtlas: ${count} paper(s)` } : undefined
    },
  })
}
