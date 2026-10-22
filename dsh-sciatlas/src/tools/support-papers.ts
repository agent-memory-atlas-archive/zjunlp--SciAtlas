/**
 * `sciatlas_support_papers`: for each candidate author, find papers of theirs
 * supporting a given topic (`POST /v1/authors/support-papers`). Author entries
 * accept names or `id:<openalex-id>` prefixes, mirroring the SciAtlas CLI.
 *
 * @module dsh-sciatlas/tools/support-papers
 */

import { defineTool } from '@deepseek-ai/dsh-tools'
import { postJson } from '../client.ts'
import { collectSupportAuthors } from '../normalize.ts'
import type { NormalizedPaper, NormalizedSupportAuthor } from '../normalize.ts'
import {
  PAPER_VALUE_SCHEMA,
  clampTopK,
  renderBudgetNotice,
  renderSupportAuthors,
  requireNonBlank,
  truncateText,
  withRanks,
} from './shared.ts'
import type { SciAtlasToolDeps } from './shared.ts'

/** Output value of one `sciatlas_support_papers` call. */
export interface SupportPapersOutput {
  query: string
  total: number
  elapsedSeconds: number
  requestedCount?: number
  effectiveCount?: number
  truncated?: boolean
  authors: (Omit<NormalizedSupportAuthor, 'supportPapers'> & { supportPapers: (NormalizedPaper & { rank: number })[] })[]
}

/** Fallback strategies the backend accepts for author lookup. */
const AUTHOR_SEARCH_FALLBACKS = ['id_then_name', 'name_then_id', 'none'] as const

/** OpenAlex author id (`A123...`) or a URL containing one. */
const OPENALEX_ID_PATTERN = /^[A-Za-z]\d+$/

/**
 * Parse one author entry into the backend's `{name}` / `{author_id}` form.
 * `id:` prefixes and OpenAlex-shaped values select `author_id`.
 *
 * @param entry - the raw author entry.
 * @returns the backend author selector.
 */
export function parseAuthorEntry(entry: string): { name: string } | { author_id: string } {
  let value = entry.trim()
  if (value.toLowerCase().startsWith('id:')) {
    value = value.slice(3).trim()
    return { author_id: requireNonBlank(value, 'author id') }
  }
  if (value.toLowerCase().includes('openalex.org') || OPENALEX_ID_PATTERN.test(value)) {
    return { author_id: value }
  }
  return { name: value }
}

/**
 * Create the `sciatlas_support_papers` tool definition.
 *
 * @param deps - plugin configuration and the call-time API key resolver.
 * @returns the registry-ready tool definition.
 */
export function createSupportPapersTool(deps: SciAtlasToolDeps) {
  return defineTool({
    name: 'sciatlas_support_papers',
    description:
      'For each candidate author, find their papers supporting a given topic, via the hosted SciAtlas knowledge graph. Use after sciatlas_related_authors to check which candidates actually publish on the topic. Author entries accept names or "id:<OpenAlex id>". Requires a SciAtlas API key.',
    parameters: {
      query: {
        type: 'string',
        required: true,
        description: 'The topic the support papers must be about, preferably in English.',
      },
      authors: {
        type: 'array',
        required: true,
        description: 'Candidate authors; each entry is a name or "id:<OpenAlex id>" (a bare A... id or openalex.org URL also selects id lookup).',
        items: { type: 'string' },
      },
      topKPerAuthor: {
        type: 'integer',
        description: 'Support papers returned per author (default 3).',
      },
      fetchAuthorStats: {
        type: 'boolean',
        description: 'Ask the backend to include author statistics (default true).',
      },
      authorSearchFallback: {
        type: 'string',
        enum: AUTHOR_SEARCH_FALLBACKS,
        description: 'Author lookup fallback when the primary selector misses (default id_then_name).',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          query: { type: 'string', required: true },
          total: { type: 'integer', required: true, description: 'Number of author entries returned.' },
          elapsedSeconds: { type: 'number', required: true },
          requestedCount: { type: 'integer' }, effectiveCount: { type: 'integer' }, truncated: { type: 'boolean' },
          authors: {
            type: 'array',
            required: true,
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                name: { type: 'string' },
                id: { type: 'string' },
                supportPapers: { type: 'array', required: true, items: PAPER_VALUE_SCHEMA },
              },
            },
          },
        },
      },
      render: (_args, value) => [
        {
          type: 'text',
          text: `SciAtlas support papers for "${truncateText(value.query, 120)}" — ${value.total} author entr(ies):\n\n${renderBudgetNotice(value)}${renderSupportAuthors(value.authors)}`,
        },
      ],
      presentationMeta: (_args, value) => ({
        query: truncateText(value.query, 120),
        count: value.total,
        authorNames: value.authors.slice(0, 5).map(entry => entry.name ?? entry.id ?? 'unknown'),
      }),
    },
    timeoutMs: deps.timeoutMs + 30_000,
    isConcurrencySafe: () => true,
    async execute(args, exec): Promise<SupportPapersOutput> {
      const query = requireNonBlank(args.query, 'query')
      const authorEntries = args.authors.map(entry => entry.trim()).filter(entry => entry.length > 0)
      if (authorEntries.length === 0) {
        throw new Error('Invalid arguments: "authors" must contain at least one non-empty entry.')
      }
      const topKPerAuthor = clampTopK(args.topKPerAuthor, 3, deps.maxTopK)
      const payload = {
        query_text: query,
        authors: authorEntries.map(parseAuthorEntry),
        options: {
          top_k_per_author: topKPerAuthor,
          fetch_author_stats: args.fetchAuthorStats ?? true,
          author_search_fallback: args.authorSearchFallback ?? 'id_then_name',
        },
      }
      const apiKey = await deps.resolveApiKey()
      const started = performance.now()
      const body = await postJson('/v1/authors/support-papers', payload, {
        baseUrl: deps.baseUrl,
        apiKey,
        timeoutMs: deps.timeoutMs,
        signal: exec.signal,
      })
      const authors = collectSupportAuthors(body, authorEntries.length, topKPerAuthor, deps.abstractMaxChars)
        .map(entry => ({ ...entry, supportPapers: withRanks(entry.supportPapers) }))
      return {
        query,
        total: authors.length,
        requestedCount: args.topKPerAuthor ?? 3, effectiveCount: topKPerAuthor, truncated: (args.topKPerAuthor ?? 3) > deps.maxTopK,
        elapsedSeconds: Math.round(((performance.now() - started) / 1000) * 1000) / 1000,
        authors,
      }
    },
    presentCall: args => ({
      card: 'generic',
      kind: 'search',
      title: `SciAtlas support papers: ${truncateText(args.query, 80)}`,
    }),
    presentResult: (_args, result) => {
      const meta = result.meta
      if (result.isError || typeof meta !== 'object' || meta === null || Array.isArray(meta)) return undefined
      const count = (meta as Record<string, unknown>)['count']
      return typeof count === 'number' ? { card: 'generic', title: `SciAtlas: support papers for ${count} author(s)` } : undefined
    },
  })
}
