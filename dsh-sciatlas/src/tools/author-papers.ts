/**
 * `sciatlas_author_papers`: list one researcher's papers from the hosted
 * SciAtlas API (`POST /v1/authors/papers`). The identifier may be a personal
 * name or an OpenAlex author id (`A...`); id detection mirrors the SciAtlas
 * CLI.
 *
 * @module dsh-sciatlas/tools/author-papers
 */

import { defineTool } from '@deepseek-ai/dsh-tools'
import { postJson } from '../client.ts'
import { collectAuthors, collectPapers } from '../normalize.ts'
import type { NormalizedPaper } from '../normalize.ts'
import {
  PAPER_VALUE_SCHEMA,
  clampTopK,
  renderBudgetNotice,
  renderPaperList,
  requireNonBlank,
  truncateText,
  withRanks,
} from './shared.ts'
import type { SciAtlasToolDeps } from './shared.ts'

/** Output value of one `sciatlas_author_papers` call. */
export interface AuthorPapersOutput {
  author?: { name?: string; id?: string }
  total: number
  elapsedSeconds: number
  requestedCount?: number
  effectiveCount?: number
  truncated?: boolean
  papers: (NormalizedPaper & { rank: number })[]
}

/** OpenAlex author id (`A123...`) or a URL containing one. */
const OPENALEX_ID_PATTERN = /^[A-Za-z]\d+$/

/**
 * Decide the SciAtlas `search_by` mode for one author identifier.
 *
 * @param identifier - author name, OpenAlex id, or OpenAlex URL.
 * @returns `'id'` for OpenAlex-shaped identifiers, else `'name'`.
 */
export function authorSearchMode(identifier: string): 'name' | 'id' {
  const trimmed = identifier.trim()
  if (trimmed.toLowerCase().includes('openalex.org')) return 'id'
  return OPENALEX_ID_PATTERN.test(trimmed) ? 'id' : 'name'
}

/**
 * Create the `sciatlas_author_papers` tool definition.
 *
 * @param deps - plugin configuration and the call-time API key resolver.
 * @returns the registry-ready tool definition.
 */
export function createAuthorPapersTool(deps: SciAtlasToolDeps) {
  return defineTool({
    name: 'sciatlas_author_papers',
    description:
      'List papers by one researcher from the hosted SciAtlas knowledge graph. Accepts a personal name or OpenAlex author id. Same-name merging and deduplication default to enabled and can be disabled. Use for publication histories. Requires a SciAtlas API key.',
    parameters: {
      mergeSameNameAuthors: { type: 'boolean', description: 'Merge same-name authors (default true).' },
      dedupePapers: { type: 'boolean', description: 'Deduplicate papers (default true).' },
      includeEmbeddings: { type: 'boolean', description: 'Return available title/abstract embeddings (default false).' },
      author: {
        type: 'string',
        required: true,
        description: 'Author name or OpenAlex author id (A... / openalex.org URL).',
      },
      limit: {
        type: 'integer',
        description: `Maximum papers to return (1–${deps.maxTopK}; default ${deps.defaultTopK}).`,
      },
      includeAbstract: {
        type: 'boolean',
        description: 'Include paper abstracts in the result (default true).',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          author: {
            type: 'object',
            additionalProperties: false,
            properties: {
              name: { type: 'string' },
              id: { type: 'string' },
            },
          },
          total: { type: 'integer', required: true },
          elapsedSeconds: { type: 'number', required: true },
          requestedCount: { type: 'integer' }, effectiveCount: { type: 'integer' }, truncated: { type: 'boolean' },
          papers: { type: 'array', required: true, items: PAPER_VALUE_SCHEMA },
        },
      },
      render: (_args, value) => {
        const label = value.author?.name ?? value.author?.id ?? 'requested author'
        return [{
          type: 'text',
          text: `SciAtlas papers by ${label} — ${value.total} paper(s):\n\n${renderBudgetNotice(value)}${renderPaperList(value.papers)}`,
        }]
      },
      presentationMeta: (_args, value) => ({
        author: value.author?.name ?? value.author?.id ?? null,
        count: value.total,
        topTitles: value.papers.slice(0, 3).map(paper => paper.title),
      }),
    },
    timeoutMs: deps.timeoutMs + 30_000,
    isConcurrencySafe: () => true,
    async execute(args, exec): Promise<AuthorPapersOutput> {
      const author = requireNonBlank(args.author, 'author')
      const limit = clampTopK(args.limit, deps.defaultTopK, deps.maxTopK)
      const payload = {
        identifier: author,
        search_by: authorSearchMode(author),
        options: {
          limit,
          merge_same_name_authors: args.mergeSameNameAuthors ?? true,
          dedupe_papers: args.dedupePapers ?? true,
          include_abstract: args.includeAbstract ?? true,
          include_embeddings: args.includeEmbeddings ?? false,
        },
      }
      const apiKey = await deps.resolveApiKey()
      const started = performance.now()
      const body = await postJson('/v1/authors/papers', payload, {
        baseUrl: deps.baseUrl,
        apiKey,
        timeoutMs: deps.timeoutMs,
        signal: exec.signal,
      })
      const papers = withRanks(collectPapers(body, { maxItems: limit, abstractMaxChars: deps.abstractMaxChars, dedupe: args.dedupePapers ?? true, includeEmbeddings: args.includeEmbeddings ?? false }))
      if (args.includeAbstract === false) for (const paper of papers) delete paper.abstract
      const resolvedAuthor = collectAuthors(body, 1)[0]
      return {
        ...resolvedAuthor !== undefined
          ? {
            author: {
              ...resolvedAuthor.name ? { name: resolvedAuthor.name } : {},
              ...resolvedAuthor.id !== undefined ? { id: resolvedAuthor.id } : {},
            },
          }
          : {},
        total: papers.length,
        requestedCount: args.limit ?? deps.defaultTopK, effectiveCount: limit, truncated: (args.limit ?? deps.defaultTopK) > deps.maxTopK,
        elapsedSeconds: Math.round(((performance.now() - started) / 1000) * 1000) / 1000,
        papers,
      }
    },
    presentCall: args => ({
      card: 'generic',
      kind: 'search',
      title: `SciAtlas author: ${truncateText(args.author, 80)}`,
    }),
    presentResult: (_args, result) => {
      const meta = result.meta
      if (result.isError || typeof meta !== 'object' || meta === null || Array.isArray(meta)) return undefined
      const count = (meta as Record<string, unknown>)['count']
      return typeof count === 'number' ? { card: 'generic', title: `SciAtlas: ${count} paper(s)` } : undefined
    },
  })
}
