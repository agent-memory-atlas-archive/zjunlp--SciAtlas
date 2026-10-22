/** Shared HTTP retrieval and paper pooling for research workflows. */
import { postJson } from '../client.ts'
import { collectAuthors, collectPapers } from '../normalize.ts'
import { buildSearchOptions, buildSearchPlan } from '../tools/shared.ts'
import type { OptionsInput, PlanInput, SciAtlasToolDeps } from '../tools/shared.ts'
import { authorSearchMode } from '../tools/author-papers.ts'
import { hash, json, object } from './json.ts'
import type { Paper, SearchResult } from './types.ts'

/** Identity uses DOI, backend id, or title, in that order. */
export function paperIdentity(paper: { doi?: string; id?: string; title: string }): string {
  return `p-${hash((paper.doi ?? paper.id ?? paper.title).replace(/^https?:\/\/doi.org\//i, '').trim().toLowerCase()).slice(0, 16)}`
}

/** Pool papers while retaining acquisition sources and richer metadata. */
export function mergePapers(groups: readonly (readonly Paper[])[], max: number): Paper[] {
  const result: Paper[] = []
  const byTitle = new Map<string, Paper>()
  const byId = new Map<string, Paper>()
  for (const group of groups) for (const paper of group) {
    const title = paper.title.toLowerCase().replace(/\s+/g, ' ').trim()
    const existing = byId.get(paper.paperId) ?? byTitle.get(title)
    if (existing) {
      existing.sources = [...new Set([...existing.sources, ...paper.sources])]
      if (!existing.abstract && paper.abstract) existing.abstract = paper.abstract
      if (!existing.pdfUrl && paper.pdfUrl) existing.pdfUrl = paper.pdfUrl
      continue
    }
    if (result.length >= max) continue
    const copy = { ...paper, sources: [...paper.sources] }
    result.push(copy)
    byId.set(copy.paperId, copy)
    byTitle.set(title, copy)
  }
  return result
}

/** Send one authenticated request and retain its reproducible request/response. */
export async function retrieve(
  deps: SciAtlasToolDeps, endpoint: string, payload: unknown, source: string, signal: AbortSignal, max: number,
): Promise<SearchResult> {
  const response = await postJson(endpoint, payload, { baseUrl: deps.baseUrl, apiKey: await deps.resolveApiKey(), timeoutMs: deps.timeoutMs, signal })
  const papers = collectPapers(response, { maxItems: max, abstractMaxChars: Number.MAX_SAFE_INTEGER }).map(paper => ({ ...paper, paperId: paperIdentity(paper), sources: [source] }))
  return { papers, authors: collectAuthors(response, max), request: object(payload), response: json(response) }
}

/** Graph search reuses the exact structured plan and bias mapping of retrieval tools. */
export function graphSearch(deps: SciAtlasToolDeps, input: PlanInput & OptionsInput, source: string, signal: AbortSignal): Promise<SearchResult> {
  const options = buildSearchOptions(input, deps)
  return retrieve(deps, '/v1/search', { plan: buildSearchPlan(input), options }, source, signal, Number(options['top_k']))
}

/** Author seed lookup used by researcher and reviewer background paths. */
export function authorPapers(deps: SciAtlasToolDeps, author: string, limit: number, signal: AbortSignal, options: { mergeSameNameAuthors?: boolean; dedupePapers?: boolean; includeAbstract?: boolean; includeEmbeddings?: boolean } = {}): Promise<SearchResult> {
  return retrieve(deps, '/v1/authors/papers', { identifier: author, search_by: authorSearchMode(author), options: {
    limit, merge_same_name_authors: options.mergeSameNameAuthors ?? true, dedupe_papers: options.dedupePapers ?? true,
    include_abstract: options.includeAbstract ?? true, include_embeddings: options.includeEmbeddings ?? false,
  } }, 'author-seed', signal, limit)
}

/** Compact model cards have stable ids; the full paper pool remains unchanged. */
export function paperCards(papers: readonly Paper[], limit: number, abstractChars = 1200) {
  return papers.slice(0, limit).map(paper => ({
    paperId: paper.paperId, title: paper.title, year: paper.year ?? null,
    citationCount: paper.citationCount ?? null, abstract: paper.abstract?.slice(0, abstractChars) ?? '',
  }))
}
