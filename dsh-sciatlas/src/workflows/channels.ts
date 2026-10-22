/** CLI research channels: retrieval presets and author-seeded graph retrieval. */
import type { ResearchRun } from '../research/runtime.ts'
import { decodePapers, smokePapers } from '../research/runtime.ts'
import { authorPapers, mergePapers } from '../research/search.ts'
import { result } from './common.ts'
import { json } from '../research/json.ts'

/** Execute the three lighter channels without introducing new generative behavior. */
export async function runChannel(run: ResearchRun) {
  const input = run.input
  if (input.command === 'researcher-review') {
    const author = input.author ?? input.query
    if (!author.trim()) throw new Error('researcher-review requires author')
    const limit = Math.max(1, Math.min(input.limit ?? 20, run.services.api.maxTopK))
    const seed = await run.stage('author-seeds', { author, limit: Math.min(limit, 5), merge: input.mergeSameNameAuthors ?? true, dedupe: input.dedupePapers ?? true }, decodePapers, async () => {
      if (input.smoke) return smokePapers(author, 3)
      const lookup = await run.request('author-seeds', () => authorPapers(run.services.api, author, Math.min(limit, 5), run.signal, { mergeSameNameAuthors: input.mergeSameNameAuthors ?? true, dedupePapers: input.dedupePapers ?? true, includeAbstract: false, includeEmbeddings: false }))
      return lookup.papers
    })
    const papers = await run.search('researcher-graph', { query: `Researcher background review for ${author}`, topK: limit, retrievalMode: 'hybrid', keywords: [{ text: author, relevance: 'high' }, { text: `${author} research`, relevance: 'middle' }], titles: seed.map(paper => ({ title: paper.title, relevance: 'high' as const })), rankingProfile: 'impact', bias: { authorship: 'high', citation: 'high', related: 'middle', exploration: 'low' } })
    const ids = new Set(seed.map(paper => paper.paperId))
    const expanded = papers.filter(paper => !ids.has(paper.paperId))
    const merged = mergePapers([seed, papers], run.services.config.maxPapers)
    if (!seed.length) run.warnings.push('Author seed lookup returned no papers; the graph search used author keywords.')
    if (expanded.length) run.warnings.push('Graph-expanded papers are related evidence; they are not verified as authored by the requested researcher.')
    return result('SciAtlas Researcher Review', `Researcher: ${author}. ${seed.length} verified seed paper(s), ${expanded.length} graph-expanded paper(s).`, merged, run.warnings, `## Author Seed Papers\n\n${seed.map(p => `- [${p.paperId}] ${p.title}`).join('\n')}\n\n## Related Graph Evidence\n\n${expanded.map(p => `- [${p.paperId}] ${p.title}`).join('\n')}`, { author, seedPapers: seed, relatedPapers: papers })
  }
  const trend = input.command === 'trend-report'
  const papers = await run.search('channel-search', { ...input, topK: input.topK ?? (trend ? 10 : 5), rankingProfile: input.rankingProfile ?? (trend ? 'impact' : 'precision'), bias: {
    ...(trend ? { citation: 'high' as const, exploration: 'middle' as const } : { keyword: 'high' as const, related: 'high' as const, citation: 'low' as const, exploration: 'low' as const }), ...input.bias,
  } })
  if (!papers.length) run.warnings.push('No papers found; widen the time range or relax the anchors.')
  if (trend) {
    const groups = new Map<string, typeof papers>()
    for (const paper of [...papers].sort((a, b) => (a.year ?? Infinity) - (b.year ?? Infinity))) {
      const year = paper.year === undefined ? 'Unknown year' : String(paper.year)
      groups.set(year, [...(groups.get(year) ?? []), paper])
    }
    return result('SciAtlas Trend Report', `${papers.length} influential paper(s) for ${input.query}. Timeline reflects retrieved evidence, not a complete history.`, papers, run.warnings, [...groups].map(([year, items]) => `## ${year}\n\n${items.map(paper => `- [${paper.paperId}] ${paper.title}; citations: ${paper.citationCount ?? 'unknown'}`).join('\n')}`).join('\n\n'), { timeline: json([...groups].map(([year, items]) => ({ year, papers: items }))) })
  }
  return result('SciAtlas Idea Grounding', `${papers.length} related paper(s) for the research idea. Retrieval similarity alone does not establish novelty.`, papers, run.warnings, `## Related Work Evidence\n\n${papers.map(paper => `### ${paper.title}\n\n[${paper.paperId}] ${paper.abstract ?? 'Abstract unavailable.'}`).join('\n\n')}\n\n## Comparison Guide\n\nCompare problem framing, mechanisms, assumptions, datasets, and evaluation design against the proposed idea.`, { query: input.query, papers })
}
