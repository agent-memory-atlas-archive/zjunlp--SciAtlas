/** Low-level title/abstract retrieval via the SciAtlas papers/search endpoint. */
import { defineTool } from '@deepseek-ai/dsh-tools'
import { postJson } from '../client.ts'
import { collectPapers } from '../normalize.ts'
import { PAPER_VALUE_SCHEMA, clampTopK, renderPaperList, requireNonBlank, truncateText, withRanks } from './shared.ts'
import type { SciAtlasToolDeps } from './shared.ts'

/** Create the low-level retrieval tool, preserving CLI mode and field controls. */
export function createPaperSearchTool(deps: SciAtlasToolDeps) {
  return defineTool({
    name: 'sciatlas_paper_search',
    description: 'Run low-level SciAtlas paper search by title or abstract. Use sciatlas_search_papers for graph retrieval. Requires a SciAtlas key.',
    parameters: {
      query: { type: 'string', required: true },
      mode: { type: 'string', description: 'Backend mode (default vector).' },
      field: { type: 'string', description: 'Backend field (default title).' },
      topK: { type: 'integer' },
    },
    output: {
      schema: { type: 'object', additionalProperties: false, properties: {
        query: { type: 'string', required: true }, total: { type: 'integer', required: true },
        requestedCount: { type: 'integer', required: true }, effectiveCount: { type: 'integer', required: true },
        truncated: { type: 'boolean', required: true }, papers: { type: 'array', required: true, items: PAPER_VALUE_SCHEMA },
      } },
      render: (_args, value) => [{ type: 'text', text: `SciAtlas low-level search: ${value.query}\n${value.truncated ? `Result budget capped at ${value.effectiveCount}.\n` : ''}${renderPaperList(value.papers)}` }],
      presentationMeta: (_args, value) => ({ count: value.total }),
    },
    timeoutMs: deps.timeoutMs + 30_000,
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const query = requireNonBlank(args.query, 'query')
      const requestedCount = args.topK ?? 5
      const effectiveCount = clampTopK(requestedCount, 5, deps.maxTopK)
      const body = await postJson('/v1/papers/search', {
        query_text: query, mode: args.mode ?? 'vector', field: args.field ?? 'title', options: { top_k: effectiveCount },
      }, { baseUrl: deps.baseUrl, apiKey: await deps.resolveApiKey(), timeoutMs: deps.timeoutMs, signal: exec.signal })
      const papers = withRanks(collectPapers(body, { maxItems: effectiveCount, abstractMaxChars: deps.abstractMaxChars }))
      return { query, total: papers.length, requestedCount, effectiveCount, truncated: requestedCount > deps.maxTopK, papers }
    },
    presentCall: args => ({ card: 'generic', kind: 'search', title: `SciAtlas: ${truncateText(args.query, 80)}` }),
    presentResult: (_args, result) => result.isError ? undefined : { card: 'generic', title: 'SciAtlas paper search' },
  })
}
