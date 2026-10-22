/** Validated model outputs and CLI-aligned research presets. */
import { array, object, text, json } from '../research/json.ts'
import type { ResearchInput, ResearchConfig, ResearchResult, Paper, Evidence } from '../research/types.ts'
import { bibliography, citations } from '../research/runtime.ts'

/** Resolve deployment-independent CLI defaults before starting a run. */
export function resolveOptions(input: ResearchInput, config: ResearchConfig) {
  const flash = input.workflow === 'flash'
  const options = {
    probeTopK: input.topK ?? (flash ? 8 : 30), roundTopK: input.topK ?? (flash ? 6 : 20),
    round1ActionLimit: flash ? 3 : 7, round2ActionLimit: flash ? 0 : 4, llmPaperLimit: flash ? 30 : 100,
    subjectDomain: 'general' as 'general' | 'chemistry' | 'biology',
    reportStopAfter: (flash ? 'packs' : 'full') as 'bibliography' | 'outline' | 'packs' | 'full',
    kgTopK: input.topK ?? (flash ? 6 : 20), s2TopK: input.topK ?? (flash ? 6 : 20),
    manifestTopK: input.topK ?? (flash ? 6 : 20), groundingFinalTopK: flash ? 4 : 8,
    groundingCandidateK: 40, groundingRerankTopN: 20, maxParagraphsPerPaperPerQuery: 2, rerankBatchSize: 4,
    maxReviewers: flash ? 1 : 3, reviewerSelectionTopK: flash ? 10 : 30,
    reviewerMaxWorkers: flash ? 1 : 6, reviewMaxWorkers: flash ? 1 : 5,
    maxEvidenceCardsPerSection: flash ? 12 : 40, maxEvidenceCardsPerQuery: flash ? 2 : 4,
    maxExperimentCardsPerSection: flash ? 6 : 20, maxEvidenceCharsPerSection: flash ? 20000 : 60000,
    shortReport: flash, disableLlmRanking: false, disableReviewLlm: false,
    seedRecentYears: 7, graphBudgetMax: flash ? 5 : 30, graphBudgetMin: flash ? 5 : 12,
    kStep1: flash ? 1 : 4, seedNumQueries: flash ? 1 : 8, seedPerQuery: flash ? 1 : 3,
    anchorTopK: flash ? 3 : 5, ideaCount: 1, maxNoveltyFeedbackRounds: flash ? 0 : 2,
    llmTimeoutMs: config.llmTimeoutMs, outlineTimeoutMs: 600000, integrationTimeoutMs: 600000,
    reviewTimeoutMs: flash ? 240000 : 600000, rubricTimeoutMs: flash ? 180000 : 300000,
    saveIntermediate: true,
    ...input.options,
  }
  for (const [key, value] of Object.entries(options)) if (typeof value === 'number') {
    const allowZero = key === 'round1ActionLimit' || key === 'round2ActionLimit' || key === 'maxNoveltyFeedbackRounds' || key === 's2TopK'
    if (!Number.isInteger(value) || value < (allowZero ? 0 : 1) || value > 2147483647) throw new Error(`Invalid research option ${key}`)
  }
  if (options.graphBudgetMin > options.graphBudgetMax) throw new Error('graphBudgetMin exceeds graphBudgetMax')
  for (const key of ['graphBudgetMax', 'llmPaperLimit', 'manifestTopK'] as const) if (options[key] > config.maxPapers) throw new Error(`${key} exceeds research.maxPapers`)
  for (const key of ['probeTopK', 'roundTopK', 'kgTopK', 'anchorTopK'] as const) if (options[key] > config.maxPapers) throw new Error(`${key} exceeds research.maxPapers`)
  if (options.kStep1 > options.graphBudgetMax) throw new Error('kStep1 exceeds graphBudgetMax')
  if (options.maxReviewers > 20 || options.seedNumQueries > 50 || options.maxNoveltyFeedbackRounds > 5 || options.ideaCount > 20) throw new Error('Research option exceeds supported reviewer/query/feedback/idea bound')
  if (input.smoke && input.command === 'literature-review') options.reportStopAfter = 'bibliography'
  if (flash && !('shortReport' in input.options)) options.shortReport = true
  return options
}

/** Validate a string list received from a model or checkpoint. */
export function strings(value: unknown, label: string): string[] { return array(value, label).map(item => text(item, label)) }

export interface Action { query: string; rationale: string; after?: string; before?: string }

/** Validate plans and enforce their action budget at the model boundary. */
export function actions(value: unknown, max: number): Action[] {
  const raw = object(value)
  const items = array(raw['actions'], 'actions')
  if (items.length > max) throw new Error('Search action count exceeds its budget')
  return items.map(item => {
    const action = object(item)
    const result: Action = { query: text(action['query'], 'query'), rationale: text(action['rationale'], 'rationale') }
    if (typeof action['after'] === 'string' && action['after']) result.after = action['after']
    if (typeof action['before'] === 'string' && action['before']) result.before = action['before']
    return result
  })
}

export interface Section { title: string; text: string; citations: string[] }

/** Validate generated prose against the acquired reference ids and URLs. */
export function sections(value: unknown, papers: readonly Paper[], evidence: readonly Evidence[] = []): Section[] {
  const raw = object(value)
  const allowed = new Set([...papers.map(paper => paper.paperId), ...evidence.map(item => item.evidenceId)])
  const links = new Set(papers.flatMap(paper => [paper.url, paper.pdfUrl, paper.doi].filter((item): item is string => !!item)))
  const items = array(raw['sections'], 'sections')
  if (!items.length) throw new Error('Report must contain at least one section')
  return items.map(item => {
    const section = object(item)
    const body = text(section['text'], 'section text')
    const refs = citations(section['citations'], papers, evidence)
    for (const match of body.matchAll(/\[([\w-]+)\]/g)) {
      if (!allowed.has(match[1]!) || !refs.includes(match[1]!)) throw new Error(`Unresolved inline citation: ${match[1]}`)
    }
    for (const match of body.matchAll(/https?:\/\/[^\s)\]]+/g)) if (!links.has(match[0])) throw new Error(`Unacquired reference URL: ${match[0]}`)
    return { title: text(section['title']), text: body, citations: refs }
  })
}

/** Convert validated paragraphs to Markdown with explicit reference handles. */
export function renderSections(items: readonly Section[]): string {
  return items.map(section => `## ${section.title}\n\n${section.text}\n\nSources: ${section.citations.map(id => `[${id}]`).join(', ') || 'No supporting citation supplied.'}`).join('\n\n')
}

/** Build a truthful terminal result, including source limitations and bibliography. */
export function result(title: string, summary: string, papers: Paper[], warnings: readonly string[], body: string, data: unknown = {}, partial = false): ResearchResult {
  return {
    status: partial ? 'partial' : 'completed', summary, papers, warnings: [...new Set(warnings)],
    report: `# ${title}\n\n${summary}\n\n${warnings.length ? `## Limitations\n\n${[...new Set(warnings)].map(warning => `- ${warning}`).join('\n')}\n\n` : ''}${body}\n\n## References\n\n${bibliography(papers)}\n`,
    data: object(json(data)),
  }
}
