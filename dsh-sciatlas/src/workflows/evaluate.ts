/** KG/optional S2 evidence, historical rubric synthesis, reviewer views, and meta-review. */
import type { ResearchRun } from '../research/runtime.ts'
import { parallelMap, citations, paperCards } from '../research/runtime.ts'
import type { Paper, Evidence } from '../research/types.ts'
import { mergePapers, retrieve, authorPapers } from '../research/search.ts'
import { object, array, text, json, errorText } from '../research/json.ts'
import { lexicalEvidence } from '../research/pdf.ts'
import { SCIENTIFIC_RULES, RUBRIC_RERANK_SCORING_GUIDE, PARAGRAPH_RERANK_SCORING_GUIDE } from '../research/prompts.ts'
import { GENERAL_PERSONA_NOTES } from '../research/persona-notes.ts'
import { reviewPrompts } from '../research/upstream-review-prompts.ts'
import { resolveOptions, result, strings, sections, renderSections } from './common.ts'

/** Upstream grounding reranker constants (sciatlas/evidence/grounding.py). */
const GROUNDING_RERANK_BATCH = 8
const RERANK_PARAGRAPH_MAX_CHARS = 1200

const RUBRIC_SECTIONS = ['Motivation', 'Method', 'Result', 'Discussion'] as const
type RubricSection = typeof RUBRIC_SECTIONS[number]
interface Standard { section: RubricSection; dimension: string; requirement: string; citations: string[]; sourceTag: string }

/**
 * The upstream rubric prompts drift between section labels (the dimensions
 * prompt says "Introduction/Methods/Results", the synthesis prompt says
 * "target_section" and "dimension_name"). Accept those aliases at the model
 * boundary so a prompt-faithful response does not burn a repair attempt.
 */
const SECTION_ALIASES: Record<string, RubricSection> = {
  motivation: 'Motivation', introduction: 'Motivation',
  method: 'Method', methods: 'Method',
  result: 'Result', results: 'Result',
  discussion: 'Discussion',
}

/** Map a model-supplied section label onto the canonical rubric section. */
function rubricSection(value: unknown): RubricSection {
  const mapped = SECTION_ALIASES[text(value, 'section').toLowerCase()]
  if (mapped === undefined) throw new Error('Invalid rubric section')
  return mapped
}

/** Keep standards idea-specific, sectioned, and linked to acquired sources. */
export function standards(value: unknown, papers: readonly Paper[]) {
  const raw = object(value)
  const result = array(raw['standards'], 'standards').map(item => {
    const standard = object(item)
    return { section: rubricSection(standard['section'] ?? standard['target_section']), dimension: text(standard['dimension'] ?? standard['dimension_name'], 'dimension'), requirement: text(standard['requirement']), citations: citations(standard['citations'], papers), sourceTag: text(standard['sourceTag']) }
  })
  if (!result.length) throw new Error('Rubric must contain standards')
  return { standards: result }
}

const STANDARD_SCHEMA = 'Return {"standards":[{"section":"Motivation|Method|Result|Discussion","dimension":"specific technical dimension","requirement":"concrete required evidence","citations":["paperId"],"sourceTag":"General|Detailed|Historical"}]}. Citations may be empty only for a standard derived solely from the user idea. Do not invent historical evidence.'

/** Schema of the upstream abstract-only PaperModuleDimensions rubric call. */
const MODULE_STANDARD_SCHEMA = 'Return {"introduction_standards":[{"dimension_name":"...","core_reason":"...","required_evidence":["..."]}],"methods_standards":[...],"results_standards":[...],"discussion_standards":[...]}. Produce 2 to 4 standards per module.'

/** Upstream module keys map onto the canonical rubric sections. */
const MODULE_SECTIONS = { introduction_standards: 'Motivation', methods_standards: 'Method', results_standards: 'Result', discussion_standards: 'Discussion' } as const

/** Evaluate idea text or a PDF using the current hosted-KG/LLM rubric path. */
export async function runEvaluate(run: ResearchRun) {
  const input = run.input
  const options = resolveOptions(input, run.services.config)
  const documents = await run.inputDocuments()
  const idea = input.query || documents.map(document => `${document.title}\n${document.abstract}\n${document.paragraphs.map(item => item.text).join('\n')}`).join('\n\n')
  const context = await run.model('review-context', { idea }, `${SCIENTIFIC_RULES}
Extract only what the submitted idea/paper states. Return {"summary":"...","problem":"...","method":"...","evaluation":"...","queries":[{"section":"Motivation|Method|Result|Discussion","query":"..."}]}. Use "Not specified" for missing mechanisms or experiments. Provide 1–8 focused evidence queries.`, value => {
    const raw = object(value)
    const queries = array(raw['queries'], 'queries').map(item => { const query = object(item); const section = text(query['section']); const found = RUBRIC_SECTIONS.find(name => name === section); if (!found) throw new Error('Invalid grounding query section'); return { section: found, query: text(query['query']) } })
    if (!queries.length || queries.length > 8) throw new Error('Grounding requires 1–8 queries')
    return { summary: text(raw['summary']), problem: text(raw['problem']), method: text(raw['method']), evaluation: text(raw['evaluation']), queries }
  }, () => ({ summary: idea, problem: idea, method: 'Synthetic method', evaluation: 'Synthetic evaluation', queries: RUBRIC_SECTIONS.map(section => ({ section, query: idea })) }))
  let papers = await run.search('review-kg-search', { ...input, query: context.summary, topK: options.kgTopK, rankingProfile: input.rankingProfile ?? 'precision', bias: { keyword: 'high', related: 'high', citation: 'low', exploration: 'low', ...input.bias } })
  let partial = false
  if (run.services.config.semanticScholarEnabled && options.s2TopK > 0 && !input.smoke) {
    const supplemental = await run.stage('review-s2-search', { query: context.summary, count: options.s2TopK }, value => { const raw = object(value); return { papers: importPapers(raw['papers']), error: typeof raw['error'] === 'string' ? raw['error'] : '' } }, async () => {
      try { return { papers: (await run.request('review-s2-search', () => run.services.semanticSearch(context.summary, options.s2TopK, run.signal))).papers, error: '' } }
      catch (error) { run.signal.throwIfAborted(); return { papers: [], error: errorText(error) } }
    })
    papers = mergePapers([papers, supplemental.papers], run.services.config.maxPapers)
    if (supplemental.error) { run.warnings.push(`Semantic Scholar evidence unavailable: ${supplemental.error}`); partial = true }
  }
  if (!papers.length) return result('SciAtlas Idea Evaluation', 'No supporting literature was retrieved. Evidence-grounded evaluation could not be completed.', [], [...run.warnings, 'No evidence available.'], '', { context }, true)
  if (!options.disableLlmRanking) {
    // Batched LLM relevance scoring, aligned with the current upstream
    // merge_search/idea_rubric path: independently scored batches, and papers
    // whose batch failed keep retrieval order below all scored papers.
    const ranked = papers.slice(0, options.llmPaperLimit)
    const batches: Paper[][] = []
    for (let start = 0; start < ranked.length; start += options.rerankBatchSize) batches.push(ranked.slice(start, start + options.rerankBatchSize))
    const scores = new Map<string, number>()
    let failedBatches = 0
    await parallelMap(batches, run.services.config.maxStageConcurrency, run.signal, async (batch, batchIndex) => {
      try {
        const ranking = await run.model(`review-paper-rerank-${batchIndex}`, json({ idea: context, papers: batch.map((paper, slot) => ({ paper_index: slot + 1, title: paper.title, abstract: paper.abstract ?? '' })) }), `${SCIENTIFIC_RULES}\n${reviewPrompts.RERANK_SYSTEM_PROMPT}\n${RUBRIC_RERANK_SCORING_GUIDE}`, value => decodeBatchScores(value, batch.length), () => ({ scores: batch.map((_, slot) => ({ paperIndex: slot + 1, score: 9 - slot, reason: 'Synthetic smoke score' })) }), options.rubricTimeoutMs)
        for (const item of ranking.scores) scores.set(batch[item.paperIndex - 1]!.paperId, item.score)
      } catch (error) {
        run.signal.throwIfAborted()
        failedBatches++
        run.warnings.push(`LLM relevance scoring failed for paper batch ${batchIndex + 1}; its papers keep retrieval order below scored papers: ${errorText(error)}`)
      }
    })
    if (failedBatches) partial = true
    const order = new Map(papers.map((paper, index) => [paper.paperId, index]))
    papers = [...papers].sort((a, b) => {
      const scoreA = scores.get(a.paperId)
      const scoreB = scores.get(b.paperId)
      if (scoreA !== undefined && scoreB !== undefined) return scoreB - scoreA || order.get(a.paperId)! - order.get(b.paperId)!
      if (scoreA !== undefined) return -1
      if (scoreB !== undefined) return 1
      return order.get(a.paperId)! - order.get(b.paperId)!
    })
  }
  const acquired = await run.evidence(papers, options.manifestTopK)
  const evidence = [...acquired, ...documents.flatMap(document => document.paragraphs)]
  const matched = await parallelMap(context.queries, run.services.config.maxStageConcurrency, run.signal, async (query, index) => {
    // Current upstream grounding path: TF-IDF lexical candidates, then a
    // batched LLM paragraph rerank over the leading pool; a failed rerank
    // keeps the lexical order for the query instead of aborting the run.
    const candidates = lexicalEvidence(query.query, acquired, options.groundingCandidateK)
    if (!candidates.length) return { section: query.section, query: query.query, evidence: [] as Evidence[] }
    let ranked = candidates.slice(0, options.groundingRerankTopN)
    try {
      const scores = new Map<string, number>()
      for (let start = 0, batchIndex = 0; start < ranked.length; start += GROUNDING_RERANK_BATCH, batchIndex++) {
        const batch = ranked.slice(start, start + GROUNDING_RERANK_BATCH)
        const ranking = await run.model(`review-grounding-${index}-batch-${batchIndex}`, json({ query: query.query, paragraphs: batch.map((item, slot) => ({ paragraph_index: slot + 1, text: truncateParagraph(item.text) })) }), `${SCIENTIFIC_RULES}\n${PARAGRAPH_RERANK_SCORING_GUIDE}`, value => decodeParagraphScores(value, batch.length), () => ({ scores: batch.map((_, slot) => ({ paragraphIndex: slot + 1, score: 8 })) }), options.rubricTimeoutMs)
        for (const item of ranking.scores) scores.set(batch[item.paragraphIndex - 1]!.evidenceId, item.score)
      }
      ranked = [...ranked].sort((a, b) => (scores.get(b.evidenceId) ?? 0) - (scores.get(a.evidenceId) ?? 0))
    } catch (error) {
      run.signal.throwIfAborted()
      partial = true
      ranked = candidates
      run.warnings.push(`Paragraph reranking failed for evaluation query ${index + 1}; lexical retrieval order was used: ${errorText(error)}`)
    }
    return { section: query.section, query: query.query, evidence: selectDiverse(ranked, Math.min(options.groundingFinalTopK, options.maxEvidenceCardsPerQuery), options.maxParagraphsPerPaperPerQuery) }
  })
  await run.store.write('grounding.json', json(matched))
  if (matched.some(item => !item.evidence.length)) { partial = true; run.warnings.push('Some evaluation queries have no matching passage evidence.') }

  const historical = await parallelMap(papers.slice(0, options.manifestTopK), run.services.config.maxStageConcurrency, run.signal, async (paper, index) => {
    // Abstract-only per-paper rubric extraction, aligned with the current
    // upstream path: norms see paper metadata plus the general reviewer-persona
    // notes, and a failed norms call deterministically falls back to those
    // notes instead of skipping the paper.
    let norms: string[]
    try {
      const generated = await run.model(`rubric-norms-${index}`, json({ paper: { title: paper.title, venue: paper.venue ?? null, publicationYear: paper.year ?? null, citationCount: paper.citationCount ?? null, abstract: paper.abstract ?? '' }, reviewerPersonaNotes: GENERAL_PERSONA_NOTES.slice(0, 8) }), `${SCIENTIFIC_RULES}\n${reviewPrompts.ABSTRACT_REVIEW_NORMS_SYSTEM_PROMPT}\nReturn {"norms":["technically specific evidence requirement"]}. Respect abstract-only limitations.`, value => ({ norms: strings(object(value)['norms'], 'norms') }), () => ({ norms: ['Test the stated mechanism with appropriate controls.'] }), options.rubricTimeoutMs)
      if (!generated.norms.length) throw new Error('Review norms generation returned no norms')
      norms = generated.norms
    } catch (error) {
      run.signal.throwIfAborted()
      partial = true
      norms = [...GENERAL_PERSONA_NOTES]
      run.warnings.push(`Review norms unavailable for ${paper.paperId}; general reviewer-persona notes were used instead: ${errorText(error)}`)
    }
    try {
      return await run.model(`rubric-dimensions-${index}`, json({ paper: { title: paper.title, abstract: paper.abstract ?? '' }, reviewNorms: norms }), `${SCIENTIFIC_RULES}\n${reviewPrompts.ABSTRACT_DIMENSIONS_SYSTEM_PROMPT}\n${MODULE_STANDARD_SCHEMA}`, value => moduleStandards(value, paper), () => ({ standards: RUBRIC_SECTIONS.map(section => ({ section, dimension: `${section} technical requirement`, requirement: norms[0]!, citations: [paper.paperId], sourceTag: 'Historical' })) }), options.rubricTimeoutMs)
    } catch (error) {
      run.signal.throwIfAborted()
      partial = true
      run.warnings.push(`Historical rubric standards skipped ${paper.paperId}: ${errorText(error)}`)
      return null
    }
  })
  const historicalStandards = historical.filter((item): item is { standards: Standard[] } => item !== null)
  const general = await run.model('rubric-general', json({ idea: context, historical: historicalStandards }), `${SCIENTIFIC_RULES}\n${reviewPrompts.GENERAL_RUBRIC_SYSTEM_PROMPT}\n${STANDARD_SCHEMA}`, value => standards(value, papers), () => ({ standards: historicalStandards.flatMap(item => item.standards).map(item => ({ ...item, sourceTag: 'General' })) }), options.rubricTimeoutMs)
  const detailed = await run.model('rubric-detailed', json({ idea: context }), `${SCIENTIFIC_RULES}\n${reviewPrompts.DETAILED_RUBRIC_SYSTEM_PROMPT}\n${STANDARD_SCHEMA}`, value => standards(value, papers), () => ({ standards: RUBRIC_SECTIONS.map(section => ({ section, dimension: `${section}: stated idea constraint`, requirement: 'Validate the concrete stated claim; missing details must be identified.', citations: [], sourceTag: 'Detailed' })) }), options.rubricTimeoutMs)
  const rubric = await run.model('rubric-synthesis', json({ idea: context, general, detailed }), `${SCIENTIFIC_RULES}\n${reviewPrompts.SECTIONED_SYNTHESIS_SYSTEM_PROMPT}\n${STANDARD_SCHEMA}\nCover all four sections. sourceTag must be General or Detailed.`, value => {
    const normalized = standards(value, papers)
    if (RUBRIC_SECTIONS.some(section => !normalized.standards.some(item => item.section === section))) throw new Error('Rubric must cover all four sections')
    if (normalized.standards.some(item => item.sourceTag !== 'General' && item.sourceTag !== 'Detailed')) throw new Error('Synthesis sourceTag must be General or Detailed')
    return normalized
  }, () => ({ standards: detailed.standards }), options.rubricTimeoutMs)
  await run.store.write('rubric.json', json(rubric))

  const reviewers = await run.stage('reviewer-candidates', { query: context.summary, count: options.reviewerSelectionTopK }, value => {
    const raw = object(value)
    return { reviewers: array(raw['reviewers']).map(item => { const author = object(item); return { name: text(author['name']), id: typeof author['id'] === 'string' ? author['id'] : '', synthetic: author['synthetic'] === true } }), error: typeof raw['error'] === 'string' ? raw['error'] : '' }
  }, async () => {
    if (input.smoke) return { reviewers: [{ name: 'Synthetic reviewer persona', id: '', synthetic: true }], error: '' }
    try {
      const response = await run.request('reviewer-candidates', () => retrieve(run.services.api, '/v1/authors/related', { plan: { query_text: context.summary, source_type: 'idea_text', source_title: null, keywords: [{ text: context.problem.slice(0, 120), score: 9 }], titles: [], reference_titles: [] }, options: { top_k: Math.min(options.reviewerSelectionTopK, run.services.api.maxTopK) } }, 'reviewer-selection', run.signal, options.reviewerSelectionTopK))
      return { reviewers: response.authors.slice(0, options.maxReviewers).map(author => ({ name: author.name, id: author.id ?? '', synthetic: false })), error: '' }
    } catch (error) { run.signal.throwIfAborted(); return { reviewers: [], error: errorText(error) } }
  })
  if (!reviewers.reviewers.length) {
    reviewers.reviewers.push({ name: 'Evidence-focused reviewer persona', id: '', synthetic: true })
    run.warnings.push('No verified researcher reviewer was available; an explicitly synthetic reviewer perspective was used.')
    partial = true
  }
  if (reviewers.error) { run.warnings.push(`Reviewer lookup failed: ${reviewers.error}`); partial = true }
  const backgrounds = await parallelMap(reviewers.reviewers.slice(0, options.maxReviewers), Math.min(options.reviewerMaxWorkers, run.services.config.maxStageConcurrency), run.signal, async (reviewer, index) => run.stage(`reviewer-background-${index}`, reviewer, value => { const raw = object(value); return { reviewer, papers: importPapers(raw['papers']), error: typeof raw['error'] === 'string' ? raw['error'] : '' } }, async () => {
    if (reviewer.synthetic) return { reviewer, papers: [], error: '' }
    try { return { reviewer, papers: (await run.request(`reviewer-background-${index}`, () => authorPapers(run.services.api, reviewer.id || reviewer.name, Math.min(5, options.kgTopK), run.signal))).papers, error: '' } }
    catch (error) { run.signal.throwIfAborted(); return { reviewer, papers: [], error: errorText(error) } }
  }))
  for (const background of backgrounds) if (background.error) { run.warnings.push(`Reviewer background unavailable: ${background.error}`); partial = true }
  const packs = RUBRIC_SECTIONS.map(section => {
    const all = matched.filter(item => item.section === section).flatMap(item => item.evidence)
    const unique = [...new Map(all.map(item => [item.evidenceId, item])).values()]
    let chars = 0
    let experiments = 0
    const selected = unique.filter(item => {
      if (/experiment|result|evaluation/i.test(item.section.join(' ')) && ++experiments > options.maxExperimentCardsPerSection) return false
      if (chars + item.text.length > options.maxEvidenceCharsPerSection) return false
      chars += item.text.length
      return true
    }).slice(0, options.maxEvidenceCardsPerSection)
    return { section, standards: rubric.standards.filter(item => item.section === section), evidence: selected }
  })
  const reviews = await parallelMap(backgrounds, Math.min(options.reviewMaxWorkers, run.services.config.maxStageConcurrency), run.signal, async (background, index) => {
    if (options.disableReviewLlm) return { reviewer: background.reviewer.name, mode: 'deterministic', sections: deterministicReview(rubric.standards, packs), error: '' }
    try {
      const review = await run.model(`review-persona-${index}`, json({ idea: context, reviewer: background.reviewer, backgroundPapers: paperCards(background.papers, 5), rubric, evidencePacks: packs }), `${SCIENTIFIC_RULES}
Use this researcher's literature to inform a simulated reviewer perspective; do not claim this is their actual opinion. Critique each rubric section, distinguish supported/unsupported/uncertain claims, state missing controls and propose falsifiable experiments. The submitted idea is not evidence that its claims work. Return {"sections":[{"title":"...","text":"...","citations":["source-id"]}]}. Cite the supplied papers/passages only.`, value => ({ sections: sections(value, papers, evidence) }), () => ({ sections: deterministicReview(rubric.standards, packs) }), options.reviewTimeoutMs)
      return { reviewer: background.reviewer.name, mode: 'simulated', sections: review.sections, error: '' }
    } catch (error) { run.signal.throwIfAborted(); return { reviewer: background.reviewer.name, mode: 'failed', sections: [], error: errorText(error) } }
  })
  for (const review of reviews) if (review.error) { partial = true; run.warnings.push(`Reviewer evaluation failed: ${review.error}`) }
  const successful = reviews.filter(review => !review.error)
  if (!successful.length) return result('SciAtlas Idea Evaluation', 'Rubric and evidence prepared, but all reviewer evaluations failed.', papers, run.warnings, '', { context, rubric, grounding: matched, reviews }, true)
  const meta = options.disableReviewLlm ? { sections: deterministicReview(rubric.standards, packs) } : await run.model('review-meta', json({ idea: context, rubric, reviews: successful, shortReport: options.shortReport, limitations: run.warnings }), `${SCIENTIFIC_RULES}
Produce a ${options.shortReport ? 'compact' : 'comprehensive'} meta-review with evidence-backed strengths, prioritized weaknesses, disagreements, concrete revision actions and experimental suggestions. Do not invent consensus or numerical scores. Return {"sections":[{"title":"...","text":"...","citations":["source-id"]}]}.`, value => ({ sections: sections(value, papers, evidence) }), () => ({ sections: deterministicReview(rubric.standards, packs) }), options.reviewTimeoutMs)
  return result('SciAtlas Idea Evaluation', `${successful.length} ${options.disableReviewLlm ? 'deterministic' : 'simulated'} reviewer perspective(s) evaluated the submitted idea.`, papers, run.warnings, renderSections(meta.sections), { context, rubric, grounding: matched, reviewers: backgrounds, reviews, meta, mode: options.disableReviewLlm ? 'deterministic' : 'llm' }, partial)
}

import { decodePapers as importPapers } from '../research/runtime.ts'

/** Trim a paragraph exactly like the upstream grounding reranker does. */
function truncateParagraph(value: string): string {
  const trimmed = value.replace(/\s+/g, ' ').trim()
  return trimmed.length > RERANK_PARAGRAPH_MAX_CHARS ? `${trimmed.slice(0, RERANK_PARAGRAPH_MAX_CHARS)} ...` : trimmed
}

/** Keep at most perPaper passages from one paper, preserving ranked order. */
function selectDiverse(ranked: readonly Evidence[], limit: number, perPaper: number): Evidence[] {
  const counts = new Map<string, number>()
  const selected: Evidence[] = []
  for (const item of ranked) {
    const count = counts.get(item.paperId) ?? 0
    if (perPaper > 0 && count >= perPaper) continue
    counts.set(item.paperId, count + 1)
    selected.push(item)
    if (selected.length >= limit) break
  }
  return selected
}

/**
 * Decode one paper-scoring batch. Invalid entries are skipped (the upstream
 * judge tolerates partial batches); a response with no usable score at all is
 * rejected so the caller can degrade the batch to retrieval order.
 */
function decodeBatchScores(value: unknown, batchSize: number) {
  const raw = object(value)
  const seen = new Set<number>()
  const scores: { paperIndex: number; score: number; reason: string }[] = []
  for (const item of array(raw['papers'] ?? raw['scores'], 'papers')) {
    const entry = object(item)
    const slot = entry['paper_index'] ?? entry['paperIndex'] ?? entry['index']
    const score = entry['score']
    if (typeof slot !== 'number' || !Number.isInteger(slot) || slot < 1 || slot > batchSize || seen.has(slot) || typeof score !== 'number') continue
    seen.add(slot)
    scores.push({ paperIndex: slot, score: Math.max(0, Math.min(10, score)), reason: typeof entry['reason'] === 'string' ? entry['reason'] : '' })
  }
  if (!scores.length) throw new Error('No valid paper scores in batch response')
  return { scores }
}

/** Decode one paragraph-scoring batch; unscored paragraphs default to 0. */
function decodeParagraphScores(value: unknown, batchSize: number) {
  const raw = object(value)
  const seen = new Set<number>()
  const scores: { paragraphIndex: number; score: number }[] = []
  for (const item of array(raw['scores'], 'scores')) {
    const entry = object(item)
    const slot = entry['paragraph_index'] ?? entry['paragraphIndex']
    const score = entry['score']
    if (typeof slot !== 'number' || !Number.isInteger(slot) || slot < 1 || slot > batchSize || seen.has(slot) || typeof score !== 'number') continue
    seen.add(slot)
    scores.push({ paragraphIndex: slot, score: Math.max(0, Math.min(10, score)) })
  }
  if (!scores.length) throw new Error('No valid paragraph scores in batch response')
  return { scores }
}

/** Decode the upstream four-module abstract-only rubric extraction. */
function moduleStandards(value: unknown, paper: Paper) {
  const raw = object(value)
  // Stage checkpoints and the in-memory value pass through this decoder
  // again, so the already-converted form must remain valid.
  if (raw['standards'] !== undefined) return standards(raw, [paper])
  const result: Standard[] = []
  for (const [key, section] of Object.entries(MODULE_SECTIONS) as [keyof typeof MODULE_SECTIONS, RubricSection][]) {
    const items = raw[key] === undefined ? [] : array(raw[key], key)
    for (const item of items) {
      const standard = object(item)
      const evidence = standard['required_evidence'] === undefined ? [] : array(standard['required_evidence'], 'required_evidence').map(entry => text(entry, 'required_evidence')).filter(entry => entry)
      const reason = text(standard['core_reason'] ?? standard['requirement'], 'core_reason')
      result.push({ section, dimension: text(standard['dimension_name'] ?? standard['dimension'], 'dimension_name'), requirement: evidence.length ? `${reason} Required evidence: ${evidence.join('; ')}` : reason, citations: [paper.paperId], sourceTag: 'Historical' })
    }
  }
  if (!result.length) throw new Error('Dimension extraction returned no standards')
  return { standards: result }
}

/** Explain requirements and evidence presence without pretending to perform LLM evaluation. */
function deterministicReview(rubric: readonly Standard[], packs: readonly { section: RubricSection; evidence: Evidence[] }[]) {
  return RUBRIC_SECTIONS.map(section => {
    const evidence = packs.find(pack => pack.section === section)?.evidence ?? []
    return { title: section, text: `Deterministic evidence inventory; no model review was performed.\n${rubric.filter(item => item.section === section).map(item => `- Required: ${item.requirement}`).join('\n')}\n${evidence.length ? `${evidence.length} matching evidence passage(s) available; satisfaction of each requirement remains unverified.` : 'No matched passage; these claims require further evidence.'}`, citations: evidence.map(item => item.evidenceId) }
  })
}
