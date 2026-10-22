/** Multi-round literature acquisition, coverage analysis, evidence packs, and review writing. */
import type { ResearchRun } from '../research/runtime.ts'
import { parallelMap, citations, paperCards } from '../research/runtime.ts'
import type { Paper } from '../research/types.ts'
import { mergePapers } from '../research/search.ts'
import { array, object, text, json } from '../research/json.ts'
import { SCIENTIFIC_RULES } from '../research/prompts.ts'
import { actions, result, resolveOptions, renderSections, sections, strings } from './common.ts'

/** Every cluster is tied to known papers; dates remain analysis of the supplied sample. */
function decodeAnalysis(value: unknown, papers: readonly Paper[], actionLimit: number) {
  const raw = object(value)
  const clusters = array(raw['methodClusters'], 'methodClusters').map(item => {
    const cluster = object(item)
    return { label: text(cluster['label']), description: text(cluster['description']), paperIds: citations(cluster['paperIds'], papers) }
  })
  const windows = array(raw['timeWindows'], 'timeWindows').map(item => {
    const window = object(item)
    if (!Number.isInteger(window['startYear']) || !Number.isInteger(window['endYear']) || Number(window['startYear']) > Number(window['endYear'])) throw new Error('Invalid time window')
    return { startYear: Number(window['startYear']), endYear: Number(window['endYear']), rationale: text(window['rationale']) }
  })
  return { methodClusters: clusters, timeWindows: windows, actions: actions(raw, actionLimit) }
}

/** Literature pipeline preserves CLI flash/full retrieval and stop-stage semantics. */
export async function runLiterature(run: ResearchRun) {
  const input = run.input
  const options = resolveOptions(input, run.services.config)
  let papers = await run.search('lr-probe', { ...input, topK: options.probeTopK, retrievalMode: input.retrievalMode ?? 'hybrid', rankingProfile: input.rankingProfile ?? 'balanced' })
  if (!papers.length) return result('SciAtlas Literature Review', 'No papers retrieved. No review was generated.', [], [...run.warnings, 'Widen the topic or remove restrictive anchors.'], '', { stopAfter: options.reportStopAfter }, true)
  const analysis = await run.model('lr-analysis', json({ topic: input.query, papers: paperCards(papers, options.llmPaperLimit), actionLimit: options.round1ActionLimit }), `${SCIENTIFIC_RULES}
Analyze method families and publication windows, then plan complementary retrieval. Keep queries on topic. Output {"methodClusters":[{"label":"...","description":"...","paperIds":["p-..."]}],"timeWindows":[{"startYear":2020,"endYear":2025,"rationale":"..."}],"actions":[{"query":"...","rationale":"...","after":"YYYY-MM-DD","before":"YYYY-MM-DD"}]}. The date fields are optional. Never exceed actionLimit.`, value => decodeAnalysis(value, papers, options.round1ActionLimit), () => ({ methodClusters: [{ label: 'Synthetic method cluster', description: 'Smoke fixture grouping', paperIds: papers.map(p => p.paperId) }], timeWindows: [], actions: Array.from({ length: Math.min(1, options.round1ActionLimit) }, () => ({ query: input.query, rationale: 'Smoke expansion' })) }))

  const searchActions = async (stage: string, planned: typeof analysis.actions) => parallelMap(planned, run.services.config.maxStageConcurrency, run.signal, (action, index) => {
    const after = [input.timeRange?.after, action.after].filter((value): value is string => !!value).sort().at(-1)
    const before = [input.timeRange?.before, action.before].filter((value): value is string => !!value).sort().at(0)
    return run.search(`${stage}-${index}`, { ...input, query: action.query, topK: options.roundTopK, timeRange: { ...(after ? { after } : {}), ...(before ? { before } : {}) } })
  })
  papers = mergePapers([papers, ...await searchActions('lr-round1', analysis.actions)], run.services.config.maxPapers)
  const guard = await run.model('lr-relevance', json({ topic: input.query, papers: paperCards(papers, options.llmPaperLimit) }), `${SCIENTIFIC_RULES}
Remove off-topic papers using topical relevance, not citation count. Return {"keepPaperIds":["p-..."],"reason":"..."}. Do not select unknown ids.`, value => { const raw = object(value); return { keepPaperIds: citations(raw['keepPaperIds'], papers), reason: text(raw['reason']) } }, () => ({ keepPaperIds: papers.map(p => p.paperId), reason: 'Smoke fixture guard' }))
  const keep = new Set(guard.keepPaperIds)
  papers = papers.filter(paper => keep.has(paper.paperId))
  if (!papers.length) return result('SciAtlas Literature Review', 'The relevance guard rejected the retrieved sample. No grounded review was generated.', [], [...run.warnings, guard.reason], '', {}, true)

  const gapDecoder = (value: unknown) => {
    const raw = object(value)
    const gaps = array(raw['gaps'], 'gaps').map(item => {
      const gap = object(item)
      if (typeof gap['blocksStop'] !== 'boolean') throw new Error('Gap blocksStop must be boolean')
      return { description: text(gap['description']), blocksStop: gap['blocksStop'] }
    })
    return { gaps, actions: actions(raw, options.round2ActionLimit) }
  }
  const gaps = await run.model('lr-gap-analysis', json({ topic: input.query, analysis, papers: paperCards(papers, options.llmPaperLimit), actionLimit: options.round2ActionLimit }), `${SCIENTIFIC_RULES}
Identify method, time-window, and representative-paper coverage gaps in this retrieved sample. Acknowledge sparse coverage rather than declaring a field-wide scientific gap. Return {"gaps":[{"description":"...","blocksStop":true}],"actions":[{"query":"...","rationale":"..."}]}, at most actionLimit actions.`, gapDecoder, () => ({ gaps: [], actions: options.round2ActionLimit > 0 ? [{ query: `${input.query} evaluation`, rationale: 'Smoke second round' }] : [] }))
  if (gaps.actions.length) papers = mergePapers([papers, ...await searchActions('lr-round2', gaps.actions)], run.services.config.maxPapers)
  const finalCoverage = await run.model('lr-final-coverage', json({ topic: input.query, gaps: gaps.gaps, papers: paperCards(papers, options.llmPaperLimit) }), `${SCIENTIFIC_RULES}
Report remaining sample coverage limitations after retrieval. Return {"limitations":["..."],"sufficient":true}. Do not treat a missing canonical paper as a scientific discovery.`, value => { const raw = object(value); if (typeof raw['sufficient'] !== 'boolean') throw new Error('sufficient must be boolean'); return { limitations: strings(raw['limitations'], 'limitations'), sufficient: raw['sufficient'] } }, () => ({ limitations: [], sufficient: true }))
  run.warnings.push(...finalCoverage.limitations)
  if (!finalCoverage.sufficient) run.warnings.push('Retrieval coverage remains incomplete; the review is limited to the acquired sample.')
  const data: Record<string, import('../research/json.ts').Json> = { analysis: json(analysis), gaps: json(gaps), coverage: json(finalCoverage), stopAfter: options.reportStopAfter }
  if (options.reportStopAfter === 'bibliography') return result('SciAtlas Literature Bibliography', `${papers.length} relevant papers collected. Stopped after bibliography.`, papers, run.warnings, '## Method Families\n\n' + analysis.methodClusters.map(cluster => `- ${cluster.label}: ${cluster.description}`).join('\n'), data, !finalCoverage.sufficient)

  const outline = await run.model('lr-outline', json({ topic: input.query, subjectDomain: options.subjectDomain, papers: paperCards(papers, options.llmPaperLimit), analysis }), `${SCIENTIFIC_RULES}
Prepare a scientific review outline for the specified subjectDomain. Chemistry must distinguish mechanisms, material systems, conditions, and metrics. Biology must distinguish systems, experimental evidence, and biological mechanisms. General reviews organize methods, evidence and limitations. Return {"title":"...","sections":[{"title":"...","focus":"...","paperIds":["p-..."]}]}. Every substantive section must have supporting papers.`, value => {
    const raw = object(value)
    const items = array(raw['sections'], 'outline sections').map(item => { const section = object(item); const paperIds = citations(section['paperIds'], papers); if (!paperIds.length) throw new Error('Outline section needs source papers'); return { title: text(section['title']), focus: text(section['focus']), paperIds } })
    if (!items.length || items.length > 12) throw new Error('Outline must have 1–12 sections')
    return { title: text(raw['title']), sections: items }
  }, () => ({ title: `SMOKE Review: ${input.query}`, sections: [{ title: 'Synthetic Findings', focus: 'Smoke pipeline validation', paperIds: papers.map(p => p.paperId) }] }), options.outlineTimeoutMs)
  data['outline'] = json(outline)
  const outlineBody = outline.sections.map(section => `## ${section.title}\n\n${section.focus}\n\nSources: ${section.paperIds.join(', ')}`).join('\n\n')
  if (options.reportStopAfter === 'outline') return result(outline.title, 'Stopped after the grounded review outline.', papers, run.warnings, outlineBody, data, !finalCoverage.sufficient)

  const evidence = await run.evidence(papers, Math.min(papers.length, options.llmPaperLimit))
  const packs = outline.sections.map(section => ({ ...section, evidence: evidence.filter(item => section.paperIds.includes(item.paperId)).slice(0, options.maxEvidenceCardsPerSection) }))
  data['evidencePacks'] = json(packs)
  await run.store.write('evidence-packs.json', json(packs))
  if (options.reportStopAfter === 'packs') return result(outline.title, 'Evidence packs prepared. No full review manuscript was requested.', papers, run.warnings, packs.map(pack => `## ${pack.title}\n\n${pack.focus}\n\n${pack.evidence.map(item => `- [${item.evidenceId}] (${item.level}, ${item.section.join(' / ')}) ${item.text}`).join('\n')}`).join('\n\n'), data, !finalCoverage.sufficient)

  const chapters = await parallelMap(packs, run.services.config.maxStageConcurrency, run.signal, (pack, index) => run.model(`lr-chapter-${index}`, json({ topic: input.query, subjectDomain: options.subjectDomain, section: pack.title, focus: pack.focus, papers: paperCards(papers.filter(paper => pack.paperIds.includes(paper.paperId)), options.llmPaperLimit), evidence: pack.evidence }), `${SCIENTIFIC_RULES}
Write the requested review section. Compare mechanisms and evidence, identify disagreements and uncertainty, and describe concrete research limitations. Use inline [paperId] or [evidenceId] citations; no bibliography or invented hyperlinks. Return {"sections":[{"title":"...","text":"...","citations":["source-id"]}]}.`, value => ({ sections: sections(value, papers, evidence) }), () => ({ sections: [{ title: pack.title, text: 'Synthetic smoke chapter.', citations: pack.paperIds }] })))
  const integrated = await run.model('lr-integration', json({ topic: input.query, outline, chapters, limitations: run.warnings }), `${SCIENTIFIC_RULES}
Integrate the supplied review chapters into a cohesive manuscript with an introduction, comparative synthesis, limitations and conclusion. Preserve grounded citations, remove repetition and maintain subject-specific rigor. Return {"sections":[{"title":"...","text":"...","citations":["source-id"]}]}. Do not claim complete literature coverage.`, value => ({ sections: sections(value, papers, evidence) }), () => ({ sections: chapters.flatMap(chapter => chapter.sections) }), options.integrationTimeoutMs)
  data['chapters'] = json(chapters)
  data['manuscript'] = json(integrated)
  return result(outline.title, `Full review generated from ${papers.length} acquired papers.`, papers, run.warnings, renderSections(integrated.sections), data, !finalCoverage.sufficient)
}
