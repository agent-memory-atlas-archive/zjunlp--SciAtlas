/** Nine-stage idea generation with graph evidence, mechanism transfer, and novelty feedback. */
import type { ResearchRun } from '../research/runtime.ts'
import { decodePapers, parallelMap, citations, paperCards } from '../research/runtime.ts'
import type { Paper } from '../research/types.ts'
import { mergePapers, paperIdentity } from '../research/search.ts'
import { object, array, text, json, errorText } from '../research/json.ts'
import { SCIENTIFIC_RULES, upstream } from '../research/prompts.ts'
import { strings, resolveOptions, result } from './common.ts'

interface Inspiration { candidateId: string; paperId: string; domain: string; radius: string; combinationPoints: string[]; combinationPlan: string; justification: string }
interface Idea { title: string; motivation: string; methodOverview: string; proposedMethod: { name: string; description: string }[]; analogyMapping: { paperId: string; concept: string; adaptation: string }[]; experimentPlan: string[]; citations: string[] }

/** Validate research proposals and retain source identities, rather than model-generated URLs. */
function decodeIdeas(value: unknown, papers: readonly Paper[], count: number) {
  const raw = object(value)
  const items = array(raw['ideas'], 'ideas')
  if (!items.length || items.length > count) throw new Error('Idea count is outside the requested budget')
  return { ideas: items.map(item => {
    const idea = object(item)
    const method = array(idea['proposedMethod'], 'proposedMethod').map(entry => { const component = object(entry); return { name: text(component['name']), description: text(component['description']) } })
    if (!method.length) throw new Error('Idea must specify a method')
    const references = citations(idea['citations'], papers)
    if (!references.length) throw new Error('Idea must have acquired supporting references')
    const analogy = array(idea['analogyMapping'], 'analogyMapping').map(entry => { const mapped = object(entry); const paperId = text(mapped['paperId']); citations([paperId], papers); return { paperId, concept: text(mapped['concept']), adaptation: text(mapped['adaptation']) } })
    const result: Idea = { title: text(idea['title']), motivation: text(idea['motivation']), methodOverview: text(idea['methodOverview']), proposedMethod: method, analogyMapping: analogy, experimentPlan: strings(idea['experimentPlan'], 'experimentPlan'), citations: references }
    if (!result.experimentPlan.length) throw new Error('Idea requires an experiment plan')
    for (const match of JSON.stringify(result).matchAll(/\[(p-[\w-]+)\]/g)) if (!references.includes(match[1]!)) throw new Error('Unresolved idea inline citation')
    if (/https?:\/\//.test(JSON.stringify(result))) throw new Error('Use acquired source ids instead of model-generated reference URLs')
    return result
  }) }
}

/** Build an evidence graph and keep predicted relations explicitly distinguished from citations. */
export async function runGenerate(run: ResearchRun) {
  const input = run.input
  const options = resolveOptions(input, run.services.config)
  const docs = await run.inputDocuments()
  const topic = input.query || docs.map(doc => doc.title).join('; ')
  const refinementInput = { raw_query: topic, attachedPapers: docs.map(doc => ({ title: doc.title, abstract: doc.abstract })) }
  const refinement = await run.model('step1-refinement', json(refinementInput), `${SCIENTIFIC_RULES}\n${upstream('QUERY_REFINEMENT', { raw_query: JSON.stringify(refinementInput) })}`, value => {
    const raw = object(value)
    return { research_problem: text(raw['research_problem']), keywords: strings(raw['keywords'], 'keywords'), refined_research_question: text(raw['refined_research_question']) }
  }, () => ({ research_problem: topic, keywords: [topic], refined_research_question: topic }))
  const queries = await run.model('step1-diversification', json({ ...refinement, count: options.seedNumQueries }), `${SCIENTIFIC_RULES}\n${upstream('SEED_QUERY_DIVERSIFICATION', { research_problem: refinement.research_problem, refined_query: refinement.refined_research_question, keywords: refinement.keywords.join(', '), k: options.seedNumQueries })}`, value => {
    const found = strings(object(value)['queries'], 'queries')
    if (!found.length || found.length > options.seedNumQueries || new Set(found).size !== found.length) throw new Error('Invalid diverse seed queries')
    return { queries: found }
  }, () => ({ queries: Array.from({ length: options.seedNumQueries }, (_, index) => `${topic} facet ${index + 1}`) }))
  const currentYear = new Date().getUTCFullYear()
  const seedsFromQuery = await parallelMap(queries.queries, run.services.config.maxStageConcurrency, run.signal, async (query, index) => {
    const candidates = await run.search(`step1-query-${index}`, { ...input, query, topK: Math.min(run.services.api.maxTopK, Math.max(options.seedPerQuery * (input.workflow === 'flash' ? 3 : 6), input.workflow === 'flash' ? 3 : 12, options.anchorTopK)), keywords: input.keywords?.length ? input.keywords : refinement.keywords.map(keyword => ({ text: keyword, relevance: 'high' as const })) })
    return candidates.filter(paper => paper.year === undefined || paper.year >= currentYear - options.seedRecentYears || (paper.citationCount ?? 0) >= 1000).slice(0, options.seedPerQuery)
  })
  const uploaded: Paper[] = docs.map((doc, index) => { const paper = { title: doc.title, abstract: doc.abstract }; return { ...paper, paperId: paperIdentity(paper), sources: [`uploaded-pdf-${index}`] } })
  const pool = mergePapers([uploaded, ...seedsFromQuery], run.services.config.maxPapers)
  if (!pool.length) return result('SciAtlas Idea Generation', 'No eligible seed papers were found. No grounded idea was generated.', [], run.warnings, '', {}, true)
  const selection = await run.model('step1-seed-selection', json({ topic, candidates: paperCards(pool, options.llmPaperLimit), count: options.kStep1 }), `${SCIENTIFIC_RULES}\n${upstream('SEED_SELECTION', { research_topic: topic, candidate_list: JSON.stringify(paperCards(pool, options.llmPaperLimit)), k: options.kStep1 })}\nUse the supplied paperId as selected_paper_ids. Reject off-topic influential papers; include uploaded seed papers when they directly serve the user's topic.`, value => {
    const raw = object(value)
    const ids = citations(raw['selected_paper_ids'], pool)
    if (!ids.length || ids.length > options.kStep1) throw new Error('Invalid seed selection count')
    return { selected_paper_ids: ids, justification: typeof raw['justification'] === 'string' ? raw['justification'] : '' }
  }, () => ({ selected_paper_ids: pool.slice(0, options.kStep1).map(p => p.paperId), justification: 'Smoke fixture selection' }))
  const seeds = pool.filter(paper => selection.selected_paper_ids.includes(paper.paperId))
  await run.store.write('step1_seed_papers.json', json({ refinement, queries, seeds }))

  let graphPapers = [...seeds]
  const retrievalEdges: { from: string; to: string; kind: string }[] = []
  const citationEdges: { from: string; to: string; kind: string }[] = []
  const visited = new Set<string>()
  let frontier = seeds
  let expansion = 0
  while (frontier.length && graphPapers.length < options.graphBudgetMax) {
    const nodes = frontier.filter(paper => !visited.has(paper.paperId))
    if (!nodes.length) break
    const expanded = await parallelMap(nodes, run.services.config.maxStageConcurrency, run.signal, async paper => {
      visited.add(paper.paperId)
      const neighbors = await run.search(`step2-expand-${expansion}-${paper.paperId}`, { query: refinement.refined_research_question, titles: [{ title: paper.title, relevance: 'high' }], topK: Math.min(run.services.api.maxTopK, input.workflow === 'flash' ? 10 : 40), rankingProfile: 'balanced', bias: { citation: 'high', related: 'high', exploration: 'low' }, ...(input.timeRange ? { timeRange: input.timeRange } : {}) })
      let references: Paper[] = []
      if (run.services.config.semanticScholarEnabled && !input.smoke && (paper.doi || paper.id)) {
        const ref = await run.stage(`step2-references-${paper.paperId}`, { paper, limit: input.workflow === 'flash' ? 10 : 40 }, value => {
          const raw = object(value); return { papers: decodePapers(raw['papers']), error: typeof raw['error'] === 'string' ? raw['error'] : '' }
        }, async () => {
          try { return { papers: (await run.request(`step2-references-${paper.paperId}`, () => run.services.referenceSearch(paper, input.workflow === 'flash' ? 10 : 40, run.signal))).papers, error: '' } }
          catch (error) { run.signal.throwIfAborted(); return { papers: [], error: errorText(error) } }
        })
        references = ref.papers
        if (ref.error) run.warnings.push(`Citation lookup unavailable for ${paper.paperId}: ${ref.error}`)
      }
      return { origin: paper, neighbors: mergePapers([references, neighbors], run.services.config.maxPapers), references }
    })
    const candidatePool = mergePapers(expanded.map(item => item.neighbors), run.services.config.maxPapers).filter(paper => !graphPapers.some(existing => existing.paperId === paper.paperId))
    if (!candidatePool.length) break
    const remaining = Math.min(options.graphBudgetMax - graphPapers.length, input.workflow === 'flash' ? nodes.length * 2 : nodes.length * 4)
    const keep = await run.model(`step2-select-${expansion}`, json({ topic, nodes: paperCards(nodes, options.llmPaperLimit), candidates: paperCards(candidatePool, options.llmPaperLimit), count: remaining }), `${SCIENTIFIC_RULES}\n${upstream('PREDECESSOR_SELECTION', { research_topic: topic, node_title: nodes.map(item => item.title).join('; '), node_abstract: nodes.map(item => item.abstract ?? '').join('\n'), candidate_list: JSON.stringify(paperCards(candidatePool, options.llmPaperLimit)), num_candidate: remaining })}\nSelect relevant predecessor/parallel/forward evidence within count. A title-anchored graph search alone does not verify a citation relation. Use supplied paperId identifiers. Return {"selected_paper_ids":["p-..."]}.`, value => {
      const ids = citations(object(value)['selected_paper_ids'], candidatePool)
      if (ids.length > remaining) throw new Error('Graph selection exceeds the remaining budget')
      return { selected_paper_ids: ids }
    }, () => ({ selected_paper_ids: candidatePool.slice(0, remaining).map(p => p.paperId) }))
    frontier = candidatePool.filter(paper => keep.selected_paper_ids.includes(paper.paperId))
    for (const item of expanded) for (const paper of frontier) if (item.neighbors.some(neighbor => neighbor.paperId === paper.paperId)) retrievalEdges.push({ from: item.origin.paperId, to: paper.paperId, kind: 'retrieval' })
    for (const item of expanded) for (const paper of frontier) if (item.references.some(reference => reference.paperId === paper.paperId)) citationEdges.push({ from: item.origin.paperId, to: paper.paperId, kind: 'citation' })
    graphPapers = mergePapers([graphPapers, frontier], options.graphBudgetMax)
    expansion++
  }
  if (graphPapers.length < options.graphBudgetMin) run.warnings.push(`Graph contains ${graphPapers.length} papers, below the requested minimum ${options.graphBudgetMin}; no unrelated nodes were added.`)
  const fields = await parallelMap(graphPapers, run.services.config.maxStageConcurrency, run.signal, (paper, index) => run.model(`step2-idea-fields-${index}`, json({ paper: paperCards([paper], 1)[0]! }), `${SCIENTIFIC_RULES}\n${upstream('IDEA_FIELDS_EXTRACTION', { paper_title: paper.title, paper_abstract: paper.abstract ?? 'Abstract unavailable; do not infer unprovided methods.' })}`, value => {
    const raw = object(value)
    return Object.fromEntries(['Problem', 'Existing Methods', 'Motivation', 'Proposed Method', 'Experiment Plan'].map(key => [key, text(raw[key], key)]))
  }, () => ({ Problem: topic, 'Existing Methods': 'Synthetic prior approach', Motivation: 'Synthetic gap', 'Proposed Method': 'Synthetic mechanism', 'Experiment Plan': 'Synthetic control experiment' })))
  const topology = await run.model('step2-topology', json({ topic, papers: paperCards(graphPapers, options.llmPaperLimit), retrievalEdges, citationEdges, fields }), `${SCIENTIFIC_RULES}
Assign research phases and candidate predecessor relations using the supplied scientific content. These are inferred scientific relations, not verified citation edges. Return {"phases":[{"paperId":"...","phase":"..."}],"inferredEdges":[{"from":"paperId","to":"paperId","rationale":"..."}]}. Use known ids only; never fabricate a bibliographic citation.`, value => {
    const raw = object(value)
    const phases = array(raw['phases']).map(item => { const node = object(item); const id = text(node['paperId']); citations([id], graphPapers); return { paperId: id, phase: text(node['phase']) } })
    const edges = array(raw['inferredEdges']).map(item => { const edge = object(item); const from = text(edge['from']); const to = text(edge['to']); citations([from, to], graphPapers); if (from === to) throw new Error('Graph self-edge is invalid'); return { from, to, rationale: text(edge['rationale']), kind: 'inferred-predecessor' } })
    return { phases, inferredEdges: edges }
  }, () => ({ phases: graphPapers.map(paper => ({ paperId: paper.paperId, phase: 'Smoke phase' })), inferredEdges: [] }))
  const graph = { papers: graphPapers, fields, ...topology, retrievalEdges, citationEdges }
  await run.store.write('step2_research_graph.json', json(graph))
  if (!citationEdges.length) run.warnings.push('Graph relations are retrieved or inferred links; no bibliographic citation edges could be independently verified.')
  const trend = await run.model('step3-trend', json(graph), `${SCIENTIFIC_RULES}\n${upstream('TREND_SUMMARY', { research_topic: topic, serialized_graph: JSON.stringify(graph) })}\nReturn JSON {"trends":"...","turningPoints":["..."],"limitations":["..."],"scientificGaps":["..."],"citations":["paperId"]}. Treat inferred graph links as hypotheses, not confirmed citations.`, value => {
    const raw = object(value)
    return { trends: text(raw['trends']), turningPoints: strings(raw['turningPoints'], 'turningPoints'), limitations: strings(raw['limitations'], 'limitations'), scientificGaps: strings(raw['scientificGaps'], 'scientificGaps'), citations: citations(raw['citations'], graphPapers) }
  }, () => ({ trends: 'Synthetic smoke trend', turningPoints: [], limitations: [], scientificGaps: ['Synthetic mechanistic gap'], citations: graphPapers.map(p => p.paperId) }))
  const rss = await run.model('step4-rss', json({ graph, trend, topic }), `${SCIENTIFIC_RULES}\n${upstream('RSS_EXTRACTION', { research_topic: topic, seed_paper_or_target_direction: topic, gor_graph: JSON.stringify(graph), trend: JSON.stringify(trend) })}`, value => {
    const raw = object(value)
    return { target_domain: text(raw['target_domain']), core_gap: text(raw['core_gap']), ...Object.fromEntries(['objects', 'relations', 'bottlenecks', 'unresolved_tensions', 'desired_solution_properties', 'possible_mechanism_types', 'inspiration_search_hints'].map(key => [key, strings(raw[key], key)])) }
  }, () => ({ target_domain: topic, core_gap: 'Synthetic scientific gap', objects: ['Synthetic object'], relations: ['Synthetic relation'], bottlenecks: ['Synthetic bottleneck'], unresolved_tensions: [], desired_solution_properties: ['Falsifiability'], possible_mechanism_types: ['Synthetic mechanism'], inspiration_search_hints: ['Synthetic domain'] }))
  const sameField = input.workflow === 'flash' ? 1 : 2
  const crossDomains = input.workflow === 'flash' ? 1 : 2
  const gateDecode = (value: unknown) => {
    const raw = object(value)
    if (typeof raw['use_same_field'] !== 'boolean' || typeof raw['use_cross_domain'] !== 'boolean') throw new Error('Invalid inspiration gate')
    return { use_same_field: raw['use_same_field'], use_cross_domain: raw['use_cross_domain'], rationale: text(raw['rationale']) }
  }
  const defaultGate = { use_same_field: true, use_cross_domain: true, rationale: 'Flash compresses the radius gate using the preset same-field and cross-domain paths.' }
  const gate = input.workflow === 'flash' ? await run.stage('step5-radius-plan', { workflow: input.workflow, rss }, gateDecode, async () => defaultGate) : await run.model('step5-radius-plan', json({ rss, trend }), `${SCIENTIFIC_RULES}\n${upstream('INSPIRATION_RADIUS_GATE', { rss_json: JSON.stringify(rss), trend: JSON.stringify(trend), same_field_top_k: sameField, num_cross_domains: crossDomains, cross_domain_top_k_per_domain: 2 })}`, gateDecode, () => defaultGate)

  let finalIdeas: Idea[] = []
  let finalNovelty: { title: string; level: string; justification: string; suggestions: string[]; checked: boolean }[] = []
  let lastInspirations: Inspiration[] = []
  let allPapers = graphPapers
  let feedback = ''
  for (let round = 0; round <= options.maxNoveltyFeedbackRounds; round++) {
    const prefix = `r${round}`
    const candidateGroups: { papers: Paper[]; domain: string; radius: string }[] = []
    if (gate.use_same_field) candidateGroups.push({ papers: await run.search(`step6-same-${prefix}`, { query: `${rss.core_gap} ${feedback}`, topK: sameField, domain: input.domain ?? rss.target_domain }), domain: 'same_field', radius: 'R1' })
    if (gate.use_cross_domain) {
      const domains = await run.model(`step6-domains-${prefix}`, json({ rss, feedback, count: crossDomains }), `${SCIENTIFIC_RULES}\n${upstream('DOMAIN_QUERY_GEN', { problem_text: `${rss.core_gap}\n${feedback}`, num_domains: crossDomains })}`, value => {
        const raw = object(value)
        const domains = array(raw['domain_queries']).map(item => { const domain = object(item); return { domain: text(domain['domain']), query_text: text(domain['query_text']) } })
        if (!domains.length || domains.length > crossDomains) throw new Error('Cross-domain plan exceeds its budget')
        return { domain_queries: domains }
      }, () => ({ domain_queries: [{ domain: 'Synthetic other domain', query_text: `${topic} synthetic mechanism transfer` }] }))
      candidateGroups.push(...await parallelMap(domains.domain_queries, run.services.config.maxStageConcurrency, run.signal, async (domain, index) => ({ papers: await run.search(`step6-cross-${prefix}-${index}`, { query: domain.query_text, domain: domain.domain, topK: input.workflow === 'flash' ? 1 : 2 }), domain: domain.domain, radius: 'R2' })))
    }
    allPapers = mergePapers([allPapers, ...candidateGroups.map(group => group.papers)], run.services.config.maxPapers)
    const candidates = candidateGroups.flatMap(group => group.papers.map(paper => ({ paper, domain: group.domain, radius: group.radius })))
    const inspirations = (await parallelMap(candidates, run.services.config.maxStageConcurrency, run.signal, async (candidate, index): Promise<Inspiration | null> => {
      try {
        const extracted = await run.model(`step7-analogy-${prefix}-${index}`, json({ problem: refinement.research_problem, candidate: paperCards([candidate.paper], 1)[0]! }), `${SCIENTIFIC_RULES}\n${upstream('COMBINATION_CHECK', { problem_text: refinement.research_problem, title: candidate.paper.title, abstract: candidate.paper.abstract ?? 'No abstract supplied.' })}`, value => {
          const raw = object(value)
          const plausible = raw['is_combination_plausible'] ?? raw['plausible']
          if (typeof plausible !== 'boolean') throw new Error('Invalid analogy plausibility')
          return { plausible, points: strings(raw['combination_points'] ?? raw['points'], 'combination_points'), plan: plausible ? text(raw['combination_plan'] ?? raw['plan']) : String(raw['combination_plan'] ?? raw['plan'] ?? ''), justification: text(raw['justification']) }
        }, () => ({ plausible: true, points: ['Synthetic mechanism transfer'], plan: 'Synthetic adaptation plan', justification: 'Smoke validation' }))
        return extracted.plausible ? { candidateId: `${prefix}-${index + 1}`, paperId: candidate.paper.paperId, domain: candidate.domain, radius: candidate.radius, combinationPoints: extracted.points, combinationPlan: extracted.plan, justification: extracted.justification } : null
      } catch (error) { run.signal.throwIfAborted(); run.warnings.push(`Analogy extraction failed for ${candidate.paper.paperId}: ${errorText(error)}`); return null }
    })).filter((item): item is Inspiration => item !== null)
    if (!inspirations.length) run.warnings.push('No plausible external inspiration survived; proposals use the acquired target-domain graph only.')
    let selected = inspirations
    if (input.workflow === 'flash') selected = [...inspirations].sort((a, b) => (a.radius === 'R2' ? 0 : 1) - (b.radius === 'R2' ? 0 : 1)).slice(0, 2)
    else if (inspirations.length > 1) {
      const best = await run.model(`step8-selection-${prefix}`, json({ rss, trend, inspirations }), `${SCIENTIFIC_RULES}\n${upstream('BEST_INSPIRATION_SELECTION', { rss_json: JSON.stringify(rss), trend: JSON.stringify(trend), inspiration_list: JSON.stringify(inspirations) })}\nUse the supplied candidateId strings, and return {"selected_candidate_ids":["candidateId"],"justification":"..."}.`, value => {
        const raw = object(value); const ids = strings(raw['selected_candidate_ids'], 'selected_candidate_ids')
        if (!ids.length || ids.length > 3 || ids.some(id => !inspirations.some(item => item.candidateId === id))) throw new Error('Invalid selected inspiration ids')
        return { selected_candidate_ids: ids, justification: text(raw['justification']) }
      }, () => ({ selected_candidate_ids: inspirations.slice(0, 2).map(item => item.candidateId), justification: 'Smoke selection' }))
      selected = inspirations.filter(item => best.selected_candidate_ids.includes(item.candidateId))
    }
    await run.store.write(`step8_selected_inspirations_${prefix}.json`, json(selected))
    lastInspirations = selected
    const generated = await run.model(`step9-generation-${prefix}`, json({ topic, graph, rss, trend, inspirations: selected, feedback, referenceCatalog: paperCards(allPapers, run.services.config.maxPapers), ideaCount: options.ideaCount }), `${SCIENTIFIC_RULES}\n${upstream('IDEA_GENERATION', { original_query: topic, serialized_graph: JSON.stringify(graph), rss_json: JSON.stringify(rss), trend: JSON.stringify(trend), inspiration_list: JSON.stringify(selected), novelty_feedback: feedback || 'None', reference_catalog: JSON.stringify(paperCards(allPapers, run.services.config.maxPapers)), idea_count: options.ideaCount })}\nFor this tool use source identifiers instead of citation URLs and normalize the result to {"ideas":[{"title":"...","motivation":"...","methodOverview":"...","proposedMethod":[{"name":"...","description":"..."}],"analogyMapping":[{"paperId":"...","concept":"...","adaptation":"..."}],"experimentPlan":["..."],"citations":["paperId"]}]}. Include one primary falsifiable hypothesis, controls and a failure criterion.`, value => decodeIdeas(value, allPapers, options.ideaCount), () => ({ ideas: Array.from({ length: options.ideaCount }, (_, index) => ({ title: `SMOKE idea ${index + 1}`, motivation: 'Synthetic scientific hypothesis for workflow testing.', methodOverview: 'A synthetic controlled mechanism-transfer experiment.', proposedMethod: [{ name: 'Controlled intervention', description: 'Vary one factor while holding others fixed.' }], analogyMapping: selected.map(item => ({ paperId: item.paperId, concept: 'Synthetic mechanism', adaptation: item.combinationPlan })), experimentPlan: ['Pre-register a falsifiable hypothesis and compare intervention with controls.'], citations: allPapers.slice(0, 3).map(p => p.paperId) })) }))
    finalIdeas = generated.ideas
    finalNovelty = await parallelMap(finalIdeas, run.services.config.maxStageConcurrency, run.signal, async (idea, index) => {
      try {
        const checked = await run.model(`step9-novelty-${prefix}-${index}`, json({ idea, references: paperCards(allPapers.filter(p => idea.citations.includes(p.paperId)), options.llmPaperLimit) }), `${SCIENTIFIC_RULES}\n${upstream('NOVELTY_CHECK', { idea_json: JSON.stringify(idea), reference_list: JSON.stringify(paperCards(allPapers.filter(p => idea.citations.includes(p.paperId)), options.llmPaperLimit)) })}`, value => {
          const raw = object(value); const level = text(raw['novelty_level'] ?? raw['level']); if (!['high', 'medium', 'low', 'none'].includes(level)) throw new Error('Invalid novelty level')
          return { level, justification: text(raw['justification']), suggestions: strings(raw['improvement_suggestions'] ?? raw['suggestions'], 'improvement_suggestions') }
        }, () => ({ level: 'medium', justification: 'Synthetic novelty fixture; not a scientific claim.', suggestions: ['Specify a more discriminating mechanism experiment.'] }))
        return { title: idea.title, ...checked, checked: true }
      } catch (error) { run.signal.throwIfAborted(); run.warnings.push(`Novelty check failed for ${idea.title}: ${errorText(error)}`); return { title: idea.title, level: 'unchecked', justification: errorText(error), suggestions: [], checked: false } }
    })
    await run.store.write(`step9_ideas_${prefix}.json`, json({ ideas: finalIdeas, novelty: finalNovelty }))
    if (!finalNovelty.some(item => item.checked && (item.level === 'low' || item.level === 'none'))) break
    feedback = finalNovelty.map(item => `${item.title}: ${item.justification}\n${item.suggestions.join('\n')}`).join('\n\n')
  }
  const body = finalIdeas.map((idea, index) => `## ${index + 1}. ${idea.title}\n\n${idea.motivation}\n\n### Proposed Method\n\n${idea.methodOverview}\n\n${idea.proposedMethod.map(item => `- **${item.name}**: ${item.description}`).join('\n')}\n\n### Mechanism Transfer\n\n${idea.analogyMapping.map(item => `- [${item.paperId}] ${item.concept}: ${item.adaptation}`).join('\n')}\n\n### Experiment Plan\n\n${idea.experimentPlan.map((item, number) => `${number + 1}. ${item}`).join('\n')}\n\n### Novelty Check\n\n${finalNovelty[index]!.checked ? finalNovelty[index]!.level : 'Not completed'}: ${finalNovelty[index]!.justification}\n\n${finalNovelty[index]!.suggestions.map(item => `- ${item}`).join('\n')}\n\nSources: ${idea.citations.join(', ')}`).join('\n\n')
  return result('SciAtlas Research Ideas', `${finalIdeas.length} research proposal(s) generated; novelty assessments are limited to the acquired reference sample.`, allPapers, run.warnings, body, { refinement, graph, trend, rss, inspirations: lastInspirations, ideas: finalIdeas, novelty: finalNovelty }, finalNovelty.some(item => !item.checked) || graphPapers.length < options.graphBudgetMin)
}
