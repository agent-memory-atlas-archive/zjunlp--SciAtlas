/** Six research tools sharing one durable runner and DSH task lifecycle. */
import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolRunContext } from '@deepseek-ai/dsh-tools'
import type { JobHandle, JobOutcome } from '@deepseek-ai/dsh-jobs'
import { createHash } from 'node:crypto'
import { join } from 'node:path'
import { RunStore, newRunId } from '../research/store.ts'
import { ResearchRun } from '../research/runtime.ts'
import { ResearchTasks, liveServices } from '../research/dsh.ts'
import type { ResearchCommand, ResearchConfig, ResearchInput, ResearchOptions, ResearchServices, ResearchResult } from '../research/types.ts'
import type { JsonObject } from '../research/json.ts'
import { object, array, text, json, errorText } from '../research/json.ts'
import { PROMPT_VERSION } from '../research/prompts.ts'
import { resolveOptions } from '../workflows/common.ts'
import { runChannel } from '../workflows/channels.ts'
import { runLiterature } from '../workflows/literature.ts'
import { runEvaluate } from '../workflows/evaluate.ts'
import { runGenerate } from '../workflows/generate.ts'
import { PAPER_VALUE_SCHEMA, RELEVANCE_VALUE_SCHEMA, buildSearchOptions, truncateText } from './shared.ts'
import type { SciAtlasToolDeps } from './shared.ts'
import type { Relevance } from '../relevance.ts'

const NUMBER_OPTIONS = ['probeTopK', 'roundTopK', 'round1ActionLimit', 'round2ActionLimit', 'llmPaperLimit', 'kgTopK', 's2TopK', 'manifestTopK', 'groundingFinalTopK', 'groundingCandidateK', 'groundingRerankTopN', 'maxParagraphsPerPaperPerQuery', 'rerankBatchSize', 'maxReviewers', 'reviewerSelectionTopK', 'reviewerMaxWorkers', 'reviewMaxWorkers', 'maxEvidenceCardsPerSection', 'maxEvidenceCardsPerQuery', 'maxExperimentCardsPerSection', 'maxEvidenceCharsPerSection', 'seedRecentYears', 'graphBudgetMax', 'graphBudgetMin', 'kStep1', 'seedNumQueries', 'seedPerQuery', 'anchorTopK', 'ideaCount', 'maxNoveltyFeedbackRounds', 'llmTimeoutMs', 'outlineTimeoutMs', 'integrationTimeoutMs', 'reviewTimeoutMs', 'rubricTimeoutMs'] as const
const BOOLEAN_OPTIONS = ['shortReport', 'disableLlmRanking', 'disableReviewLlm', 'saveIntermediate'] as const

const OPTION_SCHEMAS = {
  ...Object.fromEntries(NUMBER_OPTIONS.map(name => [name, { type: 'integer' as const, description: `${name}: override the workflow preset.` }])),
  ...Object.fromEntries(BOOLEAN_OPTIONS.map(name => [name, { type: 'boolean' as const }])),
  subjectDomain: { type: 'string', enum: ['general', 'chemistry', 'biology'] },
  reportStopAfter: { type: 'string', enum: ['bibliography', 'outline', 'packs', 'full'] },
} as const

const LONG_COMMANDS = new Set<ResearchCommand>(['literature-review', 'idea-evaluate', 'idea-generate'])

/** Validate a persisted or model-supplied option object. */
function decodeOptions(value: unknown): ResearchOptions {
  const raw = object(value)
  const options: ResearchOptions = {}
  for (const key of Object.keys(raw)) if (!(key in OPTION_SCHEMAS)) throw new Error(`Unsupported research option ${key}`)
  for (const key of NUMBER_OPTIONS) if (raw[key] !== undefined) { if (typeof raw[key] !== 'number') throw new Error(`Invalid ${key}`); options[key] = raw[key] }
  for (const key of BOOLEAN_OPTIONS) if (raw[key] !== undefined) { if (typeof raw[key] !== 'boolean') throw new Error(`Invalid ${key}`); options[key] = raw[key] }
  if (raw['subjectDomain'] !== undefined) { const domain = raw['subjectDomain']; if (domain !== 'general' && domain !== 'chemistry' && domain !== 'biology') throw new Error('Invalid subjectDomain'); options.subjectDomain = domain }
  if (raw['reportStopAfter'] !== undefined) { const stop = raw['reportStopAfter']; if (stop !== 'bibliography' && stop !== 'outline' && stop !== 'packs' && stop !== 'full') throw new Error('Invalid reportStopAfter'); options.reportStopAfter = stop }
  return options
}

/** Validate an input loaded from the manifest or built from tool arguments. */
export function decodeResearchInput(value: unknown): ResearchInput {
  const raw = object(value)
  const command = text(raw['command'])
  const commands: ResearchCommand[] = ['idea-grounding', 'trend-report', 'researcher-review', 'literature-review', 'idea-evaluate', 'idea-generate']
  const matched = commands.find(item => item === command)
  if (!matched) throw new Error('Invalid research command')
  const workflow = raw['workflow']
  if (workflow !== 'flash' && workflow !== 'full') throw new Error('Invalid workflow')
  if (typeof raw['query'] !== 'string' || typeof raw['smoke'] !== 'boolean') throw new Error('Invalid research input')
  const input: ResearchInput = { command: matched, workflow, query: raw['query'], pdfs: array(raw['pdfs']).map(item => text(item, 'PDF path')), options: decodeOptions(raw['options']), smoke: raw['smoke'] }
  if (input.pdfs.length > 10) throw new Error('At most ten input PDFs are supported')
  for (const key of ['author', 'domain', 'targetField'] as const) if (raw[key] !== undefined) input[key] = text(raw[key], key)
  for (const key of ['topK', 'limit'] as const) if (raw[key] !== undefined) { if (typeof raw[key] !== 'number' || !Number.isInteger(raw[key]) || raw[key] < 1) throw new Error(`Invalid ${key}`); input[key] = raw[key] }
  for (const key of ['mergeSameNameAuthors', 'dedupePapers', 'includeAbstract', 'includeEmbeddings'] as const) if (raw[key] !== undefined) { if (typeof raw[key] !== 'boolean') throw new Error(`Invalid ${key}`); input[key] = raw[key] }
  for (const key of ['keywords', 'titles', 'referenceTitles'] as const) if (raw[key] !== undefined) {
    const anchors = array(raw[key]).map((item): { content: string; relevance?: Relevance } => { const anchor = object(item); const relevance = anchor['relevance']; if (relevance !== undefined && relevance !== 'low' && relevance !== 'middle' && relevance !== 'high') throw new Error('Invalid anchor relevance'); return { content: text(anchor[key === 'keywords' ? 'text' : 'title']), ...(relevance ? { relevance } : {}) } })
    if (key === 'keywords') input.keywords = anchors.map(item => ({ text: item.content, ...item.relevance ? { relevance: item.relevance } : {} }))
    else input[key] = anchors.map(item => ({ title: item.content, ...item.relevance ? { relevance: item.relevance } : {} }))
  }
  if (raw['timeRange'] !== undefined) { const range = object(raw['timeRange']); input.timeRange = { ...(range['after'] !== undefined ? { after: text(range['after']) } : {}), ...(range['before'] !== undefined ? { before: text(range['before']) } : {}) } }
  if (raw['rankingProfile'] !== undefined) { const profile = raw['rankingProfile']; if (profile !== 'precision' && profile !== 'balanced' && profile !== 'discovery' && profile !== 'impact') throw new Error('Invalid rankingProfile'); input.rankingProfile = profile }
  if (raw['retrievalMode'] !== undefined) { const mode = raw['retrievalMode']; if (mode !== 'keyword' && mode !== 'semantic' && mode !== 'title' && mode !== 'hybrid') throw new Error('Invalid retrievalMode'); input.retrievalMode = mode }
  if (raw['bias'] !== undefined) {
    input.bias = {}
    const bias = object(raw['bias'])
    for (const key of ['keyword', 'nonSeedKeyword', 'citation', 'related', 'authorship', 'coauthorship', 'cooccurrence', 'exploration'] as const) if (bias[key] !== undefined) { const level = bias[key]; if (level !== 'high' && level !== 'middle' && level !== 'low') throw new Error('Invalid bias'); input.bias[key] = level }
  }
  if (!input.query.trim() && !input.pdfs.length && !input.author) throw new Error('Provide query/idea, author, or PDF input')
  if (matched === 'idea-evaluate' && (input.pdfs.length > 1 || (input.pdfs.length > 0 && input.query.trim()))) throw new Error('idea-evaluate requires exactly one of idea text or a single PDF')
  if (input.pdfs.length && matched !== 'idea-evaluate' && matched !== 'idea-generate') throw new Error('PDF input is supported by idea-evaluate and idea-generate')
  return input
}

/** Identity includes model/config, effective defaults, and PDF byte digests. */
export async function runIdentity(input: ResearchInput, services: ResearchServices, signal: AbortSignal): Promise<JsonObject> {
  const files: Record<string, string> = {}
  for (const path of input.pdfs) {
    const bytes = await services.readFile(path, signal)
    files[path] = createHash('sha256').update(bytes).digest('hex')
  }
  return object(json({ request: input, effectiveOptions: resolveOptions(input, services.config), config: services.config, api: { baseUrl: services.api.baseUrl, maxTopK: services.api.maxTopK, timeoutMs: services.api.timeoutMs, defaultRetrievalMode: services.api.defaultRetrievalMode }, route: services.route, files, promptVersion: PROMPT_VERSION, calendarYear: new Date().getUTCFullYear() }))
}

/** Execute and persist a complete run; all terminal paths release its lock. */
export async function executeResearch(store: RunStore, input: ResearchInput, services: ResearchServices, identity: JsonObject, resume: boolean, signal: AbortSignal, progress: (stage: string) => void = () => {}): Promise<ResearchResult> {
  const run = new ResearchRun(store, input, services, signal, progress)
  await run.initialize(identity, resume)
  try {
    const result = input.command === 'literature-review' ? await runLiterature(run) : input.command === 'idea-evaluate' ? await runEvaluate(run) : input.command === 'idea-generate' ? await runGenerate(run) : await runChannel(run)
    await run.finish(result)
    return result
  } catch (error) { await run.fail(error); throw error }
  finally { await store.release() }
}

/** Create one native research tool; background controls remain DSH jobs-owned. */
export function createResearchTool(command: ResearchCommand, ctx: Context, api: SciAtlasToolDeps, config: ResearchConfig, tasks: ResearchTasks) {
  const name = `sciatlas_${command.replace(/-/g, '_')}`
  const long = LONG_COMMANDS.has(command)
  return defineTool({
    name,
    description: `${command === 'literature-review' ? 'Run multiround literature retrieval, method/time clustering, evidence packs and optional full review. Flash stops at packs; full writes a manuscript.' : command === 'idea-evaluate' ? 'Evaluate idea text or one PDF using acquired literature, paragraph evidence, historical rubric and simulated reviewer perspectives.' : command === 'idea-generate' ? 'Generate grounded research ideas from a topic and optional PDFs via seeds, research graph, RSS, mechanism transfer and bounded novelty feedback.' : command === 'researcher-review' ? 'Review a researcher using author-paper seeds and title-anchored hybrid graph retrieval. Expanded papers are related evidence, not verified authorship.' : command === 'trend-report' ? 'Retrieve influential research and organize a publication timeline using acquired metadata.' : 'Ground an idea in related literature with precision graph retrieval; similarity alone does not establish novelty.'} ${long ? 'Defaults to a background job; use DSH job tools to read, wait or cancel. Set background=false to wait here.' : ''} Returns report/artifact paths. Resume with resumeRunId, optionally alone.`,
    parameters: {
      query: { type: 'string', description: 'Research topic or idea text.' },
      idea: { type: 'string', description: 'Idea text alias for query; supply only one.' },
      queryFile: { type: 'string', description: 'DSH-accessible UTF-8 text-file path.' },
      author: { type: 'string', description: 'Researcher name or OpenAlex identifier.' },
      pdf: { type: 'string', description: 'Single PDF path for evaluation or generation.' },
      pdfs: { type: 'array', items: { type: 'string' }, description: 'Generation input PDF paths.' },
      workflow: { type: 'string', enum: ['flash', 'full'], description: 'Default flash.' },
      background: { type: 'boolean' }, resumeRunId: { type: 'string' }, smoke: { type: 'boolean', description: 'Synthetic fixture run only, never a scientific result.' },
      domain: { type: 'string' }, targetField: { type: 'string' }, topK: { type: 'integer' }, limit: { type: 'integer' },
      mergeSameNameAuthors: { type: 'boolean' }, dedupePapers: { type: 'boolean' }, includeAbstract: { type: 'boolean' }, includeEmbeddings: { type: 'boolean' },
      keywords: { type: 'array', items: { type: 'object', additionalProperties: false, properties: { text: { type: 'string', required: true }, relevance: RELEVANCE_VALUE_SCHEMA } } },
      titles: { type: 'array', items: { type: 'object', additionalProperties: false, properties: { title: { type: 'string', required: true }, relevance: RELEVANCE_VALUE_SCHEMA } } },
      referenceTitles: { type: 'array', items: { type: 'object', additionalProperties: false, properties: { title: { type: 'string', required: true }, relevance: RELEVANCE_VALUE_SCHEMA } } },
      timeRange: { type: 'object', additionalProperties: false, properties: { after: { type: 'string' }, before: { type: 'string' } } },
      retrievalMode: { type: 'string', enum: ['keyword', 'semantic', 'title', 'hybrid'] },
      rankingProfile: { type: 'string', enum: ['precision', 'balanced', 'discovery', 'impact'] },
      bias: { type: 'object', additionalProperties: false, properties: { keyword: RELEVANCE_VALUE_SCHEMA, nonSeedKeyword: RELEVANCE_VALUE_SCHEMA, citation: RELEVANCE_VALUE_SCHEMA, related: RELEVANCE_VALUE_SCHEMA, authorship: RELEVANCE_VALUE_SCHEMA, coauthorship: RELEVANCE_VALUE_SCHEMA, cooccurrence: RELEVANCE_VALUE_SCHEMA, exploration: RELEVANCE_VALUE_SCHEMA } },
      options: { type: 'object', additionalProperties: false, properties: OPTION_SCHEMAS },
    },
    output: {
      schema: { type: 'object', additionalProperties: false, properties: {
        kind: { type: 'string', required: true, enum: ['background', 'result'] }, runId: { type: 'string', required: true },
        jobId: { type: 'string' }, status: { type: 'string', required: true }, summary: { type: 'string', required: true },
        artifactDir: { type: 'string', required: true }, reportPath: { type: 'string', required: true }, resultPath: { type: 'string', required: true },
        warnings: { type: 'array', required: true, items: { type: 'string' } }, papers: { type: 'array', required: true, items: PAPER_VALUE_SCHEMA },
      } },
      render: (_args, value) => [{ type: 'text', text: `${value.summary}\nRun: ${value.runId}${value.jobId ? `\nJob: ${value.jobId}; use job_output (wait:true to wait), job_list, or job_kill.` : ''}\nStatus: ${value.status}\nReport: ${value.reportPath}\nStructured result: ${value.resultPath}${value.warnings.length ? `\n${value.warnings.map(warning => `- ${warning}`).join('\n')}` : ''}` }],
      presentationMeta: (_args, value) => ({ runId: value.runId, jobId: value.jobId ?? null, status: value.status, reportPath: value.reportPath, summary: value.summary, warningCount: value.warnings.length }),
    },
    timeoutMs: config.workflowTimeoutMs + 30000,
    isConcurrencySafe: () => true,
    async execute(args, exec: ToolRunContext) {
      if (args.pdf !== undefined && args.pdfs !== undefined) throw new Error('Provide pdf or pdfs, not both')
      tasks.assertAvailable()
      exec.signal.throwIfAborted()
      if (!exec.agent) throw new Error('Research tools require a DSH session owner')
      const runId = args.resumeRunId ?? newRunId()
      const store = new RunStore(config.runsDir, runId, exec.agent.id)
      const services = liveServices(ctx, exec, api, config, runId)
      let raw: JsonObject
      const argsJson = object(json(args))
      if (args.resumeRunId && !args.query && !args.idea && !args.author && !args.pdf && !args.pdfs && !args.queryFile) {
        raw = object((await store.read()).input['request'])
        for (const [key, value] of Object.entries(argsJson)) if (!['resumeRunId', 'background'].includes(key)) raw[key] = value
      } else {
        if (args.query !== undefined && args.idea !== undefined) throw new Error('Provide query or idea, not both')
        let query = args.query ?? args.idea ?? args.author ?? ''
        if (args.queryFile !== undefined) {
          if (query) throw new Error('Provide query text or queryFile, not both')
          query = new TextDecoder('utf-8', { fatal: true }).decode(await services.readFile(args.queryFile, exec.signal))
          if (query.length > config.maxInputChars) throw new Error('Input text file exceeds research.maxInputChars')
        }
        raw = { ...argsJson, command, query: query.trim(), workflow: args.workflow ?? 'flash', smoke: args.smoke ?? false, options: argsJson['options'] ?? {}, pdfs: args.pdfs ?? (args.pdf ? [args.pdf] : []) }
      }
      const input = decodeResearchInput(raw)
      if (input.command !== command) throw new Error('Resume must use the same research tool')
      if (input.pdfs.length && !config.grobidBaseUrl) throw new Error('PDF input requires research.grobidBaseUrl')
      const options = resolveOptions(input, config)
      if (input.topK && input.topK > api.maxTopK) throw new Error(`topK exceeds configured maxTopK (${api.maxTopK})`)
      for (const count of [options.probeTopK, options.roundTopK, options.kgTopK, options.anchorTopK]) if (count > api.maxTopK) throw new Error('Workflow retrieval budget exceeds configured maxTopK')
      buildSearchOptions(input, api)
      if (!input.smoke) {
        await api.resolveApiKey()
        if (long && (!services.route || !(exec.agent.ctx ?? ctx).get('llm'))) throw new Error('Configure a DSH model route before running research workflows')
      }
      const identity = await runIdentity(input, services, exec.signal)
      if (args.resumeRunId) await store.validateResume(identity)
      const base = { runId, artifactDir: store.dir, reportPath: join(store.dir, 'report.md'), resultPath: join(store.dir, 'result.json') }
      const execute = (signal: AbortSignal, job?: JobHandle) => executeResearch(store, input, services, identity, !!args.resumeRunId, signal, stage => { job?.updateProgress(stage); job?.append(`${stage}\n`, { channel: 'log' }) })
      if (args.background ?? (long && config.background)) {
        const jobId = tasks.background(ctx, exec, runId, `SciAtlas ${command}: ${truncateText(input.query || input.pdfs.join(', '), 80)}`, async (signal, job): Promise<JobOutcome> => {
          try {
            const outcome = await execute(signal, job)
            const summary = JSON.stringify({ ...base, status: outcome.status, summary: outcome.summary, warnings: outcome.warnings })
            job.append(`${summary}\n`, { channel: 'stdout' })
            return { status: outcome.status === 'failed' ? 'failed' : 'completed', detail: outcome.status, result: summary }
          } catch (error) { return { status: signal.aborted ? 'killed' : 'failed', detail: errorText(error), result: JSON.stringify({ ...base, status: signal.aborted ? 'cancelled' : 'failed', error: errorText(error) }) } }
        })
        return { ...base, kind: 'background' as const, jobId, status: 'running', summary: `Started SciAtlas ${command}.`, warnings: input.smoke ? ['Synthetic smoke run.'] : [], papers: [] }
      }
      let outcome: ResearchResult | undefined
      const hooks = tasks.start(runId, async signal => {
        outcome = await execute(AbortSignal.any([signal, exec.signal]))
        return { status: 'completed' }
      })
      const finished = await hooks.done
      if (!outcome) throw new Error(finished.detail ?? 'Research failed')
      return { ...base, kind: 'result' as const, status: outcome.status, summary: outcome.summary, warnings: outcome.warnings, papers: outcome.papers.slice(0, 10).map((paper, index) => { const { paperId: _id, sources: _sources, ...metadata } = paper; return { ...metadata, rank: index + 1 } }) }
    },
    presentCall: args => ({ card: 'generic', kind: 'search', title: `SciAtlas ${command}: ${truncateText(args.query ?? args.idea ?? args.author ?? args.resumeRunId ?? 'PDF input', 80)}` }),
    presentResult: (_args, outcome) => {
      if (outcome.isError || !outcome.meta || typeof outcome.meta !== 'object' || Array.isArray(outcome.meta)) return undefined
      const meta = object(outcome.meta)
      return { card: 'generic', title: `SciAtlas ${command}: ${String(meta['status'])}`, content: [{ type: 'text', text: `${String(meta['summary'])}\n${String(meta['reportPath'])}` }] }
    },
  })
}

export { ResearchTasks }
