/** DSH model, filesystem, and owned background-job integration. */
import type { Context } from '@deepseek-ai/cordis'
import { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { StreamChunk, LlmCallConfig, LlmCallConfigAdapterDefaults } from '@deepseek-ai/dsh-llm'
import type { ToolRunContext } from '@deepseek-ai/dsh-tools'
import type { JobHandle, JobOutcome } from '@deepseek-ai/dsh-jobs'
import type {} from '@deepseek-ai/dsh-fs'
import type {} from '@deepseek-ai/dsh-session'
import { appendPluginRecord } from '@deepseek-ai/dsh-session'
import type { ResearchConfig, ResearchServices, ModelRoute } from './types.ts'
import type { SciAtlasToolDeps } from '../tools/shared.ts'
import type { Json } from './json.ts'
import { object, array, json } from './json.ts'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import { collectPapers } from '../normalize.ts'
import { paperIdentity } from './search.ts'

declare module '@deepseek-ai/dsh-llm' {
  interface MessageSourceMap { 'sciatlas-research': { kind: 'sciatlas-research' } }
}
declare module '@deepseek-ai/dsh-session/types' {
  interface PluginRecordMap {
    /** Exact auxiliary research input, recorded before a prepared model dispatch. */
    'plugin:sciatlas/llm-request': { runId: string; stage: string; route: ModelRoute; config: LlmCallConfig; adapterDefaults: LlmCallConfigAdapterDefaults; system: string; input: Json; maxTokens: number }
    /** Raw auxiliary chunks reconstruct the model response and token usage. */
    'plugin:sciatlas/llm-chunk': { runId: string; stage: string; chunk: StreamChunk }
  }
}
declare module '@deepseek-ai/dsh-jobs' {
  interface JobKindMap { sciatlas: 'sciatlas' }
}

/** Select the exact logged caller route unless an explicit pair is configured. */
export function resolveRoute(exec: ToolRunContext, config: ResearchConfig): ModelRoute | null {
  if (config.llmProvider && config.llmModel) return { provider: config.llmProvider, model: config.llmModel }
  const route = exec.agent?.session.requestHeader()?.config
  return route ? { provider: route.provider, model: route.model } : null
}

/** Fetch a PDF under a byte bound; stream rather than buffering an unbounded response. This request carries no credentials, so publisher/DOI redirects are followed. */
export async function fetchPdf(url: string, config: ResearchConfig, signal: AbortSignal): Promise<Uint8Array> {
  if (!/^https?:\/\//i.test(url)) throw new Error('PDF URL must use HTTP or HTTPS')
  const response = await fetch(url, { redirect: 'follow', signal: AbortSignal.any([signal, AbortSignal.timeout(config.pdfTimeoutMs)]) })
  if (!response.ok || !response.body) throw new Error(`PDF download returned HTTP ${response.status}`)
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let length = 0
  try {
    for (;;) {
      const next = await reader.read()
      if (next.done) break
      length += next.value.length
      if (length > config.maxPdfBytes) throw new Error('PDF download exceeds research.maxPdfBytes')
      chunks.push(next.value)
    }
  } finally { await reader.cancel(); reader.releaseLock() }
  const bytes = new Uint8Array(length)
  let offset = 0
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length }
  return bytes
}

/** Bind external research capabilities to the invoking agent's execution world. */
export function liveServices(ctx: Context, exec: ToolRunContext, api: SciAtlasToolDeps, config: ResearchConfig, runId: string): ResearchServices {
  const world = exec.agent?.ctx ?? ctx
  const route = resolveRoute(exec, config)
  return {
    config, api, route,
    async readFile(path, signal) {
      const fs = world.get('fs')
      if (!fs) throw new Error('PDF/text-file input requires a DSH filesystem provider')
      const target = await fs.resolve(path, { signal })
      return fs.readBytes(target, signal, config.maxPdfBytes)
    },
    fetchPdf: (url, signal) => fetchPdf(url, config, signal),
    async semanticSearch(query, limit, signal) {
      const params = new URLSearchParams({ query, limit: String(Math.min(100, limit)), fields: 'paperId,title,abstract,year,venue,authors,citationCount,externalIds,url,openAccessPdf' })
      const stored = await world.credentials.resolve(credentialRef(config.semanticScholarKeyEnv))
      const headers: Record<string, string> = { accept: 'application/json' }
      if (stored?.value.trim()) headers['x-api-key'] = stored.value.trim()
      const response = await fetch(`${config.semanticScholarBaseUrl.replace(/\/$/, '')}/paper/search?${params}`, { headers, redirect: 'error', signal: AbortSignal.any([signal, AbortSignal.timeout(api.timeoutMs)]) })
      if (!response.ok) throw new Error(`Semantic Scholar returned HTTP ${response.status}`)
      const body = object(await response.json())
      const mapped = array(body['data'], 'Semantic Scholar data').map(item => {
        const paper = object(item)
        const externalIds = paper['externalIds'] && typeof paper['externalIds'] === 'object' ? object(paper['externalIds']) : {}
        const pdf = paper['openAccessPdf'] && typeof paper['openAccessPdf'] === 'object' ? object(paper['openAccessPdf']) : {}
        return { ...paper, paper_id: paper['paperId'] ?? '', doi: externalIds['DOI'] ?? '', pdf_url: pdf['url'] ?? '', citation_count: paper['citationCount'] ?? 0 }
      })
      return { papers: collectPapers(mapped, { maxItems: limit, abstractMaxChars: Number.MAX_SAFE_INTEGER }).map(paper => ({ ...paper, paperId: paperIdentity(paper), sources: ['semantic-scholar'] })), authors: [], request: { query, limit }, response: json(body) }
    },
    async referenceSearch(paper, limit, signal) {
      const identifier = paper.doi ? `DOI:${paper.doi.replace(/^https?:\/\/doi.org\//i, '')}` : paper.id && /^[0-9a-f]{40}$/i.test(paper.id) ? paper.id : ''
      if (!identifier) throw new Error('No Semantic Scholar-compatible identifier for citation lookup')
      const params = new URLSearchParams({ limit: String(Math.min(100, limit)), fields: 'paperId,title,abstract,year,venue,authors,citationCount,externalIds,url,openAccessPdf' })
      const stored = await world.credentials.resolve(credentialRef(config.semanticScholarKeyEnv))
      const headers: Record<string, string> = { accept: 'application/json' }
      if (stored?.value.trim()) headers['x-api-key'] = stored.value.trim()
      const response = await fetch(`${config.semanticScholarBaseUrl.replace(/\/$/, '')}/paper/${encodeURIComponent(identifier)}/references?${params}`, { headers, redirect: 'error', signal: AbortSignal.any([signal, AbortSignal.timeout(api.timeoutMs)]) })
      if (!response.ok) throw new Error(`Semantic Scholar references returned HTTP ${response.status}`)
      const body = object(await response.json())
      const mapped = array(body['data']).map(item => {
        const record = object(item)
        if (record['citedPaper'] === null) return null
        const node = object(record['citedPaper'])
        const ids = node['externalIds'] && typeof node['externalIds'] === 'object' ? object(node['externalIds']) : {}
        const pdf = node['openAccessPdf'] && typeof node['openAccessPdf'] === 'object' ? object(node['openAccessPdf']) : {}
        return { ...node, paper_id: node['paperId'] ?? '', doi: ids['DOI'] ?? '', pdf_url: pdf['url'] ?? '', citation_count: node['citationCount'] ?? 0 }
      })
      return { papers: collectPapers(mapped, { maxItems: limit, abstractMaxChars: Number.MAX_SAFE_INTEGER }).map(item => ({ ...item, paperId: paperIdentity(item), sources: ['semantic-scholar-references'] })), authors: [], request: { identifier, limit }, response: json(body) }
    },
    async model(request) {
      const llm = world.get('llm')
      if (!llm || !route) throw new Error('Research requires a DSH model route; configure research.llmProvider and research.llmModel together')
      if (!exec.agent) throw new Error('Research model calls require a session owner')
      const signal = AbortSignal.any([request.signal, AbortSignal.timeout(request.timeoutMs ?? config.llmTimeoutMs)])
      const call = await llm.prepareCall({ ...route, maxTokens: config.maxOutputTokens }, signal)
      const messages = [createUserMessage({ content: [{ type: 'text', text: JSON.stringify(request.input) }], source: { kind: 'sciatlas-research' } })]
      appendPluginRecord(exec.agent.session, 'plugin:sciatlas/llm-request', { runId, stage: request.stage, route, config: call.config, adapterDefaults: call.adapterDefaults, system: request.system, input: request.input, maxTokens: config.maxOutputTokens })
      const assembler = new BlockAssembler()
      let terminal = false
      for await (const chunk of call.stream({ ...call.config, system: request.system, messages, sessionId: exec.agent.session.id, signal })) {
        appendPluginRecord(exec.agent.session, 'plugin:sciatlas/llm-chunk', { runId, stage: request.stage, chunk })
        assembler.push(chunk)
        if (chunk.type === 'finish') terminal = true
      }
      signal.throwIfAborted()
      const finish = assembler.finish
      if (!terminal) throw new Error('Research model stream ended without a terminal finish')
      if (finish.kind === 'error' || finish.kind === 'aborted') throw new Error(finish.failure.message)
      if (finish.kind !== 'stop') throw new Error(`Research model ended with ${finish.kind}; increase output budget or use a model without tool calls`)
      const content = assembler.blocks().filter(block => block.type === 'text').map(block => block.text).join('')
      if (!content.trim()) throw new Error('Research model returned no text')
      return content
    },
  }
}

/** Tracks producer-owned resources and waits for quiescence during plugin unload. */
export class ResearchTasks {
  private readonly active = new Map<string, { controller: AbortController; done: Promise<JobOutcome> }>()
  private disposed = false
  constructor(readonly max: number) {}

  /** Refuse new work once unloaded or at the configured concurrency bound. */
  assertAvailable(): void {
    if (this.disposed) throw new Error('SciAtlas plugin has been unloaded')
    if (this.active.size >= this.max) throw new Error('Research concurrent-run limit reached; wait for an existing job')
  }

  /** Execute one owned task; done always settles after producer cleanup. */
  start(runId: string, run: (signal: AbortSignal) => Promise<JobOutcome>) {
    this.assertAvailable()
    if (this.active.has(runId)) throw new Error('Research run is already active')
    const controller = new AbortController()
    const done = Promise.resolve().then(() => run(controller.signal)).catch((error: unknown): JobOutcome => ({ status: controller.signal.aborted ? 'killed' : 'failed', detail: error instanceof Error ? error.message : String(error) })).finally(() => this.active.delete(runId))
    this.active.set(runId, { controller, done })
    return { cancel: () => controller.abort(new DOMException('Research cancelled', 'AbortError')), done }
  }

  /** Start through DSH jobs so access, notices, waiting, and owner cleanup remain native. */
  background(ctx: Context, exec: ToolRunContext, runId: string, label: string, run: (signal: AbortSignal, job: JobHandle) => Promise<JobOutcome>): string {
    this.assertAvailable()
    exec.signal.throwIfAborted()
    if (!exec.agent) throw new Error('Background research requires an owning DSH agent')
    const jobs = (exec.agent.ctx ?? ctx).get('jobs')
    if (!jobs) throw new Error('Background research requires DSH jobs and its generic control tools; use background:false for foreground execution')
    return jobs.start({ kind: 'sciatlas', label, owner: exec.agent.id, run: job => this.start(runId, signal => run(signal, job)) })
  }

  /** Abort and drain every accepted task before plugin disposal returns. */
  async dispose(): Promise<void> {
    this.disposed = true
    const tasks = [...this.active.values()]
    for (const task of tasks) task.controller.abort(new DOMException('SciAtlas plugin unloaded', 'AbortError'))
    await Promise.all(tasks.map(task => task.done))
  }
}
