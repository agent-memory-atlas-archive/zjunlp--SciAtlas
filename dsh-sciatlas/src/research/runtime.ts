/** Fixed-stage execution, cumulative budgets, and resumable research runs. */
import { hash, json, object, errorText } from './json.ts'
import type { Json, JsonObject } from './json.ts'
import { RunStore, PIPELINE_VERSION } from './store.ts'
import type { RunManifest } from './store.ts'
import type { ResearchInput, ResearchServices, ResearchResult, Paper, Evidence, SearchResult } from './types.ts'
import { graphSearch, paperCards } from './search.ts'
import type { OptionsInput, PlanInput } from '../tools/shared.ts'
import { parsePdf } from './pdf.ts'

/** Decode persisted paper objects at the durable boundary. */
export function decodePapers(value: unknown): Paper[] {
  if (!Array.isArray(value)) throw new Error('Expected paper array')
  return value.map(item => {
    const raw = object(item, 'paper')
    if (typeof raw['paperId'] !== 'string' || typeof raw['title'] !== 'string' || !Array.isArray(raw['sources']) || !raw['sources'].every(source => typeof source === 'string')) throw new Error('Invalid paper checkpoint')
    const paper: Paper = { paperId: raw['paperId'], title: raw['title'], sources: raw['sources'] }
    for (const key of ['id', 'doi', 'url', 'pdfUrl', 'abstract', 'venue'] as const) if (typeof raw[key] === 'string') paper[key] = raw[key]
    for (const key of ['year', 'score', 'citationCount'] as const) if (typeof raw[key] === 'number') paper[key] = raw[key]
    if (Array.isArray(raw['authors']) && raw['authors'].every(author => typeof author === 'string')) paper.authors = raw['authors']
    return paper
  })
}

/** Decode acquired passages before they become citation targets. */
export function decodeEvidence(value: unknown): Evidence[] {
  if (!Array.isArray(value)) throw new Error('Expected evidence array')
  return value.map(item => {
    const raw = object(item, 'evidence')
    if (typeof raw['evidenceId'] !== 'string' || typeof raw['paperId'] !== 'string' || typeof raw['text'] !== 'string' || typeof raw['source'] !== 'string' || typeof raw['paragraphIndex'] !== 'number' || !Array.isArray(raw['section']) || !raw['section'].every(entry => typeof entry === 'string') || (raw['level'] !== 'fulltext' && raw['level'] !== 'abstract')) throw new Error('Invalid evidence checkpoint')
    return { evidenceId: raw['evidenceId'], paperId: raw['paperId'], text: raw['text'], source: raw['source'], paragraphIndex: raw['paragraphIndex'], section: raw['section'], level: raw['level'] }
  })
}

/** Bounded fan-out always drains workers, including after one rejects. */
export async function parallelMap<T, R>(items: readonly T[], concurrency: number, signal: AbortSignal, run: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length)
  let index = 0
  let failure: unknown
  let failed = false
  await Promise.all(Array.from({ length: Math.min(items.length, concurrency) }, async () => {
    while (!failed && index < items.length) {
      const next = index++
      try { signal.throwIfAborted(); results[next] = await run(items[next]!, next) } catch (error) { failed = true; failure = error }
    }
  }))
  if (failed) throw failure
  return results
}

/** Research stage execution context. It owns all writes and budget accounting. */
export class ResearchRun {
  readonly warnings: string[] = []
  readonly signal: AbortSignal
  manifest!: RunManifest
  private modelCount = 0
  private searchCount = 0

  constructor(readonly store: RunStore, readonly input: ResearchInput, readonly services: ResearchServices, signal: AbortSignal, readonly progress: (stage: string) => void = () => {}) {
    this.signal = AbortSignal.any([signal, AbortSignal.timeout(services.config.workflowTimeoutMs)])
    if (input.smoke) this.warnings.push('Smoke mode uses synthetic fixture data; this is not a scientific result.')
  }

  /** Open an owned run using the already-validated resume identity. */
  async initialize(identity: JsonObject, resume: boolean): Promise<void> {
    this.signal.throwIfAborted()
    await this.store.acquire()
    try {
      if (resume) {
        this.manifest = await this.store.read()
        if (this.manifest.inputHash !== hash(identity)) throw new Error('Resume input changed')
        this.modelCount = this.manifest.modelCalls
        this.searchCount = this.manifest.searchCalls
        this.manifest.status = 'running'
      } else {
        const now = new Date().toISOString()
        this.manifest = { version: PIPELINE_VERSION, runId: this.store.runId, owner: this.store.owner, command: this.input.command, inputHash: hash(identity), input: identity, status: 'running', stages: {}, modelCalls: 0, searchCalls: 0, createdAt: now, updatedAt: now }
      }
      await this.save()
    } catch (error) { await this.store.release(); throw error }
  }

  private async save(): Promise<void> {
    this.manifest.modelCalls = this.modelCount
    this.manifest.searchCalls = this.searchCount
    this.manifest.updatedAt = new Date().toISOString()
    await this.store.write('manifest.json', json(this.manifest))
  }

  /** Cache one validated stage by its own inputs, upstream data, and pipeline version. */
  async stage<T>(name: string, input: unknown, decode: (value: unknown) => T, execute: () => Promise<T>): Promise<T> {
    this.signal.throwIfAborted()
    const inputHash = hash({ version: PIPELINE_VERSION, input })
    const existing = this.manifest.stages[name]
    if (existing?.status === 'completed' && existing.inputHash === inputHash) {
      this.progress(`${name}: resumed`)
      return decode(await this.store.checkpoint(existing))
    }
    this.progress(name)
    const started = performance.now()
    const record = { status: 'running' as 'running' | 'completed' | 'failed', inputHash, file: `${name}.json`, contentHash: '', elapsedMs: 0, error: '' }
    this.manifest.stages[name] = record
    await this.save()
    try {
      const output = decode(json(await execute()))
      this.signal.throwIfAborted()
      const data = json(output)
      await this.store.write(record.file, data)
      record.contentHash = hash(data)
      record.elapsedMs = Math.round(performance.now() - started)
      record.status = 'completed'
      await this.save()
      return output
    } catch (error) {
      record.status = 'failed'
      record.error = errorText(error)
      record.elapsedMs = Math.round(performance.now() - started)
      await this.save()
      throw error
    }
  }

  /** Model call with one schema repair; transport failures are never silently repaired. */
  async model<T>(stage: string, input: Json, system: string, decode: (value: unknown) => T, smoke: () => NoInfer<T>, timeoutMs?: number): Promise<T> {
    return this.stage(stage, { input, system }, decode, async () => {
      if (this.input.smoke) return smoke()
      let previous = ''
      let problem = ''
      for (let attempt = 0; attempt < 2; attempt++) {
        const requestInput = attempt === 0 ? input : json({ original: input, invalidOutput: previous, error: problem, instruction: 'Return corrected JSON only using the requested schema.' })
        if (system.length + JSON.stringify(requestInput).length > this.services.config.maxInputChars) throw new Error(`Model input for ${stage} exceeds research.maxInputChars; lower the paper or evidence budget`)
        if (this.modelCount >= this.services.config.maxModelCalls) throw new Error('Research model-call budget exhausted')
        this.modelCount++
        await this.save()
        previous = await this.services.model({ stage: `${stage}${attempt ? '-repair' : ''}`, system, input: requestInput, signal: this.signal, timeoutMs: timeoutMs ?? this.input.options.llmTimeoutMs ?? this.services.config.llmTimeoutMs })
        if (this.input.options.saveIntermediate !== false) await this.store.write(`${stage}-response-${attempt}.txt`, previous, true)
        try {
          const raw = previous.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '')
          return decode(JSON.parse(raw))
        } catch (error) { problem = errorText(error) }
      }
      throw new Error(`Invalid structured model output for ${stage}: ${problem}`)
    })
  }

  /** Graph retrieval retains raw requests/responses alongside full paper metadata. */
  async search(stage: string, input: PlanInput & OptionsInput): Promise<Paper[]> {
    return this.stage(stage, input, decodePapers, async () => {
      if (this.input.smoke) return smokePapers(input.query, input.topK ?? 3)
      return (await this.request(stage, () => graphSearch(this.services.api, input, stage, this.signal))).papers
    })
  }

  /** Count and record any lower-level authenticated research request. */
  async request(stage: string, execute: () => Promise<SearchResult>): Promise<SearchResult> {
    if (this.searchCount >= this.services.config.maxSearchCalls) throw new Error('Research retrieval-call budget exhausted')
    this.searchCount++
    await this.save()
    const result = await execute()
    if (this.input.options.saveIntermediate !== false) {
      await this.store.write(`${stage}-request.json`, result.request)
      await this.store.write(`${stage}-response.json`, result.response)
    }
    return result
  }

  /** Parse input PDFs, with bytes included in resume identity by the caller. */
  async inputDocuments(): Promise<{ title: string; abstract: string; paragraphs: Evidence[] }[]> {
    return parallelMap(this.input.pdfs, this.services.config.maxStageConcurrency, this.signal, async (path, index) => this.stage(`input-pdf-${index}`, { path }, value => {
      const raw = object(value)
      if (typeof raw['title'] !== 'string' || typeof raw['abstract'] !== 'string') throw new Error('Invalid PDF checkpoint')
      return { title: raw['title'], abstract: raw['abstract'], paragraphs: decodeEvidence(raw['paragraphs']) }
    }, async () => {
      const document = await parsePdf(await this.services.readFile(path, this.signal), path, `input-${index}`, this.services.config, this.signal)
      if (this.input.options.saveIntermediate !== false) await this.store.write(`input-pdf-${index}.tei.xml`, document.tei, true)
      return { title: document.title, abstract: document.abstract, paragraphs: document.paragraphs }
    }))
  }

  /** Acquire passages, falling back only to explicitly labeled available abstracts. */
  async evidence(papers: readonly Paper[], limit: number): Promise<Evidence[]> {
    const results = await parallelMap(papers.slice(0, limit), this.services.config.maxStageConcurrency, this.signal, async paper => this.stage(`evidence-${paper.paperId}`, paper, value => {
      const raw = object(value)
      if (!Array.isArray(raw['warnings']) || !raw['warnings'].every(item => typeof item === 'string')) throw new Error('Invalid evidence acquisition warnings')
      return { passages: decodeEvidence(raw['passages']), warnings: raw['warnings'] }
    }, async () => {
      const warnings: string[] = []
      if (paper.pdfUrl && this.services.config.grobidBaseUrl && !this.input.smoke) {
        try {
          const bytes = await this.services.fetchPdf(paper.pdfUrl, this.signal)
          const document = await parsePdf(bytes, paper.pdfUrl, paper.paperId, this.services.config, this.signal)
          if (this.input.options.saveIntermediate !== false) await this.store.write(`${paper.paperId}.tei.xml`, document.tei, true)
          if (document.paragraphs.length) return { passages: document.paragraphs, warnings }
          warnings.push(`Full text contained no usable passages for ${paper.paperId}`)
        } catch (error) {
          this.signal.throwIfAborted()
          warnings.push(`Full text unavailable for ${paper.paperId}: ${errorText(error)}`)
        }
      }
      if (!paper.abstract) { warnings.push(`No usable passage for ${paper.paperId}`); return { passages: [], warnings } }
      return { passages: [{ evidenceId: `${paper.paperId}-abstract`, paperId: paper.paperId, section: ['Abstract'], paragraphIndex: 0, text: paper.abstract, level: 'abstract' as const, source: paper.url ?? paper.doi ?? paper.title }], warnings }
    }))
    this.warnings.push(...results.flatMap(item => item.warnings))
    const result = results.flatMap(item => item.passages)
    if (!result.some(item => item.level === 'fulltext')) this.warnings.push('Evidence is abstract-only; full-text conclusions cannot be verified.')
    return result
  }

  /** Persist terminal artifacts before releasing exclusive run ownership. */
  async finish(result: ResearchResult): Promise<void> {
    await this.store.write('papers.json', json(result.papers))
    await this.store.write('result.json', json(result))
    await this.store.write('report.md', result.report, true)
    this.manifest.status = result.status
    await this.save()
  }

  /** Keep failure and cancellation visible while retaining successful checkpoints. */
  async fail(error: unknown): Promise<void> {
    this.manifest.status = this.signal.aborted ? 'cancelled' : 'failed'
    await this.store.write('error.json', { message: errorText(error), cancelled: this.signal.aborted })
    await this.save()
  }
}

/** Clearly synthetic data used only by the explicit CLI-compatible smoke path. */
export function smokePapers(topic: string, count: number): Paper[] {
  return Array.from({ length: Math.min(count, 4) }, (_, index) => ({ paperId: `p-smoke-${index}`, title: `SMOKE fixture ${index + 1}: ${topic.slice(0, 80)}`, abstract: `Synthetic evidence about ${topic}. This fixture is for workflow validation only.`, year: 2022 + index, citationCount: 10 + index, sources: ['smoke'], authors: ['Synthetic Researcher'] }))
}

/** Validate citation ids before a model result is accepted or rendered. */
export function citations(value: unknown, papers: readonly Paper[], evidence: readonly Evidence[] = []): string[] {
  if (!Array.isArray(value) || !value.every(item => typeof item === 'string')) throw new Error('citations must be an array of source ids')
  const allowed = new Set([...papers.map(item => item.paperId), ...evidence.map(item => item.evidenceId)])
  for (const id of value) if (!allowed.has(id)) throw new Error(`Unknown citation id: ${id}`)
  return value
}

/** Citation-safe Markdown bibliography generated from acquired metadata only. */
export function bibliography(papers: readonly Paper[]): string {
  return papers.map(paper => `- [${paper.paperId}] ${paper.title.replace(/[\r\n]/g, ' ')}${paper.year ? ` (${paper.year})` : ''}. ${paper.url ?? paper.pdfUrl ?? paper.doi ?? 'No public link available.'}`).join('\n')
}

export { paperCards }
