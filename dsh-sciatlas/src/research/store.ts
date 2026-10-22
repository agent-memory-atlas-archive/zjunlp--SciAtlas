/** Private, atomic run artifacts and checksummed stage checkpoints. */
import { mkdir, readFile, writeFile, rename, open, unlink } from 'node:fs/promises'
import { resolve, join, basename } from 'node:path'
import { randomUUID } from 'node:crypto'
import { hash, json, object, text } from './json.ts'
import type { Json, JsonObject } from './json.ts'

export const PIPELINE_VERSION = '2'

/** A stage entry indexes an immutable JSON file by its input and content digests. */
export interface StageRecord {
  status: 'running' | 'completed' | 'failed'
  inputHash: string
  file: string
  contentHash: string
  elapsedMs: number
  error: string
}

export interface RunManifest {
  version: string
  runId: string
  owner: string
  command: string
  inputHash: string
  input: JsonObject
  status: 'running' | 'completed' | 'partial' | 'failed' | 'cancelled'
  stages: Record<string, StageRecord>
  modelCalls: number
  searchCalls: number
  createdAt: string
  updatedAt: string
}

/** Validate a filesystem-sourced manifest before using its paths or identities. */
function decodeManifest(value: unknown): RunManifest {
  const raw = object(value, 'manifest')
  const stages: Record<string, StageRecord> = {}
  for (const [name, item] of Object.entries(object(raw['stages'], 'stages'))) {
    safeName(name)
    const stage = object(item)
    const status = stage['status']
    if (status !== 'running' && status !== 'completed' && status !== 'failed') throw new Error('Invalid checkpoint stage status')
    const file = text(stage['file'])
    if (basename(file) !== file) throw new Error('Invalid checkpoint path')
    if (typeof stage['elapsedMs'] !== 'number') throw new Error('Invalid checkpoint duration')
    stages[name] = { status, file, elapsedMs: stage['elapsedMs'], inputHash: text(stage['inputHash']), contentHash: typeof stage['contentHash'] === 'string' ? stage['contentHash'] : '', error: typeof stage['error'] === 'string' ? stage['error'] : '' }
  }
  const status = raw['status']
  if (status !== 'running' && status !== 'completed' && status !== 'partial' && status !== 'failed' && status !== 'cancelled') throw new Error('Invalid run status')
  if (typeof raw['modelCalls'] !== 'number' || typeof raw['searchCalls'] !== 'number') throw new Error('Invalid run counters')
  return { version: text(raw['version']), runId: safeName(text(raw['runId'])), owner: text(raw['owner']), command: text(raw['command']), inputHash: text(raw['inputHash']), input: object(raw['input']), status, stages, modelCalls: raw['modelCalls'], searchCalls: raw['searchCalls'], createdAt: text(raw['createdAt']), updatedAt: text(raw['updatedAt']) }
}

/** Restrict artifact identifiers to one path component. */
export function safeName(value: string): string {
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,100}$/.test(value)) throw new Error('Invalid run or stage identifier')
  return value
}

/** Allocate a collision-resistant run id without changing the filesystem. */
export function newRunId(): string { return `research-${randomUUID()}` }

/** Store owned artifacts outside tool-supplied filesystem paths. */
export class RunStore {
  readonly dir: string
  readonly manifestPath: string
  private writes: Promise<void> = Promise.resolve()
  private unlock: (() => Promise<void>) | undefined

  constructor(readonly root: string, readonly runId: string, readonly owner: string) {
    this.dir = join(resolve(root), safeName(runId))
    this.manifestPath = join(this.dir, 'manifest.json')
  }

  /** Read a run only when its owner matches the current DSH session. */
  async read(): Promise<RunManifest> {
    const manifest = decodeManifest(JSON.parse(await readFile(this.manifestPath, 'utf8')))
    if (manifest.owner !== this.owner || manifest.runId !== this.runId) throw new Error('The research run belongs to another session')
    if (manifest.version !== PIPELINE_VERSION) throw new Error('Research checkpoint version changed; start a new run')
    return manifest
  }

  /** Validate resume inputs before publishing a background task. */
  async validateResume(input: JsonObject): Promise<void> {
    const manifest = await this.read()
    if (manifest.inputHash !== hash(input)) throw new Error('Resume input, model route, files, or configuration changed; start a new run')
  }

  /** Acquire exclusive ownership; recover a lock only after its process is gone. */
  async acquire(): Promise<void> {
    await mkdir(this.dir, { recursive: true, mode: 0o700 })
    const path = join(this.dir, '.lock')
    const create = async () => {
      const handle = await open(path, 'wx', 0o600)
      await handle.writeFile(JSON.stringify({ pid: process.pid }))
      await handle.close()
      this.unlock = async () => { await unlink(path) }
    }
    try { await create() } catch (error) {
      if (!(error instanceof Error) || !('code' in error) || error.code !== 'EEXIST') throw error
      const lock = object(JSON.parse(await readFile(path, 'utf8')), 'run lock')
      if (typeof lock['pid'] !== 'number') throw new Error('Invalid research lock; inspect the run directory')
      try { process.kill(lock['pid'], 0) } catch (probe) {
        if (probe instanceof Error && 'code' in probe && probe.code === 'ESRCH') {
          await unlink(path)
          await create()
          return
        }
        throw probe
      }
      throw new Error('This research run is already active')
    }
  }

  /** Release the run after all requests and writes have settled. */
  async release(): Promise<void> {
    await this.writes
    if (this.unlock) { const unlock = this.unlock; this.unlock = undefined; await unlock() }
  }

  /** Serialize atomic writes so parallel stage completion cannot overwrite the manifest. */
  async write(name: string, value: Json | string, raw = false): Promise<string> {
    if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(name)) throw new Error('Invalid artifact filename')
    const contents = raw ? String(value) : `${JSON.stringify(json(value), null, 2)}\n`
    const path = join(this.dir, name)
    const operation = this.writes.then(async () => {
      const temporary = join(this.dir, `.${randomUUID()}.tmp`)
      try {
        await writeFile(temporary, contents, { mode: 0o600, flag: 'wx' })
        await rename(temporary, path)
      } catch (error) {
        await unlink(temporary).catch(() => { /* A failed open may not have created the temporary file. */ })
        throw error
      }
    })
    this.writes = operation.catch(() => { /* The caller receives this write error; permit the terminal manifest write. */ })
    await operation
    return path
  }

  /** Read and validate a completed stage; corrupt data fails before dependent work. */
  async checkpoint(record: StageRecord): Promise<Json> {
    const data = json(JSON.parse(await readFile(join(this.dir, record.file), 'utf8')))
    if (hash(data) !== record.contentHash) throw new Error(`Checkpoint integrity failed: ${record.file}`)
    return data
  }
}
