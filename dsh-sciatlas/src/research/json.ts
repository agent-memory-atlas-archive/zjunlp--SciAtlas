/** JSON parsing and deterministic hashing for durable research artifacts. */
import { createHash } from 'node:crypto'

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
export type JsonObject = { [key: string]: Json }

/** Validate values acquired from files, tools, or upstream responses. */
export function json(value: unknown): Json {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (Array.isArray(value)) return value.map(json)
  if (typeof value === 'object' && value !== null) return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, json(item)]))
  throw new Error('Expected a finite JSON value')
}

/** Require an object from parsed JSON. */
export function object(value: unknown, label = 'value'): JsonObject {
  const parsed = json(value)
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) throw new Error(`${label} must be an object`)
  return parsed
}

/** Require a nonblank string field. */
export function text(value: unknown, label = 'text'): string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a non-empty string`)
  return value.trim()
}

/** Require an array acquired from JSON. */
export function array(value: unknown, label = 'items'): Json[] {
  const parsed = json(value)
  if (!Array.isArray(parsed)) throw new Error(`${label} must be an array`)
  return parsed
}

/** Parse strict JSON, optionally enclosed in one Markdown fence. */
export function parseModelJson(value: string): JsonObject {
  const clean = value.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '')
  return object(JSON.parse(clean), 'model output')
}

/** Stable JSON serialization independent of object insertion order. */
export function stable(value: unknown): string {
  const normalize = (entry: Json): Json => {
    if (Array.isArray(entry)) return entry.map(normalize)
    if (entry && typeof entry === 'object') return Object.fromEntries(Object.keys(entry).sort().map(key => [key, normalize(entry[key]!)]))
    return entry
  }
  return JSON.stringify(normalize(json(value)))
}

/** SHA-256 digest used for artifact integrity and stage identities. */
export function hash(value: unknown): string {
  return createHash('sha256').update(stable(value)).digest('hex')
}

/** Normalize an error for durable diagnostics without stack or credentials. */
export function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
