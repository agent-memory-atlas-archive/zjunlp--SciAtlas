/**
 * Minimal HTTP client for the hosted SciAtlas API. Sends the dual credential
 * headers (`Authorization: Bearer …` and `X-API-Key`) the API requires,
 * rejects redirects so credentials never leak to another origin, and maps
 * transport/HTTP failures to {@link SciAtlasApiError}. The raw SciAtlas API
 * returns the bare payload body; the `{ok, data, …}` envelope seen in CLI
 * artifacts is added by the Python CLI wrapper and does not exist on the
 * wire.
 *
 * @module dsh-sciatlas/client
 */

/** Canonical hosted SciAtlas API base URL. */
export const CANONICAL_BASE_URL = 'http://sciatlas.openkg.cn'

/** Legacy SciNet base URLs that the hosted service replaced. */
const LEGACY_BASE_URLS: Readonly<Record<string, string>> = {
  'http://scinet.openkg.cn': 'http://sciatlas.openkg.cn',
  'https://scinet.openkg.cn': 'https://sciatlas.openkg.cn',
}

/**
 * Normalize a configured base URL: trims, strips trailing slashes, rewrites
 * legacy SciNet hosts, and falls back to the canonical host when empty.
 *
 * @param value - the configured base URL.
 * @returns the base URL to send requests against.
 */
export function normalizeBaseUrl(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, '')
  return LEGACY_BASE_URLS[trimmed] ?? (trimmed || CANONICAL_BASE_URL)
}

/** Options every request shares. */
export interface SciAtlasRequestOptions {
  /** Normalized base URL. */
  baseUrl: string
  /** Resolved API key; sent as both `Authorization: Bearer` and `X-API-Key`. */
  apiKey: string
  /** Per-request timeout in milliseconds, combined with the caller signal. */
  timeoutMs: number
  /** Caller-owned cancellation signal (tool execution). */
  signal?: AbortSignal | undefined
}

/**
 * Failure talking to the SciAtlas API: transport errors, timeouts, non-2xx
 * statuses, and undecodable bodies. `status` is present only for HTTP-level
 * failures.
 */
export class SciAtlasApiError extends Error {
  /** HTTP status for HTTP-level failures; `undefined` for transport/parse failures. */
  readonly status: number | undefined

  constructor(message: string, options?: { status?: number; cause?: unknown }) {
    super(message, options?.cause === undefined ? undefined : { cause: options.cause })
    this.name = 'SciAtlasApiError'
    this.status = options?.status
  }
}

/** True for an abort/timeout rejection produced by fetch signals. */
function isSignalError(error: unknown): boolean {
  return error instanceof DOMException && (error.name === 'AbortError' || error.name === 'TimeoutError')
}

/** Cap error bodies read from the wire so a pathological upstream cannot flood memory. */
const MAX_ERROR_BODY_CHARS = 2048

/**
 * Extract a human-readable detail from an error body: prefer JSON
 * `detail`/`message`/`error` fields, then raw text, bounded in length.
 *
 * @param raw - the raw response body text.
 * @returns a short detail string, or `undefined` when the body carries none.
 */
export function extractErrorDetail(raw: string): string | undefined {
  const trimmed = raw.trim()
  if (!trimmed) return undefined
  try {
    const parsed: unknown = JSON.parse(trimmed)
    if (typeof parsed === 'object' && parsed !== null) {
      const record = parsed as Record<string, unknown>
      for (const key of ['detail', 'message', 'error']) {
        const value = record[key]
        if (typeof value === 'string' && value.trim()) return value.trim().slice(0, MAX_ERROR_BODY_CHARS)
        if (typeof value === 'object' && value !== null) {
          return JSON.stringify(value).slice(0, MAX_ERROR_BODY_CHARS)
        }
      }
    }
  } catch {
    // Not JSON: fall through to the bounded raw text.
  }
  return trimmed.slice(0, MAX_ERROR_BODY_CHARS)
}

/** Extra attempts after the first for transient failures, within the same overall timeout. */
const MAX_RETRIES = 2

/** True for HTTP statuses that are routinely transient for this read-only API. */
function isRetryableStatus(status: number): boolean {
  return status === 429 || (status >= 500 && status <= 599)
}

/**
 * True for the `redirect: 'error'` policy rejection (undici surfaces it as a
 * `TypeError` whose cause names the redirect). Retrying it can never succeed
 * and must not delay the caller.
 */
function isRedirectRejection(error: unknown): boolean {
  return error instanceof TypeError
    && typeof error.cause === 'object' && error.cause !== null
    && 'message' in error.cause && String(error.cause.message).includes('redirect')
}

/** Rethrow a signal rejection as the caller's abort or the configured timeout error. */
function throwSignalError(error: unknown, options: SciAtlasRequestOptions, timeout: AbortSignal): never {
  if (options.signal?.aborted) throw error
  if (timeout.aborted) {
    throw new SciAtlasApiError(`SciAtlas request timed out after ${options.timeoutMs} ms`, { cause: error })
  }
  throw error
}

/**
 * Wait between retries: the `Retry-After` header when present (capped at 30
 * seconds), else a 300 ms exponential backoff. Aborts with the shared
 * signal's reason so cancellation and the overall timeout stay exact.
 */
function backoff(attempt: number, retryAfter: string | null, signal: AbortSignal): Promise<void> {
  const seconds = retryAfter === null ? Number.NaN : Number(retryAfter)
  const ms = Number.isFinite(seconds) && seconds >= 0 ? Math.min(seconds * 1000, 30_000) : 300 * 2 ** attempt
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { signal.removeEventListener('abort', onAbort); resolve() }, ms)
    const onAbort = () => { clearTimeout(timer); reject(signal.reason instanceof Error ? signal.reason : new DOMException('Aborted', 'AbortError')) }
    if (signal.aborted) onAbort()
    else signal.addEventListener('abort', onAbort, { once: true })
  })
}

/**
 * POST a JSON payload to an authenticated SciAtlas endpoint and return the
 * decoded body. Redirects fail before being followed. Transient failures —
 * HTTP 429/5xx and transport errors — are retried up to {@link MAX_RETRIES}
 * times, sharing the caller's signal and the configured overall timeout; the
 * redirect policy rejection is never retried.
 *
 * @param endpoint - absolute API path such as `/v1/search`.
 * @param payload - the JSON request body.
 * @param options - shared request options.
 * @returns the decoded response body.
 */
export async function postJson(endpoint: string, payload: unknown, options: SciAtlasRequestOptions): Promise<unknown> {
  const url = `${options.baseUrl}${endpoint}`
  const timeout = AbortSignal.timeout(options.timeoutMs)
  const signal = options.signal === undefined ? timeout : AbortSignal.any([options.signal, timeout])
  const sleep = async (attempt: number, retryAfter: string | null): Promise<void> => {
    try {
      await backoff(attempt, retryAfter, signal)
    } catch (error) {
      if (isSignalError(error)) throwSignalError(error, options, timeout)
      throw error
    }
  }

  for (let attempt = 0; ; attempt++) {
    let response: Response
    try {
      response = await fetch(url, {
        method: 'POST',
        redirect: 'error',
        headers: {
          'accept': 'application/json',
          'content-type': 'application/json',
          'authorization': `Bearer ${options.apiKey}`,
          'x-api-key': options.apiKey,
        },
        body: JSON.stringify(payload),
        signal,
      })
    } catch (error: unknown) {
      if (isSignalError(error)) throwSignalError(error, options, timeout)
      const failure = new SciAtlasApiError(`SciAtlas request failed: ${error instanceof Error ? error.message : String(error)}`, { cause: error })
      if (attempt < MAX_RETRIES && !isRedirectRejection(error)) {
        await sleep(attempt, null)
        continue
      }
      throw failure
    }

    if (response.ok) {
      try {
        return await response.json()
      } catch (error: unknown) {
        if (isSignalError(error)) throwSignalError(error, options, timeout)
        throw new SciAtlasApiError('SciAtlas returned an undecodable response body', { cause: error })
      }
    }

    const status = response.status
    if (isRetryableStatus(status) && attempt < MAX_RETRIES) {
      await response.text().catch(() => { /* Drain before retrying. */ })
      await sleep(attempt, response.headers.get('retry-after'))
      continue
    }
    const raw = await response.text().catch(() => '')
    const detail = extractErrorDetail(raw)
    const hint = status === 401 || status === 403
      ? ' Check that the SciAtlas API key is present and valid (register at http://sciatlas.openkg.cn/register).'
      : ''
    throw new SciAtlasApiError(
      `SciAtlas API error (HTTP ${status})${detail === undefined ? '' : `: ${detail}`}${hint}`,
      { status },
    )
  }
}
