/**
 * Response normalization for the hosted SciAtlas API. The backend payload
 * layout is not formally documented, so detection is marker-based (like the
 * SciAtlas CLI): a paper-like object carries a title plus at least one
 * bibliographic marker, an author-like object carries a name plus at least
 * one author metric marker. The marker sets overlap across record kinds (a
 * citation count appears on both papers and authors), so each projection
 * also rejects objects carrying markers exclusive to the other kind. Unlike
 * the CLI display layer, normalization here preserves the landing page, DOI,
 * and PDF URL as separate fields and keeps the backend's ranking order.
 *
 * @module dsh-sciatlas/normalize
 */

/** One normalized paper record extracted from a SciAtlas response payload. */
export interface NormalizedPaper {
  id?: string
  titleEmbedding?: number[]
  abstractEmbedding?: number[]
  title: string
  year?: number
  venue?: string
  authors?: string[]
  citationCount?: number
  score?: number
  doi?: string
  /** Landing page (DOI resolver, OpenAlex page, or publisher page) — never a PDF. */
  url?: string
  /** Direct PDF link when the backend supplies one. */
  pdfUrl?: string
  abstract?: string
}

/** One normalized author record extracted from a SciAtlas response payload. */
export interface NormalizedAuthor {
  name: string
  id?: string
  score?: number
  worksCount?: number
  citationCount?: number
  hIndex?: number
}

/** One author entry of a support-papers response, with its supporting papers. */
export interface NormalizedSupportAuthor {
  name?: string
  id?: string
  supportPapers: NormalizedPaper[]
}

/** Bounds applied while normalizing a payload. */
export interface NormalizeOptions {
  /** Preserve duplicates when the caller explicitly disables deduplication. */
  dedupe?: boolean
  /** Keep embeddings only when explicitly requested. */
  includeEmbeddings?: boolean
  /** Maximum records returned. */
  maxItems: number
  /** Abstracts longer than this many characters are truncated with an ellipsis. */
  abstractMaxChars: number
}

type JsonObject = Record<string, unknown>

/** True for a plain JSON object (not an array, not null). */
function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** First present non-empty value among `keys`, or `undefined`. */
function firstPresent(obj: JsonObject, keys: readonly string[]): unknown {
  for (const key of keys) {
    const value = obj[key]
    if (value !== undefined && value !== null && value !== '' && !(Array.isArray(value) && value.length === 0)) {
      return value
    }
  }
  return undefined
}

/** Coerce a value to a trimmed string, or `undefined` when not string-coercible. */
function asString(value: unknown): string | undefined {
  if (typeof value === 'string') {
    const trimmed = value.trim()
    return trimmed || undefined
  }
  return undefined
}

/** Coerce a value to a finite number, or `undefined`. */
function asNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return undefined
}

/** Collapse whitespace and bound a text field. */
function compactText(value: string, maxChars: number): string {
  const compacted = value.replace(/\s+/g, ' ').trim()
  if (compacted.length <= maxChars) return compacted
  return `${compacted.slice(0, maxChars - 1)}…`
}

/** Title field candidates, in priority order. */
const TITLE_KEYS = ['title', 'paper_title', 'work_title', 'display_name', 'name'] as const

/** Marker keys that make a titled object paper-like. */
const PAPER_MARKERS = [
  'abstract',
  'abstract_text',
  'publication_year',
  'year',
  'doi',
  'paper_id',
  'openalex_id',
  'venue',
  'journal',
  'authors',
  'author_names',
  'citation_count',
  'cited_by_count',
] as const

/** Marker keys that make a named object author-like. */
const AUTHOR_MARKERS = [
  'author_id',
  'openalex_id',
  'works_count',
  'paper_count',
  'citation_count',
  'cited_by_count',
  'h_index',
  'score',
  'support_papers',
  'papers',
] as const

/**
 * Marker keys that only appear on author records. They overlap the paper
 * markers (`citation_count` etc.), so a titled object carrying one of these
 * is an author, never a paper.
 */
const AUTHOR_ONLY_MARKERS = ['author_id', 'h_index', 'works_count', 'paper_count', 'support_papers'] as const

/**
 * Marker keys that only appear on paper records. A named object carrying one
 * of these is a paper (OpenAlex-style works use `display_name`, which is
 * also an author-name key), never an author.
 */
const PAPER_ONLY_MARKERS = ['abstract', 'abstract_text', 'abstractText', 'doi', 'DOI', 'paper_id', 'venue', 'journal', 'conference', 'authors', 'author_names', 'year', 'publication_year', 'pub_year'] as const

/** Author-name keys on one author record. */
const AUTHOR_NAME_KEYS = ['author_name', 'name', 'display_name', 'full_name'] as const

/** PDF URL field candidates, checked shallowly and one nested level down. */
const PDF_URL_KEYS = [
  'pdf_url',
  'paper_pdf_url',
  'open_access_pdf_url',
  'best_oa_pdf_url',
  'oa_pdf_url',
  'fulltext_pdf_url',
  'full_text_pdf_url',
  'pdf',
  'pdfUrl',
] as const

/** Nested containers that may hold a PDF URL. */
const PDF_NESTED_KEYS = [
  'paper',
  'full_paper',
  'metadata',
  'open_access',
  'best_oa_location',
  'primary_location',
  'locations',
  'oa_locations',
] as const

/** Landing-page URL candidates; values must be HTTP(S) and not PDF links. */
const LANDING_URL_KEYS = ['url', 'landing_page_url', 'landing', 'homepage', 'link'] as const

/** True when the value is an absolute HTTP(S) URL. */
function isHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value)
}

/** True when the URL points at a PDF rather than a landing page. */
function looksLikePdfUrl(value: string): boolean {
  const lower = value.toLowerCase()
  return lower.includes('.pdf') || lower.includes('arxiv.org/pdf/') || lower.includes('openreview.net/pdf') || lower.endsWith('/pdf')
}

/** Extract a direct PDF URL from a paper-like object, or `undefined`. */
function selectPdfUrl(obj: JsonObject): string | undefined {
  const fromValue = (value: unknown): string | undefined => {
    const text = asString(value)
    if (text === undefined || !isHttpUrl(text)) return undefined
    if (text.toLowerCase().includes('openalex.org/')) return undefined
    return looksLikePdfUrl(text) || text.toLowerCase().includes('pdf') ? text : undefined
  }
  for (const key of PDF_URL_KEYS) {
    const found = fromValue(obj[key])
    if (found !== undefined) return found
  }
  // A generic `url` pointing straight at a PDF is a PDF link, not a landing page.
  const fromUrl = fromValue(obj['url'])
  if (fromUrl !== undefined && looksLikePdfUrl(fromUrl)) return fromUrl
  for (const key of PDF_NESTED_KEYS) {
    const nested = obj[key]
    if (isObject(nested)) {
      for (const innerKey of PDF_URL_KEYS) {
        const found = fromValue(nested[innerKey])
        if (found !== undefined) return found
      }
    }
    if (Array.isArray(nested)) {
      for (const entry of nested) {
        if (isObject(entry)) {
          for (const innerKey of PDF_URL_KEYS) {
            const found = fromValue(entry[innerKey])
            if (found !== undefined) return found
          }
        }
      }
    }
  }
  return undefined
}

/** Extract a landing-page URL: explicit URL fields, then DOI, then OpenAlex id. */
function selectLandingUrl(obj: JsonObject, doi: string | undefined, openalexId: string | undefined): string | undefined {
  for (const key of LANDING_URL_KEYS) {
    const text = asString(obj[key])
    if (text !== undefined && isHttpUrl(text) && !looksLikePdfUrl(text)) return text
  }
  if (doi !== undefined) {
    return isHttpUrl(doi) ? doi : `https://doi.org/${doi}`
  }
  if (openalexId !== undefined) {
    return isHttpUrl(openalexId) ? openalexId : `https://openalex.org/${openalexId}`
  }
  return undefined
}

/** Normalize the `authors`/`author_names` field into plain author names. */
function normalizeAuthorNames(value: unknown, maxAuthors: number): string[] | undefined {
  if (typeof value === 'string') {
    const text = compactText(value, 200)
    return text ? [text] : undefined
  }
  if (!Array.isArray(value)) return undefined
  const names: string[] = []
  for (const item of value) {
    if (typeof item === 'string' && item.trim()) {
      names.push(item.trim())
    } else if (isObject(item)) {
      const name = asString(firstPresent(item, AUTHOR_NAME_KEYS))
      if (name !== undefined) names.push(name)
    }
    if (names.length >= maxAuthors) break
  }
  return names.length > 0 ? names : undefined
}

/** Cap on author names kept per paper. */
const MAX_AUTHORS_PER_PAPER = 20

/**
 * Project one paper-like object into a {@link NormalizedPaper}, or return
 * `undefined` when the object lacks a title or every bibliographic marker.
 *
 * @param obj - the candidate object.
 * @param abstractMaxChars - truncation bound for abstracts.
 * @returns the normalized paper, or `undefined`.
 */
export function asNormalizedPaper(obj: JsonObject, abstractMaxChars: number): NormalizedPaper | undefined {
  const title = asString(firstPresent(obj, TITLE_KEYS))
  if (title === undefined) return undefined
  if (!PAPER_MARKERS.some(marker => marker in obj)) return undefined
  if (AUTHOR_ONLY_MARKERS.some(marker => marker in obj)) return undefined

  const year = asNumber(firstPresent(obj, ['year', 'publication_year', 'pub_year']))
  const score = asNumber(firstPresent(obj, ['score', 'rank_score', 'kg_score', 'similarity', 'similarity_score', 'final_score']))
  const citations = asNumber(firstPresent(obj, ['citation_count', 'cited_by_count', 'citations']))
  const venue = asString(firstPresent(obj, ['venue', 'journal', 'conference', 'source']))
  const authors = normalizeAuthorNames(firstPresent(obj, ['authors', 'author_names']), MAX_AUTHORS_PER_PAPER)
  const doi = asString(firstPresent(obj, ['doi', 'DOI']))
  const openalexId = asString(firstPresent(obj, ['openalex_id', 'openalexId']))
  const abstract = asString(firstPresent(obj, ['abstract', 'abstract_text', 'abstractText']))
  const id = asString(firstPresent(obj, ['paper_id', 'openalex_id', 'id']))
  const url = selectLandingUrl(obj, doi, openalexId)
  const pdfUrl = selectPdfUrl(obj)

  return {
    title: compactText(title, 300),
    ...id !== undefined ? { id } : {},
    ...year !== undefined ? { year: Math.trunc(year) } : {},
    ...venue !== undefined ? { venue: compactText(venue, 120) } : {},
    ...authors !== undefined ? { authors } : {},
    ...citations !== undefined ? { citationCount: citations } : {},
    ...score !== undefined ? { score } : {},
    ...doi !== undefined ? { doi } : {},
    ...url !== undefined ? { url } : {},
    ...pdfUrl !== undefined ? { pdfUrl } : {},
    ...abstract !== undefined ? { abstract: compactText(abstract, abstractMaxChars) } : {},
  }
}

/**
 * Project one author-like object into a {@link NormalizedAuthor}, or return
 * `undefined` when the object lacks a name or every author marker.
 *
 * @param obj - the candidate object.
 * @returns the normalized author, or `undefined`.
 */
export function asNormalizedAuthor(obj: JsonObject): NormalizedAuthor | undefined {
  const name = asString(firstPresent(obj, AUTHOR_NAME_KEYS))
  if (name === undefined) return undefined
  if (!AUTHOR_MARKERS.some(marker => marker in obj)) return undefined
  if (PAPER_ONLY_MARKERS.some(marker => marker in obj)) return undefined

  const id = asString(firstPresent(obj, ['author_id', 'id', 'openalex_id']))
  const score = asNumber(firstPresent(obj, ['score', 'rank_score', 'similarity', 'final_score']))
  const worksCount = asNumber(firstPresent(obj, ['works_count', 'paper_count']))
  const citationCount = asNumber(firstPresent(obj, ['citation_count', 'cited_by_count']))
  const hIndex = asNumber(firstPresent(obj, ['h_index']))

  return {
    name: compactText(name, 160),
    ...id !== undefined ? { id: compactText(id, 160) } : {},
    ...score !== undefined ? { score } : {},
    ...worksCount !== undefined ? { worksCount: Math.trunc(worksCount) } : {},
    ...citationCount !== undefined ? { citationCount } : {},
    ...hIndex !== undefined ? { hIndex } : {},
  }
}

/** Depth-first walk yielding every plain object in document order. */
function* walkObjects(value: unknown): Generator<JsonObject> {
  if (isObject(value)) {
    yield value
    for (const key of Object.keys(value)) {
      yield* walkObjects(value[key])
    }
  } else if (Array.isArray(value)) {
    for (const item of value) {
      yield* walkObjects(item)
    }
  }
}

/** Normalize a DOI for dedupe keys: resolver prefixes and case do not distinguish records. */
function normalizeDoiKey(value: string): string {
  return `doi:${value.replace(/^https?:\/\/(dx\.)?doi\.org\//i, '').trim().toLowerCase()}`
}

/** Dedupe key for one normalized paper: DOI, then landing URL, then title. */
function paperDedupeKey(paper: NormalizedPaper): string {
  if (paper.doi !== undefined) return normalizeDoiKey(paper.doi)
  return (paper.url ?? paper.title).toLowerCase()
}

/**
 * Collect deduplicated normalized papers from a response payload, preserving
 * the backend's document (ranking) order. Dedupe key is the DOI (ignoring
 * resolver prefix and case), then the landing URL, then the normalized title.
 *
 * @param data - the decoded response body.
 * @param options - collection bounds.
 * @returns at most `options.maxItems` papers in document order.
 */
export function collectPapers(data: unknown, options: NormalizeOptions): NormalizedPaper[] {
  const papers: NormalizedPaper[] = []
  const seen = new Set<string>()
  for (const obj of walkObjects(data)) {
    const paper = asNormalizedPaper(obj, options.abstractMaxChars)
    if (paper === undefined) continue
    const key = paperDedupeKey(paper)
    if (options.dedupe !== false && seen.has(key)) continue
    seen.add(key)
    if (options.includeEmbeddings) {
      for (const [field, key] of [['titleEmbedding', 'title_embedding'], ['abstractEmbedding', 'abstract_embedding']] as const) {
        const vector = obj[key]
        if (Array.isArray(vector) && vector.every(value => typeof value === 'number' && Number.isFinite(value))) paper[field] = vector
      }
    }
    papers.push(paper)
    if (papers.length >= options.maxItems) break
  }
  return papers
}

/**
 * Collect deduplicated normalized authors from a response payload,
 * preserving document order. Objects carrying a `support_papers` array are
 * skipped: they belong to {@link collectSupportAuthors}, and flattening them
 * here would break the ranking semantics of both lists.
 *
 * @param data - the decoded response body.
 * @param maxItems - maximum authors returned.
 * @returns at most `maxItems` authors in document order.
 */
export function collectAuthors(data: unknown, maxItems: number): NormalizedAuthor[] {
  const authors: NormalizedAuthor[] = []
  const seen = new Set<string>()
  for (const obj of walkObjects(data)) {
    if (Array.isArray(obj['support_papers'])) continue
    const author = asNormalizedAuthor(obj)
    if (author === undefined) continue
    const key = (author.id ?? author.name).toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    authors.push(author)
    if (authors.length >= maxItems) break
  }
  return authors
}

/**
 * Collect author-centric support-paper entries: objects with a
 * `support_papers` array, each carrying its own normalized papers.
 *
 * @param data - the decoded response body.
 * @param maxAuthors - maximum author entries returned.
 * @param maxPapersPerAuthor - maximum papers kept per author.
 * @param abstractMaxChars - truncation bound for paper abstracts.
 * @returns at most `maxAuthors` entries in document order.
 */
export function collectSupportAuthors(
  data: unknown,
  maxAuthors: number,
  maxPapersPerAuthor: number,
  abstractMaxChars: number,
): NormalizedSupportAuthor[] {
  const entries: NormalizedSupportAuthor[] = []
  const seen = new Set<string>()
  for (const obj of walkObjects(data)) {
    const supportPapersRaw = obj['support_papers']
    if (!Array.isArray(supportPapersRaw)) continue
    const name = asString(firstPresent(obj, AUTHOR_NAME_KEYS))
    const id = asString(firstPresent(obj, ['author_id', 'id', 'openalex_id']))
    const key = (id ?? name ?? `entry-${entries.length}`).toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)

    const supportPapers: NormalizedPaper[] = []
    const seenPapers = new Set<string>()
    for (const paperObj of supportPapersRaw) {
      if (!isObject(paperObj)) continue
      const paper = asNormalizedPaper(paperObj, abstractMaxChars)
      if (paper === undefined) continue
      const paperKey = paperDedupeKey(paper)
      if (seenPapers.has(paperKey)) continue
      seenPapers.add(paperKey)
      supportPapers.push(paper)
      if (supportPapers.length >= maxPapersPerAuthor) break
    }

    entries.push({
      ...name !== undefined ? { name: compactText(name, 160) } : {},
      ...id !== undefined ? { id: compactText(id, 160) } : {},
      supportPapers,
    })
    if (entries.length >= maxAuthors) break
  }
  return entries
}
