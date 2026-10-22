/** GROBID HTTP parsing and TEI passage extraction for PDF-based research. */
import { XMLParser, XMLValidator } from 'fast-xml-parser'
import { object, text } from './json.ts'
import type { Evidence, ResearchConfig } from './types.ts'

export interface ParsedDocument { title: string; abstract: string; paragraphs: Evidence[]; tei: string }

/** Extract text in document order, including inline reference and formula nodes. */
function nodeText(value: unknown): string {
  if (typeof value === 'string' || typeof value === 'number') return String(value)
  if (Array.isArray(value)) return value.map(nodeText).join(' ')
  if (value && typeof value === 'object') return Object.entries(value).filter(([key]) => !key.startsWith('@_')).map(([, item]) => nodeText(item)).join(' ')
  return ''
}

/** Parse ordered TEI nodes without inventing page positions. */
export function parseTei(tei: string, paperId: string, source: string): ParsedDocument {
  if (/<!DOCTYPE|<!ENTITY/i.test(tei)) throw new Error('GROBID XML declarations are not accepted')
  if (XMLValidator.validate(tei) !== true) throw new Error('Invalid GROBID TEI XML')
  const parser = new XMLParser({ preserveOrder: true, ignoreAttributes: false, removeNSPrefix: true, parseTagValue: false, processEntities: false })
  const nodes: unknown = parser.parse(tei)
  let title = ''
  let abstract = ''
  const paragraphs: Evidence[] = []
  const visit = (value: unknown, section: string[], inBody = false, inAbstract = false, inTitle = false): void => {
    if (!Array.isArray(value)) return
    let heading = section
    for (const entry of value) {
      const node = object(entry, 'TEI node')
      for (const [tag, children] of Object.entries(node)) {
        if (tag === 'head') { const label = nodeText(children).replace(/\s+/g, ' ').trim(); if (label) heading = [...section, label] }
        if (tag === 'title' && inTitle && !title) title = nodeText(children).replace(/\s+/g, ' ').trim()
        if (tag === 'abstract') abstract = nodeText(children).replace(/\s+/g, ' ').trim()
        if (tag === 'p' && (inBody || inAbstract)) {
          const content = nodeText(children).replace(/\s+/g, ' ').trim()
          if (content) paragraphs.push({ evidenceId: `${paperId}-e${paragraphs.length + 1}`, paperId, section: inAbstract ? ['Abstract'] : heading.length ? heading : ['Body'], paragraphIndex: paragraphs.length, text: content, level: inAbstract ? 'abstract' : 'fulltext', source })
        }
        if (tag !== 'p' && tag !== 'head') visit(children, tag === 'div' ? [...heading] : heading, inBody || tag === 'body', inAbstract || tag === 'abstract', inTitle || tag === 'titleStmt')
      }
    }
  }
  visit(nodes, [])
  if (!title && paragraphs.length === 0) throw new Error('No readable text found in PDF; scanned documents require OCR')
  return { title: title || 'Untitled PDF', abstract, paragraphs, tei }
}

/** Call GROBID without model or SciAtlas credentials. */
export async function parsePdf(bytes: Uint8Array, filename: string, paperId: string, config: ResearchConfig, signal: AbortSignal): Promise<ParsedDocument> {
  if (!config.grobidBaseUrl.trim()) throw new Error('PDF input requires research.grobidBaseUrl')
  if (bytes.length > config.maxPdfBytes) throw new Error('PDF exceeds research.maxPdfBytes')
  if (new TextDecoder().decode(bytes.slice(0, 1024)).indexOf('%PDF-') < 0) throw new Error('Input is not a PDF document')
  const form = new FormData()
  form.append('input', new Blob([new Uint8Array(bytes)], { type: 'application/pdf' }), filename)
  form.append('consolidateHeader', '0')
  const response = await fetch(`${config.grobidBaseUrl.replace(/\/$/, '')}/api/processFulltextDocument`, {
    method: 'POST', redirect: 'error', body: form, signal: AbortSignal.any([signal, AbortSignal.timeout(config.pdfTimeoutMs)]),
  })
  if (!response.ok) throw new Error(`GROBID returned HTTP ${response.status}`)
  const tei = await response.text()
  if (tei.length > config.maxPdfBytes * 2) throw new Error('GROBID XML exceeds the configured size bound')
  return parseTei(text(tei, 'TEI'), paperId, filename)
}

/** Lexical shortlist for subsequent LLM reranking, retaining stable evidence ids. */
export function lexicalEvidence(query: string, evidence: readonly Evidence[], topK: number): Evidence[] {
  const terms = [...new Set(query.toLowerCase().match(/[a-z0-9]+/g) ?? [])]
  const tokenized = evidence.map(item => item.text.toLowerCase().match(/[a-z0-9]+/g) ?? [])
  const frequencies = new Map<string, number>()
  for (const tokens of tokenized) for (const term of new Set(tokens)) frequencies.set(term, (frequencies.get(term) ?? 0) + 1)
  return evidence.map((item, index) => {
    const tokens = tokenized[index]!
    const counts = new Map<string, number>()
    for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1)
    return { item, index, score: terms.reduce((sum, term) => { const count = counts.get(term) ?? 0; return sum + (count ? (1 + Math.log(count)) * Math.log(1 + evidence.length / (frequencies.get(term) ?? 1)) : 0) }, 0) }
  }).filter(item => item.score > 0).sort((a, b) => b.score - a.score || a.index - b.index).slice(0, topK).map(item => item.item)
}
