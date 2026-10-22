/**
 * Relevance-level mapping for SciAtlas plan anchors. Levels and base values
 * mirror the SciAtlas CLI (`sciatlas/src/sciatlas/cli.py`): keyword
 * high/middle/low → score 9/7/4, title → confidence 0.95/0.85/0.65, and
 * reference anchors accept only high/middle (0.95/0.80) as title anchors.
 * Unlike the CLI, this plugin applies no ±1 keyword-quality adjustments: the
 * mapping must stay deterministic for a model-facing tool.
 *
 * @module dsh-sciatlas/relevance
 */

/** Three-level relevance anchor shared by keyword, title, and reference inputs. */
export type Relevance = 'low' | 'middle' | 'high'

/** All accepted relevance levels, in increasing strength. */
export const RELEVANCE_LEVELS: readonly Relevance[] = ['low', 'middle', 'high']

/** Keyword relevance → `plan.keywords[].score` (1–10). */
export const KEYWORD_SCORE_BY_RELEVANCE: Record<Relevance, number> = {
  high: 9,
  middle: 7,
  low: 4,
}

/** Title relevance → `plan.titles[].confidence` (0–1). */
export const TITLE_CONFIDENCE_BY_RELEVANCE: Record<Relevance, number> = {
  high: 0.95,
  middle: 0.85,
  low: 0.65,
}

/**
 * Reference relevance → extra title-anchor confidence, or `undefined` for
 * `low`: low-relevance references stay reference-only, matching the CLI.
 *
 * @param relevance - the reference anchor's level.
 * @returns the title-anchor confidence to add, or `undefined`.
 */
export function referenceAnchorConfidence(relevance: Relevance): number | undefined {
  if (relevance === 'high') return 0.95
  if (relevance === 'middle') return 0.8
  return undefined
}
