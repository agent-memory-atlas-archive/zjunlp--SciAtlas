/**
 * SciAtlas plugin for the DeepSeek Harness: five retrieval tools and six
 * fixed-stage research workflows. Research uses DSH models, owned jobs,
 * filesystem input, and resumable artifacts; no Python runtime is required.
 *
 * @module dsh-sciatlas
 */

import type { Context } from '@deepseek-ai/cordis'
import { credentialRef, isCredentialRefName } from '@deepseek-ai/dsh-credentials'
import { launchEnvironmentOf } from '@deepseek-ai/dsh-launch-environment'
import z from '@deepseek-ai/schemastery'
import type {} from '@deepseek-ai/dsh-tools'
import { normalizeBaseUrl } from './client.ts'
import type { SciAtlasToolDeps } from './tools/shared.ts'
import { createSearchPapersTool } from './tools/search-papers.ts'
import { createRelatedAuthorsTool } from './tools/related-authors.ts'
import { createAuthorPapersTool } from './tools/author-papers.ts'
import { createSupportPapersTool } from './tools/support-papers.ts'
import { createPaperSearchTool } from './tools/paper-search.ts'
import { createResearchTool, ResearchTasks } from './tools/research.ts'
import type { ResearchConfig, ResearchCommand } from './research/types.ts'
import { resolve } from 'node:path'

export { CANONICAL_BASE_URL, SciAtlasApiError, normalizeBaseUrl } from './client.ts'
export type { NormalizedAuthor, NormalizedPaper, NormalizedSupportAuthor } from './normalize.ts'
export type { SciAtlasToolDeps } from './tools/shared.ts'
export type { ResearchConfig, ResearchInput, ResearchResult } from './research/types.ts'

/** Cordis plugin name used by loader diagnostics. */
export const name = 'sciatlas'

/** Required services: the tool registry and the credential store. */
export const inject = ['tools', 'credentials']

/** Validated deployment defaults; every limit is configurable through Cordis. */
export const RESEARCH_DEFAULTS: ResearchConfig = {
  runsDir: 'runs/sciatlas-dsh', llmProvider: '', llmModel: '', llmTimeoutMs: 180000,
  maxOutputTokens: 8192, maxInputChars: 200000, workflowTimeoutMs: 7200000,
  maxConcurrentRuns: 2, maxStageConcurrency: 3, maxModelCalls: 150, maxSearchCalls: 100,
  maxPapers: 400, maxPdfBytes: 20971520, grobidBaseUrl: '', pdfTimeoutMs: 180000, background: true,
  semanticScholarEnabled: true, semanticScholarBaseUrl: 'https://api.semanticscholar.org/graph/v1', semanticScholarKeyEnv: 'S2_API_KEY',
}

const ResearchConfigSchema: z<ResearchConfig> = z.object({
  runsDir: z.string().default(RESEARCH_DEFAULTS.runsDir),
  llmProvider: z.string().default(''), llmModel: z.string().default(''),
  llmTimeoutMs: z.number().step(1).min(1000).max(2147483647).default(RESEARCH_DEFAULTS.llmTimeoutMs),
  maxOutputTokens: z.number().step(1).min(1).default(RESEARCH_DEFAULTS.maxOutputTokens),
  maxInputChars: z.number().step(1).min(1000).default(RESEARCH_DEFAULTS.maxInputChars),
  workflowTimeoutMs: z.number().step(1).min(1000).max(2147453647).default(RESEARCH_DEFAULTS.workflowTimeoutMs),
  maxConcurrentRuns: z.number().step(1).min(1).default(RESEARCH_DEFAULTS.maxConcurrentRuns),
  maxStageConcurrency: z.number().step(1).min(1).default(RESEARCH_DEFAULTS.maxStageConcurrency),
  maxModelCalls: z.number().step(1).min(1).default(RESEARCH_DEFAULTS.maxModelCalls),
  maxSearchCalls: z.number().step(1).min(1).default(RESEARCH_DEFAULTS.maxSearchCalls),
  maxPapers: z.number().step(1).min(1).default(RESEARCH_DEFAULTS.maxPapers),
  maxPdfBytes: z.number().step(1).min(1024).default(RESEARCH_DEFAULTS.maxPdfBytes),
  grobidBaseUrl: z.string().default(''), pdfTimeoutMs: z.number().step(1).min(1000).max(2147483647).default(RESEARCH_DEFAULTS.pdfTimeoutMs),
  background: z.boolean().default(true), semanticScholarEnabled: z.boolean().default(true),
  semanticScholarBaseUrl: z.string().default(RESEARCH_DEFAULTS.semanticScholarBaseUrl),
  semanticScholarKeyEnv: z.string().default(RESEARCH_DEFAULTS.semanticScholarKeyEnv),
}).default(RESEARCH_DEFAULTS)

/** Plugin configuration. */
export interface Config {
  research: ResearchConfig
  /** SciAtlas API base URL; legacy SciNet hosts are rewritten. */
  baseUrl: string
  /** Name of the environment variable / credential reference holding the API key (not the key itself). */
  apiKeyEnv: string
  /**
   * Per-request timeout in milliseconds. Knowledge-graph retrieval routinely
   * takes minutes, so the default is 6 minutes (the SciAtlas CLI uses 15); the
   * tool-level cooperative budget is this value plus 30 seconds.
   */
  timeoutMs: number
  /** Default result count when a call carries no explicit topK/limit. */
  defaultTopK: number
  /** Hard cap applied to every requested result count. */
  maxTopK: number
  /** Default retrieval mode for search calls. */
  defaultRetrievalMode: 'keyword' | 'semantic' | 'title' | 'hybrid'
  /** Truncation bound for abstracts in normalized papers. */
  abstractMaxChars: number
}

export const Config: z<Config> = z.object({
  research: ResearchConfigSchema,
  baseUrl: z.string().default('http://sciatlas.openkg.cn'),
  apiKeyEnv: z.string().default('SCIATLAS_API_KEY'),
  timeoutMs: z.number().step(1).min(1000).default(360_000),
  defaultTopK: z.number().step(1).min(1).default(10),
  maxTopK: z.number().step(1).min(1).default(50),
  defaultRetrievalMode: z.union(['keyword', 'semantic', 'title', 'hybrid'] as const).default('hybrid'),
  abstractMaxChars: z.number().step(1).min(200).default(4000),
})

/** Register all eleven tools and own their research-task lifecycle. */
export function apply(ctx: Context, config: Config): void {
  if (!isCredentialRefName(config.apiKeyEnv)) {
    throw new Error(
      `sciatlas: config.apiKeyEnv "${config.apiKeyEnv}" is not a valid environment-variable name (POSIX identifier, e.g. SCIATLAS_API_KEY).`,
    )
  }
  if (config.defaultTopK > config.maxTopK) {
    throw new Error(
      `sciatlas: config.defaultTopK (${config.defaultTopK}) exceeds config.maxTopK (${config.maxTopK}).`,
    )
  }
  if (!!config.research.llmProvider !== !!config.research.llmModel) throw new Error('sciatlas: research.llmProvider and research.llmModel must be configured together')
  if (!isCredentialRefName(config.research.semanticScholarKeyEnv)) throw new Error('sciatlas: research.semanticScholarKeyEnv is invalid')
  for (const [field, value] of [['baseUrl', config.baseUrl], ['grobidBaseUrl', config.research.grobidBaseUrl], ['semanticScholarBaseUrl', config.research.semanticScholarBaseUrl]]) {
    if (value && !/^https?:\/\//i.test(value)) throw new Error(`sciatlas: ${field} must use HTTP or HTTPS`)
  }

  const resolveApiKey = async (): Promise<string> => {
    const stored = await ctx.credentials.resolve(credentialRef(config.apiKeyEnv))
    if (stored !== undefined && stored.value.trim().length > 0) return stored.value.trim()
    const ambient = launchEnvironmentOf(ctx).get(config.apiKeyEnv)?.value
    if (ambient !== undefined && ambient.trim().length > 0) return ambient.trim()
    throw new Error(
      `sciatlas: no API key found. Set the ${config.apiKeyEnv} environment variable or store it in `
      + '$DSH_HOME/.credentials.yaml; register a token at http://sciatlas.openkg.cn/register.',
    )
  }

  const deps: SciAtlasToolDeps = {
    baseUrl: normalizeBaseUrl(config.baseUrl),
    timeoutMs: config.timeoutMs,
    defaultTopK: config.defaultTopK,
    maxTopK: config.maxTopK,
    defaultRetrievalMode: config.defaultRetrievalMode,
    abstractMaxChars: config.abstractMaxChars,
    resolveApiKey,
  }

  ctx.tools.register(createSearchPapersTool(deps))
  ctx.tools.register(createRelatedAuthorsTool(deps))
  ctx.tools.register(createAuthorPapersTool(deps))
  ctx.tools.register(createSupportPapersTool(deps))
  ctx.tools.register(createPaperSearchTool(deps))
  const research = { ...config.research, runsDir: resolve(config.research.runsDir) }
  const tasks = new ResearchTasks(research.maxConcurrentRuns)
  ctx.effect(() => () => tasks.dispose())
  const commands: ResearchCommand[] = ['idea-grounding', 'trend-report', 'researcher-review', 'literature-review', 'idea-evaluate', 'idea-generate']
  for (const command of commands) ctx.tools.register(createResearchTool(command, ctx, deps, research, tasks))
}
