# dsh-sciatlas

[English](README.md) · 中文

SciAtlas 的 DSH 科研插件。0.2 版提供 11 个工具：5 个检索工具和 6 个科研频道，全部使用 TypeScript，复用 DSH 的模型、凭证、文件与后台任务服务。运行时不需要 Python、PyTorch、FAISS 或本地 embedding 模型。

| 工具 | 功能 |
|---|---|
| `sciatlas_search_papers` | KG 图检索，关键词/标题/参考标题锚点、日期、严格领域、偏向及排序 |
| `sciatlas_paper_search` | 低层论文检索，独立 mode/field，默认 topK 5 |
| `sciatlas_related_authors` | 相关学者、指标与可用支撑论文 |
| `sciatlas_author_papers` | 姓名/OpenAlex ID 查论文，控制同名合并、去重、摘要与 embeddings |
| `sciatlas_support_papers` | 每位候选作者针对主题的支撑论文 |
| `sciatlas_idea_grounding` | precision 检索与已有工作对照指南 |
| `sciatlas_trend_report` | impact 检索与发表年份时间线 |
| `sciatlas_researcher_review` | 作者种子论文加标题锚定的 hybrid 图检索 |
| `sciatlas_literature_review` | 多轮检索、方法聚类、缺口、大纲、证据包与完整综述 |
| `sciatlas_idea_evaluate` | idea/PDF 上下文、段落证据、历史 rubric、模拟评审与 meta-review |
| `sciatlas_idea_generate` | 种子、研究图、趋势、RSS、灵感、类比、筛选、idea 与新颖性反馈 |

## 安装

从 npm 安装进 `web` profile：

```sh
dsh plugin --profile web add dsh-sciatlas
dsh web
```

开发时，在插件目录安装、构建；开发依赖链接到同级 DSH checkout：

```sh
cd dsh-sciatlas
pnpm install
pnpm run build
```

在 `deepseek-harness/` 启动开发 overlay：

```sh
pnpm dsh web --patch ../dsh-sciatlas/cordis.dev.yml
```

持久安装本地构建后的 bundle：

```sh
dsh plugin --profile web add /absolute/path/to/dsh-sciatlas
dsh web
```

在 <http://sciatlas.openkg.cn/register> 领取 token，通过启动环境的 `SCIATLAS_API_KEY` 或 `$DSH_HOME/.credentials.yaml` 中同名凭证提供。凭证存储优先；密钥不进入工具参数。

三个长流程需要已注册的 DSH 模型适配器，默认采用调用者会话日志中的 provider/model。显式指定模型时，必须同时配置 `research.llmProvider` 与 `research.llmModel`。后台运行需要 DSH jobs 及其通用控制工具；PDF/文本文件输入需要 DSH 文件服务，PDF 还需要 GROBID。

## 配置

插件行的 `config` 示例：

```yaml
baseUrl: http://sciatlas.openkg.cn
apiKeyEnv: SCIATLAS_API_KEY
timeoutMs: 360000
defaultTopK: 10
maxTopK: 50
abstractMaxChars: 4000
research:
  runsDir: runs/sciatlas-dsh
  background: true
  llmProvider: ''
  llmModel: ''
  grobidBaseUrl: '' # 配置成已有 GROBID 地址，例如 http://127.0.0.1:8070
  semanticScholarEnabled: true
  semanticScholarKeyEnv: S2_API_KEY
  maxConcurrentRuns: 2
  maxStageConcurrency: 3
  maxModelCalls: 150
  maxSearchCalls: 100
  maxPapers: 400
  maxInputChars: 200000
  maxOutputTokens: 8192
  llmTimeoutMs: 180000
  workflowTimeoutMs: 7200000
  maxPdfBytes: 20971520
  pdfTimeoutMs: 180000
```

相对路径以 DSH 进程工作目录为基准。Semantic Scholar 用于评审补充检索和生成流程的引用边获取，有凭证时使用配置的引用名，失败会写入报告限制。设 `semanticScholarEnabled:false` 可仅使用 SciAtlas。所有默认值与 CLI 参数映射见 [实现与覆盖表](docs/implementation.md)。

## 使用

研究频道用 `query` 输入主题/idea，`idea` 为别名，`queryFile` 接收 UTF-8 文件。默认 `workflow:"flash"`；三个长流程默认后台运行。已有检索工具的名字和原输出字段保留。

`sciatlas_literature_review` 的 flash 默认停止于证据包，full 默认生成完整综述，可显式指定停止阶段：

```json
{"query":"molecular graph generalization","workflow":"full","options":{"subjectDomain":"chemistry","reportStopAfter":"full"}}
```

`options.reportStopAfter` 可选 `bibliography/outline/packs/full`。`sciatlas_idea_evaluate` 使用文本或一个 PDF，二者互斥：

```json
{"pdf":"papers/proposal.pdf","workflow":"flash","options":{"disableReviewLlm":true}}
```

`disableReviewLlm` 生成确定性证据清单，其他规划、rubric 阶段仍调用模型。`sciatlas_idea_generate` 支持主题与多个种子 PDF：

```json
{"query":"causal molecular graph models","pdfs":["papers/seed.pdf"],"workflow":"full","options":{"ideaCount":2,"graphBudgetMin":12,"graphBudgetMax":30}}
```

后台结果包含 `kind/jobId/runId/status/artifactDir/reportPath/resultPath`。用 DSH 已安装的 jobs 工具按 `jobId` 查看、等待或取消；`background:false` 可前台等待。阶段完成前文件可能尚未存在，返回路径不表示报告已生成。

产物包括 `manifest.json`、阶段检查点、`papers.json`、`result.json`、`report.md`；失败保存 `error.json` 和已成功阶段。默认保存原始检索请求/响应、模型文本及 TEI；`options.saveIntermediate:false` 关闭诊断副本，仍保留恢复所需检查点及科研输出。模型输入和原始流事件保留于 DSH 会话日志。

在同一 DSH 会话中用原研究工具恢复：

```json
{"resumeRunId":"research-..."}
```

恢复检查归属、管线/提示词版本、有效参数、文件字节摘要、模型路由和配置，复用成功阶段、重试失败阶段，调用预算累计计算。输入变更需新建运行。重启后原生 jobId 失效，但插件检查点可在 DSH 恢复同一会话身份后显式继续。

## 证据与边界

GROBID `processFulltextDocument` 返回的 TEI 被解析为章节和段落。摘要始终标为 `abstract`，不可读扫描件报错，不生成虚构页码。生成引用必须对应已获取的论文/段落 ID，书目链接取自检索元数据。评审是模拟视角，不代表真实学者意见。研究图分别存储检索关系、推断关系与可独立获取的书目引用关系。

科研行为迁移到 TypeScript，输出不承诺逐字等同 Python。CLI 的自由文本 flag 推断改为结构化工具字段。本期不包含本地 Neo4j、embedding 空间扩展、辅助命令（`health/config/build-plan/make-report/skill`）和任意 Python 运行目录恢复。

## 验证

```sh
pnpm test
pnpm run typecheck
pnpm run build
```

离线测试使用本地 HTTP 服务、固定模型回放和真实 Cordis/DSH 模型、会话、jobs 服务。线上检索测试独立显式开启：

```sh
SCIATLAS_LIVE_ENV=../SciAtlas/.env pnpm run test:live
# 或直接导出 SCIATLAS_API_KEY
```

线上测试少量调用五个检索端点，仅保存耗时和数量到 `runs/verification/live-retrieval.json`。普通测试跳过它们；这不代表真实模型/PDF 科研质量验收。已批准的 [计划](../docs/plans/2026-10-11-dsh-sciatlas-enhancement.md) 与 [迁移源码摘要](docs/migration-baseline.json) 可供复核。`python3 scripts/sync-prompts.py` 仅用于开发时同步提示词，不是运行依赖。
