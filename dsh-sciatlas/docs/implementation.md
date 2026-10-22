# SciAtlas DSH 0.2 实现与验收记录

实施依据：[2026-10-11 完善计划](../../docs/plans/2026-10-11-dsh-sciatlas-enhancement.md)。正式 CLI 基准为 `SciAtlas/sciatlas/src/sciatlas/cli.py`。迁移读取当前工作区，包括已有用户修改；[migration-baseline.json](migration-baseline.json) 记录 10 个参考文件的 SHA256。未改动 SciAtlas 或 DSH 核心实现。

## 交付状态

| 阶段 | 实现 | 证据 |
|---|---|---|
| P0 | 计划、源码摘要、覆盖表 | 本文、migration-baseline.json、sync-prompts.py |
| P1 | 五个检索工具与兼容扩展 | tools/retrieval-parity/client/normalize/relevance 测试 |
| P2 | DSH 模型/jobs/filesystem、GROBID、检查点、预算 | dsh-research/lifecycle/pdf 测试 |
| P3 | grounding/trend/researcher 频道 | research 固定回放及六工具 smoke |
| P4 | 综述多轮检索和四种停止阶段 | bibliography/outline/packs/full 回放 |
| P5 | 评审上下文、证据、rubric、评审者、meta | flash/full、确定性报告、PDF、外源失败回放 |
| P6 | 生成九阶段和有限新颖性反馈 | flash/full、PDF、未完成 novelty 的 partial 回放 |
| P7 | 注册/卸载、双语文档、配置与 bundle | Cordis 注册、实际 DSH seam、类型检查、构建；线上状态见文末 |

## 源码结构与执行边界

- `src/tools/`：模型可见参数、结构化输出 schema、渲染与调用入口。
- `src/workflows/`：固定 TS 阶段，不启动 Python 子进程，不让 agent 自由替代管线。
- `src/research/runtime.ts`：累积预算、一次 JSON/schema 修复、证据获取、并发排空。
- `src/research/dsh.ts`：prepareCall/stream、精确会话路由、DSH 文件读取、原生 jobs 生命周期。
- `src/research/store.ts`：私有运行目录、独占锁、原子串行写入、输入/内容校验。
- `src/research/pdf.ts`：GROBID multipart、TEI 章节/段落、与当前 CLI 一致的 TF-IDF 词重叠。
- `src/research/upstream-*.ts`：开发时从 Python AST 导出的完整生成和 rubric 模板；运行时为静态 TS 常量。

内部函数共享检索能力，不伪造 DSH 工具调用。每个真实检索及模型调用都携带取消信号。后台任务归属调用者 agent，发布后由 jobs 控制，外层工具调用结束不撤销它。插件卸载取消并排空已接受任务。所有 workers 在兄弟失败后排空；未完成写入前不释放运行锁。

模型通过注册绑定的 prepared call 发送，默认 provider/model 来自 `session.requestHeader()`，不会读取可能过时的 `agent.options`。`plugin:sciatlas/llm-request` 在 dispatch 前记录有效 config、adapterDefaults、system、输入和阶段，`plugin:sciatlas/llm-chunk` 记录原始流（含 usage）。这些外部插件附属记录通过 appendPluginRecord 标记 ignorable，持久化读取器能保留并跳过，不阻碍 DSH 会话恢复；科研检查点由插件自己负责。请求/结果没有写入 API key。必须收到真实 finish 事件且 reason 为 stop；截断、错误、取消、工具调用结束不接受为有效阶段结果。

## CLI 与工具覆盖

| CLI | 工具 | 端点/行为 |
|---|---|---|
| search-papers | sciatlas_search_papers | POST /v1/search |
| related-authors | sciatlas_related_authors | POST /v1/authors/related |
| author-papers | sciatlas_author_papers | POST /v1/authors/papers |
| support-papers | sciatlas_support_papers | POST /v1/authors/support-papers |
| paper-search | sciatlas_paper_search | POST /v1/papers/search，query_text/mode/field |
| idea-grounding | sciatlas_idea_grounding | precision 预设和相关文献对照 |
| trend-report | sciatlas_trend_report | impact 预设、年份分组，未知年份独立显示 |
| researcher-review | sciatlas_researcher_review | 最多五篇作者种子，再执行标题锚定 hybrid KG 检索 |
| literature-review | sciatlas_literature_review | 下述固定综述阶段 |
| idea-evaluate | sciatlas_idea_evaluate | 下述固定评审阶段 |
| idea-generate | sciatlas_idea_generate | 下述九阶段 |

### 通用参数与运行控制

| CLI 参数 | DSH 表达 | 说明 |
|---|---|---|
| query / text / idea | query，研究工具另支持 idea 别名 | 互斥，不复制自由文本 flag 推断 |
| text-file | 研究工具 queryFile | DSH 文件服务读取 UTF-8 |
| domain | domain | 软查询提示 |
| target-field | targetField | 后端严格过滤 |
| top-k | topK | 检索数量；生成建议显式 options.anchorTopK |
| after / before / time-range | timeRange.after / before | YYYY-MM-DD；校验真实日期和顺序 |
| keywords-json | keywords: [{text,relevance}] | 明确三档，不复制 CLI 的 ±1 质量调整 |
| titles-json / reference-titles-json | titles / referenceTitles: [{title,relevance}] | reference 的 high/middle 分别附加 0.95/0.80 标题置信度 |
| retrieval-mode / ranking-profile | retrievalMode / rankingProfile | 保留后端枚举 |
| bias-* | bias.keyword/nonSeedKeyword/citation/related/authorship/coauthorship/cooccurrence/exploration | low/middle/high 原值映射 |
| workflow / preset | workflow: flash/full；smoke:true | default 归入 flash；不再另设 preset |
| smoke | smoke | 明确合成数据，报告和工具结果均标记；不能作为科研结果 |
| env / llm-api-key/base-url/model | DSH 凭证与模型注册，research.llmProvider/llmModel | 不传密钥或任意模型 URL 参数 |
| llm-timeout | options.llmTimeoutMs | 工具单位为毫秒，CLI 为秒 |
| workflow-timeout | research.workflowTimeoutMs | 整个 TS 运行预算，默认两小时 |
| runs-dir / run-id / output-json / output-text / report-max-items | research.runsDir + 自动 runId + canonical JSON + report.md | 研究结果预览最多十篇，完整元数据保留在产物；检索工具直接返回 canonical 结果 |
| resume-from-run-dir | resumeRunId | 同插件、同会话的检查点；不读取任意 Python 运行目录 |
| no-intermediate-json | options.saveIntermediate:false | 关闭原始请求/响应、模型文本、TEI 诊断副本；仍保存恢复检查点和科研产物 |
| background / cancel | background + DSH jobs | 本插件不重复注册 jobs 控制工具 |

原四工具保留已有默认值和原输出字段。search-papers 与 author-papers 的默认数量仍用插件 `defaultTopK`（10），后者与 CLI 通常为 20 的默认值有差异；显式 `limit:20` 可对齐。related-authors 默认 5、support 默认每作者 3、新增低层 paper-search 默认 5。请求超过配置上限时，检索结果附加 `requestedCount/effectiveCount/truncated`；这是预算被限幅的标记，不是对后端全集数量的推断。研究工作流的显式超预算请求在发布任务前拒绝。

作者参数：`author` 支持姓名/OpenAlex id/URL；`mergeSameNameAuthors:false`、`dedupePapers:false`、`includeAbstract:false`、`includeEmbeddings:true` 对应 CLI 开关。未请求的向量不进入结果，禁用摘要时结果也不返回摘要。支撑工具的 `authors` 是姓名/id 数组，支持 `id:` 前缀；`topKPerAuthor/fetchAuthorStats/authorSearchFallback` 对应相关选项。researcher-review 保留当前 CLI 实际路径：种子查询不用摘要/向量，后续 KG 文献不能据此认定是目标作者所著。

### 综述参数和阶段

| CLI | options 字段 |
|---|---|
| probe-top-k / round-top-k | probeTopK / roundTopK |
| round1-action-limit / round2-action-limit | round1ActionLimit / round2ActionLimit |
| llm-paper-limit | llmPaperLimit |
| subject-domain / report-stop-after | subjectDomain / reportStopAfter |
| outline-timeout / integration-timeout | outlineTimeoutMs / integrationTimeoutMs |
| enable-embedding-expansion | 本期明确排除 |

阶段：probe → 方法聚类/时间窗口/Round 1 actions → Round 1 → relevance guard → gaps/Round 2 actions → 可选 Round 2 → final coverage → bibliography 或 outline → evidence packs → 可选章节写作/全文整合。每轮 action 日期与用户日期取交集，不扩大用户限制。flash 默认 probe 8、round 6、最多 3/0 actions、LLM 论文 30、停止 packs；full 为 30/20、7/4、100、停止 full。`reportStopAfter` 可覆盖为 bibliography/outline/packs/full；smoke 与当前 CLI 一致只停 bibliography。coverage 不足保留 partial 和限制。

### 评审参数和阶段

| CLI | options 字段 |
|---|---|
| kg-top-k / s2-top-k / manifest-top-k | kgTopK / s2TopK / manifestTopK |
| grounding-final-top-k | groundingFinalTopK |
| dense-candidate-k / reranker-top-n（params-json 覆盖） | groundingCandidateK / groundingRerankTopN（默认 40 / 20） |
| max-paragraphs-per-paper | maxParagraphsPerPaperPerQuery（默认 2） |
| rerank_batch_size（params-json 覆盖） | rerankBatchSize（默认 4） |
| max-reviewers / reviewer-selection-top-k | maxReviewers / reviewerSelectionTopK |
| reviewer-max-workers / review-max-workers | reviewerMaxWorkers / reviewMaxWorkers |
| review-max-evidence-cards-per-section/query | maxEvidenceCardsPerSection / maxEvidenceCardsPerQuery |
| review-max-experiment-cards-per-section | maxExperimentCardsPerSection |
| review-max-total-evidence-chars-per-section | maxEvidenceCharsPerSection |
| review-llm-timeout-seconds / rubric-llm-timeout-seconds | reviewTimeoutMs / rubricTimeoutMs |
| short-report / disable-llm-ranking / disable-review-llm | shortReport / disableLlmRanking / disableReviewLlm |
| pdf | pdf，一个，与 idea 文本互斥 |

阶段：文本/PDF context → KG + 可选 S2 → 去重/分批 LLM 相关度打分（每批 rerankBatchSize 篇、段落级评分细则与 [0,10] 约束同上游 merge_search/idea_rubric，失败批次降级为检索序并记 partial）→ PDF/摘要证据 → TF-IDF 候选（groundingCandidateK）/前 groundingRerankTopN 段分批 LLM 段落排名（每批 8 段、单段截断 1200 字符；整查询失败降级为词法序）→ 每论文历史 norms（输入含 venue/年份/被引与 8 条 general 评审 persona 笔记；失败降级为 persona 笔记）/dimensions（上游四模块 schema，单论文失败跳过）→ General/Detailed/四分区 synthesis rubric → 相关评审者/作者文献背景 → 分区证据预算（每查询每论文最多 maxParagraphsPerPaperPerQuery 段）→ 模拟评审 → meta-review。四分区为 Motivation/Method/Result/Discussion。flash 默认为 KG/S2/manifest 各 6、一个评审者、短报告；full 为各 20、最多三个评审者。并发还受部署 `maxStageConcurrency` 限制。无 review LLM 路径生成“要求与证据存在性清单”，不会声称已完成专家评分，其他上下文/rubric 模型阶段照常执行。

模型/背景/外源局部失败保留可用 rubric、证据及 partial 标记；没有真实评审者候选时显式用通用模拟视角。人物意见不作真实归属。无证据时不生成声称已完成的文献评审。

### 生成参数和阶段

| CLI | options 字段/DSH 表达 |
|---|---|
| seed-recent-years | seedRecentYears |
| graph-budget-min/max | graphBudgetMin / graphBudgetMax |
| k-step1 / seed-num-queries / seed-per-query | kStep1 / seedNumQueries / seedPerQuery |
| anchor-top-k | anchorTopK |
| idea-count | ideaCount |
| pdf（可重复） | pdfs 数组，最多十个 |
| novelty feedback | maxNoveltyFeedbackRounds；flash 0/full 2，最大 5 |

1. query refinement → diverse queries → recent 或高被引 seed → relevance selection；输入 PDF 可进入候选。
2. 有界图扩展 → predecessor selection → 五字段 → phase/topology。S2 引用边与 KG 检索边、模型推断边分开；不能把相关度当作 CITES。
3. 图上的趋势与限制。
4. RSS 的对象、关系、瓶颈、机制需求。
5. inspiration radius gate；flash 确定性 gate，full 模型 gate。
6. 同领域与跨域检索。
7. 候选机制类比与组合可行性。
8. 灵感筛选；flash 确定性排序，full 模型筛选。
9. 具体方法/假设/实验/引用的 idea → 已检索文献内 novelty check → 有限反馈重试。

flash 图预算默认 5、一个 seed、一个 query；full 默认图 12–30、四个 seed、八个 query。达不到 graphBudgetMin 会标记限制，不填入无关论文。新颖性检查未完成时保留候选并标记 partial；检索样本的新颖性结论不作全领域证明。

## 部署配置默认值

| 字段 | 默认 |
|---|---|
| baseUrl / apiKeyEnv | http://sciatlas.openkg.cn / SCIATLAS_API_KEY |
| timeoutMs / defaultTopK / maxTopK | 360000 / 10 / 50 |
| defaultRetrievalMode / abstractMaxChars | hybrid / 4000 |
| research.runsDir / background | runs/sciatlas-dsh / true |
| research.llmProvider / llmModel | 空，继承日志路由；显式填写必须成对 |
| research.llmTimeoutMs / maxOutputTokens / maxInputChars | 180000 / 8192 / 200000 |
| research.workflowTimeoutMs | 7200000 |
| research.maxConcurrentRuns / maxStageConcurrency | 2 / 3 |
| research.maxModelCalls / maxSearchCalls / maxPapers | 150 / 100 / 400 |
| research.maxPdfBytes / pdfTimeoutMs / grobidBaseUrl | 20971520 / 180000 / 空 |
| research.semanticScholarEnabled / semanticScholarKeyEnv | true / S2_API_KEY |
| research.semanticScholarBaseUrl | https://api.semanticscholar.org/graph/v1 |

maxInputChars 包含 system 和 JSON 输入，超限在发送模型前拒绝；model/search 次数在发起前检查，格式修复也计费计数，恢复沿用累计值。阶段特定 timeout 选项可覆盖默认 LLM timeout。所有网络超时还受整个 workflow signal 限制。完整论文摘要保留在科研产物；工具预览才按显示预算处理。

## 产物、恢复与证据

每次运行默认 `<runsDir>/research-UUID/`，目录/文件以 0700/0600 新建。manifest 记录 owner、command、输入及有效选项、配置、文件 digest、路由、提示词版本、阶段状态/耗时/inputHash/contentHash、调用次数和最终状态。文件原子替换；`.lock` 只在原进程已不存在时回收。成功检查点均在读回时校验摘要和结构。返回成功之前写完 `papers.json/result.json/report.md`；失败/取消保留 `error.json`。

- 综述：lr-* 阶段、evidence-packs.json、章节/整合结果。
- 评审：review-*、rubric-*、grounding.json、rubric.json、评审者与 meta。
- 生成：step1_seed_papers.json、step2_research_graph.json、各 step 模型检查点、step8_selected_inspirations_*、step9_ideas_*。
- 原始输入/流在 DSH 会话；默认附加 API request/response、model response、TEI 诊断副本。

恢复仅接受同 owner、版本、输入、有效选项、配置、模型路由、文件字节和提示词摘要。成功阶段复用，真正 failed 阶段重试；已作为受控降级完成的 S2/PDF 失败会复用并保留警告，重新获取这些外源需新运行。不会自动恢复 Python 目录或自动在重启后继续。变更固定管线阶段/解码格式必须提升 PIPELINE_VERSION；变更 TS 自写科学提示词须提升版本前缀。导出模板与科学规则的摘要自动进入 PROMPT_VERSION。

PDF 以 HTTP multipart 发送到 GROBID `/api/processFulltextDocument`。拒绝非 PDF、超限 PDF、XML 实体/DTD 和无可读段落 TEI。证据包含 source、section、paragraphIndex、paperId/evidenceId、level。GROBID 未提供可靠页号时不添加 page。全文获取失败只回退到已有摘要，不编造全文。生成文本的引用 ID 和 URL 在模型边界校验，书目从已取得元数据构造。PDF 解析器不包含 OCR 服务。

## 有意调整与范围限制

- 不调用 Python。rubric/生成原模板静态导入；综述提示词以固定 TS JSON 阶段表达相同科研任务，因此不保证字面输出一致或随机学者抽样一致。
- CLI 智能参数推断由 DSH 模型选择明确字段代替；旧四工具的相关度映射保持确定性，不复制 ±1 keyword 质量修正。
- 评分/段落证据使用当前 KG rubric/lexical+LLM 路径。可选重型本地 embedding 空间 Round 2、Neo4j、全本地索引路径明确排除。
- 评审打分采用单次覆盖（idea_rubric 语义），不复制 merge_search 的 coverage=2 二次打分取均值；grounding 的 `--enable-grounding-refinement` 上下文精修与实验 grounding（上游默认关闭）不在本期。
- 上游 2026-10-11 下午的七个 LLM 配置解析修复（`LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL` 别名链）不适用于本插件：模型调用经 DSH 模型服务路由，不直接拼接触达 LLM endpoint。
- author name 合并可能有同名歧义；图相关文献不当作已验证著作。文章的 DOI/title 稳定身份与去重也不能消除所有元数据错误。
- health/config/build-plan/make-report/skill 管理属于辅助命令，本期不注册为科研工具。DSH 的配置、jobs 和文件工具承接运行管理。
- 科研报告保留证据层级和 partial 限制；结构与引用校验不等同事实正确性或论文质量验收。

## 验证记录

离线套件覆盖真实 HTTP 请求映射、五检索端点、十一工具注册/卸载、canonical 输出、六工具 smoke/恢复、固定模型回放、综述四停止阶段、评审/生成 flash/full、PDF 输入和 byte digest 失效、独占锁、取消、卸载排空、并发目录隔离、预算、一次格式修复、伪引用拒绝、损坏检查点和局部失败。真实 DSH LlmRuntime/SessionStore/LocalJobRegistry 用于路由、dispatch 前日志、流结束与后台任务检查。

2026-10-11 最终离线验收：`pnpm test` **94 passed / 3 skipped**（11 个通过文件，线上套件默认跳过）；`pnpm run typecheck`、`pnpm run build` 均通过。构建后的 `lib/index.js` 在真实 Cordis/ToolRuntime 中注册 **11** 个工具，卸载后剩 **0** 个。`pnpm dsh --profile web --patch ../dsh-sciatlas/cordis.dev.yml --dump-config` 成功包含插件层。`pnpm pack --pack-destination runs/verification` 生成 0.2.0 bundle。10 个迁移源码摘要复核未变化。线上测试通过 `pnpm run test:live` 单独执行，仅获取小规模真实检索；默认离线测试显式 skip。GROBID/模型线上科研端到端验证与人工质量验收必须单独记录，不能由 mock 或 smoke 宣称完成。

2026-10-11 晚（CLI 去本地模型化对齐轮）：评审工作流对齐当前上游行为——分批论文相关度打分（含评分细则正文、失败批次降级检索序）、段落 grounding 改为候选 40/重排池 20/每批 8 段/整查询词法兜底/每论文 2 段多样性、norms 阶段补 venue/年份/被引与 8 条 general persona 笔记（persona 笔记静态移植自 `persona_prompts.json`，含失败兜底）、dimensions 切到上游四模块 schema。新增 `groundingCandidateK`/`groundingRerankTopN`/`maxParagraphsPerPaperPerQuery`/`rerankBatchSize` 四个选项；阶段结构变化使 PIPELINE_VERSION 升至 `2`、PROMPT_VERSION 前缀升至 `sciatlas-ts-2-`。上游同日下午的 LLM 配置解析修复与插件无关（插件经 DSH 模型服务路由）。验证：`pnpm test` **109 passed / 3 skipped**（新增打分批次失败降级、grounding 词法兜底、norms persona 兜底三个回放），`pnpm run typecheck` 与 `pnpm run build` 通过；migration-baseline.json 刷新至当前工作区。


本轮显式线上验证实际使用了本地已有 SciAtlas 凭证：paper-search、search-papers、related-authors 三个请求各在 **300000 ms** 后超时，三个线上测试失败。由于相关学者前置请求未完成，作者论文及支撑论文检查未到达。站点 HEAD 返回 200，未认证 API 请求返回 401；这只能证明站点/认证端点可达，不能证明检索可用，也不能确定超时的服务端原因。没有据此提高默认超时或重复发起昂贵检索。真实模型科研流程、真实 GROBID/PDF 输入及报告质量未做线上验收；已通过的是其离线固定回放与真实 DSH 服务接缝。验证摘要保存在 `runs/verification/implementation-verification.json` 和 `live-retrieval.json`，不包含密钥。
