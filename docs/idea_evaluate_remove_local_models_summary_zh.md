# idea-evaluate 工作流"去本地模型化"改造总结

日期：2026-10-11
状态：**全部完成。** 代码改造、离线验证、全部 7 个 CLI 工作流的真实后端端到端验收均已通过。

## 1. 任务目标

将 SciAtlas idea-evaluate（自动化评审）工作流中的本地模型依赖全部移除：

- 删除 sentence-transformers（BGE embedding / reranker）与 FAISS 等本地模型依赖，不再下载任何模型权重；
- 检索改为调用托管 KG 检索服务（`http://sciatlas.openkg.cn`）；
- grounding 段落的重排改为"纯 Python TF-IDF 词法检索 + LLM 重排"，不引入任何本地向量模型。

## 2. 主要改动

### 2.1 核心：grounding 检索重写

`sciatlas/evidence/grounding.py`（净减约 500 行，改动最大的文件）：

- 移除 dense embedding 检索（sentence-transformers + numpy 向量缓存 `.npy`），替换为纯 Python 实现的 TF-IDF `lexical_search`；
- 新增 `LlmParagraphReranker`：通过 `GroundingLlmClient` 调用 LLM 对候选段落分批打分，调用失败时自动降级为词法排序结果；
- corpus 缓存不再保存向量文件；
- 删除 `--embedding-model` / `--reranker-model` / `--query-prefix` / `--device` 等 CLI 参数。

### 2.2 接线清理

- `review_pipeline/pipeline.py`：删除 `--grounding-device` 参数及 CUDA 设备透传；把 `from review.author_background import ...` 移入 try 块内，neo4j 缺失时降级为 LLM 兜底而不再崩溃。
- `review_pipeline/grounding/grounding_pipeline.py`、`sciatlas/tasks/_shared.py`、`sciatlas/core/schemas.py`：同步移除设备/模型相关参数与字段。

### 2.3 LLM 配置解析统一修复（7 个文件）

原代码多处硬编码 DMX 中转站的 URL（`https://www.dmxapi.cn/v1`）与模型（`DeepSeek-V3.2`），且配置默认值非空导致 `.env` 里的 `LLM_BASE_URL` / `LLM_MODEL` 永远不生效。统一修复为同一语义：**配置值等于硬编码默认即视为未设置，改从环境变量解析；显式传参仍可覆盖**。解析链：

- API key：`LLM_API_KEY` → `DMX-API-KEY` → `DMX_API_KEY` → `OPENAI_API_KEY`
- URL：`SEARCH_LLM_API_URL` → `LLM_CHAT_COMPLETIONS_URL` → `LLM_API_URL` → `OPENAI_CHAT_COMPLETIONS_URL`，缺省再由 `LLM_BASE_URL` / `OPENAI_BASE_URL` 拼 `/chat/completions`
- 模型：`SEARCH_LLM_MODEL` → `LLM_MODEL` → `OPENAI_MODEL`

涉及文件：

1. `review_pipeline/s2api/kw_extractor.py`（关键词提取）
2. `review_pipeline/search/merge_search.py`（search 阶段 LLM 重排打分器）
3. `review_pipeline/review/evaluation.py`（review 阶段）
4. `review_pipeline/review/report.py`（report 阶段）
5. `review_pipeline/review/author_background.py`（reviewers 阶段）
6. `review_pipeline/review/reviewer_background_fallback.py`（reviewers 兜底）
7. `literature_review_pipeline/downstream/lr_search/literature_review_search.py`（`DmxJsonClient`，一处修复同时覆盖文献综述的检索与报告两个子命令）

另将修复后的 `kw_extractor.py` 同步覆盖到 `literature_review_pipeline/s2api/kw_extractor.py`（原为修复前旧版，接口完全兼容）。

### 2.4 删除的文件（本地模型资产相关，已不再需要）

- `scripts/download_rubric_assets.py`
- `scripts/build_rubric_asset_pack.py`
- `scripts/check_rubric_setup.py`
- `requirements-rubric.txt`

仓库不再存在 `models/` 目录，符合"零本地模型"目标。

### 2.5 文档同步更新

- `README.md`、`README_zh.md`
- `agent-skill/sciatlas-idea-evaluate/SKILL.md`
- `review_pipeline/README.md`
- `.env.example`

## 3. 验证结果

### 3.1 离线验证（全部通过）

- grounding 词法检索与重排：14 个单元测试通过（`sciatlas/tests/test_grounding_lexical.py`，新增）；
- `review_pipeline/review`：11 个测试通过；
- `review_pipeline/s2api`：3 个测试通过；
- 所有改动文件 `py_compile` 通过；
- 离线 smoke 管线（`--smoke`）9 个阶段全部 `ok`；
- `run_rubric.py --help` 与 sciatlas CLI 导入正常。

测试统一使用 `unittest`（`/Users/interstellar888/anaconda3/envs/innoeval/bin/python`，该环境无 pytest）。

### 3.2 真实后端端到端验收（2026-10-11，托管 KG 恢复后）

| CLI 功能 | 入口 | 结果 |
|---|---|---|
| quick-paper-search | `run_sciatlas.py search-papers`（precision 档案） | ✅ |
| researcher-review | `search-papers`（impact + authorship 档案） | ✅ |
| trend-report | `search-papers`（impact + 时间窗档案） | ✅ |
| idea-grounding | `search-papers`（precision + 相关度档案） | ✅ |
| literature-review | `run_sciatlas.py literature-review --workflow flash` | ✅ 25 篇论文，综述大纲 + report.md |
| idea-generate | `python -m sciatlas_idea_gen.main --workflow flash` | ✅ idea seeds + 新颖性分析 + 参考文献 |
| rubric 独立管线 | `run_rubric.py` | ✅ 72s 生成 idea_rubric.json |
| idea-evaluate | `run_sciatlas.py idea-evaluate --workflow flash` | ✅ 端到端 302s，产出 meta-review 报告 |

idea-evaluate 各 stage 实测：

```
idea_context    ok           2.7s
search          partial_error 20.2s   # 仅 S2 分支缺 API key（设计内容忍，KG 分支正常，6 篇全部打分）
manifest        ok          175.9s    # 6 篇中 5 篇 PDF 下载 + GROBID 解析成功；1 篇 RSC 出版商反爬 403
grounding       ok           23.3s
reviewers       partial_error  0.0s   # 仅 neo4j 未安装（设计内 LLM 兜底，实际选出 1 位审稿人并生成画像）
rubric_sources  ok
rubric_llm      ok
review          ok            8.3s
report          ok           17.6s
```

两个 `partial_error` 都是可选外部依赖缺失导致的设计内降级（S2 API key、neo4j)，核心链路全部 ok，最终报告正常产出。如需全绿：在 `.env` 补 `S2_API_KEY`，并 `pip install neo4j`。

### 3.3 GROBID 本地化部署（manifest 阶段依赖）

完整 idea-evaluate 的 manifest 阶段需要 GROBID 解析论文 PDF。本机无 Docker，采用源码构建：

- 源码：GROBID 0.9.1（GitHub tag zip），位于 `~/sciatlas-runtime/grobid`（仓库外）；
- 构建：`JAVA_HOME=/opt/homebrew/opt/openjdk@21 ./gradlew installDist`（自带的 `mac_arm-64` 版 libwapiti.dylib 与 pdfalto，全 arm64 原生，无需 Rosetta；构建时需给 `installDist` 任务补 `duplicatesStrategy = EXCLUDE` 以兼容 Gradle 9.6）；
- 启动：`grobid-service/build/install/grobid-service/bin/grobid-service server grobid-home/config/grobid.yaml`（工作目录须为 `~/sciatlas-runtime/grobid`)，默认监听 `http://127.0.0.1:8070`，与管线默认值一致，无需额外配置；
- 注意：GROBID 服务是常驻进程，重启电脑后需重新启动。

## 4. 环境备注

- Python 固定使用 `/Users/interstellar888/anaconda3/envs/innoeval/bin/python`；
- 凭证位于仓库根目录 `.env`（gitignored，权限 600）：`LLM_API_KEY` / `LLM_BASE_URL=https://api.deepseek.com/v1` / `LLM_MODEL=deepseek-chat` / `SCIATLAS_API_KEY` / `SCIATLAS_API_BASE_URL=http://sciatlas.openkg.cn` / `SCIATLAS_TIMEOUT=900`；
- 管线只从 `.env` 读取 LLM 配置，不依赖 shell 环境变量；
- 所有代码改动尚未 commit，待确认后提交。
