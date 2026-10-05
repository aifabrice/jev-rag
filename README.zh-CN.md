# Jev RAG

**Jev RAG 是一个开源的本地知识库检索工具，提供七种可切换路径：默认 BM25 + Jev、无向量 Agentic Search、Embedding 混合检索、多轮 Agentic Hybrid、知识分类树路由、统一 Jev Passage Gate，以及两级 Jev Line Search。**

它可以搜索本地文件夹，用 Jev 对候选文段重排，再由回答模型输出带文件引用的答案。默认路径不需要 Embedding 或向量数据库。准确地说，Jev RAG 是“本地优先”而不是“完全离线”：选中的候选文段会发送给配置的 Jev 与回答模型服务。

[![CI](https://github.com/aifabrice/jev-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/aifabrice/jev-rag/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/aifabrice/jev-rag?include_prereleases)](https://github.com/aifabrice/jev-rag/releases)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776ab)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-17624f)](LICENSE)

[English](README.md) · [中文技术文章](docs/CONTENT_SERIES_ZH.md) · [架构](docs/ARCHITECTURE.md) · [安全说明](SECURITY.md) · [参与贡献](CONTRIBUTING.md)

**[中文项目介绍 →](https://aifabrice.github.io/jev-rag/zh/)**
· [Jev 怎么做 RAG：完整搭建教程](https://aifabrice.github.io/jev-rag/zh/jev-rag-guide.html)
· [互动式公开 Benchmark](https://aifabrice.github.io/jev-rag/)
· [常见问题](https://aifabrice.github.io/jev-rag/faq.html)
· [机器可读项目说明](https://aifabrice.github.io/jev-rag/llms.txt)

## 公开测评结果

BEIR NFCorpus 完整测试集：3,633 个文档、323 个查询。所有结果使用相同语料、
查询、相关性标注与指标实现；候选数量和远程阶段均在方案名称中明确标注。

| 方案 | nDCG@10 | MRR@10 | Recall@10 | 中位延迟 | P95 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 Top 30 | 0.305654 | 0.512697 | 0.147309 | 1.92 ms | 6.60 ms |
| BM25 Top 30 + Jev | 0.353235 | 0.585817 | 0.158667 | 1.08 秒 | 16.95 秒 |
| BM25 Top 50 + Jev | 0.362468 | 0.593023 | 0.164474 | 1.92 秒 | 17.84 秒 |
| 两级 Jev Line Search（60 个优胜候选） | 0.366280 | **0.657660** | 0.169397 | 11.74 秒 | 35.75 秒 |
| Hybrid Top 50 + 统一 Passage Gate | 0.376298 | 0.618043 | 0.166977 | 1.56 秒 | 17.47 秒 |
| Agentic 词法检索 Top 50 | 0.380168 | 0.597940 | 0.185464 | 6.99 秒 | 24.39 秒 |
| BM25 Top 50 + Embedding Top 50 + RRF | 0.396712 | 0.632089 | 0.193977 | 1.38 秒 | 1.49 秒 |
| 多轮 Agentic Hybrid Top 50 | 0.424145 | 0.637722 | 0.206303 | 6.99 秒 | 24.39 秒 |
| **Agentic 词法检索 Top 50 + Jev** | **0.430969** | **0.644041** | **0.204138** | 7.52 秒 | 26.35 秒 |
| 混合召回 Top 50 + Jev | 0.444327 | **0.654583** | 0.214907 | 3.17 秒 | 17.99 秒 |
| **Agentic Hybrid + Jev/原召回排名融合** | **0.450750** | **0.652606** | **0.220885** | 9.24 秒 | 38.72 秒 |

延迟只统计单次 query 的检索和重排，不包含一次性全库建索引与最终回答生成。
不同方案分别采用完整运行、冷启动小样本或逐 query 阶段合成估算；每行的口径见
[`nfcorpus-pipeline-latency.json`](benchmarks/nfcorpus-pipeline-latency.json)。
远程供应商的实际延迟可能随时间和负载明显变化。

Line Search 的首条命中表现最好（`nDCG@1=0.554180`），但多文档排序质量与召回率
都低于 Agentic 和 Hybrid。全语料冷跑的供应商费用为 `$4.553288`，因此它属于
实验性深度检索路径，不作为默认方案。

统一 Passage Gate 是一个如实披露的负结果：按 Cookbook 启发的固定阈值，
它排除了 16,150 个候选中的 13,631 个（84.4%），nDCG@10 低于裸
Hybrid（0.396712）和 Hybrid + Jev（0.444327）。它仍作为提示注入筛查与
问题前提检查的实验模式，但不是默认精度路径。

分类树和未做最终融合的 Agentic Hybrid 仍保留在完整报告中，但因为都处在
`0.44-0.45` 的相近区间，不再放入首页主表。主表只保留代表性的普通
Hybrid 和当前最高分方案。

最高分方案在 Jev 之后使用开发集选定的本地 RRF（Jev 排名权重 `1.0`，
原召回排名权重 `0.25`），不增加模型调用，测试集 nDCG@10 为 `0.450750`。
它的冷跑检索中位延迟为 `6.99 秒`，P95 为 `24.39 秒`。

以 2026-09-26 查看到的 [MTEB NFCorpus 页面](https://mteb-leaderboard.hf.space/tasks/NFCorpus)数值进行插入比较，
`0.450750` 约为 **251 个结果中第 4（Top 1.6%）**。这是一个**非官方的数值比较**，
不是 MTEB 官方榜单排名：这套多阶段方案尚未提交 MTEB，而且 Top 50 参数是在同一测试集上观察的。

[完整结果、精确配置、费用、局限和复现命令](benchmarks/NFCORPUS_RESULTS.md)
· [Agentic 机器可读结果](benchmarks/nfcorpus-agentic-summary.json)
· [Line Search 机器可读结果](benchmarks/nfcorpus-line-search-summary.json)
· [Passage Gate 机器可读结果](benchmarks/nfcorpus-passage-gate-summary.json)
· [知识分类树机器可读结果](benchmarks/nfcorpus-taxonomy-summary.json)
· [Agentic Hybrid 机器可读结果](benchmarks/nfcorpus-agentic-hybrid-summary.json)

![Jev RAG 本地网页界面](docs/assets/demo-ui.png)

```text
默认：本地文件 → SQLite BM25 ──────────────────→ Jev → MiniMax
Agentic：本地文件 → MiniMax 规划 → 多路 BM25/RRF → Jev → MiniMax
混合：本地文件 → BM25 + OpenRouter Embedding/RRF → Jev → MiniMax
Agentic Hybrid：两轮规划 → 多路 BM25 + Embedding/RRF → Jev + 原召回先验 → MiniMax
分类树：本地文件 → 语料分类树 → Hybrid Top 50 + 路由补充 → Jev → MiniMax
Gate：本地文件 → BM25 + Embedding/RRF → 统一 Jev Gate → MiniMax
Line：本地文件 → 并行 Jev Choice 窗口 → 全局 Choice → MiniMax
```

Jev RAG 默认只用 SQLite FTS5/BM25，不需要 Embedding、向量数据库或 GPU。
需要更强召回时，可在网页或 CLI 切换到 Agentic 模式，让 MiniMax 规划两轮
本地关键词搜索，完全不建立向量索引；也可以切换到 BM25 + Embedding + RRF。
多轮 Agentic Hybrid 则保留两轮规划，让多路 BM25 与原问题的
Embedding 查询并行，再按 `0.65:1.0` 加权 RRF 融合后交给 Jev。
然后以 `1.0:0.25` 融合 Jev 排名和原召回排名；这一步完全本地执行，
不增加模型调用。
前三种模式复用同一个 Jev 证据重排；Passage Gate 以一轮四项判断
取代普通重排；Line Search 则让所有索引文段进入
Jev 窗口 Choice，再对每个窗口的优胜文段执行第二级全局 Choice。
分类树模式只用语料 embedding 构建确定性两层树，查询时路由到
4 个叶子节点，在不改变原 Hybrid Top 50 的前提下最多补充 20 条候选。
构建分类树不使用测试问题或标准答案。

> 当前状态：Alpha。适合本地试用和二次开发，但 1.0 之前接口与数据库结构可能调整。

## 核心特点

- 默认 BM25 + Jev，无向量、无 Embedding、无外部索引服务。
- 可选两轮 Agentic Search + Jev，无需向量索引即可改善同义词召回。
- 可选 BM25 + Embedding 倒数排名融合（RRF），再进入 Jev。
- 可选多轮 Agentic BM25 + 原问题 Embedding 并行融合后进入 Jev。
  开发集选定的 Jev/原召回排名融合不增加模型调用，但规划延迟仍较高。
- 可选两层知识分类树路由，为 Hybrid 扩展候选；公开测试提高了
  候选召回，但没有提高 nDCG@10。
- 可选 Hybrid + 统一 Jev Passage Gate，同时判断相关性、可用证据、
  事实前提矛盾和提示注入；固定阈值在 NFCorpus 上未提升 nDCG@10。
- 可选两级 Jev Line Search：每个窗口最多 255 段，结构容量
  `255 × 255 = 65,025` 段，不使用 BM25 或 Embedding；实际费用与耗时会随语料规模增长。
- 不需要向量数据库或 GPU，混合模式向量缓存于 `.knowledge/`。
- 支持 Markdown、文本、HTML、JSON、CSV、YAML、DOCX 和 PDF。
- 中文二字切词与 SQLite FTS5/BM25 检索。
- BM25 召回后使用 Jev 进行证据相关度重排。
- Jev 候选自动分批，避免长请求超过上下文限制。
- 可设置相关度阈值；没有证据时明确拒答。
- MiniMax 流式回答，并显示来源编号。
- 页面展示 BM25、Embedding/RRF、Jev、首 Token、生成和总耗时。
- SQLite 保存索引、Jev 缓存和问答运行记录。

## 与向量 RAG 的区别

| | Jev RAG | 常规向量 RAG |
|---|---|---|
| 第一阶段检索 | SQLite FTS5/BM25 | Embedding 相似度 |
| 额外基础设施 | SQLite 以外无 | Embedding 模型与向量库 |
| 擅长的查询 | 准确术语、ID、名称和领域语言 | 语义相似与改写 |
| 第二阶段 | Jev 证据重排 | 可选重排器 |
| 主要取舍 | 同义词可能造成词汇失配 | Embedding 成本、建索引和运维 |

这是一种有意识的检索架构选择，不是宣称关键词检索永远优于向量检索。请用自己的文档和问题进行测试。

## 无需 API Key，先试本地检索

先使用仓库自带的公开文档验证 BM25 检索，只需要带 SQLite FTS5 的
Python 3.9+，无需安装依赖、配置 Key、生成向量或请求模型。
下面仅索引 `./docs`，不会扫描你的个人 Documents 目录。

```bash
git clone https://github.com/aifabrice/jev-rag.git
cd jev-rag
python3 local_kb.py --documents ./docs --db .knowledge/quickstart.db index
python3 local_kb.py --documents ./docs --db .knowledge/quickstart.db search 'retrieval' --no-jev
```

结果包含排序后的文段、来源路径和检索耗时。这只是 **BM25 基线**，
不包含 Jev 重排或大模型回答。两条命令请保留相同的 `--documents` 和
`--db`，该演示数据库与默认知识库分开。完整问答流程请继续按下面配置。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[documents]'
# 如果需要混合检索：
python -m pip install -e '.[documents,embeddings]'
cp .env.example .env
```

在 `.env` 中填写：

```dotenv
OPENROUTER_API_KEY=
TYPESAFE_API_KEY=
```

默认端到端流程只需要 `OPENROUTER_API_KEY`。`.env` 已被 Git 忽略；如果密钥曾被公开，必须立即撤销并重新创建。

## 快速开始

默认会自动发现并索引当前用户的 `~/Documents`（macOS 的“文稿”、
Windows/Linux 的 Documents）。首次可以直接执行：

```bash
jev-rag serve
```

浏览器打开 <http://127.0.0.1:8765>。
页面可切换 **BM25 + Jev（默认）**、**Agentic Search + Jev**、
**BM25 + Embedding + Jev**、**多轮 Agentic + BM25 + Embedding + Jev**
和其他实验性路由/检索模式。
启动时会自动更新本地 BM25 索引；默认模式不生成 Embedding，
也不需要向量数据库。

不安装命令行入口也可以直接运行：

```bash
python3 local_kb.py index
python3 local_kb.py serve
```

也可直接使用命令行搜索：

```bash
jev-rag search '我的文档里提到了哪些 AI 产品？'
```

如果要固定另一个默认目录，可在 `.env` 中设置：

```dotenv
JEV_RAG_DOCUMENTS=/你的/文档目录
```

## 搜索其他目录

```bash
jev-rag \
  --documents /你的/文档目录 \
  --db .knowledge/documents.db \
  --exclude 'private/**' \
  index --rebuild

jev-rag \
  --documents /你的/文档目录 \
  --db .knowledge/documents.db \
  --exclude 'private/**' \
  serve
```

当文档目录包含本项目时，应使用 `--exclude` 排除项目目录。

## 当前默认流程

- 文档目录自动发现为 `~/Documents`；不存在时回退到项目的 `knowledge/`。
- `serve` 启动时自动扫描新增、更新和删除的文档。
- 默认检索模式是 `bm25`。
- BM25 最多召回 30 个文段。
- 混合模式使用 BM25 Top 50 + Embedding Top 50，通过 RRF 保留 50 个候选（`rrf_k=60`）。
- Agentic Hybrid 使用两轮规划、每轮最多 5 组词法查询；
  多路 BM25 与原问题 Embedding 查询并行，通过 `0.65:1.0`
  加权 RRF 保留 50 个候选交给 Jev，再通过零调用的
  Jev/原召回排名融合（`1.0:0.25`）生成最终顺序。
- 分类树模式保留原 Hybrid Top 50，通过本地缓存的两层语料树路由，
  再最多补充 20 条节点内候选交给 Jev。
- Agentic 模式使用两轮规划，每轮最多 5 组检索词，每组 BM25 Top 100，通过 RRF 保留 50 个候选。
- Line Search 模式每个窗口最多 255 个文段，每个窗口默认保留 4 个优胜文段，
  最多 255 个窗口；第二级 Choice 在所有优胜文段中完成全局排序。
- 该路径基于 TypeSafe 官方
  [Semantic Find Cookbook](https://docs.typesafe.ai/cookbooks/semantic_find)，
  额外增加并行窗口 fan-out 和全局 reduce 层。
- Agentic 规划模型默认为 OpenRouter 上的 `minimax/minimax-m3`，规划结果缓存在本地。
- 默认 Embedding 模型是 OpenRouter 上的 `openai/text-embedding-3-large`。
- Jev 每批处理 10 个候选，多批并行执行。
- 最多向回答模型提供 10 个证据文段。
- 默认回答模型为 OpenRouter 上的 `minimax/minimax-m3`。
- 默认阈值为 `0.0`，不会根据分数删除文段。

如需过滤弱相关证据：

```bash
jev-rag serve --threshold 0.20
```

阈值只是应用策略，不代表正确性保证，应当用自己的问题集进行校准。

## 分块策略

```bash
# 默认：短文件整篇保留，长文件按标题和段落分块。
jev-rag index --chunking auto --rebuild

# 完全不分块，一个文件一条记录。
jev-rag index --chunking none --rebuild

# 总是使用标题感知的段落分块。
jev-rag index --chunking paragraph --rebuild
```

大型文件建议分块。`none` 模式可以使用，但引用只能定位到文件，检索与回答精度通常较低。

## 常用命令

```bash
# 仅运行 BM25，不调用 Jev。
jev-rag search '问题' --no-jev

# 混合召回后使用 Jev 重排。
jev-rag search '问题' --retrieval-mode hybrid

# 知识分类树路由 + Hybrid 候选扩展（实验性）。
jev-rag search '问题' --retrieval-mode taxonomy

# 混合召回后使用一轮统一 Jev Passage Gate（实验性）。
jev-rag search '问题' --retrieval-mode hybrid-gate

# 两轮 Agentic 本地词法检索后使用 Jev，不建立向量索引。
jev-rag search '问题' --retrieval-mode agentic

# 两轮 Agentic BM25 与原问题 Embedding 并行，再使用 Jev。
jev-rag search '问题' --retrieval-mode agentic-hybrid

# 全量文段进入并行 Jev 窗口，再对窗口优胜段做第二级全局 Choice。
jev-rag search '问题' --retrieval-mode line-search

# 调整窗口大小和每个窗口进入第二级的文段数。
jev-rag search '问题' --retrieval-mode line-search \
  --line-search-window-size 255 --line-search-beam 4

# 启动后页面默认选中混合模式。
jev-rag serve --retrieval-mode hybrid

# 过滤低于阈值的 Jev 结果。
jev-rag search '问题' --threshold 0.20

# 输出 JSON。
jev-rag search '问题' --json

# 查看索引状态。
jev-rag status

# 单独测试 Jev 通道。
jev-rag-smoke-test --provider openrouter
jev-rag-smoke-test --provider typesafe
jev-rag-smoke-test --dry-run
```

## 隐私与安全

- BM25 建库和召回完全在本地执行。
- 默认只索引支持的文本文档，不上传整个文件夹。
- 指向所选知识目录之外的文件软链接会被忽略。
- 混合模式首次建索引会向 OpenRouter 发送文段，每次查询会发送查询文本；向量缓存在本地。
- Agentic 模式会把问题和最多 8 条首轮命中片段发送给 OpenRouter 规划模型；检索规划缓存在本地。
- Line Search 会把每个索引文段的受限长度表示分批发送给 Jev，并再次发送
  窗口优胜文段做全局 Choice；处理私人文件前，应先用 `--documents` 与 `--exclude` 缩小范围。
- Jev 会收到问题和候选文段内容。
- OpenRouter 会收到问题和最终证据，用于生成答案。
- 索引、缓存和问答记录默认保存在 `.knowledge/`。
- 本地网页没有身份验证，不要直接暴露到公网。
- 文档内容属于不可信输入，当前提示词并不能构成完整的提示注入安全边界。

处理敏感文档前请阅读 [SECURITY.md](SECURITY.md)。

## 开发与验证

```bash
python -m pip install -e '.[dev,documents]'
python scripts/check_release.py
python -m unittest discover -s tests -v
python -m compileall -q local_kb.py jev_test.py tests
python -m build
python -m twine check dist/*
```

普通测试不会调用收费 API。

## 可复现评测

使用项目自带的公开示例文档运行 BM25 冒烟评测，不会调用收费 API：

```bash
python scripts/benchmark.py
```

如需对同一批问题比较 Jev 重排，需要显式启用服务商调用：

```bash
python scripts/benchmark.py --use-jev --provider openrouter
```

公开测评复现方法、评测文件格式和私有文档集测试方法请参考
[NFCorpus 完整结果](benchmarks/NFCORPUS_RESULTS.md) 和 [Evaluation](docs/EVALUATION.md)。

## 社区与路线图

- 在 [Discussions](https://github.com/aifabrice/jev-rag/discussions) 交流使用场景、问题和设计想法。
- 在 [Issues](https://github.com/aifabrice/jev-rag/issues) 提交可复现的缺陷和边界清晰的功能需求。
- 适合首次贡献的方向包括 OCR 适配、更多文档加载器、评测数据集、模型服务商适配和打包改进。
- 路线图在 [Issue 列表](https://github.com/aifabrice/jev-rag/issues) 中跟踪。

## 项目说明

GitHub 上存在其他相似名称的项目。本项目以 SQLite FTS5/BM25 默认检索、可选混合召回、Jev 重排和 MiniMax 流式回答为主要区别。本项目是独立社区项目，与 TypeSafe AI、OpenRouter 和 MiniMax 均无隶属或官方背书关系。

## 许可证

[MIT](LICENSE)
