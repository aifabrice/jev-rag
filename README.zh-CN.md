# Jev RAG

**基于 SQLite BM25、Jev 重排与流式引用回答的无向量本地知识库。**

[English](README.md) · [架构](docs/ARCHITECTURE.md) · [安全说明](SECURITY.md) · [参与贡献](CONTRIBUTING.md)

```text
本地文件 → SQLite FTS5 / BM25 → Jev 证据重排 → MiniMax 引用回答
```

Jev RAG 不需要 Embedding、向量数据库或 GPU。它扫描本地文件夹，用 SQLite FTS5/BM25 召回候选文段，再让 Jev 判断哪些文段真正有助于回答问题，最后可通过 OpenRouter 调用回答模型生成带引用的流式答案。

> 当前状态：Alpha。适合本地试用和二次开发，但 1.0 之前接口与数据库结构可能调整。

## 核心特点

- 无向量、无 Embedding、无外部索引服务。
- 支持 Markdown、文本、HTML、JSON、CSV、YAML、DOCX 和 PDF。
- 中文二字切词与 SQLite FTS5/BM25 检索。
- BM25 召回后使用 Jev 进行证据相关度重排。
- Jev 候选自动分批，避免长请求超过上下文限制。
- 可设置相关度阈值；没有证据时明确拒答。
- MiniMax 流式回答，并显示来源编号。
- 页面展示 BM25、Jev、首 Token、生成和总耗时。
- SQLite 保存索引、Jev 缓存和问答运行记录。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[documents]'
cp .env.example .env
```

在 `.env` 中填写：

```dotenv
OPENROUTER_API_KEY=
TYPESAFE_API_KEY=
```

默认端到端流程只需要 `OPENROUTER_API_KEY`。`.env` 已被 Git 忽略；如果密钥曾被公开，必须立即撤销并重新创建。

## 快速开始

把文档放入 `knowledge/`，然后执行：

```bash
jev-rag index
jev-rag search '退款需要多久？'
jev-rag serve
```

浏览器打开 <http://127.0.0.1:8765>。

不安装命令行入口也可以直接运行：

```bash
python3 local_kb.py index
python3 local_kb.py search '退款需要多久？'
python3 local_kb.py serve
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

- BM25 最多召回 30 个文段。
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
jev-rag search '问题' --lexical-only

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

## 项目说明

GitHub 上存在其他相似名称的项目。本项目以 SQLite FTS5/BM25、无向量本地目录索引和 MiniMax 流式回答为主要区别。本项目是独立社区项目，与 TypeSafe AI、OpenRouter 和 MiniMax 均无隶属或官方背书关系。

## 许可证

[MIT](LICENSE)
