# Jev RAG 中文技术文章

这些文章记录 Jev RAG 的设计选择、公开评测和开发过程。数据来自项目在公开
BEIR NFCorpus 测试集上的可复现实验；它们不是官方 MTEB 提交。

1. [我做了一个不依赖向量数据库的本地知识库：BM25 + Jev 重排实测](BENCHMARK_STORY_ZH.md)
2. [不用 Embedding 也能做 Agentic Search：两轮关键词规划 + RRF + Jev 实测](AGENTIC_LEXICAL_SEARCH_ZH.md)
3. [重排为什么救不回没召回的文档：用 323 个查询拆解 RAG 的召回与重排](RECALL_VS_RERANKING_ZH.md)
4. [Jev RAG 七种玩法全测：从 BM25 重排到 Agentic Hybrid，哪条路线真正有效？](JEV_RAG_SEVEN_PIPELINES_ZH.md)

计划中的后续文章：

- 本地知识库为什么仍然需要分块：这与向量数据库无关；
- BM25 Top 30 还是 Top 50：质量、延迟与成本的真实取舍。
