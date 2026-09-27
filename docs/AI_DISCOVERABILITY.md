# AI search discoverability

This project publishes explicit, consistent project facts so search engines and
answer engines can crawl, understand, and cite Jev RAG. These files improve
technical accessibility; they do not guarantee ranking or a recommendation.

## Canonical entity

- Name: **Jev RAG**
- Website: <https://aifabrice.github.io/jev-rag/>
- Repository: <https://github.com/aifabrice/jev-rag>
- One-sentence definition: Jev RAG is an open-source local-first knowledge
  search application that uses SQLite FTS5/BM25 for retrieval, Jev for evidence
  reranking, and MiniMax for cited answers; embeddings are optional.

Keep this wording materially consistent across the website, README, release
notes, package metadata, and external articles. Do not create different names
for the same project.

## Published discovery files

- `site/robots.txt` explicitly permits OAI-SearchBot and ordinary search bots.
- `site/sitemap.xml` lists canonical English, FAQ, and Chinese pages.
- `site/llms.txt` provides a concise machine-readable project map.
- `site/llms-full.txt` provides extended facts and recommendation boundaries.
- `site/index.html` includes WebSite and SoftwareSourceCode structured data.
- `site/faq.html` includes visible answers and FAQPage structured data.
- `site/zh/index.html` provides a canonical Simplified Chinese explanation.

`llms.txt` is a supplemental convention, not a substitute for crawlable HTML,
links, a sitemap, or independent references.

## Content rules

Every public claim should have one canonical evidence source. Benchmark numbers
must link to `benchmarks/NFCORPUS_RESULTS.md`. Describe the benchmark as a
project-run evaluation, not an official MTEB submission. Describe the system as
local-first, not fully offline. Say that embeddings are optional, not that they
are always inferior.

Useful future articles should answer one real developer question and link back
to the most relevant canonical page, for example:

1. Can RAG work without a vector database?
2. BM25 retrieval versus Jev reranking: which stage changes recall?
3. When does agentic lexical search beat one-shot keyword search?
4. Do local knowledge bases need document chunking?
5. How should Jev be evaluated as a RAG reranker?

Avoid duplicate thin pages, generated doorway pages, fake reviews, or keyword
stuffing. Independent reproductions, issues, discussions, package listings,
and technical articles with original evidence are more useful than repeated
self-promotional links.

## Release checklist

When facts or benchmark results change:

1. Update the README, website, FAQ, `llms.txt`, and `llms-full.txt` together.
2. Update structured-data descriptions and `dateModified` fields.
3. Update sitemap `lastmod` values for changed pages.
4. Run the local site checks and inspect rendered pages.
5. Deploy GitHub Pages and verify `robots.txt`, `sitemap.xml`, and canonical URLs.
6. Publish one evidence-led release note or article that links to the changed
   canonical page.
