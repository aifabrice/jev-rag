const pipelines = {
  bm25: { label: "BM25 top 30", ndcg: 0.305654, mrr: 0.512697, recall: 0.147309, medianMs: 1.27, p95Ms: 6.08, latencyBasis: "10-query unified cold audit", note: "SQLite FTS5/BM25 retrieves the top 30 passages with no remote model call." },
  jev30: { label: "BM25 top 30 + Jev", ndcg: 0.353235, mrr: 0.585817, recall: 0.158667, medianMs: 1366.74, p95Ms: 4344.85, latencyBasis: "10-query unified cold audit", note: "Jev reranks the same 30 lexical candidates; remote Jev dominates the measured latency." },
  jev50: { label: "BM25 top 50 + Jev", ndcg: 0.362468, mrr: 0.593023, recall: 0.164474, medianMs: 1392.62, p95Ms: 2348.26, latencyBasis: "10-query unified cold audit", note: "A wider BM25 candidate pool gives Jev more evidence to recover before the top 10 cutoff." },
  lineSearch: { label: "Two-level Jev Line Search", ndcg: 0.366280, mrr: 0.657660, recall: 0.169397, medianMs: 11742.93, p95Ms: 35751.55, latencyBasis: "10-query cold pilot", note: "The best first-hit behavior, but weaker multi-document ranking than Agentic or Hybrid; 5,168 Jev requests cost $4.553288 in the full run." },
  passageGate: { label: "Hybrid + Unified Passage Gate", ndcg: 0.376298, mrr: 0.618043, recall: 0.166977, medianMs: 1555.05, p95Ms: 17469.24, latencyBasis: "323-query cold run", note: "A disclosed negative result: fixed thresholds excluded 84.4% of candidates. Useful to study injection and premise routing, but weaker than bare Hybrid and ordinary Jev reranking." },
  agentic: { label: "Agentic lexical top 50", ndcg: 0.380168, mrr: 0.597940, recall: 0.185464, medianMs: 3848.86, p95Ms: 5452.60, latencyBasis: "10-query unified cold audit", note: "Generated lexical search terms expand the local FTS5 search; remote Agentic planning dominates the measured latency." },
  rrf: { label: "BM25 + embedding RRF", ndcg: 0.396712, mrr: 0.632089, recall: 0.193977, medianMs: 2333.26, p95Ms: 2578.39, latencyBasis: "10-query unified cold audit", note: "Reciprocal-rank fusion combines BM25 top 50 with embedding top 50 before reranking; query embedding dominates latency." },
  agenticJev: { label: "Agentic lexical top 50 + Jev", ndcg: 0.430969, mrr: 0.644041, recall: 0.204138, medianMs: 6009.40, p95Ms: 8687.05, latencyBasis: "10-query unified cold audit", note: "Agent-generated search terms widen local recall, then Jev reranks the top 50 passages." },
  hybrid: { label: "Hybrid top 50 + Jev", ndcg: 0.444327, mrr: 0.654583, recall: 0.214907, medianMs: 3887.37, p95Ms: 5666.57, latencyBasis: "10-query unified cold audit", note: "BM25 and embedding rankings are fused with RRF before Jev reranks the evidence." },
  agenticHybrid: { label: "Agentic Hybrid + Jev/retrieval fusion", ndcg: 0.450750, mrr: 0.652606, recall: 0.220885, medianMs: 6140.84, p95Ms: 9351.55, latencyBasis: "10-query unified cold audit", note: "A dev-selected zero-call rank fusion improved the Jev-only order; Agentic planning and embedding are the dominant stages." },
  agenticHybridTop200: { label: "Agentic Hybrid high-recall top 200 + Jev", ndcg: 0.466171, mrr: 0.669516, recall: 0.228912, medianMs: 6521.02, p95Ms: 7897.30, latencyBasis: "10-query unified cold audit; caches disabled", note: "Benchmark-only high-recall operating point: Agentic and vector branches retrieve 200 candidates, then Jev scores batches of 100." },
};
const fields = {
  label: document.querySelector("#selected-label"), score: document.querySelector("#selected-score"),
  ndcg: document.querySelector("#ndcg-value"), mrr: document.querySelector("#mrr-value"), recall: document.querySelector("#recall-value"),
  median: document.querySelector("#latency-median"), p95: document.querySelector("#latency-p95"), latencyBasis: document.querySelector("#latency-basis"),
  ndcgBar: document.querySelector("#ndcg-bar"), mrrBar: document.querySelector("#mrr-bar"), recallBar: document.querySelector("#recall-bar"), note: document.querySelector("#pipeline-note"),
};
function formatLatency(milliseconds) {
  if (milliseconds < 1000) return `${milliseconds.toFixed(milliseconds < 10 ? 2 : 0)} ms`;
  return `${(milliseconds / 1000).toFixed(2)} s`;
}
function renderPipeline(key) {
  const p = pipelines[key];
  fields.label.textContent = p.label; fields.score.textContent = p.ndcg.toFixed(6);
  fields.ndcg.textContent = p.ndcg.toFixed(6); fields.mrr.textContent = p.mrr.toFixed(6); fields.recall.textContent = p.recall.toFixed(6);
  fields.median.textContent = formatLatency(p.medianMs); fields.p95.textContent = formatLatency(p.p95Ms); fields.latencyBasis.textContent = p.latencyBasis;
  fields.ndcgBar.style.width = `${p.ndcg * 100}%`; fields.mrrBar.style.width = `${p.mrr * 100}%`; fields.recallBar.style.width = `${p.recall * 100}%`; fields.note.textContent = p.note;
  document.querySelectorAll(".pipeline-option").forEach((button) => { const active = button.dataset.pipeline === key; button.classList.toggle("active", active); button.setAttribute("aria-selected", active ? "true" : "false"); });
}
document.querySelectorAll(".pipeline-option").forEach((button) => button.addEventListener("click", () => renderPipeline(button.dataset.pipeline)));
document.querySelector("#copy-command").addEventListener("click", async (event) => {
  try { await navigator.clipboard.writeText("git clone https://github.com/aifabrice/jev-rag.git"); event.currentTarget.textContent = "Copied"; window.setTimeout(() => { event.currentTarget.textContent = "Copy"; }, 1400); }
  catch { event.currentTarget.textContent = "Select text"; }
});
renderPipeline("agenticHybridTop200");
