const pipelines = {
  bm25: { label: "BM25 top 30", ndcg: 0.305654, mrr: 0.512697, recall: 0.147309, note: "SQLite FTS5/BM25 retrieves the top 30 passages with no remote model call." },
  jev30: { label: "BM25 top 30 + Jev", ndcg: 0.353235, mrr: 0.585817, recall: 0.158667, note: "Jev reranks the same 30 lexical candidates, improving all three recorded metrics." },
  jev50: { label: "BM25 top 50 + Jev", ndcg: 0.362468, mrr: 0.593023, recall: 0.164474, note: "A wider BM25 candidate pool gives Jev more evidence to recover before the top 10 cutoff." },
  agentic: { label: "Agentic lexical top 50", ndcg: 0.380168, mrr: 0.597940, recall: 0.185464, note: "Generated lexical search terms expand the local FTS5 search without creating embeddings." },
  rrf: { label: "BM25 + embedding RRF", ndcg: 0.396712, mrr: 0.632089, recall: 0.193977, note: "Reciprocal-rank fusion combines BM25 top 50 with embedding top 50 before reranking." },
  agenticJev: { label: "Agentic lexical top 50 + Jev", ndcg: 0.430969, mrr: 0.644041, recall: 0.204138, note: "Agent-generated search terms widen local recall, then Jev reranks the top 50 passages." },
  hybrid: { label: "Hybrid top 50 + Jev", ndcg: 0.444327, mrr: 0.654583, recall: 0.214907, note: "BM25 and embedding rankings are fused with RRF before Jev reranks the evidence." },
};
const fields = {
  label: document.querySelector("#selected-label"), score: document.querySelector("#selected-score"),
  ndcg: document.querySelector("#ndcg-value"), mrr: document.querySelector("#mrr-value"), recall: document.querySelector("#recall-value"),
  ndcgBar: document.querySelector("#ndcg-bar"), mrrBar: document.querySelector("#mrr-bar"), recallBar: document.querySelector("#recall-bar"), note: document.querySelector("#pipeline-note"),
};
function renderPipeline(key) {
  const p = pipelines[key];
  fields.label.textContent = p.label; fields.score.textContent = p.ndcg.toFixed(6);
  fields.ndcg.textContent = p.ndcg.toFixed(6); fields.mrr.textContent = p.mrr.toFixed(6); fields.recall.textContent = p.recall.toFixed(6);
  fields.ndcgBar.style.width = `${p.ndcg * 100}%`; fields.mrrBar.style.width = `${p.mrr * 100}%`; fields.recallBar.style.width = `${p.recall * 100}%`; fields.note.textContent = p.note;
  document.querySelectorAll(".pipeline-option").forEach((button) => { const active = button.dataset.pipeline === key; button.classList.toggle("active", active); button.setAttribute("aria-selected", active ? "true" : "false"); });
}
document.querySelectorAll(".pipeline-option").forEach((button) => button.addEventListener("click", () => renderPipeline(button.dataset.pipeline)));
document.querySelector("#copy-command").addEventListener("click", async (event) => {
  try { await navigator.clipboard.writeText("git clone https://github.com/aifabrice/jev-rag.git"); event.currentTarget.textContent = "Copied"; window.setTimeout(() => { event.currentTarget.textContent = "Copy"; }, 1400); }
  catch { event.currentTarget.textContent = "Select text"; }
});
renderPipeline("hybrid");
