# DocMind — Project Notes (Interview Prep)

## What it does
Domain RAG over LangChain/LangGraph docs (304 chunks from 44 docs). Ingest → sentence-transformer embeddings → FAISS `IndexFlatIP` (cosine) → cross-encoder rerank → LangGraph agent (router → retrieval vs fallback keyword search) → Groq (or Ollama) generation → Redis cache → MLflow/W&B tracking.

**Why LangChain docs as corpus:** Chosen per Setup Checklist; covers RAG, agents, vectorstores, MCP — good for demonstrating multi-tool routing. 50-100 docs target met.

## Why each component

- **LangGraph (`src/agent.py:1`):** Stateful graph for router → retrieval/fallback → generate. Better than plain chain for multi-tool orchestration and future memory/persistence. `StateGraph` with `router_node`, `retrieval_node`, `fallback_node`, `generation_node`.
- **MCP (`src/mcp_server.py:1`, `src/tools.py:1`):** Standardizes retrieval tool over stdio via `mcp` SDK + `langchain-mcp-adapters`. Allows tool to be served/consumed as protocol, not ad-hoc function. Fallback tool kept as plain LangChain `tool` for comparison.
- **FAISS + cross-encoder (`src/embed_index.py:1`):** Two-stage retrieval — FAISS recall `k=15` then `ms-marco-MiniLM-L-6-v2` rerank to top 3-5. Gives "recent advancement" bullet vs plain vector search. FAISS `IndexFlatIP` with L2-normalized vectors = cosine.
- **Redis (`src/memory.py:1`):** `get_cached`/`set_cached` via `REDIS_URL` (Redis Cloud free 30MB). Hashes normalized query (`sha256` 16 hex) + TTL 86400. Hash before retrieval, cache full pipeline result.
- **Groq / Ollama (`src/generate.py:1`):** Swappable `generate_groq`/`generate_ollama` same signature. Groq `openai/gpt-oss-20b` (fast, free-tier), Ollama local fallback (auto GPU). Retry with backoff for 429.
- **MLflow + W&B (`src/benchmark.py:1`):** MLflow tracks latency, relevance (cosine), hit/miss per query; W&B logs similarity distribution histogram. Both run per `benchmark.py` batch.

## Final metrics (25 queries, 16 corpus-answerable, 9 fallback-only — hand-verified)

| Metric | Value | Notes |
|---|---|---|
| **Corpus-answerable** | **16 / 25** (`data/processed/eval_labels.json:1`) | 9 fallback-only `answerable:false` (cross-encoder, Redis cache, MLflow/W&B, fallback tool, router, latency, agentic memory, Redis Cloud, eval relevance) — hand-checked 0/304 chunks contain `cross-encoder` |
| **Precision@3 FAISS-only** | **6.2%** (1/16) | Hand-verified labels (first chunk from expected source); all-25 = 4.0% |
| **Precision@3 FAISS+rerank** | **6.2%** (1/16) | `k=15→3` cross-encoder; all-25 = 4.0% |
| **Lift** | **0.0pp** (heuristic earlier -11.8pp / -4.0pp) | Hand-verified replaces keyword heuristic; lift 0 shows rerank didn't help/hurt on this noisy corpus |
| **Relevance (cosine >0.35)** | **52.0%** avg 0.321 | Per `benchmark.py` |
| **Latency cold (0% hit)** | **clean 8258ms / wall 12878ms** (`data/processed/bench_cold.json:1`) | Clean = Groq API excl. retry; wall incl. TPD 200k limit retries. `n=25`, `cuda`, fallback 44% (11/25) |
| **Latency warm (100% hit)** | **1593ms** (`data/processed/bench_warm.json:1`) | Remote Redis RTT 300-2019ms dominates; ideal local <10ms. Cold 339s wall → warm 57s wall |
| **Cache hit rate** | **0% cold / 100% warm** |  |
| **Fallback routing (new)** | **44% (11/25)** | Threshold router `FAISS top-1 < 0.30 → fallback` gives 14/11 split vs intended 16/9 (tuned from 0.30 vs 0.25 gave 19/6) |
| **GPU** | **GTX 1650 4GB, CUDA 12.4, torch 2.6.0+cu124** | `torch.cuda.is_available() True` |

**Honest caveats:**
- **Warm latency still high:** Remote Redis `retroradiant-blissful-celery:19885` RTT 300-2019ms dominates warm hits (1593ms); local Docker Redis would be <10ms.
- **Cold wall inflated:** Groq TPD 200k limit hit after ~13 queries (429 retries); clean 8258ms is API only, wall 12878ms includes backoff. TPD limit, not code, is bottleneck.
- **Precision low (6.2%):** Corpus is noisy — many pages share `Agent = Model + Harness` header, `FAISS` term in 0/304 chunks (vectorstore pages 404), so even hand-verified labels (first chunk from expected source) are often not in top-3. Larger/cleaner corpus needed.
- **Rerank lift 0:** Hand-verified replaces heuristic -11.8pp; cross-encoder didn't help/hurt on this corpus+labels.
- **Ollama not benchmarked** — server not installed, `Latency (Ollama)` N/A.

## Key engineering decisions & trade-offs

- **Cosine lift was wrong → precision@3:** Original `rerank_lift_pct` compared cosine before/after rerank — FAISS already optimizes cosine, so rerank always looked worse (-8.5%). Replaced with `precision@3` against `eval_labels.json` (does correct chunk appear in top 3?). Still heuristic, but correct yardstick.
- **Sentence-transformers segfault workaround:** `sentence-transformers==5.5.1` segfaults on import (`exit -1073741819`) on Windows + torch 2.6. Replaced with `transformers.AutoModel` mean-pooling + `AutoModelForSequenceClassification` for cross-encoder, with manual `normalize` + `faiss.normalize_L2` (`src/embed_index.py:76,187`). Functionally equivalent, avoids ST.
- **Groq model swap:** `llama-3.1-8b-instant` returns 404 on this key (`Groq.models.list()` shows `openai/gpt-oss-20b`, `groq/compound`, `qwen/*`). Switched `configs/config.yaml:6` to `openai/gpt-oss-20b`.
- **Normalization fix:** Added explicit `faiss.normalize_L2` in `build_index()` and `search()` (`src/embed_index.py:78,187`) even though torch already normalized — sanity `norms mean=1.0000`. Verified `k=15`.
- **Perf: model caching:** `src/embed_index.py:11` caches `_EMB_TOK/_CE_TOK` module-level; `src/benchmark.py:56` loads embedding once. Cut per-query reload from ~3 loads to 1.
- **Latency measurement fix:** `src/generate.py:18` now tracks `LAST_CLEAN_LATENCY_MS` (API only excl. `sleep`), `src/benchmark.py:89` logs `clean` vs `wall` and cache-hit as Redis wall (not stale generation time).
- **Corpus filtering + hand-verified labels:** 9/25 fallback queries excluded; `eval_labels.json:1` now marks `answerable` and hand-verified by reading chunk text (e.g. `cross-encoder` moved to fallback after checking 0/304 chunks contain term, 16/25 remain). Previous keyword heuristic gave -11.8pp, hand-verified gives 0.0pp — still low due to noisy corpus.
- **Router threshold fix:** Replaced naive `keyword:`/`len<=2` heuristic (`src/agent.py:39`) with similarity-threshold `FAISS top-1 < 0.30 → fallback` (`src/agent.py:45`). Gives 14/11 split vs intended 16/9 (tuned: 0.30 → 14/11, 0.25 → 19/6). Stores `_router_hits` to avoid double retrieval.

## Likely interview Q&A

**Q: Why MCP over plain tool functions?**
- Standardizes tool serving (stdio/SSE) via `mcp` protocol, so retrieval can be swapped/hosted remotely without changing agent. `langchain-mcp-adapters` loads it as LangChain tool. Plain functions work for demo fallback, but MCP shows protocol awareness.

**Q: Why did reranking underperform initially?**
- First metric was cosine lift (-8.5%) — wrong yardstick (FAISS optimizes cosine). Replaced with `precision@3` on heuristic labels (-11.8pp). After hand-verifying 16/25 labels by reading chunk text, lift is **0.0pp (6.2% vs 6.2%)** — cross-encoder neither helped nor hurt on this noisy corpus (many pages share same header, `FAISS` in 0 chunks). True lift needs larger/cleaner corpus + manual labels.

**Q: Router was always 0% fallback — how fixed?**
- Naive heuristic (`keyword:` or `len<=2`, `src/agent.py:39`) sent `25/25` to retrieval vs intended `16/9`. Replaced with similarity-threshold `FAISS top-1 < 0.30 → fallback` (`src/agent.py:45`), now **14/11 split** (tuned: 0.30 → 14/11, 0.25 → 19/6) and caches `_router_hits` to avoid double retrieval.

**Q: How would you scale this?**
- Shard FAISS (IVF/PQ) for >10k chunks, use async embedding, add Redis cluster or local cache, batch Groq calls, add eval set with human labels, move MCP to SSE/HTTP for remote, add LangGraph persistence (`checkpointer`).

**Q: Trade-offs of Groq vs Ollama?**
- Groq: <300ms ideal, but 8000 TPM limit throttles benchmarks (429 retries). Ollama: local, no limit, but GTX 1650 4GB slow for 8B and not installed. Code swaps via `prefer` arg; benchmark honest about both.
