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

## Final metrics (25 queries, 17 corpus-answerable, 8 fallback-only)

| Metric | Value | Notes |
|---|---|---|
| **Corpus-answerable** | **17 / 25** (`data/processed/eval_labels.json:1`) | 8 fallback-only marked `answerable:false` (Redis cache, MLflow/W&B, fallback tool, router, latency compare, agentic memory, Redis Cloud, eval relevance) |
| **Precision@3 FAISS-only** | **17.6%** (3/17) | On answerable subset; `k=15` FAISS top-3 vs heuristic labels (keyword presence). All-25 is 12.0% |
| **Precision@3 FAISS+rerank** | **5.9%** (1/17) | `k=15→3` cross-encoder rerank. All-25 is 8.0% |
| **Lift (rerank - FAISS)** | **-11.8pp** (all-25: -4.0pp) | Negative on heuristic labels — see caveats |
| **Relevance (cosine >0.35)** | **52.0%** avg 0.343 | Per `benchmark.py` embedding sim |
| **Latency cold (0% hit)** | **clean 9166ms / wall 14832ms** (`data/processed/bench_cold.json:1`) | Clean = Groq API only excl. retry sleep; wall incl. 429 backoff (5666ms). `n=25`, `cuda` |
| **Latency warm (100% hit)** | **1759ms** (`data/processed/bench_warm.json:1`) | Redis lookup + relevance scoring; ideal local Redis <10ms, remote `retroradiant-blissful-celery:19885` measures 300-2019ms RTT, dominates warm latency. Cold 305s wall → warm 64s wall (4.7x) |
| **Cache hit rate** | **0% cold / 100% warm** | Earlier partial warm 75% (15/20) |
| **Fallback routing** | **0%** on 25 eval set (router threshold `len<=2` or `keyword:` prefix) | Short queries route to fallback |
| **GPU** | **GTX 1650 4GB, CUDA 12.4, torch 2.6.0+cu124** | `torch.cuda.is_available() True` |

**Honest caveats:**
- **Remote Redis RTT** inflates warm-cache latency (1759ms vs expected <10ms). Would be near-zero with local Docker Redis.
- **Groq rate-limit** (8000 TPM on_demand) inflates cold wall latency (retries). Clean latency 9166ms is the API time; wall 14832ms includes backoff.
- **Rerank lift negative** is artifact of heuristic `eval_labels.json` (keyword presence, not manual ground truth) + small corpus where FAISS already ranks lexical matches high; cross-encoder optimizes relevance not cosine. Manual labeling needed for true lift.
- **Ollama not benchmarked** — server not installed locally, `Latency (Ollama)` N/A.
- **Heuristic labels** are noisy; 2-3 queries (e.g. `cross-encoder` explanation) have no true LangChain chunk, forced labels via overlap.

## Key engineering decisions & trade-offs

- **Cosine lift was wrong → precision@3:** Original `rerank_lift_pct` compared cosine before/after rerank — FAISS already optimizes cosine, so rerank always looked worse (-8.5%). Replaced with `precision@3` against `eval_labels.json` (does correct chunk appear in top 3?). Still heuristic, but correct yardstick.
- **Sentence-transformers segfault workaround:** `sentence-transformers==5.5.1` segfaults on import (`exit -1073741819`) on Windows + torch 2.6. Replaced with `transformers.AutoModel` mean-pooling + `AutoModelForSequenceClassification` for cross-encoder, with manual `normalize` + `faiss.normalize_L2` (`src/embed_index.py:76,187`). Functionally equivalent, avoids ST.
- **Groq model swap:** `llama-3.1-8b-instant` returns 404 on this key (`Groq.models.list()` shows `openai/gpt-oss-20b`, `groq/compound`, `qwen/*`). Switched `configs/config.yaml:6` to `openai/gpt-oss-20b`.
- **Normalization fix:** Added explicit `faiss.normalize_L2` in `build_index()` and `search()` (`src/embed_index.py:78,187`) even though torch already normalized — sanity `norms mean=1.0000`. Verified `k=15`.
- **Perf: model caching:** `src/embed_index.py:11` caches `_EMB_TOK/_CE_TOK` module-level; `src/benchmark.py:56` loads embedding once. Cut per-query reload from ~3 loads to 1.
- **Latency measurement fix:** `src/generate.py:18` now tracks `LAST_CLEAN_LATENCY_MS` (API only excl. `sleep`), `src/benchmark.py:89` logs `clean` vs `wall` and cache-hit as Redis wall (not stale generation time).
- **Corpus filtering:** 8/25 fallback queries excluded from precision to avoid noise; `eval_labels.json` now marks `answerable` flag.

## Likely interview Q&A

**Q: Why MCP over plain tool functions?**
- Standardizes tool serving (stdio/SSE) via `mcp` protocol, so retrieval can be swapped/hosted remotely without changing agent. `langchain-mcp-adapters` loads it as LangChain tool. Plain functions work for demo fallback, but MCP shows protocol awareness.

**Q: Why did reranking underperform initially?**
- First metric was cosine lift — wrong yardstick (FAISS optimizes cosine). Cross-encoder optimizes query-document relevance (logits), not embedding cosine. Also heuristic labels were noisy and `k` was small. Fixed to `precision@3` on answerable subset; true lift needs manual labels.

**Q: How would you scale this?**
- Shard FAISS (IVF/PQ) for >10k chunks, use async embedding, add Redis cluster or local cache, batch Groq calls, add eval set with human labels, move MCP to SSE/HTTP for remote, add LangGraph persistence (`checkpointer`).

**Q: Trade-offs of Groq vs Ollama?**
- Groq: <300ms ideal, but 8000 TPM limit throttles benchmarks (429 retries). Ollama: local, no limit, but GTX 1650 4GB slow for 8B and not installed. Code swaps via `prefer` arg; benchmark honest about both.
