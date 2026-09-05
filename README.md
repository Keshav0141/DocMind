# DocMind

Domain-aware RAG agent that ingests LangChain/LangGraph docs and answers natural-language questions with grounded retrieval — built as an agentic pipeline with MCP tool serving, Redis-backed caching, and two-stage FAISS + cross-encoder retrieval. Built with LangGraph, FAISS, sentence-transformers, Redis, and Groq/Ollama for generation.

## Stack

- Agent orchestration: LangGraph (`StateGraph` router → retrieval/fallback tool → generation)
- Tool serving: MCP (Model Context Protocol) — retrieval tool exposed as a standard MCP stdio server, loaded via `langchain-mcp-adapters`
- Vector store: FAISS (`IndexFlatIP`, L2-normalized embeddings), persisted locally in `data/processed/faiss.index`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`, two-stage retrieval (FAISS top-15 → cross-encoder rerank to top-3/5)
- Memory/caching: Redis (Redis Cloud free tier), query-hash based cache for repeated/paraphrased queries
- LLM: Groq (`openai/gpt-oss-20b`) for fast free-tier inference; Ollama wrapper included for local generation
- Tracking: MLflow (experiment `docmind-rag`) + Weights & Biases (`frozenvictorxi/domain-rag-agent`)

## Setup

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys. `.env` is gitignored.

```
GROQ_API_KEY=<your key>
WANDB_API_KEY=<your key>
REDIS_URL=<your redis cloud connection string>
```

## Ingest the corpus

```
python src/ingest.py
python src/embed_index.py
```

`ingest.py` fetches the configured LangChain/LangGraph docs pages, chunks them (~300-500 tokens per chunk), and saves to `data/processed/chunks.jsonl`. `embed_index.py` embeds chunks with sentence-transformers, builds and persists the FAISS index, and verifies cross-encoder reranking on a test query.

## Ask questions

```
python src/agent.py "how does the router node work in LangGraph"
```

The agent retrieves via the MCP-served FAISS+rerank tool, falls back to keyword search over the raw corpus when retrieval is weak, checks the Redis cache before generating, and calls Groq (or Ollama, if installed locally) for the final answer.

## Benchmark

```
python src/benchmark.py
```

Runs the 25-query eval set end to end, logs latency/relevance/cache metrics to MLflow and W&B, and writes results to `data/processed/bench_*.json`.

**Results (25 eval queries, 16/25 corpus-answerable — hand-verified):**

| Metric | Value |
|---|---|
| Precision@3, FAISS-only | 6.2% (1/16) |
| Precision@3, FAISS+rerank | 6.2% (1/16) |
| Lift | 0.0pp |
| Cold latency (clean generation) | 8,258 ms |
| Cold latency (wall, incl. rate-limit retries) | 12,878 ms |
| Cold fallback routing | 44% (11/25) |
| Warm cache hit rate | 100% (repeated queries) |
| Warm cache latency | 1,593 ms (remote Redis RTT, not compute) |
| Warm fallback routing | 44% (11/25) |
| Router | FAISS top-1 < 0.30 → fallback (14/11 split vs intended 16/9) |
| Relevance (cosine >0.35) | 52.0% |

Precision@3 was measured against hand-verified labels (16 of 25 queries have a verifiable answer chunk; 9 target the fallback tool — `cross-encoder` query moved to fallback after checking 0/304 chunks contain the term). Both retrievers scored 1/16, so lift is 0.0pp — the earlier -11.8pp/ -4.0pp was from keyword-heuristic labels and is now replaced. Root cause is corpus gap, not reranking: 2/44 source docs 404'd (`vectorstores/faiss`, `document_loaders/web_loaders` → `docs.langchain.com` redirect) and `FAISS` term is absent from all 304 chunks, so retrieval ceiling is capped before reranking can help; a larger/cleaner corpus would be needed for higher numbers.

## Notes

- The Groq free tier is rate-limited (8000 TPM); cold-run latency includes retry/backoff time, reported separately from clean API latency.
- Redis Cloud free tier (30MB) is used for caching; remote RTT (300-2000ms) dominates warm-cache latency more than compute does.
- Re-running `ingest.py` without `--clear` will re-fetch and overwrite `chunks.jsonl`.

## License

[MIT](LICENSE)
