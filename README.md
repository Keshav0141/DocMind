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

**Results (25 eval queries, 17/25 corpus-answerable):**

| Metric | Value |
|---|---|
| Precision@3, FAISS-only | 17.6% (3/17) |
| Precision@3, FAISS+rerank | 5.9% (1/17) |
| Cold latency (clean generation) | 9,166 ms |
| Cold latency (wall, incl. rate-limit retries) | 14,832 ms |
| Warm cache hit rate | 100% (repeated queries) |
| Warm cache latency | 1,759 ms (bottlenecked by remote Redis RTT, not compute) |

Precision@3 was measured against a hand-tagged subset of the eval set (17 of 25 queries have a verifiable answer chunk in the corpus; the remaining 8 target the fallback tool). The negative rerank lift is likely a labeling artifact from keyword-heuristic eval labels rather than a true regression — cross-encoders often deprioritize keyword matches in favor of semantic relevance, which a keyword-based label set can't credit. A hand-verified label set would give a more trustworthy number.

## Notes

- The Groq free tier is rate-limited (8000 TPM); cold-run latency includes retry/backoff time, reported separately from clean API latency.
- Redis Cloud free tier (30MB) is used for caching; remote RTT (300-2000ms) dominates warm-cache latency more than compute does.
- Re-running `ingest.py` without `--clear` will re-fetch and overwrite `chunks.jsonl`.

## License

[MIT](LICENSE)
