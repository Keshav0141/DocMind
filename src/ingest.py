"""
ingest.py — Day 1: load + chunk corpus (LangChain docs domain)
Saves:
  data/raw/*.txt
  data/processed/chunks.jsonl
  data/processed/eval_queries.txt
"""
import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict

import yaml
import requests
from bs4 import BeautifulSoup

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"

# ~60 LangChain docs URLs — stable, covers core concepts (enough for 50-100 docs target)
LANGCHAIN_URLS = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/concepts/architecture/",
    "https://python.langchain.com/docs/concepts/chat_models/",
    "https://python.langchain.com/docs/concepts/prompt_templates/",
    "https://python.langchain.com/docs/concepts/output_parsers/",
    "https://python.langchain.com/docs/concepts/retrievers/",
    "https://python.langchain.com/docs/concepts/vectorstores/",
    "https://python.langchain.com/docs/concepts/text_splitters/",
    "https://python.langchain.com/docs/concepts/embedding_models/",
    "https://python.langchain.com/docs/concepts/example_selectors/",
    "https://python.langchain.com/docs/concepts/tools/",
    "https://python.langchain.com/docs/concepts/tool_calling/",
    "https://python.langchain.com/docs/concepts/agents/",
    "https://python.langchain.com/docs/concepts/memory/",
    "https://python.langchain.com/docs/concepts/callbacks/",
    "https://python.langchain.com/docs/concepts/rag/",
    "https://python.langchain.com/docs/how_to/chatbots_memory/",
    "https://python.langchain.com/docs/how_to/document_loader_pdf/",
    "https://python.langchain.com/docs/how_to/document_loader_web/",
    "https://python.langchain.com/docs/how_to/text_splitters/",
    "https://python.langchain.com/docs/how_to/vectorstores/",
    "https://python.langchain.com/docs/how_to/embeddings/",
    "https://python.langchain.com/docs/how_to/retrieval/",
    "https://python.langchain.com/docs/how_to/question_answering/",
    "https://python.langchain.com/docs/how_to/tool_calling/",
    "https://python.langchain.com/docs/how_to/tools/",
    "https://python.langchain.com/docs/how_to/output_parser_structured/",
    "https://python.langchain.com/docs/how_to/langgraph/",
    "https://python.langchain.com/docs/how_to/mcp/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/tutorials/extraction/",
    "https://python.langchain.com/docs/tutorials/summarization/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/integrations/chat/openai/",
    "https://python.langchain.com/docs/integrations/chat/groq/",
    "https://python.langchain.com/docs/integrations/chat/ollama/",
    "https://python.langchain.com/docs/integrations/vectorstores/faiss/",
    "https://python.langchain.com/docs/integrations/vectorstores/chroma/",
    "https://python.langchain.com/docs/integrations/vectorstores/pinecone/",
    "https://python.langchain.com/docs/integrations/document_loaders/web_loaders/",
    "https://python.langchain.com/docs/integrations/retrievers/",
    "https://python.langchain.com/docs/integrations/tools/",
    "https://langchain-ai.github.io/langgraph/concepts/why-langgraph/",
    "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    "https://langchain-ai.github.io/langgraph/concepts/high_level/",
    "https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/",
    "https://langchain-ai.github.io/langgraph/how-tos/tool-calling/",
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/rag/",
    "https://python.langchain.com/api_reference/",
    "https://python.langchain.com/docs/versions/migrating_chains/",
    "https://python.langchain.com/docs/versions/migrating_memory/",
]

HEADERS = {"User-Agent": "DocMind/1.0 (research project; contact: local)"}

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # remove very short boilerplate lines
    return text

def fetch_url(url: str, timeout=15) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        # remove nav, script, style
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()
        # main article content
        main = soup.find("main") or soup.find("article") or soup
        text = main.get_text(separator=" ", strip=True)
        text = clean_text(text)
        # truncate very long pages
        if len(text) < 500:
            return ""
        return text[:20000]
    except Exception as e:
        print(f"[warn] fetch failed {url}: {e}")
        return ""

def chunk_text(text: str, chunk_size=400, overlap=50) -> List[str]:
    """Chunk by ~tokens (approx 4 chars/token). 400 tokens ~ 1600 chars."""
    chars_per_chunk = chunk_size * 4
    overlap_chars = overlap * 4
    chunks = []
    start = 0
    while start < len(text):
        end = start + chars_per_chunk
        chunk = text[start:end].strip()
        if len(chunk) >= 200:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap_chars
    return chunks

def ingest(urls: List[str] = None):
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    chunks_path = Path(cfg["paths"]["chunks_path"])
    eval_path = Path(cfg["benchmark"]["eval_queries_path"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    urls = urls or LANGCHAIN_URLS
    print(f"[ingest] fetching {len(urls)} URLs...")
    all_chunks: List[Dict] = []
    success = 0

    for i, url in enumerate(urls, 1):
        slug = hashlib.md5(url.encode()).hexdigest()[:8]
        raw_file = raw_dir / f"{slug}.txt"
        text = ""
        if raw_file.exists() and raw_file.stat().st_size > 500:
            text = raw_file.read_text(encoding="utf-8", errors="ignore")
        else:
            text = fetch_url(url)
            if text:
                raw_file.write_text(f"SOURCE: {url}\n\n{text}", encoding="utf-8")
                time.sleep(0.3)  # be nice
        if not text:
            continue
        # strip SOURCE line if present
        if text.startswith("SOURCE:"):
            text = text.split("\n\n", 1)[-1]
        chunks = chunk_text(text, chunk_size=cfg["chunking"]["chunk_size"], overlap=cfg["chunking"]["chunk_overlap"])
        for j, c in enumerate(chunks):
            all_chunks.append({"id": f"{slug}_{j}", "source": url, "text": c, "slug": slug})
        success += 1
        print(f"  [{i}/{len(urls)}] {url} -> {len(chunks)} chunks")

    # fallback: if too few docs (network blocked), create synthetic fallback
    if len(all_chunks) < 30:
        print(f"[warn] only {len(all_chunks)} chunks, adding synthetic fallback docs")
        fallback = [
            "LangChain is a framework for developing applications powered by language models. It provides components for prompts, chains, agents, memory, and retrieval.",
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with graph-based orchestration and persistence.",
            "Vector stores like FAISS store embeddings and allow similarity search. Retrieval-augmented generation (RAG) uses vector stores to ground LLM answers in documents.",
            "Sentence-Transformers provide embedding models like all-MiniLM-L6-v2. Cross-encoders rerank retrieved results for higher precision.",
            "MCP (Model Context Protocol) standardizes tool serving. Servers expose tools over stdio/SSE, clients load them as LangChain tools.",
            "Groq provides fast LLM inference. Ollama runs local quantized models. Redis caches query results for low latency.",
        ]
        for idx, t in enumerate(fallback * 10):
            all_chunks.append({"id": f"fallback_{idx}", "source": "synthetic://fallback", "text": t + " " + t, "slug": "fallback"})

    # save chunks
    with open(chunks_path, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"[ingest] saved {len(all_chunks)} chunks to {chunks_path} from {success} docs")

    # generate eval queries (diverse, domain-relevant)
    eval_queries = [
        "What is LangChain and what are its main components?",
        "How does LangGraph differ from LangChain?",
        "What is a vector store and how is FAISS used?",
        "Explain RAG and its retrieval step",
        "How do you chunk documents for embeddings?",
        "What is a cross-encoder and why rerank after FAISS?",
        "How does MCP expose tools to an agent?",
        "What is the difference between Groq and Ollama for generation?",
        "How do you use Redis as a cache for RAG?",
        "What are LangChain retrievers and how do they work?",
        "How to load a PDF document in LangChain?",
        "How to build a chatbot with memory in LangGraph?",
        "What is tool calling in LangChain?",
        "How to use prompt templates?",
        "What is an embedding model?",
        "Explain FAISS similarity search",
        "What is the purpose of text splitters?",
        "How to track RAG experiments with MLflow or W&B?",
        "What is the fallback tool in DocMind?",
        "How does the agent decide which tool to use?",
        "Compare latency of Groq vs Ollama",
        "What is agentic memory and how is it implemented?",
        "How to evaluate retrieval relevance?",
        "What is the typical chunk size for RAG?",
        "How to connect to Redis Cloud?",
    ]
    eval_path.write_text("\n".join(eval_queries), encoding="utf-8")
    print(f"[ingest] wrote {len(eval_queries)} eval queries to {eval_path}")
    return all_chunks

if __name__ == "__main__":
    ingest()
