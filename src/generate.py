"""
generate.py — Day 2: Ollama / Groq wrappers + RAG helper
GPU: Ollama auto-uses GPU if present; embeddings check device elsewhere
"""
import os
import time
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

LAST_CLEAN_LATENCY_MS = None  # clean Groq API time excl. backoff, set per call
LAST_TOTAL_LATENCY_MS = None    # wall including retries (for reference)

def generate_groq(prompt: str, model: str = None, retries=3) -> str:
    global LAST_CLEAN_LATENCY_MS, LAST_TOTAL_LATENCY_MS
    cfg = load_config()
    model = model or cfg["models"]["groq"]
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    from groq import Groq
    client = Groq(api_key=api_key)
    last_err = None
    total_t0 = time.time()
    for attempt in range(retries):
        try:
            api_t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            clean = (time.time() - api_t0) * 1000
            LAST_CLEAN_LATENCY_MS = clean
            LAST_TOTAL_LATENCY_MS = (time.time() - total_t0) * 1000
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "limit" in msg:
                wait = 2 ** attempt
                print(f"[groq] rate limited, retry {attempt+1}/{retries} in {wait}s: {e}")
                time.sleep(wait)
            else:
                time.sleep(1)
                if attempt == retries - 1:
                    raise
    LAST_CLEAN_LATENCY_MS = None
    LAST_TOTAL_LATENCY_MS = (time.time() - total_t0) * 1000
    raise last_err

def generate_ollama(prompt: str, model: str = None) -> str:
    cfg = load_config()
    model = model or cfg["models"]["ollama"]
    try:
        import ollama
    except ImportError:
        raise ImportError("ollama package not installed. pip install ollama or use Groq path.")
    # Ollama auto-detects GPU; no device flag needed
    try:
        resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"].strip()
    except Exception as e:
        # if model not pulled, give friendly error
        raise RuntimeError(f"Ollama failed (is Ollama running? model pulled? {model}): {e}")

def generate(prompt: str, prefer="groq") -> str:
    """Unified entry — prefer groq, fallback to ollama or vice versa"""
    if prefer == "groq":
        try:
            return generate_groq(prompt)
        except Exception as e:
            print(f"[generate] groq failed: {e}, trying ollama...")
            return generate_ollama(prompt)
    else:
        try:
            return generate_ollama(prompt)
        except Exception as e:
            print(f"[generate] ollama failed: {e}, trying groq...")
            return generate_groq(prompt)

# --- Day 2: retrieve-then-generate helper ---

RAG_TEMPLATE = """You are DocMind, a helpful assistant answering over LangChain/LangGraph docs.
Use the CONTEXT below to answer. If context is insufficient, say so briefly.

CONTEXT:
{context}

QUESTION: {question}

Answer concisely (3-6 sentences) and cite sources by URL when helpful.
"""

def rag_answer(question: str, prefer="groq"):
    """Embed->FAISS->rerank->stuff->generate. No graph/cache yet — Day 2 baseline."""
    from .embed_index import search
    cfg = load_config()
    hits = search(question, k=cfg["retrieval"]["faiss_k"], rerank_k=cfg["retrieval"]["rerank_k"])
    context = "\n\n---\n\n".join([f"[{h['source']}] {h['text']}" for h in hits])
    prompt = RAG_TEMPLATE.format(context=context, question=question)
    t0 = time.time()
    try:
        answer = generate(prompt, prefer=prefer)
    except Exception as e:
        answer = f"[generation failed: {e}]"
    latency = (time.time() - t0) * 1000
    return {"question": question, "answer": answer, "hits": hits, "latency_ms": latency}

if __name__ == "__main__":
    for q in ["What is LangChain?", "How does LangGraph differ from LangChain?", "What is FAISS?"]:
        out = rag_answer(q)
        print(f"\nQ: {q}\nA: {out['answer'][:400]}...\nlatency {out['latency_ms']:.0f}ms hits={len(out['hits'])}")
