"""
agent.py — Day 3 + Day 4: LangGraph graph with router -> retrieval/fallback -> generation
Day 4 wires Redis cache before retrieval.
"""
import os
import json
import time
import asyncio
from pathlib import Path
from typing import TypedDict, List, Dict, Literal
import yaml
from dotenv import load_dotenv

load_dotenv()
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# State
class AgentState(TypedDict, total=False):
    question: str
    route: str  # "retrieval" or "fallback"
    hits: List[Dict]
    answer: str
    latency_ms: float
    cache_hit: bool
    provider: str

# Router: similarity-threshold (FAISS top-1 < threshold → fallback)
# Replaces naive keyword/length heuristic. Runs FAISS recall first, checks top-1 score.
def router_node(state: AgentState):
    q = state["question"]
    # explicit keyword prefix still forces fallback for demo
    if q.lower().startswith("keyword:"):
        state["route"] = "fallback"
        return state
    try:
        from src.embed_index import search
        cfg = load_config()
        # use FAISS recall k=15, check top-1 faiss_score (cosine, L2-normalized)
        hits = search(q, k=cfg["retrieval"]["faiss_k"], rerank_k=cfg["retrieval"]["rerank_k"])
        top_score = hits[0]["faiss_score"] if hits else 0.0
        threshold = 0.30  # tuned on 25-query eval: 14/11 split vs intended 16/9; 0.25 gave 19/6, 0.30 is closer and more conservative
        # also consider rerank_score? but faiss_score is cosine, more stable for threshold
        route = "fallback" if top_score < threshold else "retrieval"
        # store hits to avoid double retrieval in next node (optimization)
        state["route"] = route
        state["_router_hits"] = hits
        state["_router_score"] = top_score
        return state
    except Exception as e:
        print(f"[router] threshold check failed: {e}, defaulting to retrieval")
        state["route"] = "retrieval"
        return state

def retrieval_node(state: AgentState):
    # reuse hits from router if available (threshold router already did retrieval)
    if "_router_hits" in state and state.get("route") == "retrieval":
        state["hits"] = state.pop("_router_hits")
        state.pop("_router_score", None)
        return state
    from .embed_index import search
    cfg = load_config()
    hits = search(state["question"], k=cfg["retrieval"]["faiss_k"], rerank_k=cfg["retrieval"]["rerank_k"])
    state["hits"] = hits
    return state

def fallback_node(state: AgentState):
    from .tools import keyword_search
    hits = keyword_search(state["question"])
    # mark tool
    for h in hits:
        h["tool"] = "fallback"
    state["hits"] = hits
    return state

def generation_node(state: AgentState):
    from .generate import generate
    cfg = load_config()
    hits = state.get("hits", [])
    context = "\n\n---\n\n".join([f"[{h['source']}] {h['text']}" for h in hits]) if hits else "No context retrieved."
    prompt = f"""You are DocMind answering over LangChain/LangGraph docs.

CONTEXT:
{context}

QUESTION: {state['question']}

Answer concisely (3-6 sentences). Cite sources when useful."""
    provider = state.get("provider", "groq")
    t0 = time.time()
    try:
        ans = generate(prompt, prefer=provider)
    except Exception as e:
        ans = f"[generation error: {e}]"
    state["answer"] = ans
    state["latency_ms"] = (time.time() - t0) * 1000
    return state

# Build LangGraph graph
def build_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError("langgraph not installed")
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("fallback", fallback_node)
    g.add_node("generate", generation_node)

    g.set_entry_point("router")

    def route_fn(state):
        return state.get("route", "retrieval")

    g.add_conditional_edges("router", route_fn, {"retrieval": "retrieval", "fallback": "fallback"})
    g.add_edge("retrieval", "generate")
    g.add_edge("fallback", "generate")
    g.add_edge("generate", END)
    return g.compile()

# Cached wrapper (Day 4)
def answer_question(question: str, provider="groq", use_cache=True):
    """Check Redis cache -> run graph -> cache result"""
    from .memory import get_cached, set_cached
    if use_cache:
        cached = get_cached(question)
        if cached:
            cached["cache_hit"] = True
            return cached
    graph = build_graph()
    state: AgentState = {"question": question, "provider": provider}
    out = graph.invoke(state)
    result = {
        "question": question,
        "answer": out.get("answer", ""),
        "hits": out.get("hits", []),
        "route": out.get("route", ""),
        "latency_ms": out.get("latency_ms", 0),
        "cache_hit": False,
        "provider": provider,
    }
    if use_cache:
        try:
            set_cached(question, result)
        except Exception:
            pass
    return result

# MCP-aware variant (optional, shows MCP server usage)
async def answer_via_mcp(question: str, provider="groq"):
    """Demo MCP path: connect to mcp_server over stdio, call retrieve, then generate"""
    import sys
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    # We'll just call the server directly for retrieval, then generate
    server_params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server"], env=None)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            # tools[0] is retrieve
            retrieve_tool = tools[0]
            hits_json = await retrieve_tool.ainvoke({"query": question})
            hits = json.loads(hits_json) if isinstance(hits_json, str) else hits_json
            # generate
            from .generate import generate
            context = "\n\n".join([f"[{h['source']}] {h['text']}" for h in hits[:5]])
            prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\nAnswer concisely."
            t0 = time.time()
            ans = generate(prompt, prefer=provider)
            return {"question": question, "answer": ans, "hits": hits, "route": "mcp_retrieval", "latency_ms": (time.time()-t0)*1000}

if __name__ == "__main__":
    for q in ["What is LangChain?", "keyword: vector store", "Explain RAG with FAISS"]:
        r = answer_question(q)
        print(f"\nQ: {q}\n route={r['route']} cache_hit={r['cache_hit']} latency={r['latency_ms']:.0f}ms\n A: {r['answer'][:300]}...")
