"""
tools.py — Day 3: MCP client + fallback tool
- MCP retrieval tool loaded via langchain-mcp-adapters
- fallback: simple keyword search over raw corpus
"""
import json
import re
from pathlib import Path
from typing import List, Dict
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# --- Fallback tool: keyword search over processed chunks ---

def keyword_search(query: str, top_k=5) -> List[Dict]:
    """Single plain fallback tool — keyword overlap over corpus"""
    cfg = load_config()
    chunks_path = Path(cfg["paths"]["chunks_path"])
    if not chunks_path.exists():
        return []
    # load chunks
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    q_terms = set(re.findall(r"\w+", query.lower()))
    scored = []
    for ch in chunks:
        t_terms = set(re.findall(r"\w+", ch["text"].lower()))
        overlap = len(q_terms & t_terms)
        if overlap > 0:
            scored.append((overlap, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, ch in scored[:top_k]:
        out.append({"id": ch["id"], "source": ch["source"], "text": ch["text"], "faiss_score": 0.0, "rerank_score": float(score), "tool": "fallback_keyword"})
    return out

# LangChain tool wrapper for fallback (so LangGraph can use it)
try:
    from langchain_core.tools import tool as lc_tool
    @lc_tool
    def fallback_search_tool(query: str) -> str:
        """Fallback keyword search over corpus when retrieval is not needed or MCP fails. Input: query string."""
        hits = keyword_search(query)
        return json.dumps(hits, ensure_ascii=False, indent=2)
except Exception:
    # if langchain not installed, expose plain function
    def fallback_search_tool(query: str) -> str:
        return json.dumps(keyword_search(query), ensure_ascii=False, indent=2)

# --- MCP retrieval tool loader ---

async def load_mcp_retrieval_tool():
    """Connect to stdio MCP server and return LangChain tools list"""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from langchain_mcp_adapters.tools import load_mcp_tools
        import sys
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_server"],
            env=None,
        )
        # Note: caller must manage async context; this helper shows pattern
        # For simplicity, agent.py will handle the full connection
        return server_params
    except Exception as e:
        print(f"[tools] MCP load failed: {e}")
        return None

def get_fallback_tool():
    return fallback_search_tool
