"""
mcp_server.py — Day 3: MCP server exposing retrieval tool over stdio
Uses `mcp` SDK. Tool: retrieve(query, k)
"""
import asyncio
import json
from pathlib import Path

# mcp imports (install: pip install mcp)
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# reuse search logic
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embed_index import search as faiss_search

server = Server("docmind-retrieval")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="retrieve",
            description="Retrieve relevant chunks from LangChain/LangGraph corpus using FAISS + cross-encoder reranking. Input: query string. Returns top reranked chunks with source URLs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "user question"},
                    "k": {"type": "integer", "description": "FAISS recall k (default 15)"},
                    "rerank_k": {"type": "integer", "description": "final top_k after rerank (default 5)"}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name != "retrieve":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    query = arguments.get("query", "")
    k = arguments.get("k", 15)
    rk = arguments.get("rerank_k", 5)
    if not query:
        return [TextContent(type="text", text="Empty query")]
    try:
        hits = faiss_search(query, k=k, rerank_k=rk)
        # return as JSON string
        payload = json.dumps(hits, ensure_ascii=False, indent=2)
        return [TextContent(type="text", text=payload)]
    except Exception as e:
        return [TextContent(type="text", text=f"Retrieval error: {e}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
