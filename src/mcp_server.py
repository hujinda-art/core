"""MCP Server for memorystore — exposes memory tools to LLM clients.

Usage:
    # stdio transport (for Claude Desktop, opencode, etc.)
    python -m memorystore.mcp_server

    # Or via entry point after pip install:
    memorystore-mcp

Environment variables (required):
    OPENAI_API_KEY         — LLM API key (or "ollama" for local)
    OPENAI_BASE_URL        — LLM API base URL
    OPENAI_MODEL            — Model name (e.g. "qwen2.5:7b")

Environment variables (optional):
    LONG_MEM_BACKEND        — chromadb (default) or hnsw
    HNSW_EMBEDDING          — bge-m3 (default), bge-large-zh, sentence-transformers
    HNSW_EMBEDDING_DEVICE   — cuda (default) or cpu
    LONG_MEM_ANTI_POLLUTION — true (default) or false
    PROMPTS_DIR             — path to prompt markdown files
    AGENT_PROMPTS           — agent prompt filename (default: agent.md)
"""

import os
import sys

_CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP

from src.memory import LongMem
from src.agents.message_dto import Role
from src.agents.message_enum import Message

mcp = FastMCP("memorystore")

_strategy = os.getenv("MEMORY_STRATEGY", "LongMem").strip()
_backend = os.getenv("LONG_MEM_BACKEND", "chromadb").strip().lower()
_session_id = os.getenv("MCP_SESSION_ID", "mcp_default").strip()

_mem = LongMem(session_id=_session_id)


@mcp.tool()
def memory_store(question: str, answer: str) -> str:
    """Store a piece of memory (a Q&A pair). The question serves as a retrieval key, the answer is the content to remember.

    Args:
        question: The user question or context key.
        answer: The assistant answer or information to remember.
    """
    _mem.update_mem(question, answer)
    return f"Stored. (backend={_backend}, session={_session_id})"


@mcp.tool()
def memory_search(query: str) -> str:
    """Search memory for relevant past conversations. Returns the most relevant stored memories.

    Args:
        query: The search query, typically a question or topic to look up.
    """
    items = _mem.get_mem(query)
    if not items:
        return "No relevant memories found."
    lines = []
    for item in items:
        role_tag = item.role.value if hasattr(item.role, "value") else str(item.role)
        lines.append(f"[{role_tag}] {item.content}")
    return "\n\n".join(lines)


@mcp.tool()
def memory_reset() -> str:
    """Clear all stored memory for the current session."""
    _mem.reset()
    return f"Memory cleared. (backend={_backend}, session={_session_id})"


@mcp.tool()
def memory_status() -> str:
    """Check the current memory module status: backend, session, strategy, and rough item count."""
    count = _mem.get_mem("").__len__() if hasattr(_mem, "get_mem") else -1
    return (
        f"Strategy: {_strategy}\n"
        f"Backend: {_backend}\n"
        f"Session: {_session_id}\n"
        f"Stored items (approx): {count}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")


def main() -> None:
    """Entry point for `memorystore-mcp` command."""
    mcp.run(transport="stdio")