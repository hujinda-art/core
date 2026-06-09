"""Interactive chat CLI — talk to the memory-augmented agent in real time.

Usage:
    python chat.py                     # LongMem + ChromaDB (default)
    python chat.py --strategy ShortMem # ShortMem only
    python chat.py --backend hnsw     # LongMem + HNSW
    python chat.py --no-memory        # NoMem (no memory at all)

Type 'quit' or Ctrl+C to exit.
Type 'reset' to clear memory for the current session.
Type 'history' to show conversation history in memory.
"""

import asyncio
import os
import sys

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from dotenv import load_dotenv

load_dotenv()

from src.agents.Agent import AsyncAgent
from src.memory import NoMem, ShortMem, LongMem, CombinedMem

_STRATEGY_MAP = {
    "NoMem": NoMem,
    "ShortMem": ShortMem,
    "LongMem": LongMem,
    "CombinedMem": CombinedMem,
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Interactive chat with memory-augmented agent")
    parser.add_argument(
        "--strategy",
        default="LongMem",
        choices=list(_STRATEGY_MAP.keys()),
        help="Memory strategy (default: LongMem)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="LongMem backend: chromadb or hnsw (default: from env or chromadb)",
    )
    args = parser.parse_args()

    if args.backend:
        os.environ["LONG_MEM_BACKEND"] = args.backend

    mem_class = _STRATEGY_MAP[args.strategy]
    try:
        mem = mem_class(session_id="interactive_chat")
    except TypeError:
        mem = mem_class()

    agent = AsyncAgent(mem_module=mem)

    strategy = args.strategy
    backend = os.getenv("LONG_MEM_BACKEND", "chromadb")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    print(f"=== Memory Chat ===")
    print(f"  Strategy: {strategy} | Backend: {backend} | Model: {model}")
    print(f"  Type 'quit' to exit, 'reset' to clear memory, 'history' to view stored memory.")
    print()

    async def chat_loop() -> None:
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not q:
                continue
            if q.lower() == "quit":
                print("Bye!")
                break
            if q.lower() == "reset":
                agent.mem.reset()
                print("[Memory cleared]\n")
                continue
            if q.lower() == "history":
                mem_items = agent.mem.get_mem("")
                if not mem_items:
                    print("[No memory stored]\n")
                else:
                    print(f"[Memory: {len(mem_items)} items]")
                    for item in mem_items:
                        role = item.role.value if hasattr(item.role, "value") else item.role
                        content = item.content[:200] + ("..." if len(item.content) > 200 else "")
                        print(f"  [{role}] {content}")
                    print()
                continue

            reply = await agent.chat(q)
            print(f"\nAgent: {reply}\n")

    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()