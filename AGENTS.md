# AGENTS.md — AI Agent Memory Evaluation Framework

## Running Tests

```bash
# PYTHONPATH must point to this core/ directory (required for src/ and hnsw_memorystore/ imports)
export PYTHONPATH=$(pwd)

# Default: runs quick_test.json with all strategies
python main.py

# Specific test file + strategy
python main.py --test-file conversation_tests_consistency.json --strategies LongMem

# Multiple test files (each flag is separate)
python main.py --test-file quick_test.json --test-file conversation_tests_consistency.json

# Filter by test group ID
python main.py --test-id quick_01 --test-id quick_02

# Dry run without saving results
python main.py --no-save
```

## Required Setup

- Copy `.env.example` → `.env` and fill in `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`
- `pip install -r requirements.txt` (ChromaDB backend); for HNSW also: `pip install hnswlib FlagEmbedding` or `pip install hnswlib sentence-transformers`
- `memorystore/` directory is runtime data (ChromaDB/HNSW stores), gitignored, auto-created

## Architecture

- **Entry point**: `main.py` → argparse → `asyncio.gather` over (strategy × test_file) combos
- **Package layout**: `src/` (agents, memory, evaluator) + `hnsw_memorystore/` (optional HNSW backend)
- **Memory strategies** all subclass `BaseMem` (`src/memory/base_mem.py`): `NoMem`, `ShortMem`, `LongMem`, `CombinedMem`
- Strategy class names (`"NoMem"`, `"ShortMem"`, etc.) are used as CLI args and env var values — not module paths
- `LongMem` uses a strategy-pattern internal dispatch: `_LongMemChromadb` or `_LongMemHNSW` based on `LONG_MEM_BACKEND` env var
- **Evaluator** (`src/evaluator.py`) has independent config (`EVAL_OPENAI_*`) that falls back to `OPENAI_*`
- Anti-pollution logic lives in `src/memory/long_mem.py` as module-level functions, called from both backends

## Key Env Vars

| Var | Default | Notes |
|---|---|---|
| `LONG_MEM_BACKEND` | `chromadb` | `chromadb` or `hnsw` |
| `HNSW_EMBEDDING` | `bge-m3` | `bge-m3` / `bge-large-zh` / `sentence-transformers` |
| `LONG_MEM_ANTI_POLLUTION` | `true` | `true`/`false`; disables LLM refinement on write when false |
| `MEMORY_STRATEGIES` | all except `BaseMem` | Comma-separated; used when `--strategies` flag is omitted |
| `SHORT_MEM_SECOND_WATER_LEVEL` | `0.5` | Sliding window compression threshold ratio |
| `PROMPTS_DIR` | `prompts` | Directory containing prompt markdown files |
| `AGENT_PROMPTS` | `agent.md` | Filename in PROMPTS_DIR for agent system prompt + mem write prompt |
| `EVAL_BASIC_PROMPTS` | `evaluator_basic.md` | Filename in PROMPTS_DIR for consistency/forgetting evaluation prompts |
| `EVAL_PROGRAMMING_PROMPTS` | `evaluator_programming.md` | Filename in PROMPTS_DIR for programming evaluation prompts |

## Important Quirks

- `dotenv` is loaded in multiple files (`main.py`, `Agent.py`, `evaluator.py`, `long_mem.py`) — env vars must be in `.env`, not just shell exports, for fallback defaults to work correctly
- Ollama models ending in `:7b` trigger `extra_body={"options": {"num_gpu_layers": 35}}` in both `AsyncAgent` and `AsyncEvaluator`
- There is no test runner or unit test suite — `test/` contains JSON test case files (conversation scenarios), not pytest modules
- `main.py` modifies `sys.path` at import time to add the core directory; the `PYTHONPATH` export ensures this works from any cwd
- Results output format: `{Strategy}_{test_file_stem}_{backend}_{timestamp}.json` in `results/`

## 记忆规则

每次会话开始时，先用 `memorystore_memory_search` 搜索相关记忆。
每次会话结束时，用 `memorystore_memory_store` 存储本次关键信息。