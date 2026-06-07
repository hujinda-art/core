# Core — AI Agent 记忆策略评测框架

精简版的 AI Agent 长期记忆评测系统，支持多种记忆策略和向量库后端，内置 LLM 语义评估和防污染机制。

## 目录结构

```
core/
├── main.py                  # 测试运行入口
├── experiments.ipynb         # 实验笔记本
├── experiment_results/       # 实验结果缓存
├── hnsw_memorystore/         # HNSW 向量库包
│   ├── __init__.py
│   ├── store.py
│   ├── adapter.py
│   ├── embedding.py
│   └── __main__.py
├── .env                     # 环境变量配置
├── .env.example             # 环境变量模板
├── requirements.txt         # 依赖
├── results/                 # 评测结果输出目录
├── test/                    # 测试用例
│   ├── quick_test.json
│   ├── conversation_tests_consistency.json
│   └── conversation_tests_forgetting.json
└── src/
    ├── agents/
    │   ├── Agent.py          # AsyncAgent（异步对话）
    │   ├── message_dto.py    # 消息数据模型
    │   └── message_enum.py   # 系统提示词 + 防污染提示词
    ├── evaluator.py           # LLM 语义评估器
    └── memory/
        ├── __init__.py       # 策略注册
        ├── base_mem.py       # BaseMem 抽象基类
        ├── constant.py       # 常量配置
        ├── no_mem.py         # 无记忆策略
        ├── short_mem.py      # 短期记忆（滑动窗口）
        ├── long_mem.py       # 长期记忆（ChromaDB / HNSW 双后端）
        └── combined_mem.py   # 组合记忆（短期 + 长期）
```

## 快速开始

### 1. 安装依赖

```bash
cd /path/to/memorystore/core
pip install -r requirements.txt

# 如果使用 HNSW 后端，额外安装：
pip install hnswlib FlagEmbedding   # HNSW_EMBEDDING=bge-m3 或 bge-large-zh
# 或
pip install hnswlib sentence-transformers  # HNSW_EMBEDDING=sentence-transformers
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 API 地址和模型
```

`.env` 示例（使用 ollama 本地模型）：

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5:7b

MEMORY_STRATEGIES=ShortMem,LongMem,CombinedMem
LONG_MEM_BACKEND=chromadb

# 评估模型（可独立配置）
EVAL_OPENAI_BASE_URL=http://localhost:11434/v1
EVAL_OPENAI_API_KEY=ollama
EVAL_OPENAI_MODEL=qwen2.5:7b
EVAL_CONCURRENCY=5

# 防污染机制开关
LONG_MEM_ANTI_POLLUTION=true
```

### 3. 运行测试

```bash
# PYTHONPATH 需要指向 core 目录（用于 src 和 hnsw_memorystore 导入）
export PYTHONPATH=/path/to/memorystore/core

# 运行默认测试（quick_test.json，所有策略）
python main.py

# 运行指定测试文件和策略
python main.py --test-file conversation_tests_consistency.json --strategies LongMem

# 运行多个测试文件
python main.py --test-file quick_test.json --test-file conversation_tests_consistency.json

# 只运行指定 ID 的测试组
python main.py --test-id quick_01 --test-id quick_02

# 不保存结果（仅输出日志）
python main.py --no-save
```

## 环境变量

### 核心 LLM 配置

| 变量 | 必需 | 默认值 | 说明 |
|---|---|---|---|
| `OPENAI_API_KEY` | 是 | — | API Key（ollama 填 `ollama`） |
| `OPENAI_BASE_URL` | 否 | `https://api.openai.com/v1` | API 地址 |
| `OPENAI_MODEL` | 否 | `gpt-4o-mini` | 模型名称 |

### 记忆策略配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MEMORY_STRATEGIES` | `ShortMem,LongMem,CombinedMem` | 逗号分隔的策略列表 |
| `LONG_MEM_BACKEND` | `chromadb` | 长期记忆后端：`chromadb` 或 `hnsw` |
| `LONG_MEM_ANTI_POLLUTION` | `true` | 防污染机制开关：`true`/`false` |
| `HNSW_EMBEDDING` | `bge-m3` | HNSW 嵌入模型：`bge-m3`、`bge-large-zh`、`sentence-transformers` |
| `HNSW_EMBEDDING_DEVICE` | `cuda` | HNSW 嵌入设备：`cuda` 或 `cpu` |
| `SHORT_MEM_SECOND_WATER_LEVEL` | `0.5` | 短期记忆滑动窗口水位线比例 |
| `TEST_FILES` | `quick_test.json` | 默认测试文件（逗号分隔） |

### 评估配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EVAL_OPENAI_BASE_URL` | 回退到 `OPENAI_BASE_URL` | 评估用 LLM 地址（可与主 Agent 不同） |
| `EVAL_OPENAI_API_KEY` | 回退到 `OPENAI_API_KEY` | 评估用 API Key |
| `EVAL_OPENAI_MODEL` | 回退到 `OPENAI_MODEL` | 评估用模型（可用更强模型独立打分） |
| `EVAL_CONCURRENCY` | `5` | 评估 LLM 并发数 |

## 记忆策略

| 策略 | 说明 |
|---|---|
| **NoMem** | 无记忆，每次对话独立 |
| **ShortMem** | 短期记忆，滑动窗口（token 水位线压缩） |
| **LongMem** | 长期记忆，向量检索，支持 ChromaDB / HNSW 双后端 |
| **CombinedMem** | 组合记忆 = ShortMem + LongMem |

### 长期记忆后端

**ChromaDB**（默认）：

```bash
LONG_MEM_BACKEND=chromadb
```

- 数据存储在 `core/memorystore/` 目录
- 每个 session 独立 collection（`long_mem_{session_id}`）
- 无需额外依赖

**HNSW**：

```bash
LONG_MEM_BACKEND=hnsw
HNSW_EMBEDDING=bge-m3       # 可选：bge-m3 | bge-large-zh | sentence-transformers
HNSW_EMBEDDING_DEVICE=cuda   # 可选：cuda | cpu
```

- 需要 `hnsw_memorystore` 包（同目录下 `core/hnsw_memorystore/`）
- 需要安装对应嵌入模型的依赖
- 数据存储在 `core/memorystore/hnsw_{session_id}/`

## 防污染机制

长期记忆写入时自动执行防污染处理（`LONG_MEM_ANTI_POLLUTION=true`）：

### 工作原理

```
用户输入 q + 助手回复 ans
        │
        ▼
┌─ _process_before_write(q, ans) ─────────┐
│                                           │
│  有旧记忆？                               │
│  ├─ 否 → _extract_clean_q(q)             │
│  │   单轮提炼：去除自我纠正，保留最终结论    │
│  │                                        │
│  └─ 是 → 一次 LLM 调用同时完成：           │
│      1. 提炼陈述 → CLEAN_Q: <提炼结果>      │
│      2. 检测矛盾 → DELETE_IDS: <id列表>     │
│         矛盾旧记忆会被自动删除               │
│                                           │
└───────────────────────────────────────────┘
        │
        ▼
  写入 clean_q + ans 到向量库
```

### 示例

- **单轮提炼**：用户说 "我叫张三，不对，我叫李四" → 写入时提炼为 "我叫李四"
- **跨轮矛盾检测**：旧记忆 "张三住在北京" + 新信息 "张三搬去深圳了" → 删除旧记忆，写入 "张三住在深圳"

### 与后端的关系

防污染逻辑是**后端无关**的，ChromaDB 和 HNSW 共享同一套 LLM 调用代码：
- ChromaDB：`collection.query()` 检索相近记忆，`collection.delete()` 删除矛盾记忆
- HNSW：`store.search()` 检索，`store.delete_memory()` 删除

### 关闭防污染

```bash
LONG_MEM_ANTI_POLLUTION=false
```

关闭后，`update_mem(q, ans)` 直接写入原始 `q`，不做任何 LLM 调用。

## 语义评估

评测采用 LLM 语义评估取代简单子串匹配：

### 评估类型

| 测试类型 | 评估方式 | 说明 |
|---|---|---|
| **consistency** | 语义等价判定 | 判断模型回答是否与期望答案语义一致（允许不同表达） |
| **forgetting** | 知识点召回率 | 计算模型回答中包含的期望知识点比例（N/M 格式） |

### 评估提示词设计要点

- **一致性评估**：允许同义替换、精简/详述、语序差异；只关注核心含义是否一致
- **遗忘评估**：每个知识点独立计分，语义包含即算回忆到
- 批量评估优先（一次 LLM 调用处理多条），失败时自动回退到逐条评估
- 子串匹配为最终回退（LLM 调用完全失败时）

### 评估方法标记

结果 JSON 中每条 evaluation 包含 `eval_method` 字段：

| 值 | 含义 |
|---|---|
| `llm_semantic` | LLM 语义评估成功 |
| `substring_fallback` | LLM 调用失败，回退到子串匹配 |

### 单独配置评估模型

可以使用不同的模型做评估（如用强模型评估弱模型的输出）：

```env
# 主 Agent 用本地小模型快速推理
OPENAI_MODEL=qwen2.5:7b

# 评估用强模型确保评分质量
EVAL_OPENAI_MODEL=deepseek-v4-flash
EVAL_OPENAI_BASE_URL=https://api.deepseek.com/v1
EVAL_OPENAI_API_KEY=sk-xxxxxx
```

## 测试文件格式

```json
[
  {
    "id": "test_01",
    "type": "consistency",
    "turns": [
      {"role": "introduction", "q": "接下来我会告诉你一些信息，请记住。"},
      {"role": "information", "q": "我的名字是张三。"},
      {"role": "distractor", "q": "今天天气怎么样？"},
      {"role": "evaluation", "q": "我叫什么名字？", "expected": "张三"}
    ]
  }
]
```

角色说明：

| role | 说明 | 是否触发记忆更新 |
|---|---|---|
| `introduction` | 开场白 | 是 |
| `information` | 提供信息 | 是 |
| `distractor` | 干扰对话 | 是 |
| `evaluation` | 评估提问 | 是（且延迟到组结束后统一语义评估） |

## 结果输出

结果保存到 `core/results/` 目录，文件名格式：

```
{策略}_{测试文件名}_{后端}_{时间戳}.json
```

示例：`LongMem_quick_test_chromadb_20260607_153000.json`

结果结构：

```json
{
  "strategy": "LongMem",
  "backend": "chromadb",
  "test_file": "quick_test.json",
  "timestamp": "2026-06-07T15:30:00",
  "results": [
    {
      "id": "quick_01",
      "type": "consistency",
      "evaluations": [
        {
          "turn": 1,
          "q": "我叫什么名字？",
          "reply": "您叫张三。",
          "expected": "张三",
          "passed": true,
          "eval_method": "llm_semantic"
        }
      ]
    }
  ]
}
```

## 并行执行

多个策略和测试文件的组合会通过 `asyncio.gather` 并行执行：

```bash
# 并行运行 3 策略 × 2 测试文件 = 6 个任务
python main.py \
  --strategies ShortMem --strategies LongMem --strategies CombinedMem \
  --test-file quick_test.json --test-file conversation_tests_consistency.json
```

注意：并行运行时每个（策略, 测试文件）组合使用独立的 `session_id`，互不干扰。

## 常见问题

### Q: ollama 模型名带 `:7b` 后缀有什么影响？

Agent 检测到模型名以 `:7b` 结尾时，会自动传入 `extra_body={"options": {"num_gpu_layers": 35}}` 以启用 GPU 加速。

### Q: 如何切换不同的 HNSW 嵌入模型？

```bash
# BGE-M3（默认，多语言，1024 维）
HNSW_EMBEDDING=bge-m3

# BGE-Large-ZH（中文优化，1024 维）
HNSW_EMBEDDING=bge-large-zh

# Sentence-Transformers（通用）
HNSW_EMBEDDING=sentence-transformers
```

### Q: 长期记忆数据存储在哪里？

- ChromaDB：`core/memorystore/long_mem_{session_id}/`
- HNSW：`core/memorystore/hnsw_{session_id}/`

`memorystore/` 目录已在 `.gitignore` 中，不会被提交。

### Q: 如何清理已有记忆数据？

删除对应的 `core/memorystore/` 子目录即可。测试运行时每个策略会自动调用 `mem.reset()` 清空记忆。