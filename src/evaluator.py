"""LLM-based semantic evaluator for test results.

Uses a separate OpenAI-compatible client (configurable via EVAL_OPENAI_* env vars)
to judge whether model answers match expected answers in meaning, not just in
exact string matching.
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger("AILongTermMem")

_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "5"))
_SEMAPHORE = asyncio.Semaphore(_CONCURRENCY)


@dataclass
class EvalConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "EvalConfig":
        return cls(
            base_url=os.getenv("EVAL_OPENAI_BASE_URL", "").strip()
            or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=(os.getenv("EVAL_OPENAI_API_KEY", "").strip()
                     or os.getenv("OPENAI_API_KEY", "")) or "ollama",
            model=os.getenv("EVAL_OPENAI_MODEL", "").strip()
            or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )


# =========================================================
# Prompt templates (Chinese, matching the project language)
# =========================================================

_CONSISTENCY_PROMPT_SINGLE = """\
你是一个严格的语义评估员。请判断模型的回答是否与期望答案在语义上等价。

规则：
- 重点关注核心意思是否一致，而非字面完全相同
- 允许不同的表达方式、同义替换、精简/详述，只要核心含义一致即算通过
- 允许模型回答比期望答案更详细，只要包含了期望答案的要点即可
- 如果模型回答缺少关键信息、给出矛盾的答案、或答非所问，则不通过
- 不要因为语气、标点、口语化表达等差异而判定不通过

问题：{question}

期望答案：{expected}

模型回答：{reply}

请只输出 1（通过）或 0（不通过），不要输出任何解释。"""

_FORGETTING_PROMPT_SINGLE = """\
你是一个记忆召回评估员。请检查模型回答中包含了多少个期望的知识点。

规则：
- 每个知识点独立计分
- 只要模型回答在语义上包含了这个知识点（允许换用不同说法），就算回忆到了
- 允许更详细或更简略的表达，只要核心含义对就计为回忆到
- 不要因为顺序不同而扣分

问题：{question}

期望知识点：{expected}

模型回答：{reply}

请输出 回忆到的知识点数 / 总知识点数，格式为 N/M（例如 2/3）。只输出这个分数，不要任何解释。"""

_CONSISTENCY_PROMPT_BATCH = """\
你是一个严格的语义评估员。请逐一判断每条模型回答是否与期望答案在语义上等价。

规则：
- 重点关注核心意思是否一致，而非字面完全相同
- 允许不同的表达方式、同义替换、精简/详述，只要核心含义一致即算通过
- 允许模型回答比期望答案更详细，只要包含了期望答案的要点即可
- 如果模型回答缺少关键信息、给出矛盾的答案、或答非所问，则不通过
- 不要因为语气、标点、口语化表达等差异而判定不通过

{items}

请严格按顺序输出一个 JSON 数组，每个元素为 1（通过）或 0（不通过）。
只输出数组，例如 [1, 0, 1]，不要任何解释。"""

_FORGETTING_PROMPT_BATCH = """\
你是一个记忆召回评估员。请逐一检查每条模型回答中包含了多少个期望的知识点。

规则：
- 每个知识点独立计分
- 只要模型回答在语义上包含了这个知识点（允许换用不同说法），就算回忆到了
- 允许更详细或更简略的表达，只要核心含义对就计为回忆到
- 不要因为顺序不同而扣分

{items}

请严格按顺序输出一个 JSON 数组，每个元素为 "N/M" 格式的字符串（例如 ["2/3", "1/1"]）。
只输出数组，不要任何解释。"""


def _format_consistency_items(entries: list[dict[str, str]]) -> str:
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(f"Test {i}:")
        lines.append(f"问题：{e['question']}")
        lines.append(f"期望答案：{e['expected']}")
        lines.append(f"模型回答：{e['reply']}")
        lines.append("")
    return "\n".join(lines)


def _format_forgetting_items(entries: list[dict[str, str]]) -> str:
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(f"Test {i}:")
        lines.append(f"问题：{e['question']}")
        lines.append(f"期望知识点：{e['expected']}")
        lines.append(f"模型回答：{e['reply']}")
        lines.append("")
    return "\n".join(lines)


class AsyncEvaluator:
    """Asynchronous LLM evaluator with batch + fallback support."""

    def __init__(self, config: EvalConfig | None = None):
        cfg = config or EvalConfig.from_env()
        self.client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.model = cfg.model

    async def _call(self, prompt: str, temperature: float = 0.0) -> str:
        async with _SEMAPHORE:
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个语义评估助手，严格按照格式要求输出。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                if self.model.endswith(":7b"):
                    kwargs["extra_body"] = {"options": {"num_gpu_layers": 35}}
                resp = await self.client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                logger.warning("评估 LLM 调用失败: %s", e)
                return ""

    # =========================================================
    # Single-item evaluation (used as fallback from batch)
    # =========================================================

    async def eval_consistency(
        self, question: str, reply: str, expected: str
    ) -> tuple[bool, str]:
        """Evaluate a single consistency item. Returns (passed, eval_method)."""
        prompt = _CONSISTENCY_PROMPT_SINGLE.format(
            question=question, expected=expected, reply=reply
        )
        result = await self._call(prompt)
        if result:
            if "1" in result and "0" not in result.replace("10", ""):
                content = result.replace("0", "")
                if "1" in content:
                    return True, "llm_semantic"
            if result.strip() == "1":
                return True, "llm_semantic"
            if result.strip() == "0":
                return False, "llm_semantic"
            first_digit = re.search(r"[01]", result)
            if first_digit:
                return first_digit.group() == "1", "llm_semantic"
        return expected in reply, "substring_fallback"

    async def eval_forgetting(
        self, question: str, reply: str, expected: str
    ) -> tuple[float, str]:
        """Evaluate a single forgetting item. Returns (score, eval_method)."""
        prompt = _FORGETTING_PROMPT_SINGLE.format(
            question=question, expected=expected, reply=reply
        )
        result = await self._call(prompt)
        if result:
            m = re.search(r"(\d+)\s*/\s*(\d+)", result)
            if m:
                n, d = int(m.group(1)), int(m.group(2))
                if d > 0:
                    return n / d, "llm_semantic"
        return (1.0 if expected in reply else 0.0), "substring_fallback"

    # =========================================================
    # Batch evaluation (preferred, falls back to single-item)
    # =========================================================

    async def batch_eval_consistency(
        self, items: list[dict[str, str]]
    ) -> list[tuple[bool, str]]:
        """Batch consistency evaluation. Each item: {question, reply, expected}.
        Returns list of (passed, eval_method)."""
        if not items:
            return []

        prompt = _CONSISTENCY_PROMPT_BATCH.format(
            items=_format_consistency_items(items)
        )
        result = await self._call(prompt)

        parsed = self._parse_json_array(result, len(items))
        if parsed is not None:
            results = []
            for val in parsed:
                passed = val == 1 or (isinstance(val, str) and val.strip() == "1")
                results.append((passed, "llm_semantic"))
            return results

        logger.warning("批量一致性评估解析失败，回退到逐条评估")
        coros = [
            self.eval_consistency(i["question"], i["reply"], i["expected"])
            for i in items
        ]
        return list(await asyncio.gather(*coros))

    async def batch_eval_forgetting(
        self, items: list[dict[str, str]]
    ) -> list[tuple[float, str]]:
        """Batch forgetting evaluation. Each item: {question, reply, expected}.
        Returns list of (score, eval_method)."""
        if not items:
            return []

        prompt = _FORGETTING_PROMPT_BATCH.format(
            items=_format_forgetting_items(items)
        )
        result = await self._call(prompt)

        parsed = self._parse_json_array(result, len(items))
        if parsed is not None:
            results = []
            for val in parsed:
                if isinstance(val, str):
                    m = re.search(r"(\d+)\s*/\s*(\d+)", val)
                    if m:
                        n, d = int(m.group(1)), int(m.group(2))
                        results.append((n / d if d > 0 else 0.0, "llm_semantic"))
                    else:
                        results.append((0.0, "llm_semantic"))
                elif isinstance(val, (int, float)):
                    results.append((float(val), "llm_semantic"))
                else:
                    results.append((0.0, "llm_semantic"))
            return results

        logger.warning("批量遗忘评估解析失败，回退到逐条评估")
        coros = [
            self.eval_forgetting(i["question"], i["reply"], i["expected"])
            for i in items
        ]
        return list(await asyncio.gather(*coros))

    # =========================================================
    # Parsing helpers
    # =========================================================

    @staticmethod
    def _parse_json_array(
        text: str, expected_len: int
    ) -> list[Any] | None:
        """Try to parse a JSON array from LLM output. Returns None on failure."""
        if not text:
            return None
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        if not m:
            return None
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and len(arr) == expected_len:
                return arr
        except json.JSONDecodeError:
            pass
        return None