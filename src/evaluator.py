"""LLM-based semantic evaluator for test results.

Uses a separate OpenAI-compatible client (configurable via EVAL_OPENAI_* env vars)
to judge whether model answers match expected answers in meaning, not just in
exact string matching.

Prompts are loaded from markdown files in the prompts/ directory.
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROMPTS_DIR = os.getenv("PROMPTS_DIR", os.path.join(_THIS_DIR, "prompts"))


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
# Prompt loading from markdown files
# =========================================================

def _load_prompts_from_md(filename: str) -> dict[str, str]:
    """Load prompt templates from a markdown file.

    Expects ## section_name headers followed by the prompt text.
    Returns a dict mapping section names to stripped prompt text.
    """
    path = os.path.join(_PROMPTS_DIR, filename)
    if not os.path.exists(path):
        logger.warning("Prompt file not found: %s", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    prompts = {}
    current_name = None
    current_lines: list[str] = []

    for line in content.split("\n"):
        if line.startswith("## "):
            if current_name is not None:
                prompts[current_name] = "\n".join(current_lines).strip()
            current_name = line[3:].strip()
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        prompts[current_name] = "\n".join(current_lines).strip()

    return prompts


_BASIC_PROMPTS = _load_prompts_from_md(os.getenv("EVAL_BASIC_PROMPTS", "evaluator_basic.md"))
_PROGRAMMING_PROMPTS = _load_prompts_from_md(os.getenv("EVAL_PROGRAMMING_PROMPTS", "evaluator_programming.md"))


# =========================================================
# Item formatters
# =========================================================

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


def _format_programming_items(entries: list[dict[str, str]]) -> str:
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(f"Test {i}:")
        lines.append(f"问题：{e['question']}")
        lines.append(f"期望答案要点：{e['expected']}")
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
        template = _BASIC_PROMPTS.get("consistency_single", "")
        if not template:
            return expected in reply, "substring_fallback"
        prompt = template.format(question=question, expected=expected, reply=reply)
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
        template = _BASIC_PROMPTS.get("forgetting_single", "")
        if not template:
            return (1.0 if expected in reply else 0.0), "substring_fallback"
        prompt = template.format(question=question, expected=expected, reply=reply)
        result = await self._call(prompt)
        if result:
            m = re.search(r"(\d+)\s*/\s*(\d+)", result)
            if m:
                n, d = int(m.group(1)), int(m.group(2))
                if d > 0:
                    return n / d, "llm_semantic"
        return (1.0 if expected in reply else 0.0), "substring_fallback"

    async def eval_programming(
        self, question: str, reply: str, expected: str
    ) -> tuple[float, str]:
        """Evaluate a single programming item. Returns (score, eval_method).
        Score is 0.0-1.0 reflecting code correctness + naming consistency."""
        template = _PROGRAMMING_PROMPTS.get("programming_single", "")
        if not template:
            return (1.0 if expected in reply else 0.0), "substring_fallback"
        prompt = template.format(question=question, expected=expected, reply=reply)
        result = await self._call(prompt)
        if result:
            m = re.search(r"(\d+\.?\d*)", result)
            if m:
                score = float(m.group(1))
                score = max(0.0, min(1.0, score))
                return score, "llm_programming"
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

        template = _BASIC_PROMPTS.get("consistency_batch", "")
        if not template:
            coros = [
                self.eval_consistency(i["question"], i["reply"], i["expected"])
                for i in items
            ]
            return list(await asyncio.gather(*coros))

        prompt = template.format(items=_format_consistency_items(items))
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

        template = _BASIC_PROMPTS.get("forgetting_batch", "")
        if not template:
            coros = [
                self.eval_forgetting(i["question"], i["reply"], i["expected"])
                for i in items
            ]
            return list(await asyncio.gather(*coros))

        prompt = template.format(items=_format_forgetting_items(items))
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

    async def batch_eval_programming(
        self, items: list[dict[str, str]]
    ) -> list[tuple[float, str]]:
        """Batch programming evaluation. Each item: {question, reply, expected}.
        Returns list of (score, eval_method). Score is 0.0-1.0."""
        if not items:
            return []

        template = _PROGRAMMING_PROMPTS.get("programming_batch", "")
        if not template:
            coros = [
                self.eval_programming(i["question"], i["reply"], i["expected"])
                for i in items
            ]
            return list(await asyncio.gather(*coros))

        prompt = template.format(items=_format_programming_items(items))
        result = await self._call(prompt)

        parsed = self._parse_json_array(result, len(items))
        if parsed is not None:
            results = []
            for val in parsed:
                if isinstance(val, (int, float)):
                    score = max(0.0, min(1.0, float(val)))
                    results.append((score, "llm_programming"))
                elif isinstance(val, str):
                    m = re.search(r"(\d+\.?\d*)", val)
                    if m:
                        score = max(0.0, min(1.0, float(m.group(1))))
                        results.append((score, "llm_programming"))
                    else:
                        results.append((0.0, "llm_programming"))
                else:
                    results.append((0.0, "llm_programming"))
            return results

        logger.warning("批量编程评估解析失败，回退到逐条评估")
        coros = [
            self.eval_programming(i["question"], i["reply"], i["expected"])
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