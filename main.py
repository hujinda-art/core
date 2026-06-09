"""Core test runner — saves raw evaluation results for later analysis."""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from src.agents.Agent import AsyncAgent
from src.evaluator import AsyncEvaluator
from src.memory import __all__ as STRATEGY_NAMES

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("AILongTermMem")

RESULTS_DIR = os.path.join(_CORE_DIR, "results")


# =========================================================
# 测试加载与执行
# =========================================================

def load_test_file(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


async def run_tests(
    agent: AsyncAgent,
    test_groups: list[dict],
    evaluator: AsyncEvaluator,
) -> list[dict]:
    """运行测试，evaluation 轮延迟到组结束后统一发给 LLM 语义评估。"""
    all_results: list[dict] = []

    for group in test_groups:
        test_id = group.get("id", "unknown")
        test_type = group.get("type", "unknown")
        turns = group.get("turns", [])

        agent.mem.reset()
        evaluations: list[dict] = []
        eval_items: list[dict[str, str]] = []
        turn_map: list[int] = []
        eval_counter = 0

        for turn in turns:
            role = turn.get("role", "")
            q = turn.get("q", "")
            expected = turn.get("expected", "")

            if role == "introduction":
                await agent.chat(q)
                continue

            if role == "evaluation":
                eval_counter += 1
                reply = await agent.chat(q)
                evaluations.append({
                    "turn": eval_counter,
                    "q": q,
                    "reply": reply,
                    "expected": expected,
                })
                if expected:
                    eval_items.append({
                        "question": q,
                        "reply": reply,
                        "expected": expected,
                    })
                    turn_map.append(eval_counter - 1)
                continue

            await agent.chat(q)

        # Batch semantic evaluation
        if eval_items:
            test_type_key = test_type.lower() if test_type else "consistency"
            if test_type_key == "forgetting":
                batch_results = await evaluator.batch_eval_forgetting(eval_items)
            elif test_type_key == "programming":
                batch_results = await evaluator.batch_eval_programming(eval_items)
            else:
                batch_results = await evaluator.batch_eval_consistency(eval_items)

            for idx, (value, method) in enumerate(batch_results):
                eidx = turn_map[idx]
                if test_type_key in ("forgetting", "programming"):
                    evaluations[eidx]["passed"] = value >= 0.5
                    evaluations[eidx]["score"] = value
                    evaluations[eidx]["eval_method"] = method
                else:
                    evaluations[eidx]["passed"] = value
                    evaluations[eidx]["eval_method"] = method

        passed = sum(1 for e in evaluations if e.get("passed", False))
        total = len(evaluations)
        score = passed / total if total else 0.0
        if any("score" in e for e in evaluations):
            score = sum(e.get("score", 0.0) for e in evaluations) / total if total else 0.0

        logger.info(
            "[%s] %s | %d/%d | score=%.4f",
            test_id, test_type, passed, total, score,
        )

        all_results.append({
            "id": test_id,
            "type": test_type,
            "evaluations": evaluations,
        })

    return all_results


# =========================================================
# 结果保存
# =========================================================

def save_results(
    strategy_name: str,
    test_file_name: str,
    results: list[dict],
) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_name = test_file_name.replace(".json", "")
    backend = os.getenv("LONG_MEM_BACKEND", "chromadb").strip().lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{strategy_name}_{safe_name}_{backend}_{ts}.json"
    path = os.path.join(RESULTS_DIR, filename)

    payload = {
        "strategy": strategy_name,
        "backend": backend,
        "test_file": test_file_name,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("结果已保存: %s", path)
    return path


# =========================================================
# 主流程
# =========================================================

def resolve_test_files(test_files_arg: list[str]) -> list[str]:
    if test_files_arg:
        return test_files_arg
    env = os.getenv("TEST_FILES", "").strip()
    if env:
        return [f.strip() for f in env.split(",") if f.strip()]
    return ["quick_test.json"]


def resolve_strategies(strategies_arg: list[str]) -> list[str]:
    if strategies_arg:
        return strategies_arg
    env = os.getenv("MEMORY_STRATEGIES", "").strip()
    if env:
        return [s.strip() for s in env.split(",") if s.strip()]
    return [name for name in STRATEGY_NAMES if name != "BaseMem"]


async def _run_single(
    strategy_name: str,
    test_file_name: str,
    test_ids_filter: set[str] | None = None,
    no_save: bool = False,
) -> list[dict] | None:
    """运行单个（策略, 测试文件）组合。返回 results 列表（不保存时为 None）。"""
    test_path = os.path.join(_CORE_DIR, "test", test_file_name)
    if not os.path.exists(test_path):
        logger.warning("测试文件不存在: %s", test_path)
        return None

    test_groups = load_test_file(test_path)
    if test_ids_filter:
        matched = [g for g in test_groups if g.get("id") in test_ids_filter]
        skipped = [g.get("id") for g in test_groups if g.get("id") not in test_ids_filter]
        if skipped:
            logger.info("跳过测试组（未匹配 --test-id）: %s", skipped)
        test_groups = matched
        if not test_groups:
            logger.warning("没有匹配的测试组，跳过")
            return None

    from src.memory import LongMem, NoMem, ShortMem, CombinedMem
    mem_class = {"NoMem": NoMem, "ShortMem": ShortMem, "LongMem": LongMem, "CombinedMem": CombinedMem}.get(strategy_name)
    if mem_class is None:
        logger.warning("未知策略: %s", strategy_name)
        return None

    backend = os.getenv("LONG_MEM_BACKEND", "chromadb").strip().lower()
    embed = os.getenv("HNSW_EMBEDDING", "").strip().lower()
    suffix = f"{backend}_{embed}" if embed else backend
    session_id = f"{os.path.splitext(test_file_name)[0]}_{strategy_name}_{suffix}"
    try:
        mem = mem_class(session_id=session_id)
    except TypeError:
        mem = mem_class()

    agent = AsyncAgent(mem_module=mem)
    evaluator = AsyncEvaluator()
    results = await run_tests(agent, test_groups, evaluator)

    if no_save:
        return results
    save_results(strategy_name, test_file_name, results)
    return None


async def main_async(args: argparse.Namespace) -> None:
    test_files = resolve_test_files(args.test_file)
    strategy_names = resolve_strategies(args.strategies)
    backend = os.getenv("LONG_MEM_BACKEND", "chromadb").strip().lower()
    logger.info("后端: %s | 策略: %s | 测试文件: %s", backend, strategy_names, test_files)

    test_ids_filter: set[str] | None = set(args.test_id) if args.test_id else None

    async def _wrapper(s: str, tf: str) -> None:
        await _run_single(s, tf, test_ids_filter=test_ids_filter, no_save=args.no_save)

    tasks = [_wrapper(s, tf) for s in strategy_names for tf in test_files]
    await asyncio.gather(*tasks)

    print()
    print("=" * 60)
    if args.no_save:
        print("【运行完成】（未保存，--no-save 已启用）")
    else:
        print("【运行完成】结果已保存到 core/results/")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Agent 记忆策略评测")
    parser.add_argument("--test-file", action="append", dest="test_file", default=[], help="测试文件名（可多次）")
    parser.add_argument("--strategies", action="append", dest="strategies", default=[], help="记忆策略名（可多次）")
    parser.add_argument("--test-id", action="append", dest="test_id", default=[], help="只运行指定 ID 的测试组（可多次）")
    parser.add_argument("--no-save", action="store_true", help="不保存结果文件，仅输出到日志")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()