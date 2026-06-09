import os
from enum import Enum

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.join(_FILE_DIR, "..", "..")
_PROMPTS_DIR = os.getenv("PROMPTS_DIR", os.path.join(_CORE_DIR, "prompts"))


def _load_agent_prompts() -> dict[str, str]:
    """Load agent prompts from a markdown file (configured via AGENT_PROMPTS env var).

    Expects ## section_name headers followed by the prompt text.
    Returns a dict mapping section names to stripped prompt text.
    """
    filename = os.getenv("AGENT_PROMPTS", "agent.md")
    path = os.path.join(_PROMPTS_DIR, filename)
    if not os.path.exists(path):
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


_AGENT_PROMPTS = _load_agent_prompts()


class Message(Enum):
    SYSTEM_PROMPT = _AGENT_PROMPTS.get(
        "system_prompt",
        "你是一个有记忆的AI助手，你的回答风格应该保持简洁冷静，不要过多解释，给出简短回答："
    )
    LONG_MEM_CONTEXT = "[以下是长期记忆回顾]"
    MEM_WRITE_PROCESS = _AGENT_PROMPTS.get(
        "mem_write_process",
        "你是一个记忆管理助手，负责两件事：\n\n"
        "【任务一】从新对话中提炼最终结论，去除中途的纠正、否定和噪音，只保留最终成立的事实。\n"
        "输出格式：CLEAN_Q: <提炼后的用户陈述>\n\n"
        "【任务二】判断已存储的记忆中是否有被新对话替代或矛盾的条目。\n"
        "如有，输出：DELETE_IDS: <id1,id2,...>\n"
        "如无矛盾，输出：DELETE_IDS: 无\n\n"
        "注意：已存储的记忆可能包含完整对话或对话片段（格式：「用户问题 | 片段内容」），\n"
        "请根据「|」之前的部分判断是否矛盾，「|」之后是回答的具体片段。\n\n"
        "已存储的记忆（格式：[ID] 内容）：\n{existing}\n\n"
        "新的对话内容：\nUser: {q}\nAssistant: {ans}\n\n"
        "严格按格式输出两行，不要任何额外解释。"
    )
    CLEAN_Q_EXTRACT = _AGENT_PROMPTS.get(
        "clean_q_extract",
        "从以下用户陈述中提炼最终结论，去除中途的纠正和否定，只保留最终成立的事实。\n"
        "若无需提炼（陈述本身已清晰），原文返回。只输出结果，不要解释。\n\n"
        "用户陈述：{q}"
    )