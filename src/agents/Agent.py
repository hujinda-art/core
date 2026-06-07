import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.agents.message_dto import MessageDTO
from src.agents.message_enum import Message
from src.memory.base_mem import BaseMem

load_dotenv()


class AsyncAgent:
    def __init__(self, mem_module: BaseMem):
        api_key = os.getenv("OPENAI_API_KEY", "") or "ollama"
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.mem = mem_module

    async def chat(self, q: str) -> str:
        messages = self._build_messages(q, self.mem.get_mem(q))
        kwargs = dict(model=self.model, messages=messages)
        if self.model.endswith(":7b"):
            kwargs["extra_body"] = {"options": {"num_gpu_layers": 35}}
        response = await self.client.chat.completions.create(**kwargs)
        ans = response.choices[0].message.content or ""
        self.mem.update_mem(q, ans)
        return ans

    def _build_messages(self, q: str, mem: list[MessageDTO]) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": Message.SYSTEM_PROMPT.value}
        ]
        for item in mem:
            messages.append({"role": item.role.value, "content": item.content})
        messages.append({"role": "user", "content": q})
        return messages
