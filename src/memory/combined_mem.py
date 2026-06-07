from src.agents.message_dto import MessageDTO, Role
from src.agents.message_enum import Message
from src.memory.base_mem import BaseMem
from src.memory.long_mem import LongMem
from src.memory.short_mem import ShortMem


class CombinedMem(BaseMem):
    def __init__(self, session_id: str | None = None):
        self.short_mem = ShortMem()
        self.long_mem = LongMem(session_id=session_id)

    def get_mem(self, q: str) -> list[MessageDTO]:
        combined: list[MessageDTO] = []

        long_results = self.long_mem.get_mem(q)
        if long_results:
            context = "\n".join(f"{m.role.value}: {m.content}" for m in long_results)
            combined.append(MessageDTO(
                role=Role.SYSTEM,
                content=f"{Message.LONG_MEM_CONTEXT.value}\n{context}",
            ))

        combined.extend(self.short_mem.get_mem(q))
        return combined

    def update_mem(self, q: str, ans: str) -> None:
        self.short_mem.update_mem(q, ans)
        self.long_mem.update_mem(q, ans)

    def reset(self) -> None:
        self.short_mem.reset()
        self.long_mem.reset()
