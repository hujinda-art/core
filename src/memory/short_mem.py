import logging

import tiktoken

from src.agents.message_dto import MessageDTO, Role
from src.memory.base_mem import BaseMem
from src.memory.constant import MAX_CONTEXT_WINDOW, SECOND_WATER_LEVEL

logger = logging.getLogger("AILongTermMem")

_MODEL = "gpt-4o"


class ShortMem(BaseMem):
    def __init__(self):
        self.mem: list[MessageDTO] = []
        self.enc = tiktoken.encoding_for_model(_MODEL)

    def get_mem(self, q: str) -> list[MessageDTO]:
        return self.mem

    def update_mem(self, q: str, ans: str) -> None:
        self.mem.append(MessageDTO(role=Role.USER, content=q))
        self.mem.append(MessageDTO(role=Role.ASSISTANT, content=ans))

        total_tokens = self._count_tokens()
        threshold = MAX_CONTEXT_WINDOW * SECOND_WATER_LEVEL
        if total_tokens > threshold:
            logger.warning("短期记忆超水位线，触发滑动窗口压缩")
            self._compress_mem()
            logger.info("压缩后约 %d tokens", self._count_tokens())

    def _count_tokens(self) -> int:
        return sum(len(self.enc.encode(m.content)) for m in self.mem)

    def _compress_mem(self) -> None:
        threshold = MAX_CONTEXT_WINDOW * SECOND_WATER_LEVEL
        while self.mem and self._count_tokens() > threshold:
            self.mem.pop(0)

    def reset(self) -> None:
        self.mem.clear()
