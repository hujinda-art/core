from abc import ABCMeta, abstractmethod

from src.agents.message_dto import MessageDTO


class BaseMem(metaclass=ABCMeta):
    @abstractmethod
    def get_mem(self, q: str) -> list[MessageDTO]:
        ...

    @abstractmethod
    def update_mem(self, q: str, ans: str) -> None:
        ...

    def reset(self) -> None:
        pass
