from src.memory.base_mem import BaseMem


class NoMem(BaseMem):
    def get_mem(self, q: str) -> list:
        return []

    def update_mem(self, q: str, ans: str) -> None:
        return

    def reset(self) -> None:
        pass
