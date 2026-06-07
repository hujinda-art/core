"""
HnswMemoryStore → LongMem 适配层。

实现与 AILongTermMemPj/src/memory/long_mem.py 中 LongMem 完全相同的公开接口，
可供 AILongTermMemPj 直接替换 ChromaDB LongMem 使用。
"""

import logging
import os
import shutil
import uuid

from hnsw_memorystore.embedding import BgeM3Embedding
from hnsw_memorystore.store import HnswMemoryStore

logger = logging.getLogger("AILongTermMem")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AILongTermMemPj"))
_GLOBAL_SESSION_TAG = "__global__"


def _memorystore_dir() -> str:
    override = os.getenv("MEMORYSTORE_PATH", "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(_PROJECT_ROOT, "memorystore")


_EMBEDDING_INSTANCE = None


def _get_default_embedding():
    global _EMBEDDING_INSTANCE
    if _EMBEDDING_INSTANCE is None:
        device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") else "cpu"
        _EMBEDDING_INSTANCE = BgeM3Embedding(device=device)
    return _EMBEDDING_INSTANCE


class _Role:
    def __init__(self, value: str):
        self.value = value


class _MessageDTO:
    """与 AILongTermMemPj MessageDTO + Role 兼容的轻量替代。"""
    def __init__(self, role: str, content: str):
        self.role = _Role(role)
        self.content = content


class LongMemHNSW:
    """长期记忆策略（HNSW 实现）：替换 ChromaDB LongMem，接口完全一致。

    用法:
        mem = LongMemHNSW(session_id="xxx")
        mem.get_mem(q)
        mem.update_mem(q, ans)
        mem.list_mem_items()
        mem.delete_mem_item(item_id)
        mem.reset()
    """

    _logged_store: str | None = None

    def __init__(
        self,
        session_id: str | None = None,
        embedding_model=None,
    ) -> None:
        self._session_key = (session_id or _GLOBAL_SESSION_TAG).strip() or _GLOBAL_SESSION_TAG

        store = _memorystore_dir()
        os.makedirs(store, exist_ok=True)
        if LongMemHNSW._logged_store != store:
            logger.info("LongMemHNSW 持久化目录: %s", store)
            LongMemHNSW._logged_store = store

        self.store_path = os.path.join(store, f"long_mem_hnsw_{self._session_key}")
        self.collection_name = f"long_mem_hnsw_{self._session_key}"

        self._embed = embedding_model or _get_default_embedding()
        dim = self._embed.embed(["test"]).shape[1]

        self._store = HnswMemoryStore(
            dim=dim,
            space="cosine",
            max_elements=100_000,
            M=16,
            ef_construction=200,
            store_path=self.store_path,
            embedding_model=self._embed,
        )

        # 用于维护 uuid → int_id 映射的 SQLite 表（在 store 的同 DB 中）
        self._ensure_uuid_table()

    def _ensure_uuid_table(self) -> None:
        self._store._db.execute(
            """CREATE TABLE IF NOT EXISTS uuid_map (
                uuid TEXT PRIMARY KEY,
                int_id INTEGER NOT NULL
            )"""
        )
        self._store._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_uuid_int ON uuid_map(int_id)"
        )
        self._store._db.commit()

    def _uuid_to_int(self, item_id: str) -> int | None:
        cur = self._store._db.execute(
            "SELECT int_id FROM uuid_map WHERE uuid=?", (item_id,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def _int_to_uuid(self, int_id: int) -> str | None:
        cur = self._store._db.execute(
            "SELECT uuid FROM uuid_map WHERE int_id=?", (int_id,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def _save_uuid(self, item_uuid: str, int_id: int) -> None:
        self._store._db.execute(
            "INSERT OR REPLACE INTO uuid_map (uuid, int_id) VALUES (?, ?)",
            (item_uuid, int_id),
        )
        self._store._db.commit()

    def _remove_uuid(self, int_id: int) -> None:
        self._store._db.execute("DELETE FROM uuid_map WHERE int_id=?", (int_id,))
        self._store._db.commit()

    # =========================================================
    # BaseMem 接口
    # =========================================================

    def get_mem(self, q: str) -> list:
        """从长期记忆中检索最相关的消息。"""
        if self._store.count() == 0:
            return []
        results = self._store.search(q, k=5, session_id=self._session_key)
        if not results:
            return []
        messages = []
        for r in results:
            meta = r.metadata or {}
            ans = meta.get("answer", "")
            full = f"[用户] {r.text}\n[助手] {ans}"
            messages.append(_MessageDTO(role="user", content=full))
        logger.info("  [HNSW 向量搜索] 根据提问 '%s' 检索出 %d 条相关的长期记忆", q, len(messages))
        return messages

    def update_mem(self, q: str, ans: str) -> None:
        """将本轮对话合并为一条记忆写入向量库。只嵌入用户问题，但存储完整的问答。"""
        item_uuid = uuid.uuid4().hex
        items = self._store.add_memories(
            texts=[q],
            session_ids=[self._session_key],
            types=["turn"],
            metadatas=[{"role": "pair", "uuid": item_uuid, "answer": ans}],
        )
        self._save_uuid(item_uuid, items[0].id)

    def clear_mem(self) -> None:
        """清空本会话的向量记录：删除整个 store 目录并重建。"""
        dim = self._store.dim
        space = self._store.space
        max_elements = self._store._max_elements
        M = self._store._M
        ef_construction = self._store._ef_construction
        self._store.close()
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path, ignore_errors=True)
        os.makedirs(self.store_path, exist_ok=True)
        self._store = HnswMemoryStore(
            dim=dim,
            space=space,
            max_elements=max_elements,
            M=M,
            ef_construction=ef_construction,
            store_path=self.store_path,
            embedding_model=self._embed,
        )
        self._ensure_uuid_table()

    def reset(self) -> None:
        """重置长期记忆。"""
        self.clear_mem()

    # =========================================================
    # 管理接口
    # =========================================================

    def list_mem_items(self) -> list[dict]:
        """列出本会话的全部记录。"""
        all_items = self._store.list_all(session_id=self._session_key)
        out = []
        for item in all_items:
            item_uuid = self._int_to_uuid(item.id) or ""
            meta = item.metadata or {}
            out.append({
                "id": item_uuid,
                "role": meta.get("role", "pair"),
                "type": meta.get("type", "turn"),
                "content": item.text,
            })
        return out

    def delete_mem_item(self, item_id: str) -> bool:
        """按 UUID 删除一条记录。"""
        int_id = self._uuid_to_int(item_id)
        if int_id is None:
            logger.warning("delete_mem_item: uuid=%s 未找到", item_id[:12])
            return False
        ok = self._store.delete_memory(int_id)
        if ok:
            self._remove_uuid(int_id)
            logger.info("已删除长期记忆条目 uuid=%s… session=%s", item_id[:12], self._session_key[:8])
        return ok


def nuclear_reset_hnsw_collection(agents_mem: list) -> None:
    """清空所有 LongMemHNSW store 目录并重实例化（相当于 ChromaDB 版 nuclear_reset）。"""
    store = _memorystore_dir()
    for root in agents_mem:
        if isinstance(root, LongMemHNSW):
            targets = [root]
        elif hasattr(root, "long_mem") and isinstance(getattr(root, "long_mem", None), LongMemHNSW):
            targets = [root.long_mem]
        else:
            continue
        for lm in targets:
            dim = lm._store.dim
            space = lm._store.space
            max_elements = lm._store._max_elements
            M = lm._store._M
            ef_construction = lm._store._ef_construction
            try:
                lm._store.close()
            except Exception:
                pass
            if os.path.exists(lm.store_path):
                shutil.rmtree(lm.store_path, ignore_errors=True)
            os.makedirs(lm.store_path, exist_ok=True)
            lm._store = HnswMemoryStore(
                dim=dim, space=space,
                max_elements=max_elements, M=M, ef_construction=ef_construction,
                store_path=lm.store_path, embedding_model=lm._embed,
            )
            lm._ensure_uuid_table()
    logger.info("已清空并重建所有 HNSW 长期记忆存储目录")
