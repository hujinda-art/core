"""
基于 HNSW 的向量记忆库，搭配 SQLite 元数据存储。

遵循 docs/AGENT_MEMORY_BUILD.md 的设计：
  - hnswlib 负责 ANN 检索（返回 id + 距离）
  - SQLite 负责 id → {text, type, session_id, created_at, ...} 回表
"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import hnswlib
import numpy as np

from hnsw_memorystore.embedding import BaseEmbedding

logger = logging.getLogger("HnswMemoryStore")


@dataclass
class MemoryItem:
    id: int
    text: str
    type: str = "general"
    session_id: str = ""
    source: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class HnswMemoryStore:
    """HNSW 向量记忆库。

    架构:
      hnswlib.Index —— ANN 检索，返回 int id
      SQLite        —— id → 元数据（text, type, session_id, ...）
    """

    def __init__(
        self,
        dim: int = 128,
        space: str = "cosine",
        max_elements: int = 100_000,
        M: int = 16,
        ef_construction: int = 200,
        random_seed: int = 42,
        store_path: str = "./memorystore_data",
        embedding_model: BaseEmbedding | None = None,
    ):
        self.dim = dim
        self.space = space
        self._max_elements = max_elements
        self._M = M
        self._ef_construction = ef_construction
        self._random_seed = random_seed
        self._store_path = store_path
        assert embedding_model is not None, "HnswMemoryStore 需要传入 embedding_model"
        self._embed = embedding_model
        self._lock = threading.Lock()

        os.makedirs(store_path, exist_ok=True)

        # --- HNSW index ---
        index_path = os.path.join(store_path, "hnsw_index.bin")
        self.index: hnswlib.Index = hnswlib.Index(space=space, dim=dim)
        self._default_ef = max(80, ef_construction)
        self._M = M
        self._ef_construction = ef_construction
        need_rebuild = False
        if os.path.exists(index_path):
            self.index.load_index(index_path, max_elements=max_elements)
            current_count = self.index.get_current_count()
            # 初始化 SQLite 以检查数据一致性
            db_path_check = os.path.join(store_path, "metadata.db")
            if os.path.exists(db_path_check):
                try:
                    db_check = sqlite3.connect(db_path_check)
                    sqlite_count = db_check.execute("SELECT COUNT(*) FROM memory_meta").fetchone()[0]
                    db_check.close()
                except Exception:
                    sqlite_count = 0
            else:
                sqlite_count = 0
            if current_count == 0 and sqlite_count > 0:
                logger.warning("索引为空但 SQLite 有 %d 条记录，将重建索引", sqlite_count)
                need_rebuild = True
            elif current_count < sqlite_count * 0.5:
                logger.warning("索引 %d 条 << SQLite %d 条，可能不同步，将重建索引", current_count, sqlite_count)
                need_rebuild = True
            if need_rebuild:
                self.index = hnswlib.Index(space=space, dim=dim)
                self.index.init_index(
                    max_elements=max_elements,
                    M=M,
                    ef_construction=ef_construction,
                    random_seed=random_seed,
                    allow_replace_deleted=False,
                )
                self.index.set_ef(self._default_ef)
            else:
                self._default_ef = max(80, min(ef_construction, current_count)) if current_count > 0 else 80
                self.index.set_ef(self._default_ef)
                logger.info("加载已有索引: %s (max_elements=%d, current=%d, ef=%d)", index_path, max_elements, current_count, self._default_ef)
        else:
            self.index.init_index(
                max_elements=max_elements,
                M=M,
                ef_construction=ef_construction,
                random_seed=random_seed,
                allow_replace_deleted=False,
            )
            self.index.set_ef(self._default_ef)
            # 检查是否 SQLite 有数据但索引文件丢失（未正常 close）
            db_path_check = os.path.join(store_path, "metadata.db")
            if os.path.exists(db_path_check):
                try:
                    db_check = sqlite3.connect(db_path_check)
                    sqlite_count = db_check.execute("SELECT COUNT(*) FROM memory_meta").fetchone()[0]
                    db_check.close()
                except Exception:
                    sqlite_count = 0
            else:
                sqlite_count = 0
            if sqlite_count > 0:
                logger.warning("索引文件不存在但 SQLite 有 %d 条记录，将重建索引", sqlite_count)
                need_rebuild = True
            else:
                logger.info("初始化新索引: dim=%d, space=%s, M=%d, ef_construction=%d", dim, space, M, ef_construction)

        # --- SQLite ---
        db_path = os.path.join(store_path, "metadata.db")
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS memory_meta (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                type TEXT DEFAULT 'general',
                session_id TEXT DEFAULT '',
                source TEXT DEFAULT '',
                created_at REAL DEFAULT (strftime('%s','now')),
                metadata TEXT DEFAULT '{}'
            )"""
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON memory_meta(session_id)"
        )
        self._db.commit()
        self._next_id: int = self._load_next_id()
        logger.info("SQLite 元数据库就绪: %s | 当前最大 id=%d", db_path, self._next_id - 1)

        # 如果索引与数据不同步，从 SQLite 重建索引
        if need_rebuild:
            self._rebuild_index_from_sqlite()

    # =========================================================
    # 写入
    # =========================================================

    def add_memory(
        self,
        text: str,
        vector: np.ndarray | None = None,
        type_: str = "general",
        session_id: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        items = self.add_memories(
            texts=[text],
            vectors=None if vector is None else np.expand_dims(vector, axis=0),
            types=[type_],
            session_ids=[session_id],
            sources=[source],
            metadatas=[metadata or {}],
        )
        return items[0]

    def add_memories(
        self,
        texts: list[str],
        vectors: np.ndarray | None = None,
        types: list[str] | None = None,
        session_ids: list[str] | None = None,
        sources: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[MemoryItem]:
        n = len(texts)

        if vectors is None:
            vectors = self._embed.embed(texts)
        else:
            vectors = np.asarray(vectors, dtype=np.float32)

        types = types or (["general"] * n)
        session_ids = session_ids or ([""] * n)
        sources = sources or ([""] * n)
        metadatas = metadatas or ([{}] * n)

        with self._lock:
            ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
            self._next_id += n

            self.index.add_items(vectors, ids, num_threads=4)

            current_count = self.index.get_current_count()
            self._default_ef = max(80, min(self._default_ef, current_count))
            self.index.set_ef(self._default_ef)

            now = time.time()
            rows = []
            for i in range(n):
                rows.append(
                    (
                        int(ids[i]),
                        texts[i],
                        types[i],
                        session_ids[i],
                        sources[i],
                        now,
                        json.dumps(metadatas[i], ensure_ascii=False),
                    )
                )
            self._db.executemany(
                "INSERT OR REPLACE INTO memory_meta (id, text, type, session_id, source, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._db.commit()

            self._save_next_id()

        items = [
            MemoryItem(
                id=int(ids[i]),
                text=texts[i],
                type=types[i],
                session_id=session_ids[i],
                source=sources[i],
                created_at=now,
                metadata=metadatas[i],
            )
            for i in range(n)
        ]
        logger.info("写入 %d 条记忆 (id %d~%d)", n, ids[0], ids[-1])
        return items

    # =========================================================
    # 检索
    # =========================================================

    def search(
        self,
        query: str | np.ndarray,
        k: int = 10,
        ef: int | None = None,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        if isinstance(query, str):
            q_vec = self._embed.embed([query])
        else:
            q_vec = np.asarray(query, dtype=np.float32)
            if q_vec.ndim == 1:
                q_vec = np.expand_dims(q_vec, axis=0)

        n_total = self.count()
        k = min(k, n_total) if n_total > 0 else 0
        if k == 0:
            return []

        with self._lock:
            actual_ef = ef if ef is not None else max(80, k * 4)
            actual_ef = max(actual_ef, k + 1)
            current_count = self.index.get_current_count()
            if current_count > 0:
                actual_ef = min(actual_ef, current_count)
            self.index.set_ef(actual_ef)

            labels, distances = self.index.knn_query(q_vec, k=k)

        results: list[MemoryItem] = []
        for mem_id, dist in zip(labels[0], distances[0]):
            if mem_id == -1:
                continue
            meta = self._get_meta(int(mem_id))
            if meta is None:
                continue
            if session_id and meta["session_id"] != session_id:
                continue
            results.append(
                MemoryItem(
                    id=int(mem_id),
                    text=meta["text"],
                    type=meta["type"],
                    session_id=meta["session_id"],
                    source=meta["source"],
                    created_at=meta["created_at"],
                    metadata=json.loads(meta["metadata"]) if isinstance(meta["metadata"], str) else meta["metadata"],
                    score=float(dist),
                )
            )

        logger.info("检索 '%s' (k=%d) → %d 条结果", query if isinstance(query, str) else "<vector>", k, len(results))
        return results

    # =========================================================
    # 更新 / 删除
    # =========================================================

    def update_memory(self, mem_id: int, text: str, vector: np.ndarray | None = None) -> bool:
        with self._lock:
            cur = self._db.execute("SELECT id FROM memory_meta WHERE id=?", (mem_id,))
            if cur.fetchone() is None:
                return False
            if vector is not None:
                v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
                self.index.add_items(v, np.array([mem_id], dtype=np.int64), num_threads=1)
            self._db.execute("UPDATE memory_meta SET text=? WHERE id=?", (text, mem_id))
            self._db.commit()
        logger.info("更新记忆 id=%d", mem_id)
        return True

    def delete_memory(self, mem_id: int) -> bool:
        with self._lock:
            cur = self._db.execute("SELECT id FROM memory_meta WHERE id=?", (mem_id,))
            if cur.fetchone() is None:
                return False
            self.index.mark_deleted(mem_id)
            self._db.execute("DELETE FROM memory_meta WHERE id=?", (mem_id,))
            self._db.commit()
            current_count = self.index.get_current_count()
            self._default_ef = max(80, min(self._default_ef, current_count))
            self.index.set_ef(self._default_ef)
        logger.info("删除记忆 id=%d", mem_id)
        return True

    def delete_by_session(self, session_id: str) -> int:
        with self._lock:
            cur = self._db.execute("SELECT id FROM memory_meta WHERE session_id=?", (session_id,))
            ids = [row[0] for row in cur.fetchall()]
            for mid in ids:
                self.index.mark_deleted(mid)
            self._db.execute("DELETE FROM memory_meta WHERE session_id=?", (session_id,))
            self._db.commit()
            current_count = self.index.get_current_count()
            self._default_ef = max(80, min(self._default_ef, current_count))
            self.index.set_ef(self._default_ef)
        logger.info("删除 session='%s' 的 %d 条记忆", session_id, len(ids))
        return len(ids)

    # =========================================================
    # 持久化
    # =========================================================

    def save(self) -> None:
        path = os.path.join(self._store_path, "hnsw_index.bin")
        with self._lock:
            self.index.save_index(path)
            self._db.commit()
        logger.info("索引已保存: %s", path)

    def close(self) -> None:
        self.save()
        self._db.close()
        logger.info("记忆库已关闭")

    # =========================================================
    # 内部方法
    # =========================================================

    def _rebuild_index_from_sqlite(self) -> None:
        """从 SQLite 元数据重建 HNSW 索引（当索引文件损坏或不同步时）。"""
        rows = self._db.execute(
            "SELECT id, text FROM memory_meta ORDER BY id"
        ).fetchall()
        if not rows:
            logger.info("无需重建：SQLite 无记录")
            return

        logger.info("开始从 SQLite 重建索引，共 %d 条记录...", len(rows))
        batch_size = 64
        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start:batch_start + batch_size]
            ids = np.array([r[0] for r in batch], dtype=np.int64)
            texts = [r[1] for r in batch]
            vectors = self._embed.embed(texts)
            self.index.add_items(vectors, ids, num_threads=4)
            if (batch_start + batch_size) % 256 == 0 or batch_start + batch_size >= len(rows):
                logger.info("  重建进度: %d/%d", min(batch_start + batch_size, len(rows)), len(rows))

        current_count = self.index.get_current_count()
        self._default_ef = max(80, min(self._default_ef, current_count)) if current_count > 0 else 80
        self.index.set_ef(self._default_ef)
        self.save()
        logger.info("索引重建完成: %d 条记录, ef=%d", current_count, self._default_ef)

    def _get_meta(self, mem_id: int) -> dict[str, Any] | None:
        cur = self._db.execute(
            "SELECT id, text, type, session_id, source, created_at, metadata FROM memory_meta WHERE id=?",
            (mem_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "text": row[1],
            "type": row[2],
            "session_id": row[3],
            "source": row[4],
            "created_at": row[5],
            "metadata": row[6],
        }

    def _load_next_id(self) -> int:
        cur = self._db.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM memory_meta")
        return cur.fetchone()[0]

    def _save_next_id(self) -> None:
        pass  # 每次 INSERT 时自动更新，无需额外存储

    def count(self) -> int:
        cur = self._db.execute("SELECT COUNT(*) FROM memory_meta")
        return cur.fetchone()[0]

    def list_all(self, session_id: str | None = None) -> list[MemoryItem]:
        if session_id:
            cur = self._db.execute(
                "SELECT id, text, type, session_id, source, created_at, metadata FROM memory_meta WHERE session_id=? ORDER BY id",
                (session_id,),
            )
        else:
            cur = self._db.execute(
                "SELECT id, text, type, session_id, source, created_at, metadata FROM memory_meta ORDER BY id"
            )
        items = []
        for row in cur.fetchall():
            items.append(
                MemoryItem(
                    id=row[0],
                    text=row[1],
                    type=row[2],
                    session_id=row[3],
                    source=row[4],
                    created_at=row[5],
                    metadata=json.loads(row[6]) if isinstance(row[6], str) else row[6],
                )
            )
        return items

    @property
    def store_path(self) -> str:
        return self._store_path
