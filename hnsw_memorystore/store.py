"""
基于 HNSW 的向量记忆库，搭配 SQLite 元数据存储。

遵循 docs/AGENT_MEMORY_BUILD.md 的设计：
  - hnswlib 负责 ANN 检索（返回 id + 距离）
  - SQLite 负责 id → {text, type, session_id, created_at, ...} 回表
  - 可选 SparseInvertedIndex 负责 SPLADE 词项检索（混合模式）
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

from hnsw_memorystore.embedding import BaseEmbedding, BgeM3HybridEmbedding, SparseInvertedIndex

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
      hnswlib.Index       —— ANN 检索，返回 int id
      SQLite              —— id → 元数据（text, type, session_id, ...）
      SparseInvertedIndex —— 可选，SPLADE 稀疏检索（混合模式）
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

        self._hybrid = isinstance(embedding_model, BgeM3HybridEmbedding)
        self._sparse_index: SparseInvertedIndex | None = None
        if self._hybrid:
            self._sparse_index = SparseInvertedIndex()
            logger.info("混合检索模式已启用 (Dense + Sparse)")

        os.makedirs(store_path, exist_ok=True)

        # --- HNSW index ---
        index_path = os.path.join(store_path, "hnsw_index.bin")
        self.index: hnswlib.Index = hnswlib.Index(space=space, dim=dim)
        if os.path.exists(index_path):
            self.index.load_index(index_path, max_elements=max_elements)
            self.index.set_ef(max(80, ef_construction))
            logger.info("加载已有索引: %s (max_elements=%d)", index_path, max_elements)
        else:
            self.index.init_index(
                max_elements=max_elements,
                M=M,
                ef_construction=ef_construction,
                random_seed=random_seed,
                allow_replace_deleted=False,
            )
            self.index.set_ef(max(80, ef_construction))
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

        self._sparse_built = False
        if self._hybrid:
            doc_count = self.count()
            if doc_count == 0:
                self._sparse_built = True
            else:
                logger.info("稀疏索引将在首次混合检索时惰性重建 (%d 条文档)", doc_count)

    @property
    def hybrid(self) -> bool:
        return self._hybrid

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

        if self._hybrid and vectors is None:
            hybrid_output = self._embed.encode_hybrid(texts)
            vectors = hybrid_output["dense_vecs"]
            sparse_weights_list = hybrid_output["sparse_weights"]
        elif vectors is None:
            vectors = self._embed.embed(texts)
            sparse_weights_list = None
        else:
            vectors = np.asarray(vectors, dtype=np.float32)
            sparse_weights_list = None

        types = types or (["general"] * n)
        session_ids = session_ids or ([""] * n)
        sources = sources or ([""] * n)
        metadatas = metadatas or ([{}] * n)

        with self._lock:
            ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
            self._next_id += n

            self.index.add_items(vectors, ids, num_threads=4)

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

            if self._hybrid and self._sparse_index is not None and sparse_weights_list is not None:
                for i in range(n):
                    sw_dict = BgeM3HybridEmbedding.sparse_weights_to_dict(sparse_weights_list[i])
                    self._sparse_index.add(int(ids[i]), sw_dict)
                self._sparse_built = True

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
            if ef is not None:
                self.index.set_ef(ef)
            else:
                self.index.set_ef(max(80, k * 4))

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

    def search_hybrid(
        self,
        query: str,
        k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        ef: int | None = None,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        """混合检索：Dense (HNSW ANN) + Sparse (倒排索引)，加权融合。

        Args:
            query: 查询文本
            k: 返回结果数量
            dense_weight: Dense 分数权重
            sparse_weight: Sparse 分数权重
            rrf_k: RRF 常数（越大越平滑）
            ef: HNSW ef 参数
            session_id: 可选，过滤会话

        Returns:
            融合排序后的 MemoryItem 列表
        """
        if not self._hybrid or self._sparse_index is None:
            return self.search(query, k=k, ef=ef, session_id=session_id)

        if not self._sparse_built:
            self._rebuild_sparse_from_db()
            self._sparse_built = True

        n_total = self.count()
        k_dense = min(k * 3, n_total) if n_total > 0 else 0
        k_sparse = min(k * 3, n_total) if n_total > 0 else 0
        if k_dense == 0:
            return []

        hybrid_output = self._embed.encode_hybrid([query])
        q_dense = hybrid_output["dense_vecs"]
        q_sparse_dict = BgeM3HybridEmbedding.sparse_weights_to_dict(hybrid_output["sparse_weights"][0])

        with self._lock:
            if ef is not None:
                self.index.set_ef(ef)
            else:
                self.index.set_ef(max(80, k_dense * 4))

            labels, distances = self.index.knn_query(q_dense, k=k_dense)

        dense_scores: dict[int, float] = {}
        for mem_id, dist in zip(labels[0], distances[0]):
            if mem_id == -1:
                continue
            dense_scores[int(mem_id)] = float(dist)

        sparse_results = self._sparse_index.search(q_sparse_dict, top_k=k_sparse)
        sparse_scores: dict[int, float] = {}
        for doc_id, score in sparse_results:
            sparse_scores[doc_id] = score

        all_candidate_ids = set(dense_scores.keys()) | set(sparse_scores.keys())

        filtered_candidates: dict[int, dict] = {}
        for cid in all_candidate_ids:
            meta = self._get_meta(cid)
            if meta is None:
                continue
            if session_id and meta["session_id"] != session_id:
                continue
            filtered_candidates[cid] = meta

        dense_ranked = sorted(dense_scores.keys(), key=lambda x: dense_scores[x])
        sparse_ranked = sorted(sparse_scores.keys(), key=lambda x: sparse_scores[x], reverse=True)

        rrf_scores: dict[int, float] = {}
        for rank, cid in enumerate(dense_ranked):
            if cid in filtered_candidates:
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_weight / (rrf_k + rank + 1)
        for rank, cid in enumerate(sparse_ranked):
            if cid in filtered_candidates:
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + sparse_weight / (rrf_k + rank + 1)

        final_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results: list[MemoryItem] = []
        for cid, score in final_ranked:
            meta = filtered_candidates[cid]
            results.append(
                MemoryItem(
                    id=cid,
                    text=meta["text"],
                    type=meta["type"],
                    session_id=meta["session_id"],
                    source=meta["source"],
                    created_at=meta["created_at"],
                    metadata=json.loads(meta["metadata"]) if isinstance(meta["metadata"], str) else meta["metadata"],
                    score=score,
                )
            )

        logger.info("混合检索 '%s' (k=%d) → %d 条结果", query, k, len(results))
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
            if self._hybrid and self._sparse_index is not None:
                self._sparse_index.remove(mem_id)
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
            if self._hybrid and self._sparse_index is not None:
                for mid in ids:
                    self._sparse_index.remove(mid)
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

    def _rebuild_sparse_from_db(self) -> None:
        """从 SQLite 重建稀疏索引。"""
        if not self._hybrid or self._sparse_index is None:
            return
        logger.info("从数据库重建稀疏索引…")
        all_items = self.list_all()
        if not all_items:
            return
        texts = [item.text for item in all_items]
        hybrid_output = self._embed.encode_hybrid(texts)
        for i, item in enumerate(all_items):
            sw_dict = BgeM3HybridEmbedding.sparse_weights_to_dict(hybrid_output["sparse_weights"][i])
            self._sparse_index.add(item.id, sw_dict)
        logger.info("稀疏索引重建完成，共 %d 条", len(all_items))

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