import hashlib
import logging
import os
import shutil
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from dotenv import load_dotenv
from openai import OpenAI
from src.agents.message_dto import MessageDTO, Role
from src.agents.message_enum import Message
from src.memory.base_mem import BaseMem
from src.memory.chunker import DialogChunker
from src.memory.constant import (
    LONG_MEM_N,
    CHUNK_THRESHOLD,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP,
    CHUNK_SEARCH_MULTIPLIER,
)

load_dotenv()

logger = logging.getLogger("AILongTermMem")

_GLOBAL_SESSION_TAG = "__global__"

_ANTI_POLLUTION = os.getenv("LONG_MEM_ANTI_POLLUTION", "true").strip().lower() in (
    "true", "1", "yes",
)

_async_executor = ThreadPoolExecutor(max_workers=1)
_async_lock = Lock()

_sync_client: OpenAI | None = None
_sync_model: str = ""


def _get_sync_client() -> tuple[OpenAI, str]:
    global _sync_client, _sync_model
    if _sync_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "") or "ollama"
        _sync_client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        _sync_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return _sync_client, _sync_model


def _store_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "memorystore"))


def _build_hnsw_embedding():
    kind = os.getenv("HNSW_EMBEDDING", "bge-m3").strip().lower()
    device = os.getenv("HNSW_EMBEDDING_DEVICE", "cuda").strip().lower()

    if kind == "bge-large-zh":
        from hnsw_memorystore.embedding import FlagModelEmbedding
        return FlagModelEmbedding(model_name="BAAI/bge-large-zh-v1.5", device=device)
    elif kind == "sentence-transformers":
        from hnsw_memorystore.embedding import SentenceEmbedding
        return SentenceEmbedding()
    else:
        from hnsw_memorystore.embedding import BgeM3Embedding
        return BgeM3Embedding(device=device)


def _hash_parent(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


# =========================================================
# Anti-pollution logic (backend-agnostic)
# =========================================================

def _llm_call(prompt: str) -> str:
    client, model = _get_sync_client()
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    if model.endswith(":7b"):
        kwargs["extra_body"] = {"options": {"num_gpu_layers": 35}}
    try:
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("反污染 LLM 调用失败: %s", e)
        return ""


def _extract_clean_q(q: str) -> str:
    prompt = Message.CLEAN_Q_EXTRACT.value.format(q=q)
    result = _llm_call(prompt)
    return result if result else q


def _process_before_write(
    q: str,
    ans: str,
    existing_texts: list[tuple[str, str]],
) -> tuple[str, list[str]]:
    if not existing_texts:
        clean_q = _extract_clean_q(q)
        return clean_q, []

    existing_block = "\n".join(
        f"[{eid}] {doc}" for eid, doc in existing_texts
    )
    prompt = Message.MEM_WRITE_PROCESS.value.format(
        existing=existing_block, q=q, ans=ans
    )
    response = _llm_call(prompt)
    if not response:
        return q, []

    clean_q = q
    ids_to_delete: list[str] = []

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("CLEAN_Q:"):
            extracted = line[len("CLEAN_Q:"):].strip()
            if extracted:
                clean_q = extracted
        elif line.startswith("DELETE_IDS:"):
            raw_ids = line[len("DELETE_IDS:"):].strip()
            if raw_ids and raw_ids != "无":
                ids_to_delete = [
                    eid.strip() for eid in raw_ids.split(",") if eid.strip()
                ]

    return clean_q, ids_to_delete


# =========================================================
# Shared chunking logic
# =========================================================

_chunker = DialogChunker()


def _chunk_answer(ans: str, clean_q: str) -> list[dict]:
    """对助手回答进行切片，返回存储条目列表。

    Returns:
        list[dict], 每个含: doc (被 embedding 的文本), metadata (所有元数据)
    """
    chunks = _chunker.chunk(ans)

    if len(chunks) == 1 and chunks[0].chunk_type == "whole":
        return [{
            "doc": clean_q,
            "metadata": {
                "role": "pair",
                "type": "whole",
                "parent_id": "",
                "chunk_idx": 0,
                "original_q": clean_q,
                "original_ans": ans,
            },
        }]

    parent_id = _hash_parent(clean_q)
    entries = []
    for chunk in chunks:
        doc_text = f"{clean_q} | {chunk.content}"
        entries.append({
            "doc": doc_text,
            "metadata": {
                "role": "chunk",
                "type": chunk.chunk_type,
                "parent_id": parent_id,
                "chunk_idx": chunk.chunk_index,
                "original_q": clean_q,
                "original_ans": ans,
            },
        })
    return entries


def _aggregate_chunks(
    raw_items: list[tuple[str, dict]],
    top_k: int,
) -> list[MessageDTO]:
    """将按 parent_id 聚合切片，返回 top_k 个对话回合。

    Args:
        raw_items: list of (text, metadata) pairs from vector search
        top_k: 返回的对话回合数

    Returns:
        list of MessageDTO, 每条是一条完整对话回合
    """
    groups: dict[str, list[tuple[float, str, dict]]] = defaultdict(list)
    single_counter = 0

    for i, (text, meta) in enumerate(raw_items):
        pid = meta.get("parent_id", "")
        score = 0.0
        if not pid:
            pid = f"_single_{single_counter}"
            single_counter += 1
        groups[pid].append((score, text, meta))

    # 需要重新搜索以获取分数 -- 由各后端实现传入分数
    # 这里假设 raw_items 已按相似度排序
    for items in groups.values():
        pass

    scored_groups = []
    for pid, items in groups.items():
        # 用第一条的 metadata 代表整组
        best_meta = items[0][2]
        scored_groups.append((pid, items, best_meta))

    # 取 top_k 组
    result = []
    seen_groups = scored_groups[:top_k]

    for pid, items, best_meta in seen_groups:
        original_q = best_meta.get("original_q", "")
        original_ans = best_meta.get("original_ans", "")

        if original_q and original_ans:
            full = f"[用户] {original_q}\n[助手] {original_ans}"
        else:
            # 兼容旧数据：没有 original_q/original_ans
            doc_text = items[0][1]
            full = f"[用户] {doc_text}"

        result.append(MessageDTO(role=Role.USER, content=full))

    return result


# =========================================================
# Backend implementations
# =========================================================

class _LongMemChromadb(BaseMem):
    """ChromaDB 后端"""

    def __init__(self, session_key: str, store: str) -> None:
        import chromadb
        self._session_key = session_key
        self.client = chromadb.PersistentClient(path=store)
        self.collection_name = f"long_mem_{session_key}"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def get_mem(self, q: str) -> list[MessageDTO]:
        if self.collection.count() == 0:
            return []
        k = LONG_MEM_N * CHUNK_SEARCH_MULTIPLIER
        total = self.collection.count()
        k = min(k, total)
        if k == 0:
            return []

        results = self.collection.query(
            query_texts=[q],
            n_results=k,
            include=["documents", "metadatas"],
        )
        if not results or not results["documents"] or not results["documents"][0]:
            return []

        raw_items = []
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            raw_items.append((doc, meta))

        messages = _aggregate_chunks(raw_items, LONG_MEM_N)
        if messages:
            logger.info("  [ChromaDB] 检索出 %d 条长期记忆 (聚合后)", len(messages))
        return messages

    def update_mem(self, q: str, ans: str) -> None:
        clean_q = q
        if _ANTI_POLLUTION:
            existing = self._search_existing(q)
            clean_q, ids_to_delete = _process_before_write(q, ans, existing)
            if ids_to_delete:
                valid_ids = [eid for eid in ids_to_delete if eid in self._all_ids()]
                if valid_ids:
                    self.collection.delete(ids=valid_ids)
                    # 同时删除同一 parent 的所有切片
                    logger.info("已删除 %d 条被替代的旧记忆 (ChromaDB)", len(valid_ids))

            # 删除同一 parent 的旧切片（如果此问题之前已存过）
            self._delete_by_parent(clean_q)

        entries = _chunk_answer(ans, clean_q)
        ids = [uuid.uuid4().hex for _ in entries]
        docs = [e["doc"] for e in entries]
        metas = [e["metadata"] for e in entries]
        metas_stripped = []
        for m in metas:
            ms = dict(m)
            # ChromaDB metadata 值有大小限制，original_ans 可能很长
            # 超过 4KB 时截断并标记
            if "original_ans" in ms and len(str(ms["original_ans"])) > 4000:
                ms["original_ans_truncated"] = True
                ms["original_ans"] = str(ms["original_ans"])[:4000]
            else:
                ms["original_ans_truncated"] = False
            metas_stripped.append(ms)

        self.collection.add(
            documents=docs,
            ids=ids,
            metadatas=metas_stripped,
        )

    def _delete_by_parent(self, clean_q: str) -> None:
        """删除同一 parent_id 的旧切片。"""
        parent_id = _hash_parent(clean_q)
        try:
            all_data = self.collection.get(include=["metadatas"])
            ids_to_del = []
            for i, meta in enumerate(all_data["metadatas"]):
                if meta and meta.get("parent_id") == parent_id:
                    ids_to_del.append(all_data["ids"][i])
            if ids_to_del:
                self.collection.delete(ids=ids_to_del)
                logger.info("删除 parent_id=%s 的 %d 条旧切片 (ChromaDB)", parent_id, len(ids_to_del))
        except Exception:
            pass

    def _all_ids(self) -> set[str]:
        count = self.collection.count()
        if count == 0:
            return set()
        results = self.collection.get(include=[])
        return set(results.get("ids", []))

    def _search_existing(self, q: str) -> list[tuple[str, str]]:
        count = self.collection.count()
        if count == 0:
            return []
        k = min(LONG_MEM_N * 2, count)
        results = self.collection.query(
            query_texts=[q],
            n_results=k,
            include=["documents"],
        )
        if not results or not results.get("documents") or not results["documents"][0]:
            return []
        ids = results.get("ids", [[]])[0]
        docs = results["documents"][0]
        return list(zip(ids, docs))

    def reset(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)


class _LongMemHNSW(BaseMem):
    """HNSW 后端"""

    def __init__(self, session_key: str, store: str) -> None:
        from hnsw_memorystore.embedding import BaseEmbedding
        from hnsw_memorystore.store import HnswMemoryStore

        self._session_key = session_key
        self._store_path = os.path.join(store, f"hnsw_{session_key}")
        self._embed = _build_hnsw_embedding()

        dim = self._embed.embed(["test"]).shape[1]
        self._store = HnswMemoryStore(
            dim=dim,
            space="cosine",
            max_elements=100_000,
            M=16,
            ef_construction=200,
            store_path=self._store_path,
            embedding_model=self._embed,
        )

    def get_mem(self, q: str) -> list[MessageDTO]:
        if self._store.count() == 0:
            return []
        k = LONG_MEM_N * CHUNK_SEARCH_MULTIPLIER
        results = self._store.search(q, k=k, session_id=self._session_key)
        if not results:
            return []

        raw_items = []
        for r in results:
            meta = r.metadata or {}
            raw_items.append((r.text, meta))

        messages = _aggregate_chunks(raw_items, LONG_MEM_N)
        if messages:
            logger.info("  [HNSW] 检索出 %d 条长期记忆 (聚合后)", len(messages))
        return messages

    def update_mem(self, q: str, ans: str) -> None:
        def _do_write():
            clean_q = q
            if _ANTI_POLLUTION:
                existing = self._search_existing(q)
                clean_q, ids_to_delete = _process_before_write(q, ans, existing)
                if ids_to_delete:
                    deleted = 0
                    for eid in ids_to_delete:
                        try:
                            int_id = int(eid)
                            if self._store.delete_memory(int_id):
                                deleted += 1
                        except (ValueError, Exception):
                            pass
                    if deleted:
                        logger.info("已删除 %d 条被替代的长期记忆 (HNSW)", deleted)

                self._delete_by_parent(clean_q)

            entries = _chunk_answer(ans, clean_q)
            texts = [e["doc"] for e in entries]
            metadatas = [e["metadata"] for e in entries]
            types = [e["metadata"].get("type", "general") for e in entries]

            self._store.add_memories(
                texts=texts,
                session_ids=[self._session_key] * len(entries),
                types=types,
                metadatas=metadatas,
            )

        # 异步写入：embedding 和入库在后台线程执行，不阻塞对话
        _async_executor.submit(_do_write)

    def _delete_by_parent(self, clean_q: str) -> None:
        """删除同一 parent_id 的旧切片。"""
        parent_id = _hash_parent(clean_q)
        try:
            all_items = self._store.list_all(session_id=self._session_key)
            ids_to_del = []
            for item in all_items:
                if item.metadata and item.metadata.get("parent_id") == parent_id:
                    ids_to_del.append(item.id)
            if ids_to_del:
                for mid in ids_to_del:
                    self._store.delete_memory(mid)
                logger.info("删除 parent_id=%s 的 %d 条旧切片 (HNSW)", parent_id, len(ids_to_del))
        except Exception:
            pass

    def _search_existing(self, q: str) -> list[tuple[str, str]]:
        if self._store.count() == 0:
            return []
        k = LONG_MEM_N * 2
        results = self._store.search(q, k=k, session_id=self._session_key)
        if not results:
            return []
        return [(str(r.id), r.text) for r in results]

    def reset(self) -> None:
        self._store.close()
        if os.path.exists(self._store_path):
            shutil.rmtree(self._store_path, ignore_errors=True)
        os.makedirs(self._store_path, exist_ok=True)

        from hnsw_memorystore.store import HnswMemoryStore
        dim = self._embed.embed(["test"]).shape[1]
        self._store = HnswMemoryStore(
            dim=dim, space="cosine",
            max_elements=100_000, M=16, ef_construction=200,
            store_path=self._store_path, embedding_model=self._embed,
        )


class LongMem(BaseMem):
    """长期记忆：通过 LONG_MEM_BACKEND 环境变量切换后端。

    环境变量:
      LONG_MEM_BACKEND=chromadb|hnsw  （默认 chromadb）
      HNSW_EMBEDDING=bge-m3|bge-large-zh|sentence-transformers  （默认 bge-m3）
      HNSW_EMBEDDING_DEVICE=cuda|cpu  （默认 cuda）
      LONG_MEM_ANTI_POLLUTION=true|false  （默认 true，写入前提炼+矛盾检测）
      LONG_MEM_CHUNK_THRESHOLD=500  （回答短于此值不切片）
      LONG_MEM_CHUNK_MAX_CHARS=1500  （每块最大字符数）
      LONG_MEM_CHUNK_OVERLAP=200  （块间重叠字符数）
      LONG_MEM_CHUNK_SEARCH_MULTIPLIER=3  （检索时 k 的放大倍率）
    """

    def __init__(self, session_id: str | None = None) -> None:
        session_key = (session_id or _GLOBAL_SESSION_TAG).strip() or _GLOBAL_SESSION_TAG
        store = _store_dir()
        os.makedirs(store, exist_ok=True)

        backend = os.getenv("LONG_MEM_BACKEND", "chromadb").strip().lower()
        logger.info(
            "LongMem 后端: %s | session=%s | anti_pollution=%s | chunk_threshold=%d",
            backend, session_key[:12], _ANTI_POLLUTION, CHUNK_THRESHOLD,
        )

        if backend == "hnsw":
            self._impl = _LongMemHNSW(session_key, store)
        else:
            self._impl = _LongMemChromadb(session_key, store)

    def get_mem(self, q: str) -> list[MessageDTO]:
        return self._impl.get_mem(q)

    def update_mem(self, q: str, ans: str) -> None:
        self._impl.update_mem(q, ans)

    def reset(self) -> None:
        self._impl.reset()