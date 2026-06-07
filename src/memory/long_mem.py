import logging
import os
import shutil
import uuid

from dotenv import load_dotenv
from openai import OpenAI
from src.agents.message_dto import MessageDTO, Role
from src.agents.message_enum import Message
from src.memory.base_mem import BaseMem
from src.memory.constant import LONG_MEM_N

load_dotenv()

logger = logging.getLogger("AILongTermMem")

_GLOBAL_SESSION_TAG = "__global__"

_ANTI_POLLUTION = os.getenv("LONG_MEM_ANTI_POLLUTION", "true").strip().lower() in (
    "true", "1", "yes",
)

_sync_client: OpenAI | None = None
_sync_model: str = ""


def _get_sync_client() -> tuple[OpenAI, str]:
    """Lazy-init the synchronous LLM client so .env is loaded first."""
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
    """根据环境变量 HNSW_EMBEDDING 构建嵌入模型。"""
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


# =========================================================
# Anti-pollution logic (backend-agnostic)
# =========================================================

def _llm_call(prompt: str) -> str:
    """Synchronous LLM call for anti-pollution. Uses the main agent's model."""
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
    """无旧记忆时的轻量提炼：只清理单轮内的自我纠正，返回最终结论。"""
    prompt = Message.CLEAN_Q_EXTRACT.value.format(q=q)
    result = _llm_call(prompt)
    return result if result else q


def _process_before_write(
    q: str,
    ans: str,
    existing_texts: list[tuple[str, str]],
) -> tuple[str, list[str]]:
    """一次 LLM 调用同时完成：提炼用户陈述 + 检测矛盾旧记忆。

    Args:
        q: 用户输入
        ans: 助手回复
        existing_texts: list of (id, text) pairs from vector search

    Returns:
        (clean_q, ids_to_delete) — 提炼后的用户陈述 + 需要删除的记忆ID列表
    """
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
        results = self.collection.query(
            query_texts=[q],
            n_results=LONG_MEM_N,
            include=["documents", "metadatas"],
        )
        if not results or not results["documents"]:
            return []
        messages = []
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else []
        for i, user_q in enumerate(docs):
            ans = metas[i].get("answer", "") if i < len(metas) else ""
            full = f"[用户] {user_q}\n[助手] {ans}"
            messages.append(MessageDTO(role=Role.USER, content=full))
        if messages:
            logger.info("  [ChromaDB] 检索出 %d 条长期记忆", len(messages))
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
                    logger.info("已删除 %d 条被替代的长期记忆 (ChromaDB)", len(valid_ids))

        self.collection.add(
            documents=[clean_q],
            ids=[uuid.uuid4().hex],
            metadatas=[{"role": "pair", "type": "turn", "answer": ans}],
        )

    def _all_ids(self) -> set[str]:
        count = self.collection.count()
        if count == 0:
            return set()
        results = self.collection.get(include=[])
        return set(results.get("ids", []))

    def _search_existing(self, q: str) -> list[tuple[str, str]]:
        """搜索语义相近的旧记忆，返回 (id, text) 列表。"""
        count = self.collection.count()
        if count == 0:
            return []
        k = min(LONG_MEM_N, count)
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
        results = self._store.search(q, k=LONG_MEM_N, session_id=self._session_key)
        if not results:
            return []
        messages = []
        for r in results:
            meta = r.metadata or {}
            ans = meta.get("answer", "")
            full = f"[用户] {r.text}\n[助手] {ans}"
            messages.append(MessageDTO(role=Role.USER, content=full))
        if messages:
            logger.info("  [HNSW] 检索出 %d 条长期记忆", len(messages))
        return messages

    def update_mem(self, q: str, ans: str) -> None:
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

        self._store.add_memories(
            texts=[clean_q],
            session_ids=[self._session_key],
            types=["turn"],
            metadatas=[{"role": "pair", "type": "turn", "answer": ans}],
        )

    def _search_existing(self, q: str) -> list[tuple[str, str]]:
        """搜索语义相近的旧记忆，返回 (id, text) 列表。"""
        if self._store.count() == 0:
            return []
        results = self._store.search(q, k=LONG_MEM_N, session_id=self._session_key)
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
    """

    def __init__(self, session_id: str | None = None) -> None:
        session_key = (session_id or _GLOBAL_SESSION_TAG).strip() or _GLOBAL_SESSION_TAG
        store = _store_dir()
        os.makedirs(store, exist_ok=True)

        backend = os.getenv("LONG_MEM_BACKEND", "chromadb").strip().lower()
        logger.info(
            "LongMem 后端: %s | session=%s | anti_pollution=%s",
            backend, session_key[:12], _ANTI_POLLUTION,
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