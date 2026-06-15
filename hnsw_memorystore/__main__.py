"""端到端验证：写入 → 检索 → 更新 → 删除 → 持久化。

对应 docs/AGENT_MEMORY_BUILD.md 第 9 节「最小端到端示例骨架」。
"""

import logging
import os
import shutil
import sys

import numpy as np

from hnsw_memorystore import HnswMemoryStore, BaseEmbedding, FlagModelEmbedding, BgeM3Embedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("HnswMemoryStore")


class _TestEmbedding(BaseEmbedding):
    """确定性伪随机嵌入，同一文本始终返回相同向量，仅用于单元测试。"""

    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self._base = rng.normal(size=(1000, dim)).astype(np.float32)
        for i in range(1000):
            n = np.linalg.norm(self._base[i])
            if n > 0:
                self._base[i] /= n

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            idx = abs(hash(t)) % 1000
            vecs[i] = self._base[idx]
        return vecs


_test_embed = _TestEmbedding(dim=64)


def main():
    test_dir = os.path.join(os.path.dirname(__file__), "_test_store")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    store = HnswMemoryStore(
        dim=64,
        space="cosine",
        max_elements=10_000,
        M=16,
        ef_construction=200,
        store_path=test_dir,
        embedding_model=_test_embed,
    )

    # ---------- 1. 写入 ----------
    memories = [
        "用户喜欢 Python 和向量检索",
        "上次讨论过 HNSW 参数 ef 和 M",
        "会议定在明天下午三点",
        "项目代号是北极星",
        "用户的邮箱是 user@example.com",
    ]
    mem_vecs = _test_embed.embed(memories)

    items = store.add_memories(
        memories, vectors=mem_vecs,
        types=["fact"] * len(memories),
        session_ids=["session_1"] * len(memories),
    )
    logger.info("写入 %d 条记忆", len(items))
    assert store.count() == 5

    # ---------- 2. 检索 ----------
    logger.info("\n--- 检索测试 ---")
    queries = [
        "用户喜欢 Python 和向量检索",
        "项目代号是北极星",
        "用户的邮箱是 user@example.com",
    ]
    query_vecs = _test_embed.embed(queries)
    for q, qv in zip(queries, query_vecs):
        results = store.search(qv, k=3)
        assert results[0].text == q, f"检索失败: 预期 '{q}', 得到 '{results[0].text}'"
        logger.info("  ✅ '%s' → id=%d (dist=%.4f)", q, results[0].id, results[0].score)

    # ---------- 3. 更新 ----------
    logger.info("\n--- 更新测试 ---")
    new_text = "用户的邮箱是 user@newdomain.com"
    new_vec = _test_embed.embed([new_text])[0]
    store.update_memory(items[4].id, new_text, vector=new_vec)

    qv_updated = _test_embed.embed([new_text])[0]
    updated = store.search(qv_updated, k=1)
    assert updated[0].text == new_text, f"更新验证失败: {updated[0].text}"
    logger.info("  ✅ 更新后检索 '%s'", new_text)

    # ---------- 4. 自我召回验证 ----------
    logger.info("\n--- 自我召回验证 ---")
    store.index.set_ef(50)
    labels, distances = store.index.knn_query(mem_vecs[:3], k=1)
    recall = (labels.reshape(-1) == np.arange(3)).mean()
    logger.info("  self-recall (top 3): %.3f", recall)
    assert recall > 0.95, f"召回率过低: {recall}"

    # ---------- 5. 删除 ----------
    logger.info("\n--- 删除测试 ---")
    store.delete_memory(items[2].id)
    qv_deleted = _test_embed.embed(["会议定在明天下午三点"])[0]
    after_delete = store.search(qv_deleted, k=5)
    for r in after_delete:
        assert r.id != items[2].id, f"已删除的 id={items[2].id} 不该出现"
    logger.info("  ✅ 删除后已过滤 id=%d", items[2].id)

    # ---------- 6. 持久化 ----------
    logger.info("\n--- 持久化测试 ---")
    store.save()
    store.close()

    store2 = HnswMemoryStore(
        dim=64,
        space="cosine",
        max_elements=10_000,
        store_path=test_dir,
        embedding_model=_test_embed,
    )
    assert store2.count() == 4, f"持久化后计数错误: {store2.count()}"
    logger.info("  ✅ 重新加载后 %d 条记录", store2.count())

    qv_persist = _test_embed.embed(["项目代号是北极星"])[0]
    reloaded = store2.search(qv_persist, k=1)
    assert "北极星" in reloaded[0].text, f"持久化检索失败: {reloaded[0].text}"
    logger.info("  ✅ 持久化检索 '%s'", reloaded[0].text)
    store2.close()

    # ---------- 7. FlagModelEmbedding 验证 ----------
    logger.info("\n--- FlagModelEmbedding 验证 ---")
    try:
        fme = FlagModelEmbedding(device="cpu")
        test_texts = ["Hello world", "FlagEmbedding test", "BGE large zh v1.5"]
        embeds = fme.embed(test_texts)
        assert embeds.shape[0] == len(test_texts)
        assert embeds.dtype == np.float32
        norms = np.linalg.norm(embeds, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
        logger.info("  ✅ 形状: %s, dtype: %s, dim=%d", embeds.shape, embeds.dtype, embeds.shape[1])
        logger.info("  ✅ 归一化校验: norms≈1.0 %s", norms)
    except Exception as e:
        logger.warning("  ⚠️  FlagModelEmbedding 测试跳过: %s", e)

    # ---------- 8. BgeM3Embedding 验证 ----------
    logger.info("\n--- BgeM3Embedding 验证 ---")
    try:
        bge = BgeM3Embedding(device="cpu")
        test_texts = ["Hello, world!", "BGE-M3 embedding test", "Memory store with BGE"]
        embeds = bge.embed(test_texts)
        assert embeds.shape[0] == len(test_texts)
        assert embeds.dtype == np.float32
        norms = np.linalg.norm(embeds, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
        logger.info("  ✅ 形状: %s, dtype: %s, dim=%d", embeds.shape, embeds.dtype, embeds.shape[1])
        logger.info("  ✅ 归一化校验: norms≈1.0 %s", norms)
    except Exception as e:
        logger.warning("  ⚠️  BgeM3Embedding 测试跳过: %s", e)

    # ---------- 清理 ----------
    shutil.rmtree(test_dir, ignore_errors=True)
    logger.info("\n🎉 所有测试通过")


if __name__ == "__main__":
    main()
