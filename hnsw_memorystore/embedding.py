import numpy as np
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any


class BaseEmbedding(metaclass=ABCMeta):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class SentenceEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class FlagModelEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", device: str = "cuda"):
        from FlagEmbedding import FlagModel
        self._model = FlagModel(
            model_name,
            use_fp16=(device == "cuda"),
            devices=device,
            normalize_embeddings=True,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts)


class BgeM3Embedding(BaseEmbedding):
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=(device == "cuda"),
            devices=device,
            normalize_embeddings=True,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        output = self._model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return np.asarray(output["dense_vecs"], dtype=np.float32)


class BgeM3HybridEmbedding(BaseEmbedding):
    """BGE-M3 混合嵌入：同时输出 Dense 向量和 Sparse 词项权重。

    检索时使用 Dense（HNSW ANN）+ Sparse（倒排索引）分数融合，
    兼顾语义泛化和关键词精确匹配。
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=(device == "cuda"),
            devices=device,
            normalize_embeddings=True,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        output = self._model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return np.asarray(output["dense_vecs"], dtype=np.float32)

    def encode_hybrid(self, texts: list[str]) -> dict[str, Any]:
        output = self._model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense_vecs = np.asarray(output["dense_vecs"], dtype=np.float32)
        sparse_weights = output["lexical_weights"]
        if isinstance(sparse_weights, list):
            sparse_weights_list = sparse_weights
        else:
            sparse_weights_list = sparse_weights
        return {
            "dense_vecs": dense_vecs,
            "sparse_weights": sparse_weights_list,
        }

    @staticmethod
    def sparse_weights_to_dict(sparse_weights_item: dict) -> dict[int, float]:
        result: dict[int, float] = {}
        for token_id, weight in sparse_weights_item.items():
            result[int(token_id)] = float(weight)
        return result


class SparseInvertedIndex:
    """轻量稀疏倒排索引，用于 BM25/SPLADE 风格的词项检索。"""

    def __init__(self):
        self._documents: dict[int, dict[int, float]] = {}
        self._inverted: dict[int, dict[int, float]] = defaultdict(dict)
        self._next_id = 0

    def add(self, doc_id: int, sparse_weights: dict[int, float]) -> None:
        self._documents[doc_id] = sparse_weights
        for token_id, weight in sparse_weights.items():
            if weight > 0:
                self._inverted[token_id][doc_id] = weight

    def search(
        self,
        query_weights: dict[int, float],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        candidate_scores: dict[int, float] = defaultdict(float)
        for token_id, q_weight in query_weights.items():
            if q_weight <= 0:
                continue
            postings = self._inverted.get(token_id, {})
            for doc_id, d_weight in postings.items():
                candidate_scores[doc_id] += q_weight * d_weight
        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def remove(self, doc_id: int) -> None:
        sparse_weights = self._documents.pop(doc_id, None)
        if sparse_weights is None:
            return
        for token_id in sparse_weights:
            self._inverted.pop(token_id, None)

    def count(self) -> int:
        return len(self._documents)