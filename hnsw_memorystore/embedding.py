import numpy as np
from abc import ABCMeta, abstractmethod


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
