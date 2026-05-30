"""Подключаемый слой эмбеддингов.

Два бэкенда (Задание 1 — сравнение моделей эмбеддингов):

* ``st``   — Sentence-Transformers (paraphrase-multilingual-MiniLM-L12-v2).
             Качественные плотные эмбеддинги, понимают синонимы и русский язык.
             Требует torch (тяжёлый), модель скачивается с HF Hub.
* ``tfidf`` — TF-IDF + TruncatedSVD (LSA) на scikit-learn.
             Лёгкий офлайн-вариант без скачивания моделей и без torch,
             даёт плотные нормированные векторы фиксированной длины.

Оба возвращают L2-нормированные float32-векторы, поэтому скалярное произведение
в FAISS (IndexFlatIP) эквивалентно косинусной близости.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Sequence

import numpy as np


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class BaseEmbedder:
    backend: str = "base"
    dim: int = 0

    def fit(self, texts: Sequence[str]) -> "BaseEmbedder":  # noqa: D401
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path) -> "BaseEmbedder":
        raise NotImplementedError


class STEmbedder(BaseEmbedder):
    """Sentence-Transformers backend (ленивая загрузка модели)."""

    backend = "st"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.dim = vecs.shape[1]
        return vecs.astype("float32")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"backend": self.backend, "model_name": self.model_name, "dim": self.dim}
        (directory / "embedder.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "STEmbedder":
        meta = json.loads((directory / "embedder.json").read_text(encoding="utf-8"))
        emb = cls(meta["model_name"])
        emb.dim = meta.get("dim", 0)
        return emb


class TfidfEmbedder(BaseEmbedder):
    """Лёгкий офлайн-эмбеддер: TF-IDF + усечённый SVD (латентно-семантический анализ)."""

    backend = "tfidf"

    def __init__(self, n_components: int = 256):
        self.n_components = n_components
        self._vectorizer = None
        self._svd = None

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = list(texts)
        self._vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95,
        )
        tfidf = self._vectorizer.fit_transform(texts)
        n_features = tfidf.shape[1]
        # SVD: компонентов не больше, чем признаков/документов
        n_comp = max(2, min(self.n_components, n_features - 1, max(2, len(texts) - 1)))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self._svd.fit(tfidf)
        self.dim = n_comp
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("TfidfEmbedder не обучен: сначала вызовите fit().")
        tfidf = self._vectorizer.transform(list(texts))
        reduced = self._svd.transform(tfidf)
        return _l2_normalize(reduced)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "embedder.pkl").open("wb") as fh:
            pickle.dump(
                {"vectorizer": self._vectorizer, "svd": self._svd, "dim": self.dim}, fh
            )
        (directory / "embedder.json").write_text(
            json.dumps({"backend": self.backend, "dim": self.dim}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "TfidfEmbedder":
        with (directory / "embedder.pkl").open("rb") as fh:
            state = pickle.load(fh)
        emb = cls()
        emb._vectorizer = state["vectorizer"]
        emb._svd = state["svd"]
        emb.dim = state["dim"]
        return emb


def build_embedder(backend: str, st_model: str) -> BaseEmbedder:
    """Фабрика эмбеддера с мягким фоллбэком st → tfidf, если нет torch/модели."""
    backend = (backend or "st").lower()
    if backend == "tfidf":
        return TfidfEmbedder()
    try:
        import sentence_transformers  # noqa: F401

        return STEmbedder(st_model)
    except Exception as exc:  # torch/модель недоступны — переходим на tfidf
        print(
            f"[embeddings] sentence-transformers недоступен ({exc.__class__.__name__}), "
            "переключаюсь на лёгкий TF-IDF-эмбеддер."
        )
        return TfidfEmbedder()


def load_embedder(directory: Path) -> BaseEmbedder:
    meta = json.loads((directory / "embedder.json").read_text(encoding="utf-8"))
    if meta["backend"] == "st":
        return STEmbedder.load(directory)
    return TfidfEmbedder.load(directory)
