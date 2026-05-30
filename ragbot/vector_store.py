"""Векторное хранилище на FAISS (Задание 3).

Используем ``IndexFlatIP`` поверх L2-нормированных векторов — скалярное
произведение становится косинусной близостью (как в уроке про FAISS).
Метаданные чанков и артефакты эмбеддера лежат рядом с индексом, плюс manifest
с хэшами исходных файлов (нужен для инкрементального обновления, Задание 6).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chunking import Chunk
from .embeddings import BaseEmbedder, load_embedder

INDEX_FILE = "faiss.index"
META_FILE = "meta.pkl"
MANIFEST_FILE = "manifest.json"


class VectorStore:
    def __init__(self, embedder: BaseEmbedder, index=None, metas: Optional[List[Dict]] = None):
        self.embedder = embedder
        self.index = index
        self.metas: List[Dict] = metas or []
        self.manifest: Dict = {}

    # ── построение ───────────────────────────────────────────
    @classmethod
    def build(cls, embedder: BaseEmbedder, chunks: Sequence[Chunk]) -> "VectorStore":
        import faiss

        texts = [c.text for c in chunks]
        embedder.fit(texts)                       # no-op для st, обучение для tfidf
        vectors = embedder.encode(texts)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        store = cls(embedder, index, [c.to_meta() for c in chunks])
        store.manifest = {
            "embedder_backend": embedder.backend,
            "dim": int(dim),
            "n_chunks": len(chunks),
            "n_docs": len({c.source for c in chunks}),
        }
        return store

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Догружает новые чанки в существующий индекс (для update_index)."""
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        vectors = self.embedder.encode(texts)
        self.index.add(vectors)
        self.metas.extend(c.to_meta() for c in chunks)
        self.manifest["n_chunks"] = len(self.metas)
        self.manifest["n_docs"] = len({m["source"] for m in self.metas})
        return len(chunks)

    # ── поиск ────────────────────────────────────────────────
    def search(self, query: str, k: int) -> List[Tuple[float, Dict]]:
        if self.index is None or not self.metas:
            return []
        qv = self.embedder.encode([query])
        k = min(k, len(self.metas))
        scores, ids = self.index.search(qv, k)
        out: List[Tuple[float, Dict]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            out.append((float(score), self.metas[idx]))
        return out

    # ── сохранение / загрузка ────────────────────────────────
    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        with (directory / META_FILE).open("wb") as fh:
            pickle.dump(self.metas, fh)
        self.embedder.save(directory)
        (directory / MANIFEST_FILE).write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        import faiss

        embedder = load_embedder(directory)
        index = faiss.read_index(str(directory / INDEX_FILE))
        with (directory / META_FILE).open("rb") as fh:
            metas = pickle.load(fh)
        store = cls(embedder, index, metas)
        manifest_path = directory / MANIFEST_FILE
        if manifest_path.exists():
            store.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return store

    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / INDEX_FILE).exists() and (directory / META_FILE).exists()
