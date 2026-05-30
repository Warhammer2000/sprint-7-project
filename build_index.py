#!/usr/bin/env python3
"""Задание 3: построение векторного индекса базы знаний.

Читает документы из knowledge_base/, режет их на чанки с перекрытием,
строит эмбеддинги выбранной моделью и сохраняет FAISS-индекс + метаданные
в index/. В конце печатает статистику (модель, число чанков, время) и
прогоняет пример запроса.

Запуск:
    python build_index.py
    python build_index.py --query "Кто уничтожил Звезду Гибели?"
"""
from __future__ import annotations

import sys
import time
from datetime import datetime

from ragbot.chunking import chunk_documents, load_documents
from ragbot.config import settings
from ragbot.embeddings import build_embedder
from ragbot.vector_store import VectorStore


def main() -> None:
    settings.ensure_dirs()
    t0 = time.perf_counter()

    docs = load_documents(settings.kb_dir)
    if not docs:
        print(f"❌ В {settings.kb_dir} нет документов. Сначала: python scripts/build_kb.py")
        sys.exit(1)

    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    print(f"Документов: {len(docs)} | чанков: {len(chunks)} "
          f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    embedder = build_embedder(settings.embeddings_backend, settings.st_model)
    print(f"Эмбеддер: backend={embedder.backend}", end="", flush=True)

    store = VectorStore.build(embedder, chunks)
    store.manifest["built_at"] = datetime.now().isoformat(timespec="seconds")
    store.manifest["chunk_size"] = settings.chunk_size
    store.manifest["chunk_overlap"] = settings.chunk_overlap
    store.save(settings.index_dir)

    dt = time.perf_counter() - t0
    print(f", dim={store.manifest['dim']}")
    print(f"✅ Индекс сохранён в {settings.index_dir} за {dt:.2f} c")
    print(f"   Источников: {store.manifest['n_docs']}, чанков: {store.manifest['n_chunks']}")

    # Пример запроса к индексу
    query = "Кто уничтожил Звезду Гибели?"
    if "--query" in sys.argv:
        i = sys.argv.index("--query")
        if i + 1 < len(sys.argv):
            query = sys.argv[i + 1]
    print(f"\nПример запроса: «{query}»")
    for score, meta in store.search(query, 3):
        snippet = meta["text"][:90].replace("\n", " ")
        print(f"  score={score:.3f} | {meta['title']} ({meta['source']}) | {snippet}…")


if __name__ == "__main__":
    main()
