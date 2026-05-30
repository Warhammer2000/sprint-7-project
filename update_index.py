#!/usr/bin/env python3
"""Задание 6: автоматическое обновление векторного индекса.

Сканирует источник (по умолчанию knowledge_base/), находит новые и изменённые
документы по sha1, нарезает их на чанки, считает эмбеддинги и инкрементально
добавляет в FAISS-индекс. Пишет лог (время старта/финиша, число новых чанков,
размер индекса, ошибки).

Запуск вручную:   python update_index.py
По расписанию:    см. scripts/update.sh и cron-строку в README/Project_template.
"""
from __future__ import annotations

import hashlib
import sys
import traceback
from datetime import datetime
from pathlib import Path

from ragbot.chunking import Document, chunk_documents
from ragbot.config import settings
from ragbot.vector_store import VectorStore

LOG = settings.log_dir / "update.log"


def log_line(msg: str) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {msg}"
    print(line)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def current_hashes() -> dict:
    hashes = {}
    for path in sorted(settings.kb_dir.glob("**/*")):
        if path.suffix.lower() in {".md", ".txt"}:
            hashes[path.name] = hashlib.sha1(path.read_bytes()).hexdigest()
    return hashes


def load_doc(name: str) -> Document:
    path = settings.kb_dir / name
    text = path.read_text(encoding="utf-8").strip()
    title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
    return Document(text=text, source=name, title=title)


def main() -> None:
    log_line("=== запуск обновления индекса ===")
    errors = 0
    try:
        if not VectorStore.exists(settings.index_dir):
            log_line("ОШИБКА: индекс не найден, сначала запустите build_index.py")
            sys.exit(1)

        store = VectorStore.load(settings.index_dir)
        baseline = store.manifest.get("sources_hash", {})
        current = current_hashes()

        new_files = [f for f in current if f not in baseline]
        changed_files = [f for f in current if f in baseline and current[f] != baseline[f]]
        log_line(f"просканировано файлов: {len(current)}; "
                 f"новых: {len(new_files)}; изменённых: {len(changed_files)}")

        to_index = new_files + changed_files
        if changed_files:
            log_line(f"внимание: изменённые файлы переиндексируются добавлением "
                     f"(старые чанки FAISS-flat остаются; для полной чистки — build_index.py): "
                     f"{', '.join(changed_files)}")

        new_chunks = 0
        if to_index:
            docs = [load_doc(f) for f in to_index]
            chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
            new_chunks = store.add_chunks(chunks)
            store.manifest["sources_hash"] = current
            store.manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
            store.save(settings.index_dir)
            log_line(f"добавлено чанков: {new_chunks}; "
                     f"файлы: {', '.join(to_index)}")
        else:
            log_line("новых или изменённых документов нет — индекс актуален")

        log_line(f"index updated: {len(to_index)} files added, "
                 f"{new_chunks} new chunks, index size {len(store.metas)} chunks, {errors} errors")
    except Exception as exc:  # noqa: BLE001
        errors += 1
        log_line(f"ОШИБКА обновления: {exc}")
        log_line(traceback.format_exc().strip())
        sys.exit(1)
    finally:
        log_line("=== обновление завершено ===")


if __name__ == "__main__":
    main()
