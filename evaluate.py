#!/usr/bin/env python3
"""Задание 7: автоматическая оценка покрытия и качества RAG-бота.

1. Строит индекс С ИСКУССТВЕННЫМИ ПРОБЕЛАМИ — из базы знаний исключаются
   документы из gap_files (удалённые сущности).
2. Прогоняет «золотой набор» вопросов (tests/golden_questions.json):
     known — бот обязан ответить (нужное ключевое слово в ответе);
     gap   — ответа нет, бот должен честно отказать и не выдумывать факт.
3. Логирует каждый запрос (logs/eval_results.jsonl + logs/queries.jsonl) и
   печатает сводку: точность по known, корректность отказов по gap.

Запуск:  python evaluate.py
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ragbot.chunking import chunk_documents, load_documents
from ragbot.config import settings as base
from ragbot.embeddings import build_embedder
from ragbot.logging_utils import log_query
from ragbot.rag import RagBot
from ragbot.vector_store import VectorStore

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "tests" / "golden_questions.json"
EVAL_LOG = base.log_dir / "eval_results.jsonl"


def build_gapped_bot(gap_files: list[str]) -> RagBot:
    docs = [d for d in load_documents(base.kb_dir) if d.source not in set(gap_files)]
    chunks = chunk_documents(docs, base.chunk_size, base.chunk_overlap)
    embedder = build_embedder("tfidf", base.st_model)
    store = VectorStore.build(embedder, chunks)
    s = dataclasses.replace(base, llm_backend="offline", security_enabled=True)
    return RagBot(store, s), len(docs)


def contains_any(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def main() -> None:
    base.ensure_dirs()
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    gap_files = data.get("gap_files", [])
    questions = data["questions"]

    bot, n_docs = build_gapped_bot(gap_files)
    print(f"Индекс с пробелами: {n_docs} документов "
          f"(исключены: {', '.join(gap_files)}), чанков: {len(bot.store.metas)}\n")

    if EVAL_LOG.exists():
        EVAL_LOG.unlink()

    known_total = known_pass = 0
    gap_total = gap_pass = gap_refused = 0
    rows = []

    for item in questions:
        q, qtype = item["q"], item["type"]
        r = bot.answer(q)  # пишет в logs/queries.jsonl
        ok = False
        if qtype == "known":
            known_total += 1
            ok = (not r.refused) and contains_any(r.answer, item["expect_keywords"])
            known_pass += int(ok)
        else:  # gap
            gap_total += 1
            leaked = contains_any(r.answer, item.get("removed_keywords", []))
            ok = r.refused or not leaked
            gap_pass += int(ok)
            gap_refused += int(r.refused)

        rec = {
            "q": q, "type": qtype, "refused": r.refused, "ok": ok,
            "answer_len": len(r.answer), "top_score": round(r.top_score, 3),
            "sources": [s["source"] for s in r.sources],
        }
        rows.append(rec)
        with EVAL_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        mark = "✅" if ok else "❌"
        print(f"{mark} [{qtype:5}] {q}")
        print(f"     → {r.answer[:110]}{'…' if len(r.answer) > 110 else ''}")
        if r.security_notes:
            print(f"     notes: {'; '.join(r.security_notes)}")

    print("\n" + "=" * 60)
    print(f"KNOWN: {known_pass}/{known_total} верных ответов")
    print(f"GAP:   {gap_pass}/{gap_total} корректно обработанных пробелов "
          f"(из них честных отказов: {gap_refused}/{gap_total})")
    total_ok = known_pass + gap_pass
    print(f"ИТОГО: {total_ok}/{len(questions)} "
          f"({100 * total_ok / len(questions):.0f}%)")
    print(f"Логи: {EVAL_LOG} и {base.log_dir / 'queries.jsonl'}")


if __name__ == "__main__":
    main()
