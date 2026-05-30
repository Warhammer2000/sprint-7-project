"""RAG-конвейер (Задание 4): от запроса пользователя до проверенного ответа.

Повторяет 8 шагов из урока «Архитектура RAG»:
  1. препроцессинг запроса
  2. эмбеддинг + (safety-проверка запроса)
  3. recall — широкий ANN-поиск (recall_k)
  4. отсев по порогу близости (min_score) + санитизация чанков (Задание 5)
  5. trim — оставляем top_k чанков в контекст
  6. prompt compose (few-shot + Chain-of-Thought, безопасный System)
  7. ответ LLM (offline / openai / ollama)
  8. post-processing + guard вывода + логирование
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import security
from .config import Settings, settings as default_settings
from .llm import IDK, build_llm
from .logging_utils import log_query
from .vector_store import VectorStore


@dataclass
class RagResult:
    query: str
    answer: str
    sources: List[Dict] = field(default_factory=list)
    used_chunks: List[Dict] = field(default_factory=list)
    refused: bool = False
    blocked: bool = False
    success: bool = False
    security_notes: List[str] = field(default_factory=list)
    top_score: float = 0.0

    def format(self) -> str:
        lines = [self.answer]
        if self.sources:
            lines.append("\nИсточники:")
            for s in self.sources:
                lines.append(f"  [{s['n']}] {s['title']} ({s['source']}, score={s['score']:.3f})")
        if self.security_notes:
            lines.append("\n[security] " + "; ".join(self.security_notes))
        return "\n".join(lines)


def _preprocess(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


class RagBot:
    def __init__(self, store: VectorStore, settings: Settings):
        self.store = store
        self.settings = settings
        self.llm = build_llm(settings)

    @classmethod
    def from_settings(cls, settings: Settings = default_settings) -> "RagBot":
        if not VectorStore.exists(settings.index_dir):
            raise FileNotFoundError(
                f"Индекс не найден в {settings.index_dir}. Сначала запустите build_index.py."
            )
        store = VectorStore.load(settings.index_dir)
        return cls(store, settings)

    def answer(self, query: str, log: bool = True) -> RagResult:
        s = self.settings
        query = _preprocess(query)
        result = RagResult(query=query, answer="")
        notes: List[str] = []

        # Шаг 2a. Safety-проверка запроса (Слой safety_in)
        if s.security_enabled:
            ok, reason = security.is_query_safe(query)
            if not ok:
                result.blocked = True
                result.answer = "Запрос отклонён политикой безопасности."
                notes.append(reason)
                result.security_notes = notes
                self._maybe_log(log, result, chunks_found=0)
                return result

        # Шаги 2b–3. Эмбеддинг + recall
        hits = self.store.search(query, s.recall_k)

        # Шаг 4. Отсев по порогу близости
        relevant = [(score, meta) for score, meta in hits if score >= s.min_score]
        result.top_score = hits[0][0] if hits else 0.0

        # Нет релевантного контекста → честный отказ
        if not relevant:
            result.refused = True
            result.answer = IDK
            self._maybe_log(log, result, chunks_found=0)
            return result

        # Шаг 5. Trim до top_k
        relevant = relevant[: s.top_k]
        chunks = [dict(meta, score=score) for score, meta in relevant]

        # Шаг 4 (security). Санитизация чанков от инъекций
        if s.security_enabled:
            chunks, flagged = security.filter_chunks(chunks)
            if flagged:
                notes.append(
                    f"очищено инъекций в чанках: {len(flagged)} "
                    f"(источники: {', '.join(sorted({f['source'] for f in flagged}))})"
                )

        # Шаги 6–7. Промпт + генерация
        raw_answer = self.llm.answer(
            query, chunks, use_few_shot=s.use_few_shot, use_cot=s.use_cot
        )

        # Шаг 8. Guard вывода (Слой safety_out)
        if s.security_enabled:
            raw_answer, touched, reason = security.guard_output(raw_answer)
            if touched:
                notes.append(reason)

        result.answer = raw_answer.strip() or IDK
        result.used_chunks = chunks
        result.sources = [
            {"n": i + 1, "title": c.get("title", ""), "source": c.get("source", ""),
             "score": float(c.get("score", 0.0))}
            for i, c in enumerate(chunks)
        ]
        result.refused = result.answer.startswith("Я не знаю")
        result.success = not result.refused and not result.blocked
        result.security_notes = notes
        self._maybe_log(log, result, chunks_found=len(chunks))
        return result

    def _maybe_log(self, log: bool, result: RagResult, chunks_found: int) -> None:
        if not log:
            return
        log_query(
            self.settings.log_dir,
            query=result.query,
            chunks_found=chunks_found,
            answer=result.answer,
            sources=[s["source"] for s in result.sources],
            success=result.success,
            refused=result.refused or result.blocked,
            extra={"top_score": round(result.top_score, 4), "llm": self.llm.name},
        )
