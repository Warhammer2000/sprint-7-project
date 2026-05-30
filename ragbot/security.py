"""Слои защиты RAG от prompt-injection и утечек (Задание 5).

Реализованы три слоя из урока «Архитектура RAG и безопасность»:

  Слой 1. Фильтрация входа/корпуса  — scan_document / strip_injection
  Слой 2. Жёсткий шаблон промпта     — см. prompts.SAFE_SYSTEM_PROMPT
  Слой 3. Post-generation guard      — guard_output

Плюс safety-проверка пользовательского запроса (is_query_safe).
Все слои включаются флагом SECURITY_ENABLED, чтобы можно было
продемонстрировать поведение «с защитой / без защиты».
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ── Сигнатуры инъекций (директивы внутри документов) ─────────
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all|any|previous|prior)\s+instructions", re.I),
    re.compile(r"disregard\s+(all|the|previous|prior)", re.I),
    re.compile(r"forget\s+(all|everything|previous)", re.I),
    re.compile(r"\bsystem\s*prompt\b", re.I),
    re.compile(r"/system\b", re.I),
    re.compile(r"\boutput\s*:\s*", re.I),
    re.compile(r"забудь\s+(все|всё|предыдущие|инструкции)", re.I),
    re.compile(r"проигнориру\w*\s+(все|инструкции)", re.I),
    re.compile(r"ты\s+теперь\b", re.I),
    re.compile(r"<\s*python\s*>", re.I),
    re.compile(r"os\.system|rm\s+-rf|subprocess", re.I),
]

# ── Попытки вытащить системный промпт / сделать jailbreak ─────
JAILBREAK_PATTERNS = [
    re.compile(r"(покажи|раскрой|выведи|повтори)\s+(свой|системн\w+)\s*(промпт|инструкц\w+)", re.I),
    re.compile(r"(reveal|show|print|repeat)\s+(your|the)\s+(system\s*)?prompt", re.I),
    re.compile(r"ignore\s+(all|previous)\s+instructions", re.I),
    re.compile(r"act\s+as\s+(dan|developer\s*mode)", re.I),
]

# ── Что считаем секретом и вырезаем из ответа ────────────────
SECRET_PATTERNS = [
    re.compile(r"(супер)?паро\w*\s*[:=].*", re.I),
    re.compile(r"\bpassword\s*[:=].*", re.I),
    re.compile(r"\broot\s*[:=].*", re.I),
    re.compile(r"\bswordfish\b", re.I),
    re.compile(r"\b(api[_-]?key|token|secret)\s*[:=].*", re.I),
]

REDACTION = "[скрыто политикой безопасности]"


def contains_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def scan_document(text: str) -> List[str]:
    """Слой 1: возвращает список найденных подозрительных сигнатур (для лога на ingest)."""
    findings = []
    for p in INJECTION_PATTERNS:
        m = p.search(text)
        if m:
            findings.append(m.group(0))
    return findings


def strip_injection(text: str) -> str:
    """Слой 1: нейтрализует строки-директивы внутри текста чанка."""
    cleaned_lines = []
    for line in text.splitlines():
        if any(p.search(line) for p in INJECTION_PATTERNS):
            cleaned_lines.append("[инструкция из документа удалена слоем безопасности]")
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def is_query_safe(query: str) -> Tuple[bool, str]:
    """Safety-классификатор входа: блокирует явные jailbreak/extraction-запросы."""
    for p in JAILBREAK_PATTERNS:
        if p.search(query):
            return False, "Запрос похож на попытку обойти инструкции/раскрыть системный промпт."
    return True, ""


def filter_chunks(chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Post-retrieval фильтр: чистит инъекции в чанках, помечает подозрительные.

    Возвращает (очищенные_чанки, список_подозрительных).
    """
    cleaned: List[Dict] = []
    flagged: List[Dict] = []
    for c in chunks:
        if contains_injection(c.get("text", "")):
            flagged.append(c)
            safe = dict(c)
            safe["text"] = strip_injection(c["text"])
            cleaned.append(safe)
        else:
            cleaned.append(c)
    return cleaned, flagged


def guard_output(answer: str) -> Tuple[str, bool, str]:
    """Слой 3: вырезает секреты и следы системных инструкций из финального ответа.

    Возвращает (очищенный_ответ, был_ли_тронут, причина).
    """
    flagged = False
    reason = ""
    cleaned = answer

    # вырезаем внутренние рассуждения CoT, если просочились
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S | re.I).strip()

    for p in SECRET_PATTERNS:
        if p.search(cleaned):
            cleaned = p.sub(REDACTION, cleaned)
            flagged = True
            reason = "Обнаружен и скрыт потенциальный секрет (пароль/токен)."

    # если в ответе осталась инъекционная директива — нейтрализуем
    if contains_injection(cleaned):
        cleaned = strip_injection(cleaned)
        flagged = True
        reason = reason or "Обнаружена и удалена инъекционная директива в ответе."

    return cleaned, flagged, reason
