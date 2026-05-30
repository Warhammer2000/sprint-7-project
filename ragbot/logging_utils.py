"""Логирование запросов к боту в JSONL (Задание 7 — аналитика покрытия).

Каждая строка лога — один запрос с полями: время, текст запроса, были ли
найдены чанки, длина ответа, флаг успешного ответа, найденные источники.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def log_query(
    log_dir: Path,
    query: str,
    chunks_found: int,
    answer: str,
    sources: List[str],
    success: bool,
    refused: bool = False,
    extra: Dict | None = None,
    filename: str = "queries.jsonl",
) -> Dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "chunks_found": chunks_found,
        "answer_len": len(answer or ""),
        "success": bool(success),
        "refused": bool(refused),
        "sources": sources,
    }
    if extra:
        record.update(extra)
    with (log_dir / filename).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_log(log_dir: Path, filename: str = "queries.jsonl") -> List[Dict]:
    path = log_dir / filename
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
