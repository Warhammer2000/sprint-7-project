#!/usr/bin/env python3
"""Задание 4: REST-интерфейс RAG-бота на FastAPI.

Запуск:
    python build_index.py
    uvicorn api:app --host 0.0.0.0 --port 8000
    # или: python api.py

Эндпоинты:
    GET  /health        — статус, бэкенды, размер индекса
    POST /ask {query}   — ответ бота с источниками
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from ragbot.config import settings
from ragbot.rag import RagBot

app = FastAPI(title="QuantumForge RAG bot", version="1.0.0")
_bot: Optional[RagBot] = None


def get_bot() -> RagBot:
    global _bot
    if _bot is None:
        _bot = RagBot.from_settings(settings)
    return _bot


class AskRequest(BaseModel):
    query: str


class Source(BaseModel):
    n: int
    title: str
    source: str
    score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[Source]
    refused: bool
    blocked: bool
    success: bool
    security_notes: List[str]
    top_score: float


@app.get("/health")
def health():
    bot = get_bot()
    return {
        "status": "ok",
        "llm": bot.llm.name,
        "embedder": bot.store.embedder.backend,
        "n_chunks": len(bot.store.metas),
        "security": settings.security_enabled,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    bot = get_bot()
    r = bot.answer(req.query)
    return AskResponse(
        query=r.query, answer=r.answer, sources=r.sources, refused=r.refused,
        blocked=r.blocked, success=r.success, security_notes=r.security_notes,
        top_score=r.top_score,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
