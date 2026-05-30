#!/usr/bin/env python3
"""Задание 4: консольный интерфейс RAG-бота (REPL).

Запуск:
    python build_index.py        # сначала собрать индекс
    python cli.py                # затем задавать вопросы

Команды внутри: пустая строка или 'exit'/'quit' — выход.
"""
from __future__ import annotations

import sys

from ragbot.config import settings
from ragbot.rag import RagBot


def main() -> None:
    try:
        bot = RagBot.from_settings(settings)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    print("RAG-бот QuantumForge готов.")
    print(f"  LLM: {bot.llm.name} | эмбеддер: {bot.store.embedder.backend} | "
          f"чанков: {len(bot.store.metas)} | защита: {'on' if settings.security_enabled else 'off'}")
    print("Задайте вопрос (пустая строка — выход).\n")

    while True:
        try:
            query = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"exit", "quit", "выход"}:
            break
        result = bot.answer(query)
        print("\nБот:", result.format(), "\n")


if __name__ == "__main__":
    main()
