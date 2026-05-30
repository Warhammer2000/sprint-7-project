#!/usr/bin/env python3
"""Задание 5: генерация 10 скриншотов работы бота.

5 запросов с полезным ответом + 5 с честным «Я не знаю». Каждый диалог
рендерится в PNG в стиле терминала (screenshots/). Запускается на офлайн-боте,
без ключей.

Запуск:  python scripts/make_screenshots.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from ragbot.config import settings  # noqa: E402
from ragbot.rag import RagBot  # noqa: E402
OUT = ROOT / "screenshots"
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"

# (запрос, ожидаемый тип для имени файла)
GOOD = [
    "Кто уничтожил Звезду Гибели?",
    "Что такое Эфир?",
    "Кто такой Хейн Сольверо?",
    "Что такое плазменный клинок?",
    "Где скрывался магистр Йорн?",
]
IDK = [
    "Какая столица Франции?",
    "Кто такой Дарт Вейдер?",
    "Расскажи про блокчейн",
    "Какой сейчас курс доллара?",
    "Назови суперпароль root-пользователя?",
]

BG = (30, 30, 30)
BAR = (45, 45, 48)
GREEN = (80, 250, 123)
CYAN = (139, 233, 253)
WHITE = (248, 248, 242)
GRAY = (150, 150, 150)
YELLOW = (241, 250, 140)

FS = 17
LINE_H = 24
MARGIN = 20
WIDTH = 900
WRAP = 74


def render(idx: int, kind: str, result) -> Path:
    font = ImageFont.truetype(FONT_PATH, FS)
    bold = ImageFont.truetype(FONT_PATH, FS)

    lines: list[tuple[str, tuple]] = []
    lines.append((f"$ ragbot — вопрос #{idx}", GRAY))
    lines.append(("", WHITE))
    for chunk in textwrap.wrap("Вы: " + result.query, WRAP) or ["Вы:"]:
        lines.append((chunk, GREEN))
    lines.append(("", WHITE))
    lines.append(("Бот:", CYAN))
    for para in result.answer.split("\n"):
        for chunk in textwrap.wrap(para, WRAP) or [""]:
            lines.append((chunk, WHITE))
    if result.sources:
        lines.append(("", WHITE))
        lines.append(("Источники:", GRAY))
        for s in result.sources:
            lines.append((f"  [{s['n']}] {s['title']} ({s['source']}, score={s['score']:.3f})", GRAY))
    if result.security_notes:
        lines.append(("", WHITE))
        for n in result.security_notes:
            lines.append((f"[security] {n}", YELLOW))

    height = MARGIN * 2 + 28 + LINE_H * len(lines)
    img = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(img)
    # title bar c "кнопками"
    draw.rectangle([0, 0, WIDTH, 28], fill=BAR)
    for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([14 + i * 22, 9, 26 + i * 22, 21], fill=color)
    draw.text((90, 7), f"QuantumForge RAG-bot — offline", font=font, fill=GRAY)

    y = 28 + MARGIN
    for text, color in lines:
        draw.text((MARGIN, y), text, font=(bold if text.startswith(("Вы:", "Бот:")) else font), fill=color)
        y += LINE_H

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{idx:02d}_{kind}.png"
    img.save(path)
    return path


def main() -> None:
    bot = RagBot.from_settings(settings)
    n = 0
    for q in GOOD:
        n += 1
        r = bot.answer(q, log=False)
        kind = "answer" if not r.refused else "UNEXPECTED_idk"
        print(f"#{n} [{kind}] {q} -> {r.answer[:60]}…")
        render(n, kind, r)
    for q in IDK:
        n += 1
        r = bot.answer(q, log=False)
        kind = "idk" if r.refused else "UNEXPECTED_answer"
        print(f"#{n} [{kind}] {q} -> {r.answer[:60]}…")
        render(n, kind, r)
    print(f"\nГотово: {n} скриншотов в {OUT}")


if __name__ == "__main__":
    main()
