#!/usr/bin/env python3
"""Задание 2, шаг 2: подмена терминов raw_source/ -> knowledge_base/.

Читает data/terms_map.json и для каждого исходного документа заменяет все
оригинальные термины на вымышленные. Замена — пословная (с границами слова),
longest-match-first, с сохранением регистра первой буквы. Термины из
case_sensitive заменяются только в точном регистре (чтобы не задеть обычное
слово «сила»).

После генерации выполняется проверка: ни один оригинальный термин из словаря
не должен остаться в knowledge_base/ (иначе модель сможет «угадать» вселенную).

Запуск:
    python scripts/build_kb.py            # сгенерировать + проверить
    python scripts/build_kb.py --check    # только проверить
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_source"
KB = ROOT / "knowledge_base"
TERMS = ROOT / "data" / "terms_map.json"


def load_map() -> Tuple[Dict[str, str], set]:
    data = json.loads(TERMS.read_text(encoding="utf-8"))
    return data["replacements"], set(data.get("case_sensitive", []))


def _compile(replacements: Dict[str, str], case_sensitive: set):
    """Готовит список (pattern, value, case_sensitive), longest-match-first."""
    items = sorted(replacements.items(), key=lambda kv: len(kv[0]), reverse=True)
    compiled = []
    for key, value in items:
        cs = key in case_sensitive
        flags = 0 if cs else re.IGNORECASE
        pat = re.compile(r"\b" + re.escape(key) + r"\b", flags)
        compiled.append((pat, value, cs))
    return compiled


def _make_repl(value: str, case_sensitive: bool):
    def _r(m: re.Match) -> str:
        if case_sensitive:
            return value
        matched = m.group(0)
        if matched[:1].isupper():
            return value[:1].upper() + value[1:]
        return value
    return _r


def apply_map(text: str, compiled) -> str:
    for pat, value, cs in compiled:
        text = pat.sub(_make_repl(value, cs), text)
    return text


def build() -> int:
    replacements, case_sensitive = load_map()
    compiled = _compile(replacements, case_sensitive)
    KB.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(RAW.glob("*.md")):
        out = apply_map(src.read_text(encoding="utf-8"), compiled)
        (KB / src.name).write_text(out, encoding="utf-8")
        n += 1
    print(f"Сгенерировано {n} документов базы знаний в {KB}")
    return n


# Корни оригинальных терминов (ловят пропущенные падежные формы, которых нет
# в словаре как точных ключей). Сила/Силы не включены: их обычная форма «сила»
# (мощь) — нормальное слово, она проверяется отдельно по точным ключам.
FORBIDDEN_STEMS = [
    "Скайуок", "Вейдер", "Энакин", "Амидал", "Палпатин", "Сидиус", "Кеноби",
    "Чубакк", "Джабб", "Хатт", "джеда", "ситх", "Татуин", "Корусант", "Дагоб",
    "Альдераан", "Эндор", "Набу", "вуки", "эвок", "кайбер", "бластер",
    "повстанц", "Импери", "Республик", "сокол", "X-wing", "TIE",
    "дроид", "штурмов",
]


def check() -> List[str]:
    """Проверяет, что ни один оригинальный термин/корень не остался в knowledge_base/."""
    replacements, case_sensitive = load_map()
    leaks: List[str] = []
    for doc in sorted(KB.glob("*.md")):
        text = doc.read_text(encoding="utf-8")
        # 1) точные ключи словаря
        for key in replacements:
            flags = 0 if key in case_sensitive else re.IGNORECASE
            if re.search(r"\b" + re.escape(key) + r"\b", text, flags):
                leaks.append(f"{doc.name}: остался термин «{key}»")
        # 2) корни оригинальных терминов (на случай пропущенных форм)
        for stem in FORBIDDEN_STEMS:
            m = re.search(stem + r"[а-яё]*", text, re.IGNORECASE)
            if m:
                leaks.append(f"{doc.name}: остался корень «{stem}» → «{m.group(0)}»")
    return leaks


def main() -> None:
    only_check = "--check" in sys.argv
    if not only_check:
        build()
    leaks = check()
    if leaks:
        print(f"\n⚠️  Найдены утечки оригинальных терминов ({len(leaks)}):")
        for l in leaks[:50]:
            print("  -", l)
        sys.exit(1)
    print("✅ Проверка пройдена: оригинальных терминов в базе знаний не осталось.")


if __name__ == "__main__":
    main()
