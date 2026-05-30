"""Подключаемый слой LLM (Задание 4 — генерация ответа).

Три бэкенда, выбираются через LLM_BACKEND:

* ``offline`` — без сети и ключей. Экстрактивный ответ: собирает наиболее
                релевантные предложения из найденных чанков и проставляет ссылки.
                Если релевантных чанков нет — честно отвечает «Я не знаю».
                Идеален для запуска, тестов и скриншотов без API-ключа.
* ``openai``  — chat.completions (gpt-4o-mini и совместимые). Полноценные
                few-shot и Chain-of-Thought.
* ``ollama``  — локальная модель через REST API Ollama (llama3.1 и т. п.).

Сравнение этих вариантов по качеству/скорости/стоимости — в reports/task1_research.md.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

from .prompts import build_messages

_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.I)
IDK = "Я не знаю: в предоставленной базе знаний нет информации по этому вопросу."


def _tokens(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) > 2}


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def _strip_think(text: str) -> str:
    """Убирает внутренние CoT-рассуждения <think>…</think> из ответа."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()


class BaseLLM:
    name = "base"

    def answer(
        self, question: str, contexts: Sequence[Dict],
        use_few_shot: bool = True, use_cot: bool = True,
    ) -> str:
        raise NotImplementedError


class OfflineLLM(BaseLLM):
    """Экстрактивный ответчик без внешних вызовов."""

    name = "offline"

    def answer(self, question, contexts, use_few_shot=True, use_cot=True) -> str:
        if not contexts:
            return IDK
        q = _tokens(question)
        scored = []
        for i, c in enumerate(contexts, start=1):
            for sent in _sentences(c.get("text", "")):
                overlap = len(q & _tokens(sent))
                scored.append((overlap, i, sent))
        scored.sort(key=lambda x: -x[0])
        best = [s for s in scored if s[0] > 0][:2]
        if not best:
            # нет пересечения по словам — берём первое предложение лучшего чанка
            first = _sentences(contexts[0].get("text", ""))
            if not first:
                return IDK
            best = [(0, 1, first[0])]
        answer = " ".join(sent for _, _, sent in best)
        cites = sorted({i for _, i, _ in best})
        return f"{answer} " + "".join(f"[{i}]" for i in cites)


class OpenAILLM(BaseLLM):
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str = ""):
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def answer(self, question, contexts, use_few_shot=True, use_cot=True) -> str:
        if not contexts:
            return IDK
        messages = build_messages(question, contexts, use_few_shot, use_cot)
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.1,
        )
        return _strip_think(resp.choices[0].message.content or "")


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def answer(self, question, contexts, use_few_shot=True, use_cot=True) -> str:
        import requests

        if not contexts:
            return IDK
        messages = build_messages(question, contexts, use_few_shot, use_cot)
        resp = requests.post(
            f"{self.host}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False,
                  "options": {"temperature": 0.1}},
            timeout=120,
        )
        resp.raise_for_status()
        return _strip_think(resp.json()["message"]["content"])


def build_llm(settings) -> BaseLLM:
    backend = (settings.llm_backend or "offline").lower()
    if backend == "openai":
        if not settings.openai_api_key:
            print("[llm] OPENAI_API_KEY пуст — откатываюсь в offline-режим.")
            return OfflineLLM()
        try:
            return OpenAILLM(settings.openai_api_key, settings.openai_model,
                             settings.openai_base_url)
        except Exception as exc:
            print(f"[llm] OpenAI недоступен ({exc.__class__.__name__}) — offline-режим.")
            return OfflineLLM()
    if backend == "ollama":
        return OllamaLLM(settings.ollama_host, settings.ollama_model)
    return OfflineLLM()
