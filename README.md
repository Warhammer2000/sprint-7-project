# Sprint 7 — RAG-бот для базы знаний QuantumForge Software

Чат-бот на основе **Retrieval-Augmented Generation (RAG)**: ищет релевантные
фрагменты в векторной базе знаний (FAISS) и формирует ответ с опорой на них,
применяя few-shot и Chain-of-Thought, с тремя слоями защиты от prompt-injection.

Проектная работа 7 спринта. Полное описание решения по всем 7 заданиям —
в **[`Project_template.md`](Project_template.md)**.

> 🟢 **Работает офлайн без API-ключей** (эмбеддер `tfidf` + экстрактивный
> `offline`-LLM). Для качества — Sentence-Transformers и OpenAI/Ollama
> переключаются через `.env`.

## Архитектура (8 шагов RAG)

```
запрос → препроцессинг → safety-проверка → эмбеддинг
       → ANN-поиск (FAISS) → отсев по порогу + grounding
       → top_k чанков → промпт (few-shot + CoT, безопасный System)
       → LLM (offline / OpenAI / Ollama) → guard вывода → ответ + источники
```

Подключаемые слои (`ragbot/`):
- **embeddings** — `st` (Sentence-Transformers) | `tfidf` (офлайн);
- **llm** — `offline` (экстрактивный) | `openai` | `ollama`;
- **vector_store** — FAISS `IndexFlatIP` (косинус) + метаданные;
- **security** — фильтр корпуса, безопасный промпт, guard вывода.

## Быстрый старт

```bash
make install-light        # numpy, scikit-learn, faiss-cpu, fastapi, dotenv
make index                # построить индекс (или используйте закоммиченный)
make run                  # консольный бот (REPL)
```

REST API:

```bash
make api                  # FastAPI на http://localhost:8000
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"query":"Кто уничтожил Звезду Гибели?"}'
```

Docker:

```bash
docker compose up --build # бот + FAISS, REST на :8000
```

Демонстрации:

```bash
make demo                 # защита от prompt-injection (вкл/выкл + 10 запросов)
make eval                 # автооценка на золотом наборе (known/gap)
make update               # инкрементальное обновление индекса
```

## Структура

| Путь | Назначение |
| --- | --- |
| `ragbot/` | ядро: конфиг, чанкинг, эмбеддинги, FAISS, промпты, LLM, security, пайплайн, логи |
| `build_index.py` | Задание 3 — сборка векторного индекса |
| `cli.py`, `api.py` | Задание 4 — интерфейсы (REPL / FastAPI) |
| `demo_security.py` | Задание 5 — демонстрация защиты |
| `update_index.py` | Задание 6 — авто-обновление базы |
| `evaluate.py` | Задание 7 — оценка покрытия |
| `scripts/` | `seed_raw_source.py`, `build_kb.py` (Задание 2), `update.sh` |
| `raw_source/`, `knowledge_base/` | база знаний до и после подмены терминов |
| `data/` | `terms_map.json`, `malicious/injection.txt` |
| `tests/`, `reports/`, `diagrams/` | golden set, отчёты, диаграммы PlantUML/Mermaid |
| `index/`, `logs/` | готовый индекс и примеры логов |

## Конфигурация

Скопируйте `.env.example` → `.env`. Основное:
`LLM_BACKEND` (offline/openai/ollama), `EMBEDDINGS_BACKEND` (st/tfidf),
`TOP_K`, `MIN_SCORE`, `USE_FEW_SHOT`, `USE_COT`, `SECURITY_ENABLED`.
