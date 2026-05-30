"""RAG-бот для корпоративной базы знаний QuantumForge Software (спринт 7).

Пакет реализует полный RAG-конвейер из 8 шагов (препроцессинг → эмбеддинг →
recall → rerank/trim → prompt → LLM → постобработка), три слоя защиты от
prompt-injection и подключаемый слой LLM (offline / OpenAI / Ollama).
"""

__version__ = "1.0.0"
