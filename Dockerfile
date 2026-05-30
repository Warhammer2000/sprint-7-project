# Минимальный образ RAG-бота (бот + FAISS in-process).
# По умолчанию офлайн-режим (tfidf + offline LLM) — работает без ключей.
# Для качества: пересоберите с requirements.txt и задайте LLM_BACKEND/EMBEDDINGS_BACKEND.
FROM python:3.11-slim

WORKDIR /app

# Лёгкие зависимости: numpy, scikit-learn, faiss-cpu, fastapi, uvicorn, dotenv, requests
COPY requirements-light.txt .
RUN pip install --no-cache-dir -r requirements-light.txt

COPY . .

EXPOSE 8000

# Если индекс не закоммичен/удалён — собираем его, затем поднимаем REST API.
CMD ["sh", "-c", "[ -f index/faiss.index ] || EMBEDDINGS_BACKEND=tfidf python build_index.py; uvicorn api:app --host 0.0.0.0 --port 8000"]
