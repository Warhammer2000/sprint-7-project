PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: help venv install install-light kb index run api demo eval update docker-build docker-up clean

help:
	@echo "Цели:"
	@echo "  make venv          — создать виртуальное окружение .venv"
	@echo "  make install-light — лёгкие зависимости (офлайн, без torch)"
	@echo "  make install       — полный стек (langchain, sentence-transformers, ...)"
	@echo "  make kb            — собрать базу знаний (seed + подмена терминов)"
	@echo "  make index         — построить векторный индекс FAISS"
	@echo "  make run           — консольный бот (REPL)"
	@echo "  make api           — REST API (FastAPI на :8000)"
	@echo "  make demo          — демонстрация защиты от prompt-injection"
	@echo "  make eval          — автооценка на золотом наборе"
	@echo "  make update        — инкрементальное обновление индекса"
	@echo "  make docker-up     — собрать и запустить в Docker"

venv:
	python3 -m venv .venv

install-light: venv
	$(PIP) install -q -r requirements-light.txt

install: venv
	$(PIP) install -q -r requirements.txt

kb:
	$(PY) scripts/seed_raw_source.py
	$(PY) scripts/build_kb.py

index:
	$(PY) build_index.py

run:
	$(PY) cli.py

api:
	$(PY) -m uvicorn api:app --host 0.0.0.0 --port 8000

demo:
	$(PY) demo_security.py

eval:
	$(PY) evaluate.py

update:
	$(PY) update_index.py

docker-build:
	docker compose build

docker-up:
	docker compose up --build

clean:
	rm -rf index/faiss.index index/*.pkl index/*.json logs/*.jsonl logs/*.log
	@echo "Очищены индекс и логи (исходники и база знаний сохранены)."
