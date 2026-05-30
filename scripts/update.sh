#!/usr/bin/env bash
# Задание 6: обёртка для запуска обновления индекса по расписанию (cron).
#
# Установка в cron (ежедневно в 06:00):
#   crontab -e
#   0 6 * * * /path/to/5.\ sprint-7-project/scripts/update.sh >> /path/to/logs/cron.log 2>&1
#
# При ошибке update_index.py вернёт код 1 — cron зафиксирует это в cron.log.
set -euo pipefail

# Каталог проекта = на уровень выше scripts/
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Используем venv, если он есть, иначе системный python3
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

echo "[update.sh] $(date '+%Y-%m-%d %H:%M:%S') старт обновления в $PROJECT_DIR"
"$PY" update_index.py
echo "[update.sh] $(date '+%Y-%m-%d %H:%M:%S') готово"
