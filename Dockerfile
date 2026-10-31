# Этап сборки зависимостей
FROM python:3.12-slim AS builder

# Устанавливаем системные зависимости, необходимые для LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Создаем виртуальное окружение и устанавливаем библиотеки
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --no-cache fastapi uvicorn pandas numpy scikit-learn lightgbm jinja2

# Финальный легковесный образ
FROM python:3.12-slim

# ВАЖНО: На финальный этап тоже нужно скопировать libgomp1, так как LightGBM ищет её в системе при запуске!
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем виртуальное окружение из builder-этапа
COPY --from=builder /app/.venv /app/.venv

# Активируем виртуальное окружение по умолчанию через PATH
ENV PATH="/app/.venv/bin:$PATH"

# Копируем исходный код, модель и скрипт
COPY app/ ./app/
COPY model.bin ./model.bin

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]