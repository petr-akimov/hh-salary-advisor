# Этап сборки зависимостей
FROM python:3.12-slim AS builder

# Устанавливаем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Создаем виртуальное окружение и устанавливаем необходимые библиотеки напрямую
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --no-cache fastapi uvicorn pandas numpy scikit-learn lightgbm jinja2

# Финальный легковесный образ
FROM python:3.12-slim

WORKDIR /app

# Копируем виртуальное окружение из builder-этапа
COPY --from=builder /app/.venv /app/.venv

# Активируем виртуальное окружение по умолчанию через PATH
ENV PATH="/app/.venv/bin:$PATH"

# Копируем исходный код, обученную модель из корня и приложение
COPY app/ ./app/
COPY model.bin ./model.bin
COPY train.py ./

# Открываем порт для FastAPI
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]