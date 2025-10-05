FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# чуть ускорим pip и не держим кеш
RUN pip config set global.no-cache-dir true

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app_embed.py .

# Render передаст переменную PORT — слушаем её
CMD ["sh", "-c", "uvicorn app_embed:app --host 0.0.0.0 --port $PORT"]
