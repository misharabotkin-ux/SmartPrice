# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Предскачиваем модель в образ (ускорит холодный старт на бесплатном хостинге)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

COPY app_embed.py .

EXPOSE 8000
CMD ["uvicorn", "app_embed:app", "--host", "0.0.0.0", "--port", "8000"]
