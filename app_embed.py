# app_embed.py
# Мини-сервис для эмбеддинга (E5). Возвращает только вектор для текста запроса.

from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer

# Модель полегче и быстрее: small. (Можно заменить на base, если нужно выше качество.)
MODEL_NAME = "intfloat/multilingual-e5-small"
model = SentenceTransformer(MODEL_NAME)

app = FastAPI(title="SmartPrice Embed API")

@app.get("/embed", summary="Вернёт вектор для смыслового запроса")
def embed(query: str = Query(..., description="Запрос своими словами")):
    # Важно: префикс "query: " — это часть методики E5
    vec = model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
    return {"vector": vec, "dims": len(vec), "model": MODEL_NAME}
