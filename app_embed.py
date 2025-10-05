# app_embed.py
import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastembed import TextEmbedding

app = FastAPI()

# Читаем имя модели из переменных окружения (можно задать в Render)
DEFAULT_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

def _init_embedder(model_name: str) -> TextEmbedding:
    try:
        return TextEmbedding(model_name=model_name)
    except ValueError as e:
        # если задали неподдерживаемую модель — пробуем дефолт, иначе пробрасываем
        if model_name != "intfloat/multilingual-e5-base":
            return TextEmbedding(model_name="intfloat/multilingual-e5-base")
        # собрать список поддерживаемых и показать в ошибке
        supported = TextEmbedding.list_supported_models()
        raise RuntimeError(
            f"{e}. Supported models: {supported}"
        )

embedder = _init_embedder(DEFAULT_MODEL)

class EmbedIn(BaseModel):
    texts: List[str]

@app.get("/health")
def health():
    return {"ok": True, "model": DEFAULT_MODEL}

@app.get("/models")
def models():
    return {"supported": TextEmbedding.list_supported_models()}

@app.post("/embed")
def embed(payload: EmbedIn):
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")
    # FastEmbed возвращает генератор векторов
    vectors = [v.tolist() for v in embedder.embed(payload.texts, batch_size=64)]
    dim = len(vectors[0]) if vectors else 0
    return {"vectors": vectors, "dim": dim, "model": DEFAULT_MODEL}
