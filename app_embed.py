from fastapi import FastAPI, Body
from fastembed import TextEmbedding

# один раз инициализируем модель (скачается и закешируется)
# размер вектора у e5-small = 384
MODEL_NAME = "intfloat/multilingual-e5-small"
embedder = TextEmbedding(model_name=MODEL_NAME)

app = FastAPI(title="E5 embed microservice", version="1.0")

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}

@app.post("/embed")
def embed(payload: dict = Body(...)):
    """
    payload = {
      "texts": ["копка земли механизировано", "поклейка обоев"],
      "mode": "query" | "passage"   # по умолчанию passage
    }
    """
    texts = payload.get("texts") or []
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return {"vectors": [], "dim": 0, "model": MODEL_NAME}

    mode = (payload.get("mode") or "passage").lower()
    prefix = "query: " if mode == "query" else "passage: "

    prepared = [prefix + t for t in texts]

    # FastEmbed возвращает генератор numpy-векторов
    vectors = [v.tolist() for v in embedder.embed(prepared)]
    dim = len(vectors[0]) if vectors else 0
    return {"vectors": vectors, "dim": dim, "model": MODEL_NAME}
