"""Small OpenAI-style embedding server for local Letta experiments."""
from __future__ import annotations

import argparse
import os
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    user: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="CEA-MI local embedding server")
    model_name = os.environ.get("CEA_MI_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    device = os.environ.get("CEA_MI_EMBEDDING_DEVICE")
    normalize = os.environ.get("CEA_MI_EMBEDDING_NORMALIZE", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    model = None

    def get_model():
        nonlocal model
        if model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if device:
                kwargs["device"] = device
            model = SentenceTransformer(model_name, **kwargs)
        return model

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_name}

    @app.post("/embeddings")
    def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        texts = request.input if isinstance(request.input, list) else [request.input]
        vectors = get_model().encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return {
            "object": "list",
            "model": request.model or model_name,
            "data": [
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": vector.tolist(),
                }
                for idx, vector in enumerate(vectors)
            ],
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("CEA_MI_EMBEDDING_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CEA_MI_EMBEDDING_PORT", "8290")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
