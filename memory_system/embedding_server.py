from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from .config import ensure_directories, settings
from .schemas import EmbedRequest, EmbedResponse, HealthResponse

LOGGER = logging.getLogger("memory.embedding_server")

app = FastAPI(title="BGE-M3 Embedding Service")
_MODEL: Any = None
_DEVICE: str | None = None


def _select_device() -> str:
    if settings.bge_device != "auto":
        if settings.bge_device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available")
                if ":" in settings.bge_device:
                    requested_index = int(settings.bge_device.split(":", 1)[1])
                    device_count = torch.cuda.device_count()
                    if requested_index >= device_count:
                        raise RuntimeError(
                            f"Requested {settings.bge_device}, but only {device_count} CUDA devices are visible"
                        )
            except ValueError as exc:
                raise RuntimeError(f"Invalid BGE_DEVICE value: {settings.bge_device}") from exc
        return settings.bge_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model() -> Any:
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    from sentence_transformers import SentenceTransformer

    if not settings.bge_model_dir.exists():
        raise RuntimeError(f"BGE-M3 model directory not found: {settings.bge_model_dir}")
    _DEVICE = _select_device()
    LOGGER.info("Loading BGE-M3 from %s on %s", settings.bge_model_dir, _DEVICE)
    _MODEL = SentenceTransformer(str(settings.bge_model_dir), device=_DEVICE)
    return _MODEL


@app.on_event("startup")
def startup() -> None:
    ensure_directories()
    _load_model()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _MODEL is not None
    return HealthResponse(
        ok=loaded,
        service="embedding",
        details={
            "model_dir": str(settings.bge_model_dir),
            "device": _DEVICE,
            "loaded": loaded,
        },
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    model = _load_model()
    vectors = model.encode(
        request.texts,
        batch_size=settings.bge_batch_size,
        normalize_embeddings=request.normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    return EmbedResponse(
        embeddings=vectors.tolist(),
        model=str(settings.bge_model_dir),
        dimensions=int(vectors.shape[1]),
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "memory_system.embedding_server:app",
        host=settings.embedding_api_host,
        port=settings.embedding_api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
