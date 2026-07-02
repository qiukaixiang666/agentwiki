from __future__ import annotations

import httpx

from .config import settings


class EmbeddingClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or settings.embedding_service_url).rstrip("/")
        self.timeout = timeout or settings.embedding_timeout_seconds
        self.provider = (provider or settings.embedding_provider).lower()
        self.model = model or settings.embedding_model
        self.api_key = api_key if api_key is not None else settings.embedding_api_key

    def _uses_openai_embeddings(self) -> bool:
        return (
            self.provider in {"vllm", "openai", "openai_compatible"}
            or self.base_url.endswith("/v1")
        )

    async def embed(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if self._uses_openai_embeddings():
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=self._headers(),
                    json={"model": self.model, "input": texts},
                )
                response.raise_for_status()
                payload = response.json()
                return _vectors_from_openai_payload(payload)

            response = await client.post(
                f"{self.base_url}/embed",
                json={"texts": texts, "normalize": normalize},
            )
            if response.status_code in {404, 405, 422}:
                response = await client.post(
                    f"{self.base_url}/v1/embeddings",
                    headers=self._headers(),
                    json={"model": self.model, "input": texts},
                )
            response.raise_for_status()
            payload = response.json()
            if "data" in payload:
                return _vectors_from_openai_payload(payload)
            return payload["embeddings"]

    async def embed_one(self, text: str, normalize: bool = True) -> list[float]:
        vectors = await self.embed([text], normalize=normalize)
        return vectors[0]

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if self._uses_openai_embeddings():
                response = await client.get(f"{self.base_url}/models", headers=self._headers())
                response.raise_for_status()
                payload = response.json()
                return {
                    "ok": True,
                    "service": "openai_compatible_embedding",
                    "details": {
                        "base_url": self.base_url,
                        "model": self.model,
                        "models": [
                            item.get("id")
                            for item in payload.get("data", [])
                            if isinstance(item, dict)
                        ],
                    },
                }
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}


def _vectors_from_openai_payload(payload: dict) -> list[list[float]]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("OpenAI-compatible embeddings response missing data list")
    vectors = []
    for item in sorted(data, key=lambda row: row.get("index", 0)):
        if not isinstance(item, dict) or "embedding" not in item:
            raise ValueError(f"Invalid embedding item: {item!r}")
        vectors.append(item["embedding"])
    return vectors
