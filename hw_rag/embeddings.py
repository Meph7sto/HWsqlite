from __future__ import annotations

import hashlib
import math
import os
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Embedder(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class HashingEmbedder(Embedder):
    """Deterministic local fallback embedding.

    This is intentionally simple and dependency-free. It gives the database a
    real vector channel offline, while keeping the embedder swappable for a
    stronger model later.
    """

    dims: int = 384

    @property
    def name(self) -> str:
        return f"hashing-{self.dims}"

    @property
    def dimension(self) -> int:
        return self.dims

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        tokens = tokenize_for_embedding(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:8], "little") % self.dims
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._dimension = 1536

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return list(response.data[0].embedding)


def make_embedder(provider: str = "auto") -> Embedder:
    if provider.startswith("hashing-"):
        return HashingEmbedder(dims=int(provider.split("-", 1)[1]))
    if provider.startswith("openai:"):
        return OpenAIEmbedder(model=provider.split(":", 1)[1])
    if provider == "hash":
        return HashingEmbedder()
    if provider == "openai":
        return OpenAIEmbedder()
    if provider != "auto":
        raise ValueError(f"Unknown embedding provider: {provider}")
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder()
        except Exception:
            return HashingEmbedder()
    return HashingEmbedder()


def vector_to_blob(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def blob_to_vector(blob: bytes) -> list[float]:
    if not blob:
        return []
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def tokenize_for_embedding(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    current: list[str] = []
    for char in lowered:
        if char.isalnum() or char == "_":
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current.clear()
            if "\u4e00" <= char <= "\u9fff":
                tokens.append(char)
    if current:
        tokens.append("".join(current))
    return tokens
