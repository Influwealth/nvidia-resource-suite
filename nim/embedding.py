"""
NVIDIA NIM — Embedding Client

Semantic embedding for:
  - Curriculum content search
  - Student progress matching
  - World scene retrieval
  - RAG knowledge base indexing
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from .client import NIMClient

log = logging.getLogger(__name__)


class EmbeddingClient:
    """NIM embedding client with local caching and similarity search."""

    EMBEDDING_DIM = 4096  # nv-embed-v1 dimension

    def __init__(self, nim_client: NIMClient | None = None, cache_path: str | None = None):
        self.nim = nim_client or NIMClient()
        self._cache: dict[str, list[float]] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        if self._cache_path and self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text())

    def embed_single(self, text: str, input_type: str = "query") -> list[float]:
        """Embed a single text. Uses local cache to avoid redundant API calls."""
        cache_key = f"{input_type}:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        vectors = self.nim.embed([text], input_type=input_type)
        result = vectors[0] if vectors else [0.0] * self.EMBEDDING_DIM
        self._cache[cache_key] = result
        self._save_cache()
        return result

    def embed_batch(
        self,
        texts: list[str],
        input_type: str = "passage",
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed a batch of texts, respecting rate limits."""
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self.nim.embed(batch, input_type=input_type)
            results.extend(vectors)
        return results

    def embed_curriculum(
        self,
        curriculum_items: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Embed curriculum content (title + description) for semantic search."""
        texts = [f"{item.get('title', '')} {item.get('description', '')}" for item in curriculum_items]
        vectors = self.embed_batch(texts, input_type="passage")
        return [
            {**item, "embedding": vec}
            for item, vec in zip(curriculum_items, vectors)
        ]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a))
        mag_b = math.sqrt(sum(x ** 2 for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def find_most_relevant(
        self,
        query: str,
        corpus: list[dict[str, Any]],
        text_key: str = "description",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find the most semantically relevant items from a corpus.
        Each item in corpus must have an 'embedding' key, or text_key for on-the-fly embedding.
        """
        query_vec = self.embed_single(query, input_type="query")
        scored = []
        for item in corpus:
            if "embedding" in item:
                item_vec = item["embedding"]
            else:
                item_vec = self.embed_single(item.get(text_key, ""), input_type="passage")
            score = self.cosine_similarity(query_vec, item_vec)
            scored.append({**item, "_relevance_score": score})
        scored.sort(key=lambda x: x["_relevance_score"], reverse=True)
        return scored[:top_k]

    def _save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.write_text(json.dumps(self._cache))
