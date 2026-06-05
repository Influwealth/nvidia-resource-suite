"""
NVIDIA NIM Base Client

OpenAI-compatible client for all NIM microservices.
Handles authentication, retries, SAP header propagation, and streaming.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Generator, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

AVAILABLE_MODELS = {
    "chat": [
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.3-70b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/llama-3.1-nemotron-nano-8b-v1",
        "mistralai/mixtral-8x7b-instruct-v0.1",
        "mistralai/mistral-7b-instruct-v0.3",
        "google/gemma-2-27b-it",
        "microsoft/phi-3-mini-128k-instruct",
        "qwen/qwen2.5-72b-instruct",
    ],
    "embedding": [
        "nvidia/nv-embed-v1",
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/nv-embedqa-mistral-7b-v2",
        "baai/bge-m3",
    ],
    "vision": [
        "microsoft/phi-3-vision-128k-instruct",
        "nvidia/neva-22b",
        "liuhaotian/llava-v1.6-34b",
        "adept/fuyu-8b",
    ],
    "rerank": [
        "nvidia/nv-rerankqa-mistral-4b-v3",
        "nvidia/rerank-qa-mistral-4b",
    ],
    "audio": [
        "nvidia/canary-1b",
        "nvidia/parakeet-ctc-1.1b",
    ],
    "code": [
        "meta/codellama-70b",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
    ],
}


class NIMClient:
    """Base NIM client with retry logic, streaming, and SAP header support."""

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    REQUEST_TIMEOUT = 120

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        node_id: str = "NODE_DELTA",
    ):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.base_url = (base_url or os.environ.get("NIM_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.node_id = node_id
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        return session

    def _headers(self, trace_id: str | None = None) -> dict[str, str]:
        tid = trace_id or str(uuid.uuid4())
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-sap-node-id": self.node_id,
            "x-sap-trace-id": tid,
            "x-sap-version": "3.7",
        }
        return h

    def _stream_headers(self, trace_id: str | None = None) -> dict[str, str]:
        h = self._headers(trace_id)
        h["Accept"] = "text/event-stream"
        return h

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "meta/llama-3.1-70b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        trace_id: str | None = None,
    ) -> str:
        """Non-streaming chat completion. Returns full assistant message."""
        if not self.api_key:
            return "[NIM unavailable — set NVIDIA_API_KEY]"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(trace_id),
            json=payload,
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "meta/llama-3.1-70b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        trace_id: str | None = None,
    ) -> Iterator[str]:
        """Streaming chat completion. Yields token chunks."""
        if not self.api_key:
            yield "[NIM unavailable — set NVIDIA_API_KEY]"
            return

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        with self._session.post(
            f"{self.base_url}/chat/completions",
            headers=self._stream_headers(trace_id),
            json=payload,
            stream=True,
            timeout=self.REQUEST_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError):
                        continue

    def embed(
        self,
        texts: list[str],
        model: str = "nvidia/nv-embed-v1",
        input_type: str = "query",
        trace_id: str | None = None,
    ) -> list[list[float]]:
        """Embed a list of texts. Returns list of embedding vectors."""
        if not self.api_key:
            log.warning("NIM unavailable — returning zero embeddings")
            return [[0.0] * 4096 for _ in texts]

        payload = {
            "model": model,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }
        resp = self._session.post(
            f"{self.base_url}/embeddings",
            headers=self._headers(trace_id),
            json=payload,
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def vision_chat(
        self,
        prompt: str,
        image_url: str,
        model: str = "microsoft/phi-3-vision-128k-instruct",
        max_tokens: int = 512,
        trace_id: str | None = None,
    ) -> str:
        """Vision chat — describe or analyze an image."""
        if not self.api_key:
            return "[NIM vision unavailable — set NVIDIA_API_KEY]"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(trace_id),
            json=payload,
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def rerank(
        self,
        query: str,
        passages: list[str],
        model: str = "nvidia/nv-rerankqa-mistral-4b-v3",
        top_n: int = 5,
        trace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank passages by relevance to query. Returns sorted list with scores."""
        if not self.api_key:
            return [{"index": i, "relevance_score": 0.0, "passage": p} for i, p in enumerate(passages)]

        payload = {
            "model": model,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
            "top_n": top_n,
        }
        resp = self._session.post(
            f"{self.base_url}/ranking",
            headers=self._headers(trace_id),
            json=payload,
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {"index": r["index"], "relevance_score": r["relevance_score"], "passage": passages[r["index"]]}
            for r in data.get("rankings", [])
        ]

    def list_models(self) -> dict[str, list[str]]:
        return AVAILABLE_MODELS

    def health(self) -> dict[str, Any]:
        return {
            "nim_available": bool(self.api_key),
            "base_url": self.base_url,
            "models": AVAILABLE_MODELS,
        }


if __name__ == "__main__":
    client = NIMClient()
    if client.api_key:
        resp = client.chat([{"role": "user", "content": "Hello! What can you do?"}])
        print(resp)
    else:
        print("Set NVIDIA_API_KEY to test NIM.")
