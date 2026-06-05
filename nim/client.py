"""
NVIDIA NIM Client — NVIDIA Inference Microservices
Wraps the OpenAI-compatible NIM API at api.nvidia.com.

Usage:
    from nim.client import NIMClient
    client = NIMClient()
    response = client.chat("Tell me about the ancient Silk Road")
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Iterator

import requests

from config import config


@dataclass
class NIMResponse:
    content: str
    model: str
    usage: dict[str, int]
    finish_reason: str
    request_id: str


class NIMClient:
    """OpenAI-compatible client for NVIDIA NIM inference microservices."""

    DEFAULT_CHAT_MODEL = "meta/llama-3.1-70b-instruct"
    DEFAULT_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
    DEFAULT_VISION_MODEL = "microsoft/phi-3-vision-128k-instruct"

    AVAILABLE_MODELS = {
        "chat": [
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-8b-instruct",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "microsoft/phi-3-medium-128k-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "google/gemma-2-27b-it",
        ],
        "embedding": [
            "nvidia/nv-embedqa-e5-v5",
            "snowflake/arctic-embed-l",
            "nvidia/nv-embed-v1",
        ],
        "vision": [
            "microsoft/phi-3-vision-128k-instruct",
            "nvidia/neva-22b",
        ],
        "rerank": [
            "nvidia/rerank-qa-mistral-4b",
        ],
    }

    def __init__(self) -> None:
        self.api_key = config.api_key
        self.base_url = config.nim_base_url
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def _headers(self, trace_id: str | None = None) -> dict[str, str]:
        return {
            "x-sap-node-id": config.sap_node_id,
            "x-sap-trace-id": trace_id or str(uuid.uuid4()),
            "x-sap-version": "1.0",
        }

    def chat(
        self,
        message: str,
        model: str = DEFAULT_CHAT_MODEL,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        trace_id: str | None = None,
    ) -> NIMResponse:
        """Send a chat message to a NIM language model."""
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not set. Get your key at build.nvidia.com")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        return NIMResponse(
            content=choice["message"]["content"],
            model=data["model"],
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
            request_id=data.get("id", str(uuid.uuid4())),
        )

    def stream_chat(
        self,
        message: str,
        model: str = DEFAULT_CHAT_MODEL,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream a chat response token by token."""
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not set")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        with self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    data_str = line[6:].decode()
                    if data_str.strip() == "[DONE]":
                        break
                    import json
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def embed(
        self,
        texts: list[str],
        model: str = DEFAULT_EMBED_MODEL,
        input_type: str = "query",
    ) -> list[list[float]]:
        """Generate text embeddings via NIM."""
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not set")

        payload = {
            "model": model,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }

        resp = self._session.post(
            f"{self.base_url}/embeddings",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def vision_chat(
        self,
        message: str,
        image_url: str,
        model: str = DEFAULT_VISION_MODEL,
        max_tokens: int = 1024,
    ) -> NIMResponse:
        """Send a vision + language message to a NIM vision model."""
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not set")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }

        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        return NIMResponse(
            content=choice["message"]["content"],
            model=data["model"],
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
            request_id=data.get("id", str(uuid.uuid4())),
        )

    def list_models(self) -> dict[str, list[str]]:
        """Return the catalog of available NIM models by category."""
        return self.AVAILABLE_MODELS


if __name__ == "__main__":
    client = NIMClient()
    issues = config.validate()
    if issues:
        print("Config issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("NIM client ready.")
        resp = client.chat("What is NVIDIA Omniverse and how does it enable digital world building?")
        print(f"\nResponse:\n{resp.content}")
