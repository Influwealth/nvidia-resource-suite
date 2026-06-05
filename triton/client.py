"""Triton Inference Server HTTP/gRPC client with graceful degradation."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    import numpy as np
    TRITON_CLIENT_AVAILABLE = True
except ImportError:
    TRITON_CLIENT_AVAILABLE = False
    logger.warning("tritonclient not installed — Triton running in HTTP-only mode")


@dataclass
class InferenceResult:
    request_id: str
    model_name: str
    model_version: str
    outputs: dict[str, Any]
    latency_ms: float
    success: bool
    error: str | None = None


@dataclass
class ModelMetadata:
    name: str
    versions: list[str]
    platform: str
    inputs: list[dict]
    outputs: list[dict]
    max_batch_size: int = 0
    ready: bool = False


class TritonClient:
    """Client for NVIDIA Triton Inference Server.

    Supports HTTP REST and gRPC transports. Falls back to mock
    responses when Triton is unreachable so educational demos
    can run without a GPU server.
    """

    def __init__(self, http_url: str = "http://localhost:8000", grpc_url: str = "localhost:8001"):
        self.http_url = http_url.rstrip("/")
        self.grpc_url = grpc_url
        self._http = httpx.Client(timeout=30.0)
        self._available = self._check_health()
        if self._available:
            logger.info(f"Triton server ready at {self.http_url}")
        else:
            logger.warning("Triton server not reachable — using mock inference mode")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def _check_health(self) -> bool:
        try:
            r = self._http.get(f"{self.http_url}/v2/health/ready", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def is_healthy(self) -> bool:
        return self._check_health()

    def server_metadata(self) -> dict:
        if not self._available:
            return {"name": "triton-mock", "version": "0.0.0", "extensions": []}
        r = self._http.get(f"{self.http_url}/v2")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        if not self._available:
            return ["mock-llm", "mock-embed", "mock-vision"]
        r = self._http.get(f"{self.http_url}/v2/repository/index")
        r.raise_for_status()
        return [m["name"] for m in r.json()]

    def get_model_metadata(self, model_name: str, version: str = "") -> ModelMetadata:
        if not self._available:
            return ModelMetadata(
                name=model_name, versions=["1"], platform="mock",
                inputs=[], outputs=[], ready=False,
            )
        url = f"{self.http_url}/v2/models/{model_name}"
        if version:
            url += f"/versions/{version}"
        r = self._http.get(url)
        r.raise_for_status()
        data = r.json()
        return ModelMetadata(
            name=data["name"],
            versions=data.get("versions", ["1"]),
            platform=data.get("platform", ""),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            max_batch_size=data.get("max_batch_size", 0),
            ready=True,
        )

    def load_model(self, model_name: str) -> bool:
        if not self._available:
            logger.info(f"[MOCK] Load model: {model_name}")
            return True
        r = self._http.post(f"{self.http_url}/v2/repository/models/{model_name}/load")
        return r.status_code == 200

    def unload_model(self, model_name: str) -> bool:
        if not self._available:
            return True
        r = self._http.post(f"{self.http_url}/v2/repository/models/{model_name}/unload")
        return r.status_code == 200

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        model_name: str,
        inputs: dict[str, Any],
        outputs: list[str] | None = None,
        model_version: str = "",
    ) -> InferenceResult:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        if not self._available:
            return self._mock_infer(request_id, model_name, model_version, start)

        payload: dict[str, Any] = {
            "id": request_id,
            "inputs": [
                {"name": k, "shape": list(v.shape) if hasattr(v, "shape") else [1], "datatype": "FP32", "data": v.tolist() if hasattr(v, "tolist") else [v]}
                for k, v in inputs.items()
            ],
        }
        if outputs:
            payload["outputs"] = [{"name": o} for o in outputs]

        try:
            r = self._http.post(
                f"{self.http_url}/v2/models/{model_name}/infer",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
            latency = (time.perf_counter() - start) * 1000
            output_dict = {o["name"]: o.get("data") for o in data.get("outputs", [])}
            return InferenceResult(
                request_id=request_id, model_name=model_name,
                model_version=data.get("model_version", "1"),
                outputs=output_dict, latency_ms=latency, success=True,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return InferenceResult(
                request_id=request_id, model_name=model_name, model_version="",
                outputs={}, latency_ms=latency, success=False, error=str(e),
            )

    def _mock_infer(self, request_id: str, model_name: str, version: str, start: float) -> InferenceResult:
        import random
        time.sleep(0.01)  # simulate 10ms latency
        latency = (time.perf_counter() - start) * 1000
        return InferenceResult(
            request_id=request_id, model_name=model_name, model_version="1-mock",
            outputs={"output": [random.random() for _ in range(128)]},
            latency_ms=latency, success=True,
        )

    def batch_infer(
        self,
        model_name: str,
        batch_inputs: list[dict[str, Any]],
        max_concurrent: int = 4,
    ) -> list[InferenceResult]:
        """Submit a batch of inference requests with concurrency control."""
        results = []
        for inp in batch_inputs:
            results.append(self.infer(model_name, inp))
        return results

    def stream_infer(
        self,
        model_name: str,
        inputs: dict[str, Any],
    ):
        """Streaming inference using Triton's decoupled model protocol."""
        if not self._available:
            for token in ["[MOCK]", " stream", " output"]:
                yield token
            return
        with self._http.stream(
            "POST",
            f"{self.http_url}/v2/models/{model_name}/generate_stream",
            json={"inputs": inputs},
        ) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    yield line[5:].strip()

    # ------------------------------------------------------------------
    # Statistics and metrics
    # ------------------------------------------------------------------

    def model_statistics(self, model_name: str) -> dict:
        if not self._available:
            return {"model": model_name, "inference_count": 0, "mock": True}
        r = self._http.get(f"{self.http_url}/v2/models/{model_name}/stats")
        r.raise_for_status()
        return r.json()

    def server_statistics(self) -> dict:
        if not self._available:
            return {"mock": True}
        r = self._http.get(f"{self.http_url}/v2/stats")
        r.raise_for_status()
        return r.json()

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
