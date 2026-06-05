"""
nvidia-resource-suite — MONAD NODE_DELTA: NVIDIA 6G Aerial Substrate

NODE_DELTA hosts all NVIDIA workloads:
  - NIM inference (chat, embedding, vision)
  - Omniverse world rendering
  - GPU job scheduling
  - 6G RAN integration via infraflow-ran-oai

Primary port: 7760 (FastAPI)
MONAD external port: 38412
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

MONAD_VERSION = "3.7"
NODE_ID = "NODE_DELTA"
API_PORT = 7760
MONAD_PORT = 38412

NIM_MODELS = [
    "meta/llama-3.1-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "microsoft/phi-3-vision-128k-instruct",
    "nvidia/nv-embed-v1",
]

WORLD_THEMES = [
    "ancient-silk-road",
    "harlem-renaissance",
    "brooklyn-90s",
    "great-migration",
    "indigenous-americas",
    "east-flatbush-origins",
    "greenville-sovereign",
]


@dataclass
class NODE_DELTA_Registration:
    service_id: str = "nvidia-resource-suite"
    node_id: str = NODE_ID
    api_port: int = API_PORT
    monad_port: int = MONAD_PORT
    capabilities: list[str] = field(default_factory=lambda: [
        "nim_inference",
        "nim_embedding",
        "nim_vision",
        "omniverse_rendering",
        "gpu_scheduling",
        "world_building",
        "education_tutor",
        "ran_6g_integration",
    ])
    nim_models: list[str] = field(default_factory=lambda: NIM_MODELS)
    world_themes: list[str] = field(default_factory=lambda: WORLD_THEMES)
    nim_available: bool = field(default_factory=lambda: bool(os.environ.get("NVIDIA_API_KEY")))
    monad_version: str = MONAD_VERSION
    registered_at: float = field(default_factory=time.time)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def as_sap_headers(self) -> dict[str, str]:
        return {
            "x-sap-node-id": self.node_id,
            "x-sap-trace-id": self.trace_id,
            "x-sap-version": self.monad_version,
            "x-sap-capsule": f"delta-{self.registered_at:.0f}",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "node_id": self.node_id,
            "api_port": self.api_port,
            "monad_port": self.monad_port,
            "capabilities": self.capabilities,
            "nim_models": self.nim_models,
            "world_themes": self.world_themes,
            "nim_available": self.nim_available,
            "monad_version": self.monad_version,
            "registered_at": self.registered_at,
            "trace_id": self.trace_id,
        }


class NODE_DELTA_Handler:
    """NODE_DELTA registration and health for nvidia-resource-suite."""

    SOVEREIGN_QUANT = {
        "level": 4,
        "quantization": "INT4_per_channel",
        "embedding_compression": "16x",
        "transmission_reduction": "78%",
        "vram_min_gb": 8,
    }

    def __init__(self):
        self.registration = NODE_DELTA_Registration()

    def sovereign_quant_status(self) -> dict[str, Any]:
        """Check if current hardware meets SovereignQuant Level 4 requirements."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            vram_mb = int(result.stdout.strip().split("\n")[0]) if result.returncode == 0 else 0
            vram_gb = vram_mb / 1024
            meets_requirement = vram_gb >= self.SOVEREIGN_QUANT["vram_min_gb"]
        except Exception:
            vram_gb = 0.0
            meets_requirement = False

        return {
            **self.SOVEREIGN_QUANT,
            "vram_detected_gb": round(vram_gb, 1),
            "meets_requirement": meets_requirement,
        }

    def status(self) -> dict[str, Any]:
        return {
            "service_id": "nvidia-resource-suite",
            "node_id": NODE_ID,
            "monad_version": MONAD_VERSION,
            "api_port": API_PORT,
            "monad_port": MONAD_PORT,
            "nim_available": self.registration.nim_available,
            "nim_models": NIM_MODELS,
            "world_themes": WORLD_THEMES,
            "sovereign_quant": self.sovereign_quant_status(),
        }


_handler = NODE_DELTA_Handler()


def get_handler() -> NODE_DELTA_Handler:
    return _handler
