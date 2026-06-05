"""
NVIDIA Resource Suite — Global Configuration

All configuration is loaded from environment variables.
NEVER hardcode API keys or secrets here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NIMConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"))

    # Default models
    chat_model: str = "meta/llama-3.1-70b-instruct"
    reasoning_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    embedding_model: str = "nvidia/nv-embed-v1"
    vision_model: str = "microsoft/phi-3-vision-128k-instruct"
    code_model: str = "meta/codellama-70b"
    guardrails_model: str = "nvidia/nemo-guardrails-v1"

    @property
    def available(self) -> bool:
        return bool(self.api_key)


@dataclass
class OmniverseConfig:
    nucleus_url: str = field(default_factory=lambda: os.environ.get("OMNIVERSE_NUCLEUS_URL", "omniverse://localhost"))
    farm_url: str = field(default_factory=lambda: os.environ.get("OMNIVERSE_FARM_URL", "http://localhost:8222"))
    composer_url: str = field(default_factory=lambda: os.environ.get("OMNIVERSE_COMPOSER_URL", "http://localhost:8011"))

    @property
    def available(self) -> bool:
        try:
            import omni.client  # type: ignore
            return True
        except ImportError:
            return False


@dataclass
class TritonConfig:
    http_url: str = field(default_factory=lambda: os.environ.get("TRITON_URL", "http://localhost:8000"))
    grpc_url: str = field(default_factory=lambda: os.environ.get("TRITON_GRPC_URL", "localhost:8001"))
    model_repository: str = "model_repository"

    @property
    def available(self) -> bool:
        try:
            import tritonclient.http  # type: ignore
            return True
        except ImportError:
            return False


@dataclass
class TensorRTConfig:
    workspace_gb: int = 4
    fp16_enabled: bool = True
    int8_enabled: bool = False
    int4_enabled: bool = False
    max_batch_size: int = 16

    @property
    def available(self) -> bool:
        try:
            import tensorrt  # type: ignore
            return True
        except ImportError:
            return False


@dataclass
class Earth2Config:
    api_url: str = field(default_factory=lambda: os.environ.get("EARTH2_API_URL", "https://api.earth2.nvidia.com"))
    api_key: str = field(default_factory=lambda: os.environ.get("EARTH2_API_KEY", ""))
    cache_dir: str = ".earth2_cache"

    @property
    def available(self) -> bool:
        try:
            import earth2mip  # type: ignore
            return True
        except ImportError:
            return False


@dataclass
class CloudXRConfig:
    server_url: str = field(default_factory=lambda: os.environ.get("CLOUDXR_SERVER_URL", ""))

    @property
    def available(self) -> bool:
        return bool(self.server_url)


@dataclass
class ServiceConfig:
    api_port: int = field(default_factory=lambda: int(os.environ.get("API_PORT", "7760")))
    deepflex_url: str = field(default_factory=lambda: os.environ.get("DEEPFLEX_URL", "http://localhost:8000"))
    argus_url: str = field(default_factory=lambda: os.environ.get("ARGUS_URL", "http://localhost:7700"))
    vr_room_url: str = field(default_factory=lambda: os.environ.get("VR_ROOM_URL", "http://localhost:7791"))
    environment: str = field(default_factory=lambda: os.environ.get("ENVIRONMENT", "development"))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))


@dataclass
class Config:
    nim: NIMConfig = field(default_factory=NIMConfig)
    omniverse: OmniverseConfig = field(default_factory=OmniverseConfig)
    triton: TritonConfig = field(default_factory=TritonConfig)
    tensorrt: TensorRTConfig = field(default_factory=TensorRTConfig)
    earth2: Earth2Config = field(default_factory=Earth2Config)
    cloudxr: CloudXRConfig = field(default_factory=CloudXRConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)

    def availability_report(self) -> dict[str, bool]:
        return {
            "nim": self.nim.available,
            "omniverse": self.omniverse.available,
            "triton": self.triton.available,
            "tensorrt": self.tensorrt.available,
            "earth2": self.earth2.available,
            "cloudxr": self.cloudxr.available,
        }

    def validate(self) -> list[str]:
        """Return list of warnings about missing optional config."""
        warnings = []
        if not self.nim.available:
            warnings.append("NVIDIA_API_KEY not set — NIM inference will be unavailable")
        if not self.omniverse.available:
            warnings.append("Omniverse Kit not installed — 3D rendering will use mock mode")
        if not self.triton.available:
            warnings.append("tritonclient not installed — Triton serving unavailable")
        if not self.earth2.available:
            warnings.append("earth2mip not installed — climate simulation unavailable")
        return warnings


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
