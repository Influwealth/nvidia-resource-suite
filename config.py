"""
NVIDIA Resource Suite — Configuration
Loads NVIDIA API credentials and service URLs from environment variables.
NEVER commit API keys to git.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class NvidiaConfig:
    # NVIDIA NIM API
    api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    nim_base_url: str = field(default_factory=lambda: os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"))

    # Omniverse
    nucleus_url: str = field(default_factory=lambda: os.getenv("OMNIVERSE_NUCLEUS_URL", "omniverse://localhost"))
    farm_url: str = field(default_factory=lambda: os.getenv("OMNIVERSE_FARM_URL", "http://localhost:8222"))
    kit_sdk_path: str = field(default_factory=lambda: os.getenv("OMNIVERSE_KIT_SDK_PATH", "/opt/ov/kit"))

    # Service
    port: int = field(default_factory=lambda: int(os.getenv("NVIDIA_SUITE_PORT", "7760")))
    sap_node_id: str = "nvidia-resource-suite"
    deepflex_url: str = field(default_factory=lambda: os.getenv("DEEPFLEX_BASE_URL", "http://localhost:8000"))

    def validate(self) -> list[str]:
        """Returns list of missing required config items."""
        issues = []
        if not self.api_key:
            issues.append("NVIDIA_API_KEY not set — NIM inference will fail. Get a key at build.nvidia.com")
        return issues

    def is_nim_available(self) -> bool:
        return bool(self.api_key)


# Singleton config
config = NvidiaConfig()
