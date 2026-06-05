"""
NVIDIA Omniverse Nucleus — Asset Server Client

Nucleus is the central asset server for Omniverse.
This client handles:
  - Asset upload and download
  - Directory browsing
  - Real-time collaboration subscriptions
  - Community asset library management
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

try:
    import omni.client as omni_client  # type: ignore
    OMNI_AVAILABLE = True
except ImportError:
    omni_client = None
    OMNI_AVAILABLE = False


@dataclass
class NucleusAsset:
    path: str
    name: str
    size_bytes: int = 0
    asset_type: str = "usd"
    world: str = ""
    tags: list[str] = field(default_factory=list)


class NucleusClient:
    """Client for Omniverse Nucleus asset server."""

    def __init__(self, server_url: str | None = None):
        self.server_url = server_url or os.environ.get("OMNIVERSE_NUCLEUS_URL", "omniverse://localhost")
        self._available = OMNI_AVAILABLE

        if self._available:
            result = omni_client.initialize()
            log.info("Nucleus initialized: %s", result)
        else:
            log.info("Omniverse not installed — Nucleus client in mock mode")

    def upload(
        self,
        local_path: str,
        nucleus_path: str,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Upload a local file to Nucleus."""
        if not self._available:
            return {"status": "mock", "local": local_path, "nucleus": nucleus_path}
        try:
            result = omni_client.copy(
                local_path,
                f"{self.server_url}/{nucleus_path.lstrip('/')}",
                omni_client.CopyBehavior.OVERWRITE if overwrite else omni_client.CopyBehavior.ERROR_IF_EXISTS,
            )
            return {"status": str(result), "path": nucleus_path}
        except Exception as exc:
            log.error("Nucleus upload failed: %s", exc)
            return {"status": "error", "detail": str(exc)}

    def download(
        self,
        nucleus_path: str,
        local_path: str,
    ) -> dict[str, Any]:
        """Download a file from Nucleus to local storage."""
        if not self._available:
            return {"status": "mock", "nucleus": nucleus_path, "local": local_path}
        try:
            result = omni_client.copy(
                f"{self.server_url}/{nucleus_path.lstrip('/')}",
                local_path,
            )
            return {"status": str(result), "local": local_path}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    def list_assets(
        self,
        directory: str = "/Projects/WorldInteractiveOrigins",
        recursive: bool = False,
    ) -> list[NucleusAsset]:
        """List assets in a Nucleus directory."""
        if not self._available:
            return [
                NucleusAsset(path=f"{directory}/example.usda", name="example.usda", size_bytes=4096, world="demo"),
            ]
        try:
            result, entries = omni_client.list(f"{self.server_url}/{directory.lstrip('/')}")  # type: ignore
            assets = []
            for entry in entries:
                assets.append(NucleusAsset(
                    path=f"{directory}/{entry.relative_path}",
                    name=entry.relative_path,
                    size_bytes=entry.size if hasattr(entry, "size") else 0,
                ))
            return assets
        except Exception as exc:
            log.error("Nucleus list failed: %s", exc)
            return []

    def create_world_library(
        self,
        world: str,
        subdirs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create the standard directory structure for a world's asset library."""
        base = f"/Projects/WorldInteractiveOrigins/{world}"
        dirs = subdirs or ["scenes", "props", "characters", "textures", "audio", "renders"]
        created = []
        for d in dirs:
            path = f"{base}/{d}"
            if not self._available:
                created.append({"status": "mock", "path": path})
            else:
                try:
                    omni_client.create_folder(f"{self.server_url}/{path.lstrip('/')}")  # type: ignore
                    created.append({"status": "created", "path": path})
                except Exception as exc:
                    created.append({"status": "error", "path": path, "detail": str(exc)})
        return {"world": world, "base_path": base, "directories": created}

    def status(self) -> dict[str, Any]:
        return {
            "nucleus_available": self._available,
            "server_url": self.server_url,
        }
