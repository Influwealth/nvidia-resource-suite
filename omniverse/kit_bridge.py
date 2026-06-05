"""
NVIDIA Omniverse Kit Bridge
Connects the NVIDIA Resource Suite to NVIDIA Omniverse Kit SDK.

For World Interactive Origins: builds and streams 3D educational worlds
using USD (Universal Scene Description) and Omniverse rendering.

Requirements:
- NVIDIA Omniverse installed (https://www.nvidia.com/en-us/omniverse/)
- omni-client Python package (pip install omni-client)
- Valid Nucleus server URL
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import config


@dataclass
class OmniverseScene:
    scene_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    nucleus_path: str = ""
    world_theme: str = ""  # e.g. "ancient-silk-road", "harlem-renaissance"
    resolution: tuple[int, int] = (1920, 1080)
    render_mode: str = "rtx"  # "rtx", "pathtraced", "iray"
    interactive: bool = True


class OmniverseKitBridge:
    """
    Bridge to NVIDIA Omniverse Kit SDK.

    In development/mock mode: simulates scene operations.
    In production: calls omni.client and Kit SDK APIs.
    """

    WORLD_THEMES = {
        "ancient-silk-road": {
            "description": "Ancient Silk Road trading routes, 100 BCE - 1450 CE",
            "template_usd": "omniverse://localhost/templates/silk_road_base.usd",
            "lighting": "golden_hour",
            "interactive_objects": ["merchant_tent", "camel_caravan", "spice_market", "cartographer_table"],
        },
        "harlem-renaissance": {
            "description": "Harlem Renaissance, New York City, 1920s-1930s",
            "template_usd": "omniverse://localhost/templates/harlem_1920s.usd",
            "lighting": "night_city",
            "interactive_objects": ["jazz_club", "newspaper_stand", "brownstone", "art_studio"],
        },
        "brooklyn-1990s": {
            "description": "Brooklyn, New York, 1990s — hip-hop and community origins",
            "template_usd": "omniverse://localhost/templates/brooklyn_90s.usd",
            "lighting": "afternoon_sun",
            "interactive_objects": ["basketball_court", "corner_store", "recording_studio", "community_board"],
        },
        "great-migration": {
            "description": "The Great Migration — Black American movement north, 1910-1970",
            "template_usd": "omniverse://localhost/templates/migration_era.usd",
            "lighting": "dawn",
            "interactive_objects": ["train_station", "newspaper_office", "factory_floor", "church"],
        },
        "indigenous-americas": {
            "description": "Pre-Columbian Americas — civilizations before 1492",
            "template_usd": "omniverse://localhost/templates/precolumbian.usd",
            "lighting": "tropical_noon",
            "interactive_objects": ["tenochtitlan_market", "ceremonial_center", "agricultural_terraces", "observatory"],
        },
    }

    def __init__(self, mock: bool = True) -> None:
        self.mock = mock
        self.nucleus_url = config.nucleus_url
        self.farm_url = config.farm_url
        self._omni_available = self._check_omni()

    def _check_omni(self) -> bool:
        """Check if omni.client is available."""
        if self.mock:
            return False
        try:
            import omni.client  # type: ignore[import]
            return True
        except ImportError:
            return False

    def list_worlds(self) -> list[dict[str, Any]]:
        """List available educational world themes."""
        return [
            {"theme": k, "description": v["description"], "interactive_objects": v["interactive_objects"]}
            for k, v in self.WORLD_THEMES.items()
        ]

    def create_scene(self, world_theme: str, name: str | None = None) -> OmniverseScene:
        """Create a new Omniverse scene for an educational world."""
        theme_data = self.WORLD_THEMES.get(world_theme, {})
        scene = OmniverseScene(
            name=name or f"{world_theme}-{str(uuid.uuid4())[:8]}",
            nucleus_path=theme_data.get("template_usd", f"omniverse://localhost/worlds/{world_theme}.usd"),
            world_theme=world_theme,
        )

        if self._omni_available:
            self._create_usd_stage(scene)
        else:
            print(f"[omniverse-mock] Would create scene: {scene.name} from {scene.nucleus_path}")

        return scene

    def _create_usd_stage(self, scene: OmniverseScene) -> None:
        """Real USD stage creation — requires Omniverse installation."""
        try:
            from pxr import Usd, UsdGeom  # type: ignore[import]
            stage_path = f"{self.nucleus_url}/worlds/{scene.scene_id}.usd"
            stage = Usd.Stage.CreateNew(stage_path)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.GetRootLayer().Save()
            scene.nucleus_path = stage_path
        except Exception as exc:
            print(f"[omniverse] Stage creation failed: {exc}")

    def render_scene(self, scene: OmniverseScene, output_path: str | None = None) -> dict[str, Any]:
        """Submit scene for rendering via Omniverse Farm."""
        if self._omni_available:
            return self._submit_to_farm(scene, output_path)
        return {
            "status": "mock",
            "scene_id": scene.scene_id,
            "render_mode": scene.render_mode,
            "output_path": output_path or f"/renders/{scene.scene_id}.png",
            "note": "Install NVIDIA Omniverse to enable real rendering",
        }

    def _submit_to_farm(self, scene: OmniverseScene, output_path: str | None) -> dict[str, Any]:
        """Submit to Omniverse Farm for GPU rendering."""
        import requests
        payload = {
            "usd_path": scene.nucleus_path,
            "render_mode": scene.render_mode,
            "resolution": list(scene.resolution),
            "output_path": output_path,
        }
        try:
            resp = requests.post(f"{self.farm_url}/api/v1/jobs", json=payload, timeout=10)
            return resp.json()
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def get_interactive_objects(self, world_theme: str) -> list[str]:
        """Get list of interactive objects in a world theme."""
        return self.WORLD_THEMES.get(world_theme, {}).get("interactive_objects", [])
