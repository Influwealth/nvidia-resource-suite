"""
NVIDIA Omniverse Kit Bridge

Core bridge between this platform and Omniverse Kit SDK.
Handles USD scene creation, asset management, and render dispatch
for all World Interactive Origins themes.

Graceful degradation: when omni.client or pxr (USD) are not installed,
all operations log a clear message and return mock data so the rest of
the platform continues working.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    import omni.client as omni_client  # type: ignore
    OMNI_AVAILABLE = True
except ImportError:
    omni_client = None
    OMNI_AVAILABLE = False

try:
    from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf  # type: ignore
    USD_AVAILABLE = True
except ImportError:
    Usd = UsdGeom = UsdLux = Sdf = Gf = None
    USD_AVAILABLE = False


WORLD_THEMES = {
    "ancient-silk-road": {
        "name": "Ancient Silk Road",
        "period": "100 BCE–1450 CE",
        "biome": "desert-steppe-oasis",
        "sky": "midday-desert",
        "ambient_color": (0.95, 0.85, 0.65),
        "ground_material": "sand-packed",
        "landmark": "Dunhuang Caves",
        "population_density": "medium",
        "sounds": ["camel_bells", "market_crowd", "wind"],
    },
    "harlem-renaissance": {
        "name": "Harlem Renaissance",
        "period": "1920s–1930s",
        "biome": "urban-northeast-usa",
        "sky": "evening-city-glow",
        "ambient_color": (0.85, 0.80, 0.70),
        "ground_material": "cobblestone-wet",
        "landmark": "Cotton Club, 142nd St",
        "population_density": "high",
        "sounds": ["jazz_trumpet", "street_chatter", "train_distant"],
    },
    "brooklyn-90s": {
        "name": "East Flatbush Origins",
        "period": "1990s",
        "biome": "urban-brooklyn",
        "sky": "summer-afternoon",
        "ambient_color": (0.90, 0.88, 0.82),
        "ground_material": "asphalt-cracked",
        "landmark": "458 E 94th St",
        "population_density": "high",
        "sounds": ["hip_hop_distant", "kids_playing", "ice_cream_truck", "basketball"],
    },
    "great-migration": {
        "name": "The Great Migration",
        "period": "1910–1970",
        "biome": "mixed-south-north",
        "sky": "dawn-hopeful",
        "ambient_color": (0.80, 0.75, 0.65),
        "ground_material": "red-clay-dirt",
        "landmark": "Chicago's South Side",
        "population_density": "medium",
        "sounds": ["blues_guitar", "train_whistle", "church_bell"],
    },
    "indigenous-americas": {
        "name": "Pre-Columbian Americas",
        "period": "Pre-1492",
        "biome": "diverse-forest-plains",
        "sky": "clear-preindustrial",
        "ambient_color": (0.70, 0.85, 0.75),
        "ground_material": "forest-floor",
        "landmark": "Cahokia Mounds",
        "population_density": "low",
        "sounds": ["birds", "river", "ceremonial_drums"],
    },
    "greenville-sovereign": {
        "name": "Greenville Sovereign World",
        "period": "Present + Future",
        "biome": "eastern-nc-piedmont",
        "sky": "clear-afternoon",
        "ambient_color": (0.85, 0.92, 0.80),
        "ground_material": "red-clay-grass",
        "landmark": "53-acre Monadic Site",
        "population_density": "medium",
        "sounds": ["birds", "wind", "distant_community"],
    },
}


@dataclass
class OmniverseScene:
    scene_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    theme: str = "brooklyn-90s"
    name: str = ""
    nucleus_path: str = ""
    usd_stage_path: str = ""
    render_resolution: tuple[int, int] = (1920, 1080)
    created_at: float = field(default_factory=time.time)
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "theme": self.theme,
            "name": self.name,
            "nucleus_path": self.nucleus_path,
            "usd_stage_path": self.usd_stage_path,
            "render_resolution": list(self.render_resolution),
            "created_at": self.created_at,
            "status": self.status,
            "metadata": self.metadata,
        }


class OmniverseKitBridge:
    """Bridge to Omniverse Kit SDK for 3D world-building."""

    def __init__(
        self,
        nucleus_url: str | None = None,
        farm_url: str | None = None,
    ):
        self.nucleus_url = nucleus_url or os.environ.get("OMNIVERSE_NUCLEUS_URL", "omniverse://localhost")
        self.farm_url = farm_url or os.environ.get("OMNIVERSE_FARM_URL", "http://localhost:8222")
        self._omni_available = OMNI_AVAILABLE
        self._usd_available = USD_AVAILABLE
        self._scenes: dict[str, OmniverseScene] = {}

        if not self._omni_available:
            log.info("Omniverse Kit not installed — running in mock mode. Install via NVIDIA Omniverse Launcher.")
        if not self._usd_available:
            log.info("USD Python bindings not found — install usd-core: pip install usd-core")

    # ------------------------------------------------------------------ #
    # Scene lifecycle                                                       #
    # ------------------------------------------------------------------ #

    def create_scene(
        self,
        theme: str,
        name: str = "",
        resolution: tuple[int, int] = (1920, 1080),
    ) -> OmniverseScene:
        """Create a new USD world scene for the given theme."""
        if theme not in WORLD_THEMES:
            raise ValueError(f"Unknown theme '{theme}'. Available: {list(WORLD_THEMES.keys())}")

        theme_cfg = WORLD_THEMES[theme]
        scene = OmniverseScene(
            theme=theme,
            name=name or theme_cfg["name"],
            nucleus_path=f"{self.nucleus_url}/Projects/WorldInteractiveOrigins/{theme}/{uuid.uuid4()}.usd",
            render_resolution=resolution,
            metadata={"theme_config": theme_cfg},
        )

        if self._usd_available:
            scene = self._build_usd_stage(scene, theme_cfg)
        else:
            scene.status = "mock"
            log.info("[mock] Scene created: %s (theme=%s)", scene.scene_id, theme)

        self._scenes[scene.scene_id] = scene
        return scene

    def _build_usd_stage(self, scene: OmniverseScene, theme_cfg: dict) -> OmniverseScene:
        """Build a USD stage with lighting, ground plane, and sky dome."""
        try:
            local_path = f"/tmp/wio_{scene.scene_id}.usda"
            stage = Usd.Stage.CreateNew(local_path)
            stage.SetMetadata("comment", f"World Interactive Origins — {scene.name}")

            # Root xform
            xform = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(xform.GetPrim())

            # Ground plane
            ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
            ground.CreatePointsAttr([(-500, 0, -500), (500, 0, -500), (500, 0, 500), (-500, 0, 500)])
            ground.CreateFaceVertexCountsAttr([4])
            ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

            # Ambient light
            dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
            dome.CreateIntensityAttr(1000.0)
            color = theme_cfg.get("ambient_color", (1.0, 1.0, 1.0))
            dome.CreateColorAttr(Gf.Vec3f(*color))

            # Sun (directional)
            sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
            sun.CreateIntensityAttr(3000.0)
            sun.CreateAngleAttr(0.53)

            stage.Save()
            scene.usd_stage_path = local_path
            scene.status = "created"
            log.info("USD stage created: %s", local_path)
        except Exception as exc:
            log.error("USD stage creation failed: %s", exc)
            scene.status = "error"
        return scene

    def add_asset(
        self,
        scene: OmniverseScene,
        asset_type: str,
        position: tuple[float, float, float] = (0, 0, 0),
        asset_url: str = "",
        label: str = "",
    ) -> dict[str, Any]:
        """Add an asset (character, building, prop) to an existing scene."""
        if scene.status == "mock" or not self._usd_available:
            return {
                "status": "mock",
                "asset_type": asset_type,
                "position": position,
                "label": label,
                "scene_id": scene.scene_id,
            }

        try:
            stage = Usd.Stage.Open(scene.usd_stage_path)
            prim_path = f"/World/{asset_type}_{label.replace(' ', '_')}_{int(time.time())}"
            if asset_url:
                ref_prim = stage.OverridePrim(prim_path)
                ref_prim.GetReferences().AddReference(asset_url)
            else:
                UsdGeom.Cube.Define(stage, prim_path)  # placeholder geometry

            xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(prim_path))
            xform.SetTranslate(Gf.Vec3d(*position))
            stage.Save()
            return {"status": "added", "prim_path": prim_path, "scene_id": scene.scene_id}
        except Exception as exc:
            log.error("Failed to add asset: %s", exc)
            return {"status": "error", "detail": str(exc)}

    def render_scene(
        self,
        scene: OmniverseScene,
        output_path: str = "",
        renderer: str = "rtx",
    ) -> dict[str, Any]:
        """Submit scene for rendering via Omniverse Farm."""
        if scene.status == "mock":
            return {
                "status": "mock_rendered",
                "scene_id": scene.scene_id,
                "output": output_path or f"/renders/{scene.scene_id}.png",
                "renderer": renderer,
            }
        return self._submit_to_farm(scene, output_path, renderer)

    def _submit_to_farm(self, scene: OmniverseScene, output_path: str, renderer: str) -> dict[str, Any]:
        """Submit render job to Omniverse Farm."""
        import requests
        job = {
            "job_type": "render",
            "scene_path": scene.usd_stage_path or scene.nucleus_path,
            "output_path": output_path or f"/renders/{scene.scene_id}.png",
            "renderer": renderer,
            "resolution": list(scene.render_resolution),
        }
        try:
            resp = requests.post(f"{self.farm_url}/queue/submit", json=job, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("Farm submission failed (is Farm running?): %s", exc)
            return {"status": "farm_unavailable", "job": job}

    def list_themes(self) -> dict[str, dict]:
        return WORLD_THEMES

    def get_scene(self, scene_id: str) -> OmniverseScene | None:
        return self._scenes.get(scene_id)

    def status(self) -> dict[str, Any]:
        return {
            "omniverse_kit": self._omni_available,
            "usd_python": self._usd_available,
            "nucleus_url": self.nucleus_url,
            "farm_url": self.farm_url,
            "scenes_active": len(self._scenes),
            "themes_available": list(WORLD_THEMES.keys()),
        }
