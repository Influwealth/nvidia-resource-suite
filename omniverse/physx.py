"""
NVIDIA PhysX — Real-time Physics Simulation

PhysX powers physically accurate simulation in Omniverse worlds:
  - Rigid body dynamics (buildings, props, vehicles)
  - Cloth simulation (period-accurate clothing for historical characters)
  - Fluid simulation (rivers, rain, ocean)
  - Destruction (historically appropriate)
  - Character physics (crowd simulation)

Also interfaces with NVIDIA Warp for GPU-accelerated physics kernels.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

try:
    from pxr import PhysxSchema, UsdPhysics, Gf  # type: ignore
    PHYSX_AVAILABLE = True
except ImportError:
    PhysxSchema = UsdPhysics = Gf = None
    PHYSX_AVAILABLE = False

try:
    import warp as wp  # type: ignore
    WARP_AVAILABLE = True
except ImportError:
    wp = None
    WARP_AVAILABLE = False


@dataclass
class PhysicsBody:
    prim_path: str
    mass_kg: float = 1.0
    friction: float = 0.5
    restitution: float = 0.1
    is_static: bool = False
    collision_shape: str = "convex_hull"  # convex_hull | box | sphere | mesh


@dataclass
class ClothSimConfig:
    prim_path: str
    particle_mass: float = 0.01
    stretch_stiffness: float = 10000.0
    bend_stiffness: float = 200.0
    damping: float = 0.2
    wind_velocity: tuple[float, float, float] = (0.0, 0.0, 0.5)


class PhysXClient:
    """Physics simulation client using NVIDIA PhysX via USD Python bindings."""

    GRAVITY = -9.81  # m/s^2

    def __init__(self):
        self._physx_available = PHYSX_AVAILABLE
        self._warp_available = WARP_AVAILABLE
        if not self._physx_available:
            log.info("PhysX USD schema not available — install Omniverse Kit or usd-core")

    def add_scene_physics(
        self,
        stage: Any,
        gravity: float = GRAVITY,
    ) -> bool:
        """Enable physics on a USD stage."""
        if not self._physx_available or stage is None:
            log.info("[mock] Physics enabled on stage (gravity=%.2f)", gravity)
            return True
        try:
            scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, gravity, 0))
            scene.CreateGravityMagnitudeAttr().Set(abs(gravity))
            return True
        except Exception as exc:
            log.error("Failed to add scene physics: %s", exc)
            return False

    def add_rigid_body(
        self,
        stage: Any,
        body: PhysicsBody,
    ) -> bool:
        """Add rigid body physics to a USD prim."""
        if not self._physx_available or stage is None:
            log.info("[mock] Rigid body added: %s (mass=%.1fkg)", body.prim_path, body.mass_kg)
            return True
        try:
            prim = stage.GetPrimAtPath(body.prim_path)
            if not prim.IsValid():
                raise ValueError(f"Prim not found: {body.prim_path}")
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(body.mass_kg)
            UsdPhysics.CollisionAPI.Apply(prim)
            if not body.is_static:
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            return True
        except Exception as exc:
            log.error("Rigid body setup failed: %s", exc)
            return False

    def add_cloth(
        self,
        stage: Any,
        config: ClothSimConfig,
    ) -> bool:
        """Add cloth simulation to a mesh prim (period-accurate clothing)."""
        if not self._physx_available or stage is None:
            log.info("[mock] Cloth sim added: %s", config.prim_path)
            return True
        try:
            prim = stage.GetPrimAtPath(config.prim_path)
            cloth = PhysxSchema.PhysxParticleClothAPI.Apply(prim)
            cloth.CreateStretchStiffnessAttr(config.stretch_stiffness)
            cloth.CreateBendStiffnessAttr(config.bend_stiffness)
            cloth.CreateDampingAttr(config.damping)
            return True
        except Exception as exc:
            log.error("Cloth simulation setup failed: %s", exc)
            return False

    def simulate_projectile(
        self,
        initial_pos: tuple[float, float, float],
        initial_vel: tuple[float, float, float],
        dt: float = 0.016,
        steps: int = 120,
    ) -> list[tuple[float, float, float]]:
        """
        Pure-Python projectile physics — no Omniverse required.
        Useful for in-browser physics education demos.
        """
        positions = []
        x, y, z = initial_pos
        vx, vy, vz = initial_vel
        g = self.GRAVITY
        for _ in range(steps):
            x += vx * dt
            y += vy * dt + 0.5 * g * dt * dt
            z += vz * dt
            vy += g * dt
            positions.append((round(x, 3), round(y, 3), round(z, 3)))
            if y < 0:
                break
        return positions

    def status(self) -> dict[str, Any]:
        return {
            "physx_available": self._physx_available,
            "warp_available": self._warp_available,
            "gravity": self.GRAVITY,
        }
