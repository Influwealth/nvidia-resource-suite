"""Isaac Sim bridge: USD scene, robot articulation, sensor streams, ROS2 bridge."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import omni.kit.app
    import omni.usd
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    logger.warning("Isaac Sim not found — running in headless simulation mode")

try:
    from pxr import Usd, UsdGeom, UsdPhysics
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


@dataclass
class RobotConfig:
    name: str
    usd_path: str  # path or NGC URL (omniverse://)
    position: tuple = (0.0, 0.0, 0.0)
    orientation: tuple = (0.0, 0.0, 0.0, 1.0)  # quaternion xyzw
    joint_names: list[str] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)  # camera, lidar, imu, depth


@dataclass
class SensorStream:
    sensor_type: str
    robot_name: str
    topic: str  # ROS2 topic
    data: Any = None
    timestamp: float = 0.0


class IsaacSimBridge:
    """Bridge to NVIDIA Isaac Sim for robot simulation and sensor data.

    When Isaac Sim is not running locally, provides a simulation mode
    that generates synthetic sensor data for educational use.
    """

    def __init__(self, nucleus_url: str = "omniverse://localhost"):
        self.nucleus_url = nucleus_url
        self._robots: dict[str, RobotConfig] = {}
        self._stage = None
        self._available = ISAAC_AVAILABLE
        if self._available:
            self._init_stage()
        else:
            logger.info("Isaac Sim not available — synthetic simulation active")

    def _init_stage(self):
        try:
            ctx = omni.usd.get_context()
            ctx.new_stage()
            self._stage = ctx.get_stage()
            logger.info("Isaac Sim stage initialized")
        except Exception as e:
            logger.error(f"Isaac Sim stage init failed: {e}")
            self._available = False

    def spawn_robot(
        self,
        config: RobotConfig,
        add_physics: bool = True,
    ) -> str:
        """Spawn a robot into the USD scene from a .usd file or Nucleus path."""
        robot_id = f"{config.name}_{uuid.uuid4().hex[:6]}"
        self._robots[robot_id] = config

        if not self._available or not USD_AVAILABLE:
            logger.info(f"[MOCK] Spawned robot {config.name} at {config.position}")
            return robot_id

        from pxr import Gf
        robot_prim = self._stage.DefinePrim(f"/World/{config.name}", "Xform")
        robot_prim.GetReferences().AddReference(config.usd_path)
        xform = UsdGeom.Xformable(robot_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*config.position))
        if add_physics:
            UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
        logger.info(f"Spawned robot {config.name} (id={robot_id})")
        return robot_id

    def despawn_robot(self, robot_id: str):
        if robot_id in self._robots:
            config = self._robots.pop(robot_id)
            if self._available and self._stage:
                prim = self._stage.GetPrimAtPath(f"/World/{config.name}")
                if prim.IsValid():
                    self._stage.RemovePrim(prim.GetPath())
            logger.info(f"Despawned robot {robot_id}")

    def set_joint_positions(self, robot_id: str, positions: dict[str, float]):
        if not self._available:
            logger.info(f"[MOCK] Set joints for {robot_id}: {positions}")
            return
        # Requires Isaac Sim articulation controller
        from omni.isaac.core.articulations import Articulation
        config = self._robots.get(robot_id)
        if not config:
            raise ValueError(f"Robot {robot_id} not found")
        art = Articulation(f"/World/{config.name}")
        art.initialize()
        for joint, pos in positions.items():
            art.set_joint_positions([pos], joint_indices=[art.get_dof_index(joint)])

    def get_sensor_data(
        self,
        robot_id: str,
        sensor_type: str = "camera",
    ) -> SensorStream:
        import time, random
        if not self._available:
            return SensorStream(
                sensor_type=sensor_type,
                robot_name=self._robots.get(robot_id, RobotConfig("", "")).name,
                topic=f"/isaac/{robot_id}/{sensor_type}",
                data={"mock": True, "value": random.random()},
                timestamp=time.time(),
            )
        # Real implementation reads from Isaac Sim sensor nodes
        raise NotImplementedError("Mount Isaac Sim sensor node and read from ROS2 topic")

    def enable_ros2_bridge(
        self,
        robot_id: str,
        topics: list[str] | None = None,
    ) -> dict:
        topics = topics or ["cmd_vel", "odom", "scan", "camera/image_raw"]
        return {
            "robot_id": robot_id,
            "ros2_bridge": "enabled" if self._available else "mock",
            "topics": [f"/isaac/{robot_id}/{t}" for t in topics],
            "note": "Install omni.isaac.ros2_bridge extension to activate",
        }

    def simulation_step(self, dt: float = 1.0 / 60.0):
        if self._available:
            omni.kit.app.get_app().update()
        else:
            import time
            time.sleep(dt)

    def save_usd(self, output_path: str | Path):
        if self._available and self._stage:
            self._stage.GetRootLayer().Export(str(output_path))
            logger.info(f"Stage saved to {output_path}")
        else:
            logger.info(f"[MOCK] Would save stage to {output_path}")
