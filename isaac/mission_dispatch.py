"""Isaac Mission Dispatch: robot mission queue, waypoint navigation, status polling."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
from loguru import logger


class MissionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Waypoint:
    x: float
    y: float
    z: float = 0.0
    heading: float = 0.0  # degrees
    label: str = ""
    action: str = "navigate"  # navigate | dock | inspect | capture


@dataclass
class Mission:
    id: str
    robot_id: str
    waypoints: list[Waypoint]
    status: MissionStatus = MissionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    priority: int = 5  # 1-10, higher = more urgent
    metadata: dict = field(default_factory=dict)
    current_waypoint: int = 0
    error: str | None = None


class MissionDispatch:
    """Isaac Mission Dispatch client.

    Connects to Isaac Mission Dispatch REST API when available.
    Falls back to in-process mock mission runner for simulation.
    """

    def __init__(self, dispatch_url: str = "http://localhost:5000"):
        self.dispatch_url = dispatch_url.rstrip("/")
        self._missions: dict[str, Mission] = {}
        self._available = self._check_available()
        if self._available:
            logger.info(f"Mission Dispatch connected at {self.dispatch_url}")
        else:
            logger.info("Mission Dispatch not reachable — mock mission runner active")

    def _check_available(self) -> bool:
        try:
            r = httpx.get(f"{self.dispatch_url}/health", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def create_mission(
        self,
        robot_id: str,
        waypoints: list[Waypoint],
        priority: int = 5,
        metadata: dict | None = None,
    ) -> Mission:
        mission = Mission(
            id=str(uuid.uuid4()),
            robot_id=robot_id,
            waypoints=waypoints,
            priority=priority,
            metadata=metadata or {},
        )
        if self._available:
            self._dispatch_mission(mission)
        else:
            self._missions[mission.id] = mission
            logger.info(f"[MOCK] Mission {mission.id} queued for robot {robot_id}")
        return mission

    def _dispatch_mission(self, mission: Mission):
        payload = {
            "mission_id": mission.id,
            "robot_id": mission.robot_id,
            "waypoints": [
                {"x": w.x, "y": w.y, "z": w.z, "heading": w.heading, "action": w.action}
                for w in mission.waypoints
            ],
            "priority": mission.priority,
        }
        r = httpx.post(f"{self.dispatch_url}/missions", json=payload, timeout=10.0)
        r.raise_for_status()
        self._missions[mission.id] = mission

    def get_status(self, mission_id: str) -> MissionStatus:
        if self._available:
            r = httpx.get(f"{self.dispatch_url}/missions/{mission_id}", timeout=5.0)
            r.raise_for_status()
            return MissionStatus(r.json()["status"])
        mission = self._missions.get(mission_id)
        if not mission:
            raise ValueError(f"Mission {mission_id} not found")
        self._simulate_progress(mission)
        return mission.status

    def _simulate_progress(self, mission: Mission):
        """Advance mock mission state for educational demos."""
        if mission.status == MissionStatus.PENDING:
            mission.status = MissionStatus.RUNNING
            mission.started_at = time.time()
        elif mission.status == MissionStatus.RUNNING:
            mission.current_waypoint += 1
            if mission.current_waypoint >= len(mission.waypoints):
                mission.status = MissionStatus.COMPLETED
                mission.completed_at = time.time()

    def cancel_mission(self, mission_id: str) -> bool:
        if self._available:
            r = httpx.post(f"{self.dispatch_url}/missions/{mission_id}/cancel", timeout=5.0)
            return r.status_code == 200
        mission = self._missions.get(mission_id)
        if mission and mission.status in (MissionStatus.PENDING, MissionStatus.RUNNING):
            mission.status = MissionStatus.CANCELLED
            return True
        return False

    def list_missions(self, robot_id: str | None = None) -> list[Mission]:
        missions = list(self._missions.values())
        if robot_id:
            missions = [m for m in missions if m.robot_id == robot_id]
        return sorted(missions, key=lambda m: m.created_at, reverse=True)

    def create_survey_mission(
        self,
        robot_id: str,
        area_bounds: tuple,  # (x_min, y_min, x_max, y_max)
        grid_spacing: float = 1.0,
        height: float = 2.0,
    ) -> Mission:
        """Auto-generate a grid survey mission for a rectangular area."""
        x_min, y_min, x_max, y_max = area_bounds
        waypoints = []
        x = x_min
        direction = 1
        while x <= x_max:
            ys = [y for y in [y for y in range(int(y_min * 10), int(y_max * 10) + 1, int(grid_spacing * 10))]]
            if direction == -1:
                ys = ys[::-1]
            for y_int in ys:
                waypoints.append(Waypoint(x=x, y=y_int / 10, z=height, action="inspect"))
            x += grid_spacing
            direction *= -1
        return self.create_mission(robot_id, waypoints, priority=3, metadata={"type": "survey", "area": area_bounds})
