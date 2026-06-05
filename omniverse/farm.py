"""
NVIDIA Omniverse Farm — GPU Render Dispatch

Omniverse Farm distributes rendering workloads across GPU workers.
This client handles:
  - Job submission for world scene renders
  - Job status polling
  - Output retrieval
  - Priority queue management (education jobs get elevated priority)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests

log = logging.getLogger(__name__)


class RenderJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RenderPriority(int, Enum):
    EDUCATION = 100  # Students always first
    INTERACTIVE = 80
    BATCH = 50
    BACKGROUND = 20


@dataclass
class RenderJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scene_path: str = ""
    output_path: str = ""
    renderer: str = "rtx"          # rtx | iray | rasterizer
    resolution: tuple[int, int] = (1920, 1080)
    frame_range: tuple[int, int] = (1, 1)
    priority: int = RenderPriority.EDUCATION
    world: str = ""
    submitted_at: float = field(default_factory=time.time)
    status: RenderJobStatus = RenderJobStatus.PENDING
    progress: float = 0.0
    output_url: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "scene_path": self.scene_path,
            "output_path": self.output_path,
            "renderer": self.renderer,
            "resolution": list(self.resolution),
            "frame_range": list(self.frame_range),
            "priority": self.priority,
            "world": self.world,
            "status": self.status,
            "progress": self.progress,
            "output_url": self.output_url,
        }


class OmniverseFarm:
    """Omniverse Farm client for distributed GPU rendering."""

    def __init__(self, farm_url: str | None = None):
        self.farm_url = (farm_url or os.environ.get("OMNIVERSE_FARM_URL", "http://localhost:8222")).rstrip("/")
        self._jobs: dict[str, RenderJob] = {}
        self._mock_mode = not self._check_farm_reachable()

    def _check_farm_reachable(self) -> bool:
        try:
            resp = requests.get(f"{self.farm_url}/health", timeout=3)
            return resp.ok
        except Exception:
            log.info("Omniverse Farm not reachable at %s — using mock mode", self.farm_url)
            return False

    def submit(
        self,
        scene_path: str,
        output_path: str,
        renderer: str = "rtx",
        resolution: tuple[int, int] = (1920, 1080),
        frame_range: tuple[int, int] = (1, 1),
        priority: int = RenderPriority.EDUCATION,
        world: str = "",
    ) -> RenderJob:
        """Submit a render job to the Farm queue."""
        job = RenderJob(
            scene_path=scene_path,
            output_path=output_path,
            renderer=renderer,
            resolution=resolution,
            frame_range=frame_range,
            priority=priority,
            world=world,
        )

        if self._mock_mode:
            job.status = RenderJobStatus.COMPLETED
            job.output_url = f"/renders/mock_{job.job_id}.png"
            log.info("[mock] Render job submitted: %s", job.job_id)
        else:
            payload = {
                "scene": scene_path,
                "output": output_path,
                "renderer": renderer,
                "width": resolution[0],
                "height": resolution[1],
                "start_frame": frame_range[0],
                "end_frame": frame_range[1],
                "priority": priority,
            }
            try:
                resp = requests.post(f"{self.farm_url}/queue/submit", json=payload, timeout=30)
                resp.raise_for_status()
                job.job_id = resp.json().get("job_id", job.job_id)
                job.status = RenderJobStatus.PENDING
            except Exception as exc:
                log.error("Farm submit failed: %s", exc)
                job.status = RenderJobStatus.FAILED
                job.error = str(exc)

        self._jobs[job.job_id] = job
        return job

    def poll(
        self,
        job_id: str,
        timeout: float = 300.0,
        poll_interval: float = 5.0,
    ) -> RenderJob:
        """Poll a job until completion or timeout."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if self._mock_mode:
            return job

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(f"{self.farm_url}/queue/{job_id}", timeout=10)
                resp.raise_for_status()
                data = resp.json()
                job.status = RenderJobStatus(data.get("status", "pending"))
                job.progress = data.get("progress", 0.0)
                job.output_url = data.get("output_url", "")
                if job.status in (RenderJobStatus.COMPLETED, RenderJobStatus.FAILED):
                    break
            except Exception as exc:
                log.warning("Poll error: %s", exc)
            time.sleep(poll_interval)

        return job

    def submit_world_render(
        self,
        world: str,
        scene_path: str,
        resolution: tuple[int, int] = (3840, 2160),
    ) -> RenderJob:
        """Convenience method: submit a 4K render for a world scene."""
        output = f"/renders/{world}_{int(time.time())}.png"
        return self.submit(
            scene_path=scene_path,
            output_path=output,
            renderer="rtx",
            resolution=resolution,
            priority=RenderPriority.EDUCATION,
            world=world,
        )

    def list_jobs(self, world: str = "") -> list[RenderJob]:
        jobs = list(self._jobs.values())
        if world:
            jobs = [j for j in jobs if j.world == world]
        return jobs

    def status(self) -> dict[str, Any]:
        return {
            "farm_url": self.farm_url,
            "mock_mode": self._mock_mode,
            "total_jobs": len(self._jobs),
            "pending": sum(1 for j in self._jobs.values() if j.status == RenderJobStatus.PENDING),
            "running": sum(1 for j in self._jobs.values() if j.status == RenderJobStatus.RUNNING),
            "completed": sum(1 for j in self._jobs.values() if j.status == RenderJobStatus.COMPLETED),
        }
