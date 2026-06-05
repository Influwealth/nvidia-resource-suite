"""
NVIDIA Omniverse Replicator — Synthetic Data Generation

Replicator creates annotated synthetic datasets for:
  - Training visual AI models for world navigation
  - Generating student avatar datasets
  - Creating training data for historical scene understanding
  - Accessibility testing with diverse virtual populations
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

try:
    import omni.replicator.core as rep  # type: ignore
    REP_AVAILABLE = True
except ImportError:
    rep = None
    REP_AVAILABLE = False


@dataclass
class ReplicatorDataset:
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    world: str = ""
    num_frames: int = 1000
    output_dir: str = ""
    annotations: list[str] = field(default_factory=lambda: ["rgb", "depth", "semantic", "bbox2d"])
    status: str = "pending"
    frames_generated: int = 0


class ReplicatorClient:
    """Omniverse Replicator client for synthetic data generation."""

    ANNOTATION_TYPES = [
        "rgb",           # Color image
        "depth",         # Depth map
        "semantic",      # Semantic segmentation
        "instance",      # Instance segmentation
        "bbox2d",        # 2D bounding boxes
        "bbox3d",        # 3D bounding boxes
        "normals",       # Surface normals
        "occlusion",     # Occlusion masks
        "distance",      # Distance to camera
    ]

    def __init__(self):
        self._available = REP_AVAILABLE
        self._datasets: dict[str, ReplicatorDataset] = {}
        if not self._available:
            log.info("Omniverse Replicator not installed — mock mode")

    def create_dataset(
        self,
        world: str,
        name: str,
        num_frames: int = 1000,
        output_dir: str = "",
        annotations: list[str] | None = None,
    ) -> ReplicatorDataset:
        """Define a new synthetic dataset for a world."""
        dataset = ReplicatorDataset(
            name=name,
            world=world,
            num_frames=num_frames,
            output_dir=output_dir or f"/datasets/{world}/{name}",
            annotations=annotations or ["rgb", "depth", "semantic", "bbox2d"],
        )
        self._datasets[dataset.dataset_id] = dataset
        return dataset

    def generate(
        self,
        dataset: ReplicatorDataset,
        scene_stage: Any = None,
    ) -> ReplicatorDataset:
        """Generate synthetic data frames for the dataset."""
        if not self._available or scene_stage is None:
            log.info("[mock] Generating %d frames for dataset '%s'", dataset.num_frames, dataset.name)
            dataset.frames_generated = dataset.num_frames
            dataset.status = "completed"
            return dataset

        try:
            # Real Replicator pipeline
            rep.orchestrator.set_capture_on_play(False)
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(
                output_dir=dataset.output_dir,
                rgb=True,
                depth="depth" in dataset.annotations,
                semantic_segmentation="semantic" in dataset.annotations,
                bounding_box_2d_tight="bbox2d" in dataset.annotations,
            )
            rep.orchestrator.run(num_frames=dataset.num_frames)
            dataset.frames_generated = dataset.num_frames
            dataset.status = "completed"
        except Exception as exc:
            log.error("Replicator generation failed: %s", exc)
            dataset.status = "error"
        return dataset

    def generate_avatar_dataset(
        self,
        world: str,
        num_avatars: int = 50,
        diversity_config: dict | None = None,
    ) -> ReplicatorDataset:
        """
        Generate a diverse avatar dataset for a world.
        diversity_config can specify skin tones, ages, body types, clothing styles.
        """
        config = diversity_config or {
            "skin_tones": ["light", "medium", "medium-dark", "dark", "very-dark"],
            "ages": ["child", "teen", "young-adult", "adult", "elder"],
            "body_types": ["slim", "average", "athletic", "heavy"],
            "historical_accuracy": True,
        }
        dataset = self.create_dataset(
            world=world,
            name=f"avatars_{world}",
            num_frames=num_avatars * 10,
            annotations=["rgb", "semantic", "bbox2d"],
        )
        dataset.status = "mock_complete" if not self._available else "pending"
        log.info("Avatar dataset created for world '%s': %d avatars × 10 poses", world, num_avatars)
        return dataset

    def status(self) -> dict[str, Any]:
        return {
            "replicator_available": self._available,
            "datasets": len(self._datasets),
            "annotation_types": self.ANNOTATION_TYPES,
        }
