"""
Base World — Abstract world class for all World Interactive Origins themes

All worlds inherit from BaseWorld and implement:
  - build(): construct the USD scene
  - get_spawn_points(): entry points for students
  - get_quests(): learning objectives
  - get_guide(): AI guide character
  - get_knowledge_base(): documents for RAG
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SpawnPoint:
    name: str
    position: tuple[float, float, float]
    description: str
    suggested_for: list[str] = field(default_factory=list)  # age groups


@dataclass
class WorldQuest:
    quest_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    learning_objectives: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    age_groups: list[str] = field(default_factory=list)
    completion_reward: str = ""  # EBTK token amount description
    difficulty: str = "medium"   # easy | medium | hard
    estimated_minutes: int = 20
    steps: list[dict[str, str]] = field(default_factory=list)


@dataclass
class WorldDocument:
    title: str
    content: str
    source: str
    doc_type: str = "primary"
    author: str = ""
    date: str = ""


class BaseWorld(ABC):
    """Abstract base class for all World Interactive Origins worlds."""

    THEME: str = ""
    NAME: str = ""
    PERIOD: str = ""
    SUBJECTS: list[str] = []
    LANGUAGES: list[str] = ["en"]

    def __init__(self, kit_bridge: Any = None, nim_client: Any = None):
        self.kit = kit_bridge
        self.nim = nim_client
        self._scene = None
        self._built = False

    def build(self) -> dict[str, Any]:
        """Build the world scene. Returns scene metadata."""
        if self.kit:
            self._scene = self.kit.create_scene(self.THEME, name=self.NAME)
            self._built = True
            self._populate_scene()
            log.info("World '%s' built: scene_id=%s", self.NAME, self._scene.scene_id)
            return self._scene.to_dict()
        else:
            self._built = True
            log.info("[mock] World '%s' built", self.NAME)
            return {
                "status": "mock",
                "world": self.THEME,
                "name": self.NAME,
                "period": self.PERIOD,
            }

    def _populate_scene(self) -> None:
        """Override to add world-specific assets after scene creation."""
        pass

    @abstractmethod
    def get_spawn_points(self) -> list[SpawnPoint]:
        """Return student entry points for this world."""
        ...

    @abstractmethod
    def get_quests(self, age_group: str = "middle-school") -> list[WorldQuest]:
        """Return available learning quests."""
        ...

    @abstractmethod
    def get_guide(self) -> dict[str, Any]:
        """Return AI guide character configuration."""
        ...

    @abstractmethod
    def get_knowledge_base(self) -> list[WorldDocument]:
        """Return source documents for RAG."""
        ...

    def welcome_message(
        self,
        student_name: str = "Explorer",
        language: str = "en",
    ) -> str:
        """Generate a welcome message from the world guide."""
        guide = self.get_guide()
        prompt = (
            f"You are {guide['name']}, {guide['description']}. "
            f"Welcome the student named {student_name} to {self.NAME} ({self.PERIOD}). "
            f"Keep it to 3 sentences. Be warm, inviting, and historically authentic."
        )
        if self.nim:
            try:
                return self.nim.chat(
                    [{"role": "user", "content": prompt}],
                    model="meta/llama-3.1-70b-instruct",
                    max_tokens=200,
                )
            except Exception as exc:
                log.warning("NIM welcome failed: %s", exc)
        return f"Welcome to {self.NAME}, {student_name}! I am {guide['name']}. Let's explore together."

    def describe_location(
        self,
        location_name: str,
        student_question: str = "",
    ) -> str:
        """Describe a specific location in the world."""
        guide = self.get_guide()
        prompt = (
            f"As {guide['name']} in {self.NAME} ({self.PERIOD}), "
            f"describe the {location_name}. "
        )
        if student_question:
            prompt += f"The student asks: {student_question}"
        if self.nim:
            try:
                return self.nim.chat([{"role": "user", "content": prompt}], max_tokens=300)
            except Exception:
                pass
        return f"{location_name} is a significant place in {self.NAME}."

    def status(self) -> dict[str, Any]:
        return {
            "theme": self.THEME,
            "name": self.NAME,
            "period": self.PERIOD,
            "subjects": self.SUBJECTS,
            "languages": self.LANGUAGES,
            "built": self._built,
            "spawn_points": len(self.get_spawn_points()),
        }
