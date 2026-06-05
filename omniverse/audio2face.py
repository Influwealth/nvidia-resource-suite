"""
NVIDIA Omniverse Audio2Face

Animate AI guide characters in educational worlds using Audio2Face.
Takes an audio clip (voice line) and drives realistic facial animation
for historical figures, community guides, and AI tutors.

Also includes ACE (Avatar Cloud Engine) integration for real-time
personalized avatar responses.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests

log = logging.getLogger(__name__)


@dataclass
class AnimationClip:
    clip_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    audio_path: str = ""
    character_name: str = ""
    world: str = ""
    output_path: str = ""
    duration_seconds: float = 0.0
    status: str = "pending"
    blend_shapes_path: str = ""  # USD blend shape animation curves


CHARACTER_GUIDES = {
    "east-flatbush-origins": {
        "name": "Community Elder",
        "description": "A wise elder from East Flatbush who lived through the neighborhood's transformation",
        "voice_style": "warm_caribbean",
        "age": 70,
    },
    "harlem-renaissance": {
        "name": "Langston",
        "description": "Inspired by Langston Hughes — a poet guide who speaks in verse and story",
        "voice_style": "poetic_baritone",
        "age": 30,
    },
    "ancient-silk-road": {
        "name": "Merchant Al-Rashid",
        "description": "A multilingual Silk Road merchant who has traveled from Baghdad to Chang'an",
        "voice_style": "measured_arabic",
        "age": 45,
    },
    "greenville-sovereign": {
        "name": "Elder Founder",
        "description": "One of the original founders of the Greenville Sovereign community",
        "voice_style": "southern_dignified",
        "age": 65,
    },
    "great-migration": {
        "name": "Mama Rosa",
        "description": "A Mississippi woman who made the journey north and built a new life",
        "voice_style": "southern_gospel",
        "age": 55,
    },
    "indigenous-americas": {
        "name": "Keeper of Stories",
        "description": "A guardian of oral traditions who knows the land and its history",
        "voice_style": "deliberate_gentle",
        "age": 60,
    },
}


class Audio2FaceClient:
    """Client for Audio2Face facial animation and ACE avatar services."""

    A2F_PORT = 8011  # Default Audio2Face service port

    def __init__(self, server_url: str | None = None):
        self.server_url = server_url or os.environ.get(
            "OMNIVERSE_COMPOSER_URL", f"http://localhost:{self.A2F_PORT}"
        )
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            resp = requests.get(f"{self.server_url}/status", timeout=3)
            return resp.ok
        except Exception:
            log.info("Audio2Face server not reachable — mock mode")
            return False

    def animate(
        self,
        audio_path: str,
        character_name: str,
        world: str,
        output_path: str = "",
    ) -> AnimationClip:
        """Animate a character from an audio clip."""
        clip = AnimationClip(
            audio_path=audio_path,
            character_name=character_name,
            world=world,
            output_path=output_path or f"/animations/{character_name}_{uuid.uuid4()}.usd",
        )

        if not self._available:
            clip.status = "mock_animated"
            clip.duration_seconds = 5.0
            log.info("[mock] Animated character '%s' from audio", character_name)
            return clip

        try:
            payload = {
                "audio_file": audio_path,
                "character": character_name,
                "output": clip.output_path,
            }
            resp = requests.post(f"{self.server_url}/animate", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            clip.status = "completed"
            clip.duration_seconds = data.get("duration", 0.0)
            clip.blend_shapes_path = data.get("blend_shapes", "")
        except Exception as exc:
            log.error("Audio2Face animation failed: %s", exc)
            clip.status = "error"
        return clip

    def get_world_guide(
        self,
        world: str,
    ) -> dict[str, Any]:
        """Get the AI guide character config for a given world."""
        return CHARACTER_GUIDES.get(world, {
            "name": "World Guide",
            "description": "Your AI educational companion for this world",
            "voice_style": "friendly_neutral",
            "age": 40,
        })

    def generate_guide_line(
        self,
        world: str,
        text: str,
        nim_client: Any = None,
    ) -> AnimationClip:
        """
        Full pipeline: text → TTS audio → Audio2Face animation.
        Requires a NIM audio client for TTS synthesis.
        """
        guide = self.get_world_guide(world)
        audio_path = f"/tmp/guide_{uuid.uuid4()}.wav"

        if nim_client:
            try:
                nim_client.synthesize(text, voice=guide.get("voice_style", "default"), output_path=audio_path)
            except Exception as exc:
                log.warning("TTS failed, using silent clip: %s", exc)
                audio_path = ""

        return self.animate(
            audio_path=audio_path,
            character_name=guide["name"],
            world=world,
        )

    def status(self) -> dict[str, Any]:
        return {
            "audio2face_available": self._available,
            "server_url": self.server_url,
            "world_guides": list(CHARACTER_GUIDES.keys()),
        }
