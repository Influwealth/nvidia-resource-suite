"""
NVIDIA NIM — Vision Client

Multimodal image understanding for:
  - Analyzing student-submitted artwork
  - Identifying historical objects in world scenes
  - Accessibility descriptions of 3D renders
  - Community photo archiving (describe → index)
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from .client import NIMClient

log = logging.getLogger(__name__)


class VisionClient:
    """NIM vision client for image understanding tasks."""

    def __init__(self, nim_client: NIMClient | None = None):
        self.nim = nim_client or NIMClient()
        self._default_model = "microsoft/phi-3-vision-128k-instruct"

    def _encode_image(self, image_path: str) -> str:
        """Encode a local image file as a base64 data URL."""
        path = Path(image_path)
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        mime = mime_map.get(suffix, "image/jpeg")
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def describe(
        self,
        image: str,
        context: str = "",
        model: str | None = None,
    ) -> str:
        """Describe an image. image can be a URL or local file path."""
        if not image.startswith("http"):
            image = self._encode_image(image)
        prompt = "Describe this image in detail."
        if context:
            prompt += f" Context: {context}"
        return self.nim.vision_chat(prompt, image, model=model or self._default_model)

    def identify_historical_objects(
        self,
        image: str,
        period: str,
        culture: str,
    ) -> str:
        """Identify historical artifacts, clothing, or objects in a scene."""
        if not image.startswith("http"):
            image = self._encode_image(image)
        prompt = (
            f"This image is from the {period} period in {culture} culture. "
            f"Identify all historically significant objects, clothing, architecture, and artifacts. "
            f"Explain each item's cultural and historical significance."
        )
        return self.nim.vision_chat(prompt, image, model=self._default_model)

    def generate_alt_text(
        self,
        image: str,
        audience: str = "general",
    ) -> str:
        """Generate accessibility alt text for an image."""
        if not image.startswith("http"):
            image = self._encode_image(image)
        prompt = (
            f"Write a concise, descriptive alt text for this image, appropriate for a {audience} audience. "
            f"Include all relevant details for someone who cannot see the image."
        )
        return self.nim.vision_chat(prompt, image, model=self._default_model, max_tokens=200)

    def analyze_student_artwork(
        self,
        image: str,
        assignment_context: str,
    ) -> dict[str, str]:
        """Analyze student-submitted artwork and provide encouraging feedback."""
        if not image.startswith("http"):
            image = self._encode_image(image)

        feedback_prompt = (
            f"A student submitted this artwork for the assignment: '{assignment_context}'. "
            f"Provide encouraging, specific, and constructive feedback. "
            f"Note what they did well, and offer one or two concrete suggestions for growth. "
            f"Be warm and supportive — this student is learning."
        )
        description_prompt = "Describe what you see in this image."

        return {
            "description": self.nim.vision_chat(description_prompt, image, max_tokens=200),
            "feedback": self.nim.vision_chat(feedback_prompt, image, max_tokens=400),
        }

    def extract_text_from_image(
        self,
        image: str,
        language_hint: str = "",
    ) -> str:
        """Extract and transcribe text visible in an image (OCR via vision model)."""
        if not image.startswith("http"):
            image = self._encode_image(image)
        prompt = "Extract and transcribe all text visible in this image, preserving formatting as best as possible."
        if language_hint:
            prompt += f" The text is primarily in {language_hint}."
        return self.nim.vision_chat(prompt, image, max_tokens=1024)
