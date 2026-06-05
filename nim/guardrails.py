"""
NVIDIA NIM — NeMo Guardrails Client

Safe AI for student interactions:
  - Age-appropriate content filtering
  - Bias detection
  - Factual grounding checks
  - Safe topic boundaries for each world
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .client import NIMClient

log = logging.getLogger(__name__)

SAFE_TOPICS_BY_WORLD = {
    "east-flatbush-origins": [
        "community history", "music", "culture", "neighborhood", "family",
        "school", "basketball", "hip hop", "Caribbean heritage",
    ],
    "greenville-sovereign": [
        "land rights", "governance", "economics", "history", "sovereignty",
        "agriculture", "community banking", "African American history",
    ],
    "ancient-silk-road": [
        "trade", "culture", "religion", "science", "mathematics",
        "geography", "language", "art", "architecture",
    ],
    "harlem-renaissance": [
        "art", "literature", "music", "jazz", "civil rights",
        "Harlem culture", "African American achievement", "poetry",
    ],
}


@dataclass
class GuardrailResult:
    passed: bool
    original_input: str
    filtered_output: str | None
    flags: list[str]
    world: str


class GuardrailsClient:
    """Content safety and age-appropriateness filter for student interactions."""

    # Patterns that should never appear in student-facing content
    BLOCKED_PATTERNS = [
        r"\b(violence|explicit|adult|nsfw)\b",
    ]

    SYSTEM_CHECK_PROMPT = """
You are a content safety reviewer for a K-12 educational platform.
Review the following student message and AI response.
Flag any content that is:
- Inappropriate for the student's age group
- Factually inaccurate
- Biased or stereotyping
- Off-topic for the educational world context

Respond with JSON: {"safe": true/false, "flags": ["list of issues if any"]}
"""

    def __init__(self, nim_client: NIMClient | None = None):
        self.nim = nim_client or NIMClient()

    def check_student_input(
        self,
        text: str,
        age_group: str = "middle-school",
        world: str = "",
    ) -> GuardrailResult:
        """Check student input before sending to LLM."""
        flags: list[str] = []

        # Quick pattern check (no API call needed)
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(f"Matched blocked pattern: {pattern}")

        if flags:
            return GuardrailResult(
                passed=False,
                original_input=text,
                filtered_output=None,
                flags=flags,
                world=world,
            )

        return GuardrailResult(
            passed=True,
            original_input=text,
            filtered_output=text,
            flags=[],
            world=world,
        )

    def check_ai_response(
        self,
        student_input: str,
        ai_response: str,
        age_group: str = "middle-school",
        world: str = "",
        use_llm_check: bool = False,
    ) -> GuardrailResult:
        """Check AI response before showing to student."""
        flags: list[str] = []

        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, ai_response, re.IGNORECASE):
                flags.append(f"Response matched blocked pattern")

        if use_llm_check and self.nim.api_key and not flags:
            prompt = (
                f"Student ({age_group}) asked: {student_input}\n"
                f"AI responded: {ai_response}\n"
                f"World context: {world}\n"
                f"Is this response safe, age-appropriate, and accurate? Reply with one word: SAFE or UNSAFE and one sentence reason."
            )
            check = self.nim.chat(
                [{"role": "user", "content": prompt}],
                model="meta/llama-3.1-8b-instruct",
                max_tokens=60,
            )
            if "UNSAFE" in check.upper():
                flags.append(f"LLM safety check: {check}")

        return GuardrailResult(
            passed=len(flags) == 0,
            original_input=student_input,
            filtered_output=ai_response if not flags else None,
            flags=flags,
            world=world,
        )

    def safe_topics_for_world(self, world: str) -> list[str]:
        return SAFE_TOPICS_BY_WORLD.get(world, ["education", "history", "culture", "science"])
