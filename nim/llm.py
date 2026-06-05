"""
NVIDIA NIM — LLM Client

High-level client for language model tasks:
  - tutoring and education
  - code generation and review
  - reasoning and analysis
  - multilingual support
  - world-building narrative generation
"""

from __future__ import annotations

import logging
from typing import Iterator

from .client import NIMClient

log = logging.getLogger(__name__)

EDUCATION_SYSTEM_PROMPT = """
You are an expert educator and world-guide. Your role is to:
- Explain complex topics clearly and accessibly
- Adapt your language to the student's age and background
- Connect history, science, and culture to the student's own community
- Ask questions that encourage critical thinking
- Never talk down to students; always respect their intelligence
- Use stories, analogies, and examples from the world they're exploring
"""

WORLD_BUILDING_SYSTEM_PROMPT = """
You are a master world-builder and narrative designer.
You create rich, historically accurate, culturally respectful digital worlds.
You know how to balance educational accuracy with engaging storytelling.
You draw on primary sources, community oral histories, and expert scholarship.
"""


class LLMClient:
    """High-level LLM client optimized for educational world-building."""

    def __init__(self, nim_client: NIMClient | None = None):
        self.nim = nim_client or NIMClient()

    def tutor(
        self,
        question: str,
        world_context: str = "",
        age_group: str = "middle-school",
        language: str = "en",
        model: str = "meta/llama-3.1-70b-instruct",
    ) -> str:
        """Answer a student's question in the context of the current world."""
        system = EDUCATION_SYSTEM_PROMPT
        if world_context:
            system += f"\n\nCurrent world context: {world_context}"
        if age_group == "elementary":
            system += "\n\nSpeak simply, use short sentences, be encouraging."
        elif age_group == "high-school":
            system += "\n\nYou can use technical terms, cite sources, encourage debate."
        elif age_group == "adult":
            system += "\n\nTreat as a peer. Be comprehensive and cite scholarly sources."

        lang_instruction = f" Respond in {language}." if language != "en" else ""
        messages = [
            {"role": "system", "content": system + lang_instruction},
            {"role": "user", "content": question},
        ]
        return self.nim.chat(messages, model=model)

    def stream_tutor(
        self,
        question: str,
        world_context: str = "",
        age_group: str = "middle-school",
        model: str = "meta/llama-3.1-70b-instruct",
    ) -> Iterator[str]:
        """Streaming tutor — yields tokens as they arrive."""
        system = EDUCATION_SYSTEM_PROMPT
        if world_context:
            system += f"\n\nWorld context: {world_context}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        yield from self.nim.stream_chat(messages, model=model)

    def generate_world_narrative(
        self,
        world_name: str,
        period: str,
        location: str,
        themes: list[str],
        length: str = "medium",
    ) -> str:
        """Generate a rich narrative description for a world scene."""
        length_tokens = {"short": 256, "medium": 512, "long": 1024}
        prompt = (
            f"Write a vivid, historically accurate, culturally respectful scene description "
            f"for the '{world_name}' educational world.\n"
            f"Period: {period}\n"
            f"Location: {location}\n"
            f"Themes to weave in: {', '.join(themes)}\n"
            f"Make it immersive — the student should feel they are standing there."
        )
        messages = [
            {"role": "system", "content": WORLD_BUILDING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.nim.chat(messages, max_tokens=length_tokens.get(length, 512))

    def explain_concept(
        self,
        concept: str,
        domain: str,
        examples: list[str] | None = None,
        analogy_style: str = "everyday",
    ) -> str:
        """Explain a concept with examples and analogies."""
        prompt = f"Explain '{concept}' in the domain of {domain}."
        if examples:
            prompt += f" Use these examples: {', '.join(examples)}."
        prompt += f" Use {analogy_style} analogies. Be clear and engaging."
        messages = [
            {"role": "system", "content": EDUCATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.nim.chat(messages)

    def generate_quiz(
        self,
        topic: str,
        num_questions: int = 5,
        difficulty: str = "medium",
        format: str = "multiple-choice",
    ) -> str:
        """Generate a quiz on any topic."""
        prompt = (
            f"Create a {num_questions}-question {format} quiz on '{topic}' "
            f"at {difficulty} difficulty level. "
            f"Include the correct answers at the end."
        )
        messages = [
            {"role": "system", "content": EDUCATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.nim.chat(messages, max_tokens=2048)

    def translate(
        self,
        text: str,
        target_language: str,
        preserve_cultural_terms: bool = True,
    ) -> str:
        """Translate educational content to another language."""
        instruction = (
            f"Translate the following to {target_language}. "
        )
        if preserve_cultural_terms:
            instruction += "Preserve cultural terms and proper nouns. Add a brief note if cultural context is needed."
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ]
        return self.nim.chat(messages, model="meta/llama-3.1-70b-instruct")

    def code_assist(
        self,
        task: str,
        language: str = "python",
        context: str = "",
    ) -> str:
        """Help students learn to code within world-building tasks."""
        system = (
            "You are a patient coding teacher. Explain your code step by step. "
            "Use comments. Never just give the answer — guide the student."
        )
        prompt = f"Help me write {language} code to: {task}"
        if context:
            prompt += f"\n\nContext: {context}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.nim.chat(messages, model="meta/codellama-70b")
