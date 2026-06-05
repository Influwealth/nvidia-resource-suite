"""Adaptive curriculum engine: K-12 STEM, AI literacy, sovereign economics."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class AgeGroup(str, Enum):
    K4 = "k4"         # Kindergarten-Grade 4  (5-9)
    G5_8 = "g5_8"     # Grades 5-8            (10-13)
    G9_12 = "g9_12"   # Grades 9-12           (14-17)
    ADULT = "adult"   # 18+
    ALL = "all"


@dataclass
class LearnerProfile:
    learner_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    age_group: AgeGroup = AgeGroup.G5_8
    language: str = "en"
    world: str = "east_flatbush"
    completed_lessons: list[str] = field(default_factory=list)
    ebtk_balance: float = 0.0
    gvc_balance: float = 0.0
    skills: dict[str, float] = field(default_factory=dict)  # skill → 0-100
    accessibility: list[str] = field(default_factory=list)  # dyslexia, screen_reader, etc.


@dataclass
class LessonPlan:
    lesson_id: str
    title: str
    subject: str
    age_group: AgeGroup
    world: str
    duration_min: int
    objectives: list[str]
    activities: list[dict]
    assessment: dict
    ebtk_reward: float = 0.0
    prerequisites: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=lambda: ["en"])
    nvidia_tech_used: list[str] = field(default_factory=list)


class CurriculumEngine:
    """Adaptive curriculum generator powered by NIM LLM.

    Generates personalized lesson plans anchored to the world-building
    theme of the learner’s current environment. Integrates NVIDIA tech
    (Warp, Modulus, RAPIDS, NIM) as hands-on learning tools.
    """

    SUBJECTS = [
        "mathematics", "physics", "history", "geography",
        "computer_science", "ai_literacy", "economics",
        "ecology", "language_arts", "music",
    ]

    WORLD_CONTEXTS = {
        "east_flatbush": "1990s Brooklyn — Caribbean-American community, hip-hop culture, block economics",
        "greenville": "Greenville NC — 53-acre sovereign site, community banking, sustainable agriculture",
        "ancient_silk_road": "Silk Road trade routes — mathematics, astronomy, cross-cultural exchange",
        "harlem_renaissance": "1920s-30s Harlem — art, jazz, literature, the Great Migration",
    }

    def __init__(self, nim_client=None):
        self._nim = nim_client

    def generate_lesson(
        self,
        profile: LearnerProfile,
        subject: str,
        topic: str,
        duration_min: int = 45,
    ) -> LessonPlan:
        world_ctx = self.WORLD_CONTEXTS.get(profile.world, "")
        objectives = self._generate_objectives(subject, topic, profile.age_group, world_ctx)
        activities = self._generate_activities(subject, topic, profile, world_ctx)
        assessment = self._generate_assessment(subject, topic, profile.age_group)
        ebtk = self._compute_ebtk_reward(duration_min, profile.age_group)
        nvidia_tech = self._select_nvidia_tech(subject)

        return LessonPlan(
            lesson_id=str(uuid.uuid4())[:8],
            title=f"{topic} in the World of {profile.world.replace('_', ' ').title()}",
            subject=subject,
            age_group=profile.age_group,
            world=profile.world,
            duration_min=duration_min,
            objectives=objectives,
            activities=activities,
            assessment=assessment,
            ebtk_reward=ebtk,
            languages=[profile.language],
            nvidia_tech_used=nvidia_tech,
        )

    def _generate_objectives(self, subject: str, topic: str, age: AgeGroup, world: str) -> list[str]:
        base = [
            f"Understand the core principles of {topic}",
            f"Apply {topic} concepts to real-world scenarios in {world}",
            f"Demonstrate learning through a hands-on activity",
        ]
        if age == AgeGroup.K4:
            base.insert(0, f"Recognize basic vocabulary related to {topic}")
        elif age in (AgeGroup.G9_12, AgeGroup.ADULT):
            base.append(f"Analyze and evaluate {topic} using data-driven tools")
        return base

    def _generate_activities(
        self, subject: str, topic: str, profile: LearnerProfile, world: str
    ) -> list[dict]:
        activities = [
            {
                "type": "introduction",
                "title": "World Guide Narration",
                "description": f"Your AI guide introduces {topic} through the lens of {profile.world}.",
                "duration_min": 10,
                "nvidia_tool": "NeMo TTS + NIM LLM",
            },
            {
                "type": "exploration",
                "title": "Interactive Simulation",
                "description": f"Explore {topic} using a GPU-accelerated simulation in the Omniverse world.",
                "duration_min": 20,
                "nvidia_tool": self._select_nvidia_tech(subject)[0] if self._select_nvidia_tech(subject) else "NIM",
            },
            {
                "type": "quiz",
                "title": "Knowledge Check",
                "description": f"Answer 5 questions about {topic} to earn EBTK tokens.",
                "duration_min": 10,
                "nvidia_tool": "NIM LLM",
            },
            {
                "type": "reflection",
                "title": "Community Connection",
                "description": f"Share what you learned with the {profile.world} community on Synapz.",
                "duration_min": 5,
                "nvidia_tool": "Synapz ICP",
            },
        ]
        return activities

    def _generate_assessment(
        self, subject: str, topic: str, age: AgeGroup
    ) -> dict:
        return {
            "type": "quiz" if age in (AgeGroup.K4, AgeGroup.G5_8) else "project",
            "passing_score": 70,
            "max_attempts": 3,
            "questions": [
                {"q": f"What is the main concept of {topic}?", "type": "short_answer"},
                {"q": f"Give one real-world example of {topic}.", "type": "short_answer"},
                {"q": f"How does {topic} connect to your community?", "type": "essay"},
            ],
        }

    def _compute_ebtk_reward(self, duration_min: int, age: AgeGroup) -> float:
        base = duration_min * 0.5
        multiplier = {AgeGroup.K4: 1.0, AgeGroup.G5_8: 1.2, AgeGroup.G9_12: 1.5, AgeGroup.ADULT: 2.0}.get(age, 1.0)
        return round(base * multiplier, 2)

    def _select_nvidia_tech(self, subject: str) -> list[str]:
        mapping = {
            "physics": ["NVIDIA Warp", "Modulus"],
            "mathematics": ["RAPIDS cuML", "NIM LLM"],
            "computer_science": ["NIM LLM", "TensorRT", "Triton"],
            "ai_literacy": ["NIM LLM", "NeMo Guardrails"],
            "ecology": ["Earth-2 CorrDiff", "Modulus Digital Twin"],
            "economics": ["RAPIDS cuDF", "NIM LLM"],
            "history": ["NIM LLM", "Omniverse", "NeMo ASR"],
            "geography": ["Earth-2 FourCastNet", "CorrDiff"],
        }
        return mapping.get(subject, ["NIM LLM"])

    def suggest_next_lesson(
        self,
        profile: LearnerProfile,
        available_subjects: list[str] | None = None,
    ) -> dict:
        """Suggest the next lesson based on learner progress and skill gaps."""
        subjects = available_subjects or self.SUBJECTS
        skill_scores = profile.skills
        weakest = min(
            (s for s in subjects if s in skill_scores),
            key=lambda s: skill_scores[s],
            default=subjects[0],
        )
        return {
            "recommended_subject": weakest,
            "reason": f"Skill score for '{weakest}' is {skill_scores.get(weakest, 0):.0f}/100 — room to grow!",
            "world": profile.world,
            "ebtk_potential": self._compute_ebtk_reward(45, profile.age_group),
        }

    def full_curriculum(
        self,
        profile: LearnerProfile,
        weeks: int = 12,
    ) -> list[LessonPlan]:
        """Generate a full 12-week curriculum for a learner."""
        plans = []
        subjects_cycle = (self.SUBJECTS * ((weeks * 5) // len(self.SUBJECTS) + 1))[:weeks * 5]
        topics_by_subject = {
            "mathematics": ["fractions", "algebra", "geometry", "statistics", "calculus"],
            "physics": ["motion", "forces", "energy", "waves", "electromagnetism"],
            "computer_science": ["algorithms", "data structures", "machine learning", "networks", "security"],
            "ai_literacy": ["what is AI", "neural networks", "bias in AI", "prompt engineering", "AI ethics"],
            "economics": ["supply and demand", "community banking", "cryptocurrency basics", "land value", "entrepreneurship"],
        }
        used_topics: dict[str, int] = {}
        for subject in subjects_cycle:
            topic_list = topics_by_subject.get(subject, ["core concepts", "applications", "advanced topics"])
            idx = used_topics.get(subject, 0) % len(topic_list)
            used_topics[subject] = idx + 1
            plans.append(self.generate_lesson(profile, subject, topic_list[idx]))
        return plans
