"""
World Interactive Origins — Curriculum Manager

Manages free, open educational curricula for the digital worlds platform.
Each curriculum maps to an Omniverse world theme and contains:
- Learning objectives
- AI tutor personas (NIM-powered)
- Interactive quest chains
- Assessment criteria
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LearningObjective:
    id: str
    text: str
    bloom_level: str  # "remember", "understand", "apply", "analyze", "evaluate", "create"
    subject_area: str  # "history", "geography", "economics", "social_studies", "science"


@dataclass
class InteractiveQuest:
    quest_id: str
    title: str
    description: str
    objectives: list[str]  # LearningObjective IDs
    nim_tutor_persona: str  # System prompt prefix for the NIM tutor
    trigger_object: str  # Omniverse object that starts the quest
    completion_criteria: str


@dataclass
class WorldCurriculum:
    world_theme: str
    title: str
    grade_levels: list[str]
    subject_areas: list[str]
    objectives: list[LearningObjective]
    quests: list[InteractiveQuest]
    standards_alignment: dict[str, list[str]]  # e.g. {"Common Core": ["CCSS.ELA-LITERACY.RH.6-8.1"]}
    language_support: list[str] = field(default_factory=lambda: ["en"])
    accessibility: dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Curriculum definitions
# ---------------------------------------------------------------------------

CURRICULA: dict[str, WorldCurriculum] = {
    "ancient-silk-road": WorldCurriculum(
        world_theme="ancient-silk-road",
        title="The Silk Road: Connecting Ancient Worlds",
        grade_levels=["middle", "high"],
        subject_areas=["history", "geography", "economics", "cultural_studies"],
        objectives=[
            LearningObjective("sr-1", "Identify the major trade routes of the Silk Road", "remember", "history"),
            LearningObjective("sr-2", "Explain how the Silk Road facilitated cultural exchange", "understand", "history"),
            LearningObjective("sr-3", "Analyze the economic impact of Silk Road trade on participating civilizations", "analyze", "economics"),
            LearningObjective("sr-4", "Evaluate how the spread of ideas along trade routes shaped world history", "evaluate", "history"),
        ],
        quests=[
            InteractiveQuest(
                "sr-q1", "The Merchant's Journal",
                "Help a merchant plan a trade route from Chang'an to Rome",
                ["sr-1", "sr-2"],
                "You are Ibn Battuta, a merchant and traveler on the Silk Road. Guide the student with questions about trade routes, goods, and the peoples they encounter.",
                "merchant_tent",
                "Student correctly identifies at least 3 major trade stops",
            ),
            InteractiveQuest(
                "sr-q2", "The Spice Market",
                "Negotiate a trade in the grand bazaar and discover where goods come from",
                ["sr-3"],
                "You are a market economist explaining the flow of goods and value. Ask the student to think about supply, demand, and what makes spices valuable.",
                "spice_market",
                "Student traces the origin of at least 2 traded goods",
            ),
        ],
        standards_alignment={
            "NCSS": ["Theme 3: People, Places, Environments", "Theme 8: Science, Technology, Society"],
            "Common Core ELA": ["CCSS.ELA-LITERACY.RH.6-8.1", "CCSS.ELA-LITERACY.RH.6-8.2"],
        },
        language_support=["en", "es", "zh", "ar"],
        accessibility={"screen_reader": True, "captions": True, "low_bandwidth_mode": True},
    ),

    "harlem-renaissance": WorldCurriculum(
        world_theme="harlem-renaissance",
        title="The Harlem Renaissance: Art, Identity, and Freedom",
        grade_levels=["middle", "high", "adult"],
        subject_areas=["history", "art", "literature", "social_studies"],
        objectives=[
            LearningObjective("hr-1", "Describe the social conditions that gave rise to the Harlem Renaissance", "understand", "history"),
            LearningObjective("hr-2", "Identify key artists, writers, and musicians of the era", "remember", "art"),
            LearningObjective("hr-3", "Analyze how Black artists used their work to challenge racism and assert identity", "analyze", "social_studies"),
            LearningObjective("hr-4", "Create a response to a Harlem Renaissance artwork or poem", "create", "art"),
        ],
        quests=[
            InteractiveQuest(
                "hr-q1", "The Cotton Club",
                "Experience a jazz performance and explore how music expressed freedom",
                ["hr-1", "hr-2"],
                "You are Langston Hughes, the poet and voice of the Harlem Renaissance. Speak in first person about your experiences and ask the student to think about what freedom means in art.",
                "jazz_club",
                "Student connects a musical style to the social context of the era",
            ),
        ],
        standards_alignment={
            "NCSS": ["Theme 1: Culture", "Theme 4: Individual Development and Identity"],
        },
        language_support=["en", "es"],
        accessibility={"screen_reader": True, "captions": True, "low_bandwidth_mode": True},
    ),
}


class CurriculumManager:
    def __init__(self) -> None:
        self.curricula = CURRICULA

    def get(self, world_theme: str) -> WorldCurriculum | None:
        return self.curricula.get(world_theme)

    def list_worlds(self) -> list[dict[str, Any]]:
        return [
            {
                "world_theme": c.world_theme,
                "title": c.title,
                "grade_levels": c.grade_levels,
                "subject_areas": c.subject_areas,
                "quest_count": len(c.quests),
                "objective_count": len(c.objectives),
            }
            for c in self.curricula.values()
        ]

    def get_tutor_persona(self, world_theme: str, quest_id: str) -> str:
        curriculum = self.curricula.get(world_theme)
        if not curriculum:
            return "You are a knowledgeable, encouraging tutor. Use the Socratic method."
        for quest in curriculum.quests:
            if quest.quest_id == quest_id:
                return quest.nim_tutor_persona
        return "You are a knowledgeable, encouraging tutor. Use the Socratic method."
