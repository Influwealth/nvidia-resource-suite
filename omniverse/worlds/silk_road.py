"""
Ancient Silk Road World

100 BCE – 1450 CE. From Chang'an to Constantinople.
The greatest trade network in human history.

This world explores:
  - Trade, economics, and global exchange
  - Cultural and scientific transmission
  - Mathematics, astronomy, and medicine from the Islamic Golden Age
  - Multilingual communication across cultures
"""

from __future__ import annotations

from typing import Any

from .base_world import BaseWorld, SpawnPoint, WorldDocument, WorldQuest


class SilkRoadWorld(BaseWorld):

    THEME = "ancient-silk-road"
    NAME = "Ancient Silk Road"
    PERIOD = "100 BCE – 1450 CE"
    SUBJECTS = ["history", "economics", "geography", "science", "mathematics", "languages"]
    LANGUAGES = ["en", "ar", "zh", "fa", "tr"]

    LOCATIONS = {
        "dunhuang_caves": {
            "position": (0.0, 0.0, 0.0),
            "description": "The Caves of Mogao — 40,000 scrolls, 500 chapels, a library of civilizations.",
        },
        "samarkand_bazaar": {
            "position": (200.0, 0.0, 0.0),
            "description": "Samarkand's Grand Bazaar. Every language, every spice, every idea passes through.",
        },
        "baghdad_house_of_wisdom": {
            "position": (-300.0, 0.0, 50.0),
            "description": "Bayt al-Hikma — where Al-Khwarizmi invented algebra and the world changed forever.",
        },
        "silk_workshop_changan": {
            "position": (400.0, 0.0, -100.0),
            "description": "The silk workshops of Chang'an. The most valuable textile on Earth.",
        },
    }

    def get_spawn_points(self) -> list[SpawnPoint]:
        return [
            SpawnPoint(
                name="Dunhuang Oasis",
                position=(0.0, 0.0, 0.0),
                description="Begin at the crossroads of the Silk Road — where East meets West.",
                suggested_for=["elementary", "middle-school", "high-school", "adult"],
            ),
            SpawnPoint(
                name="House of Wisdom, Baghdad",
                position=(-300.0, 0.0, 50.0),
                description="Join the scholars who preserved and advanced all human knowledge.",
                suggested_for=["middle-school", "high-school", "adult"],
            ),
        ]

    def get_quests(self, age_group: str = "middle-school") -> list[WorldQuest]:
        return [
            WorldQuest(
                title="The Merchant's Journey",
                description="Travel from Chang'an to Samarkand as a silk merchant. 5,000 miles, 3 months.",
                learning_objectives=[
                    "Understand trade routes and economic exchange",
                    "Learn how goods moved before modern transportation",
                    "Explore the cultural mixing that happened along the route",
                ],
                subjects=["history", "economics", "geography"],
                age_groups=["elementary", "middle-school", "high-school"],
                completion_reward="30 EBTK tokens",
                difficulty="easy",
                estimated_minutes=25,
                steps=[
                    {"action": "select", "task": "Choose your cargo: silk, spices, or glass"},
                    {"action": "travel", "task": "Navigate the route from Chang'an to Dunhuang to Samarkand"},
                    {"action": "trade", "location": "samarkand_bazaar", "task": "Negotiate your trade"},
                    {"action": "calculate", "task": "Determine your profit and what you learned"},
                ],
            ),
            WorldQuest(
                title="Al-Khwarizmi and the Birth of Algebra",
                description="At the House of Wisdom in Baghdad, discover how algebra was born.",
                learning_objectives=[
                    "Understand the Islamic Golden Age of science",
                    "Learn the origins of algebra and Al-Khwarizmi's contributions",
                    "Solve historical mathematical problems using the original methods",
                ],
                subjects=["mathematics", "history", "science"],
                age_groups=["middle-school", "high-school", "adult"],
                completion_reward="50 EBTK tokens",
                difficulty="hard",
                estimated_minutes=40,
                steps=[
                    {"action": "visit", "location": "baghdad_house_of_wisdom", "task": "Enter the library"},
                    {"action": "study", "task": "Read Al-Khwarizmi's original algebra problems"},
                    {"action": "solve", "task": "Solve 5 algebra problems using his original geometric method"},
                    {"action": "connect", "task": "Show how this became the algebra you use today"},
                ],
            ),
        ]

    def get_guide(self) -> dict[str, Any]:
        return {
            "name": "Merchant Al-Rashid",
            "description": "A multilingual Silk Road merchant who has traveled from Baghdad to Chang'an and back",
            "voice_style": "measured_arabic",
            "age": 45,
            "languages": ["en", "ar", "fa", "zh"],
            "specialty": "trade, culture, Islamic Golden Age science, geography",
        }

    def get_knowledge_base(self) -> list[WorldDocument]:
        return [
            WorldDocument(
                title="The Silk Road — Overview",
                content=(
                    "The Silk Road was a network of trade routes connecting China, Central Asia, "
                    "South Asia, the Middle East, East Africa, and Southern Europe from approximately "
                    "130 BCE (when the Han Dynasty officially opened trade) to 1453 CE (fall of Constantinople). "
                    "It was never a single road but a web of paths crossing deserts, mountains, and steppes. "
                    "Along it traveled not only silk, spices, and gold, but ideas: Buddhism, Islam, Christianity, "
                    "paper-making, gunpowder, the compass, algebra, and the Black Death."
                ),
                source="Historical research composite",
                doc_type="secondary",
            ),
            WorldDocument(
                title="Al-Khwarizmi and the House of Wisdom",
                content=(
                    "Muhammad ibn Musa al-Khwarizmi (c. 780–850 CE) was a Persian mathematician at "
                    "the House of Wisdom (Bayt al-Hikma) in Baghdad. His book 'Al-Kitab al-Mukhtasar "
                    "fi Hisab al-Jabr wal-Muqabala' (The Compendious Book on Calculation by Completion "
                    "and Balancing, c. 830 CE) gave us the word 'algebra' (from 'al-jabr'). "
                    "His name gave us the word 'algorithm'. He also introduced Hindu-Arabic numerals "
                    "(0-9) to the Western world, replacing Roman numerals."
                ),
                source="History of Mathematics",
                doc_type="primary",
                author="Historical composite",
                date="830 CE (original)",
            ),
        ]
