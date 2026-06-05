"""
Harlem Renaissance World

1920s–1930s Harlem, New York.
The flowering of African American art, literature, music, and intellectual life.

This world brings to life:
  - The Harlem Renaissance literary and artistic movement
  - Jazz and blues music culture
  - The Harlem Hellfighters and WWI context
  - Marcus Garvey and the Pan-African movement
  - Langston Hughes, Zora Neale Hurston, Duke Ellington
"""

from __future__ import annotations

from typing import Any

from .base_world import BaseWorld, SpawnPoint, WorldDocument, WorldQuest


class HarlemRenaissanceWorld(BaseWorld):

    THEME = "harlem-renaissance"
    NAME = "Harlem Renaissance"
    PERIOD = "1920s–1930s, Harlem, New York City"
    SUBJECTS = ["history", "literature", "art", "music", "civil rights", "social studies"]
    LANGUAGES = ["en", "fr"]  # Many Harlem Renaissance figures had French/Francophone connections

    LOCATIONS = {
        "cotton_club": {
            "position": (0.0, 0.0, 0.0),
            "description": "The Cotton Club, 142nd St and Lenox Ave. Duke Ellington plays tonight.",
        },
        "langston_library": {
            "position": (80.0, 0.0, 30.0),
            "description": "The 135th Street Branch Library. Langston Hughes reads here every Thursday.",
        },
        "garvey_liberty_hall": {
            "position": (-100.0, 0.0, 20.0),
            "description": "Liberty Hall. Marcus Garvey speaks to thousands about Black self-determination.",
        },
        "studio_arts_guild": {
            "position": (150.0, 0.0, -50.0),
            "description": "The studio where Aaron Douglas paints his iconic murals.",
        },
    }

    def get_spawn_points(self) -> list[SpawnPoint]:
        return [
            SpawnPoint(
                name="125th Street, 1925",
                position=(0.0, 0.0, -50.0),
                description="The main artery of the Harlem Renaissance. Walk with Langston.",
                suggested_for=["elementary", "middle-school", "high-school", "adult"],
            ),
            SpawnPoint(
                name="Cotton Club Entrance",
                position=(0.0, 1.0, 0.0),
                description="Hear the music that changed the world.",
                suggested_for=["middle-school", "high-school", "adult"],
            ),
        ]

    def get_quests(self, age_group: str = "middle-school") -> list[WorldQuest]:
        return [
            WorldQuest(
                title="Voices of the Renaissance",
                description="Meet the writers and artists who defined the Harlem Renaissance.",
                learning_objectives=[
                    "Understand the Harlem Renaissance as a cultural and political movement",
                    "Read and analyze key works by Langston Hughes and Zora Neale Hurston",
                    "Explore how art and literature can be acts of resistance",
                ],
                subjects=["literature", "history", "art"],
                age_groups=["middle-school", "high-school", "adult"],
                completion_reward="40 EBTK tokens",
                difficulty="medium",
                estimated_minutes=35,
                steps=[
                    {"action": "visit", "location": "langston_library", "task": "Read three poems by Langston Hughes"},
                    {"action": "analyze", "task": "What political message is hidden in 'Let America Be America Again'?"},
                    {"action": "visit", "location": "studio_arts_guild", "task": "Study Aaron Douglas's mural style"},
                    {"action": "create", "task": "Write a poem or create art in the spirit of the Renaissance"},
                ],
            ),
            WorldQuest(
                title="Jazz and Freedom",
                description="How did jazz music become a language of liberation?",
                learning_objectives=[
                    "Understand the African origins of jazz",
                    "Learn about Duke Ellington and the Cotton Club era",
                    "Explore music as cultural resistance and identity",
                ],
                subjects=["music", "history", "social studies"],
                age_groups=["elementary", "middle-school", "high-school"],
                completion_reward="35 EBTK tokens",
                difficulty="easy",
                estimated_minutes=20,
                steps=[
                    {"action": "listen", "location": "cotton_club", "task": "Hear Ellington's orchestra"},
                    {"action": "identify", "task": "Name the African musical traditions that feed into jazz"},
                    {"action": "create", "task": "Compose a simple 12-bar blues progression"},
                ],
            ),
        ]

    def get_guide(self) -> dict[str, Any]:
        return {
            "name": "Langston",
            "description": "Inspired by Langston Hughes — a poet-guide who speaks in rhythm and story",
            "voice_style": "poetic_baritone",
            "age": 30,
            "languages": ["en", "fr"],
            "specialty": "poetry, civil rights, Harlem Renaissance, jazz history",
        }

    def get_knowledge_base(self) -> list[WorldDocument]:
        return [
            WorldDocument(
                title="The Harlem Renaissance — Overview",
                content=(
                    "The Harlem Renaissance (roughly 1920–1940) was an intellectual, social, and artistic "
                    "explosion centered in Harlem, New York. African Americans who had migrated from the "
                    "South and the Caribbean created a new cultural identity through literature, music, art, "
                    "and philosophy. Key figures include Langston Hughes, Zora Neale Hurston, Duke Ellington, "
                    "Louis Armstrong, Marcus Garvey, Alain Locke, Countee Cullen, and Jacob Lawrence. "
                    "The movement laid intellectual groundwork for the Civil Rights Movement."
                ),
                source="Historical research composite",
                doc_type="secondary",
            ),
            WorldDocument(
                title="Langston Hughes — Selected Poems",
                content=(
                    "Langston Hughes (1902-1967) was the poet laureate of Harlem. "
                    "Key works: 'The Weary Blues' (1926), 'Let America Be America Again' (1936), "
                    "'A Dream Deferred' (1951). His poetry directly addressed race, identity, "
                    "and the promise and failure of American democracy. "
                    "He wrote: 'I, too, sing America. / I am the darker brother. / "
                    "They send me to eat in the kitchen / When company comes, / But I laugh, / "
                    "And eat well, / And grow strong.'"
                ),
                source="Langston Hughes Complete Poems",
                doc_type="primary",
                author="Langston Hughes",
                date="1926-1951",
            ),
        ]
