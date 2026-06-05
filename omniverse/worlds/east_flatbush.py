"""
East Flatbush Origins — Priority World

458 E 94th Street, Brooklyn, NY — 1990s.
The cultural origin point of the Monadic Archive.

This world reconstructs East Flatbush in the 1990s using:
  - Community-contributed oral histories
  - Archival photography
  - NVIDIA Omniverse photorealistic rendering
  - NIM AI guides voiced by community members

Connects to:
  - roblox-wealthbridge-east-flatbush (Roblox game layer)
  - eastflatbush-90s-unity (Unity 3D)
  - synapz (The 458 Frequency documentary)
"""

from __future__ import annotations

from typing import Any

from .base_world import BaseWorld, SpawnPoint, WorldDocument, WorldQuest


class EastFlatbushWorld(BaseWorld):

    THEME = "brooklyn-90s"
    NAME = "East Flatbush Origins"
    PERIOD = "1990–2000, East Flatbush, Brooklyn, NY"
    SUBJECTS = ["local history", "music", "community", "economics", "social studies"]
    LANGUAGES = ["en", "es", "ht"]  # English, Spanish, Haitian Creole

    # Key locations in the world
    LOCATIONS = {
        "corner_of_e94_and_flatbush": {
            "position": (0.0, 0.0, 0.0),
            "description": "The heart of the neighborhood. Flatbush Ave buzzing with life.",
        },
        "458_e_94th_entrance": {
            "position": (25.0, 0.0, 5.0),
            "description": "458 E 94th Street. The anchor building. Home to the 458 Frequency.",
        },
        "community_basketball_court": {
            "position": (-40.0, 0.0, 20.0),
            "description": "The court where generations played. Legends were born here.",
        },
        "west_indian_market": {
            "position": (60.0, 0.0, -10.0),
            "description": "The Caribbean market. Plantains, roti, doubles, and conversation.",
        },
        "record_shop": {
            "position": (15.0, 0.0, -30.0),
            "description": "The record shop where hip-hop and reggae met. Everything 12-inch.",
        },
    }

    def get_spawn_points(self) -> list[SpawnPoint]:
        return [
            SpawnPoint(
                name="Corner of E 94th and Flatbush",
                position=(0.0, 0.0, 0.0),
                description="Start at the beating heart of East Flatbush, 1993.",
                suggested_for=["elementary", "middle-school", "high-school", "adult"],
            ),
            SpawnPoint(
                name="Front Steps of 458",
                position=(25.0, 1.5, 5.0),
                description="The famous stoop of 458 E 94th. Where the 458 Frequency was born.",
                suggested_for=["middle-school", "high-school", "adult"],
            ),
            SpawnPoint(
                name="Basketball Court",
                position=(-40.0, 0.0, 20.0),
                description="The community court. Start with a pickup game and learn the neighborhood.",
                suggested_for=["elementary", "middle-school"],
            ),
        ]

    def get_quests(self, age_group: str = "middle-school") -> list[WorldQuest]:
        all_quests = [
            WorldQuest(
                title="The 458 Frequency",
                description="Discover the story of 458 E 94th Street and the music that came from its walls.",
                learning_objectives=[
                    "Understand how place shapes culture",
                    "Explore the roots of hip-hop in Caribbean communities",
                    "Learn about community resilience during the 1990s crack epidemic",
                ],
                subjects=["history", "music", "social studies"],
                age_groups=["middle-school", "high-school", "adult"],
                completion_reward="50 EBTK tokens + The 458 Frequency documentary access",
                difficulty="medium",
                estimated_minutes=30,
                steps=[
                    {"action": "visit", "location": "458_e_94th_entrance", "task": "Enter the building and find the first clue"},
                    {"action": "listen", "location": "record_shop", "task": "Hear the sounds that defined the block"},
                    {"action": "talk", "character": "Community Elder", "task": "Ask about the neighborhood's transformation"},
                    {"action": "create", "task": "Write or record your own verse inspired by what you've learned"},
                ],
            ),
            WorldQuest(
                title="Caribbean Roots",
                description="Explore the rich Caribbean heritage that shaped East Flatbush.",
                learning_objectives=[
                    "Learn about Caribbean immigration to Brooklyn",
                    "Understand how communities maintain cultural identity",
                    "Explore the economic contributions of immigrant communities",
                ],
                subjects=["history", "geography", "social studies", "economics"],
                age_groups=["elementary", "middle-school", "high-school"],
                completion_reward="30 EBTK tokens",
                difficulty="easy",
                estimated_minutes=20,
                steps=[
                    {"action": "visit", "location": "west_indian_market", "task": "Explore the market and learn what each item means"},
                    {"action": "talk", "character": "Market Vendor", "task": "Hear a story from the vendor about home"},
                    {"action": "map", "task": "Mark where the neighborhood's families originally came from"},
                ],
            ),
            WorldQuest(
                title="Block Economics",
                description="How does a neighborhood's economy work? Follow the money on the block.",
                learning_objectives=[
                    "Understand informal and formal economies",
                    "Learn about redlining and its impact on Black neighborhoods",
                    "Explore community wealth-building strategies",
                ],
                subjects=["economics", "history", "mathematics"],
                age_groups=["high-school", "adult"],
                completion_reward="75 EBTK tokens + WealthBridge financial literacy module",
                difficulty="hard",
                estimated_minutes=45,
                steps=[
                    {"action": "analyze", "task": "Review the block's business landscape in 1990 vs 2000"},
                    {"action": "calculate", "task": "Estimate the economic output of five local businesses"},
                    {"action": "research", "task": "Find the redlining map for this neighborhood"},
                    {"action": "propose", "task": "Design a community development plan for the block"},
                ],
            ),
        ]
        return [q for q in all_quests if age_group in q.age_groups] or all_quests

    def get_guide(self) -> dict[str, Any]:
        return {
            "name": "Community Elder",
            "description": "A lifelong East Flatbush resident who has seen the neighborhood through every era",
            "voice_style": "warm_caribbean",
            "age": 70,
            "languages": ["en", "ht", "es"],
            "specialty": "community history, oral tradition, Caribbean culture",
        }

    def get_knowledge_base(self) -> list[WorldDocument]:
        return [
            WorldDocument(
                title="East Flatbush Neighborhood History",
                content=(
                    "East Flatbush, Brooklyn is one of New York City's most diverse neighborhoods, "
                    "shaped heavily by Caribbean immigration beginning in the 1960s and intensifying "
                    "through the 1980s-90s. Jamaican, Haitian, Trinidadian, and Barbadian communities "
                    "transformed the neighborhood's culture, food, music, and economy. "
                    "The area encompasses Flatbush Avenue as its main commercial corridor, "
                    "with streets like E 94th serving as intimate community anchors."
                ),
                source="Community oral history composite",
                doc_type="primary",
                date="1990s",
            ),
            WorldDocument(
                title="Hip-Hop's Caribbean Roots in Brooklyn",
                content=(
                    "The hip-hop music that emerged from neighborhoods like East Flatbush in the late 1980s "
                    "and 1990s carried strong Caribbean influences — the riddim-influenced flow, "
                    "the sound system culture inherited from Jamaica, and the patois-inflected lyrics "
                    "that mixed English with Creole. Artists like Notorious B.I.G., who grew up nearby "
                    "in Bed-Stuy, and the whole Brooklyn rap scene were deeply connected to this heritage."
                ),
                source="Music history research",
                doc_type="secondary",
            ),
            WorldDocument(
                title="The 1990s Crack Epidemic's Impact on East Flatbush",
                content=(
                    "The crack cocaine epidemic of the late 1980s and early 1990s devastated many "
                    "Brooklyn neighborhoods, including East Flatbush. Families were torn apart, "
                    "incarceration rates soared under policies like the Rockefeller Drug Laws, "
                    "and community institutions faced immense pressure. Yet community organizations, "
                    "churches, and families fought back — creating mentorship programs, "
                    "after-school initiatives, and cultural spaces that preserved the neighborhood's soul."
                ),
                source="Social history archive",
                doc_type="secondary",
            ),
        ]
