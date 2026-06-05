"""
Greenville Sovereign World

53-acre Monadic anchor site, Greenville, NC.
The first buildable sovereign Black civilization site in the Monadic Archive.

This world maps 1:1 to physical land owned by the community.
Each parcel is a capsule NFT on the Internet Computer (ICP).
The GRN-USD stablecoin is backed by this land as collateral.
"""

from __future__ import annotations

from typing import Any

from .base_world import BaseWorld, SpawnPoint, WorldDocument, WorldQuest


class GreenvilleWorld(BaseWorld):

    THEME = "greenville-sovereign"
    NAME = "Greenville Sovereign World"
    PERIOD = "Pre-1865 to Present — Greenville, NC"
    SUBJECTS = ["history", "economics", "land rights", "governance", "agriculture", "finance"]
    LANGUAGES = ["en"]

    # 53-acre site zones
    ZONES = {
        "main_entrance": {
            "position": (0.0, 0.0, 0.0),
            "description": "Welcome arch. The land claims begin here.",
        },
        "community_commons": {
            "position": (100.0, 0.0, 50.0),
            "description": "The central gathering space. Governance happens here.",
        },
        "agricultural_zone": {
            "position": (-150.0, 0.0, 100.0),
            "description": "53 acres of productive land. Crops, orchards, and research plots.",
        },
        "charter_bank": {
            "position": (50.0, 0.0, -80.0),
            "description": "The Greenville Charter Community Bank. GRN-USD redeemable here.",
        },
        "history_memorial": {
            "position": (-50.0, 0.0, -120.0),
            "description": "Memorial to the land's history from pre-1865 through reconstruction and beyond.",
        },
    }

    def get_spawn_points(self) -> list[SpawnPoint]:
        return [
            SpawnPoint(
                name="Main Entrance",
                position=(0.0, 0.0, 0.0),
                description="Begin your journey at the entrance to 53 acres of sovereign land.",
                suggested_for=["elementary", "middle-school", "high-school", "adult"],
            ),
            SpawnPoint(
                name="Community Commons",
                position=(100.0, 0.0, 50.0),
                description="Join the community assembly and participate in governance.",
                suggested_for=["high-school", "adult"],
            ),
            SpawnPoint(
                name="History Memorial",
                position=(-50.0, 0.0, -120.0),
                description="Start with the land's history. Understand what came before.",
                suggested_for=["middle-school", "high-school", "adult"],
            ),
        ]

    def get_quests(self, age_group: str = "middle-school") -> list[WorldQuest]:
        return [
            WorldQuest(
                title="The Land and Its History",
                description="Walk the 53 acres and learn what this land has witnessed across 200 years.",
                learning_objectives=[
                    "Understand the history of Black land ownership in the South",
                    "Learn about land theft, the promise of 40 acres, and Reconstruction",
                    "Explore how this land was reclaimed as sovereign territory",
                ],
                subjects=["history", "social studies", "geography"],
                age_groups=["middle-school", "high-school", "adult"],
                completion_reward="50 GRN-USD + land deed NFT",
                difficulty="medium",
                estimated_minutes=30,
                steps=[
                    {"action": "visit", "location": "history_memorial", "task": "Read the land's history through each era"},
                    {"action": "research", "task": "Find how much Black-owned land was lost 1910-1970"},
                    {"action": "talk", "character": "Elder Founder", "task": "Hear how this land was reclaimed"},
                    {"action": "reflect", "task": "Write a letter to the land"},
                ],
            ),
            WorldQuest(
                title="Build the Community Bank",
                description="Design and simulate the Greenville Charter Community Bank.",
                learning_objectives=[
                    "Understand how community banks differ from commercial banks",
                    "Learn about the GRN-USD stablecoin and land-backed currency",
                    "Explore cooperative economics and community wealth",
                ],
                subjects=["economics", "mathematics", "finance"],
                age_groups=["high-school", "adult"],
                completion_reward="100 GRN-USD + banker NFT credential",
                difficulty="hard",
                estimated_minutes=60,
                steps=[
                    {"action": "analyze", "task": "Compare charter bank vs. commercial bank structures"},
                    {"action": "calculate", "task": "Model the GRN-USD reserve requirements"},
                    {"action": "simulate", "task": "Run a 10-year community bank simulation"},
                    {"action": "vote", "location": "community_commons", "task": "Submit your bank design for community vote"},
                ],
            ),
        ]

    def get_guide(self) -> dict[str, Any]:
        return {
            "name": "Elder Founder",
            "description": "One of the original founders of the Greenville Sovereign community, keeper of the land's story",
            "voice_style": "southern_dignified",
            "age": 68,
            "languages": ["en"],
            "specialty": "land rights, cooperative economics, community governance, African American history",
        }

    def get_knowledge_base(self) -> list[WorldDocument]:
        return [
            WorldDocument(
                title="The 53-Acre Monadic Site — Greenville, NC",
                content=(
                    "The 53-acre Monadic anchor site in Greenville, North Carolina represents the "
                    "first buildable terrain in the Sovereign Civilization Plan. This land is "
                    "owned and controlled by the community, with each parcel represented as a "
                    "capsule NFT on the Internet Computer (ICP). The GRN-USD stablecoin "
                    "is backed 1:1 by USD, collateralized by the land value and bank deposits, "
                    "and redeemable at the Greenville Charter Community Bank (in development)."
                ),
                source="Monadic Archive Sovereign Ecosystem Specification",
                doc_type="primary",
                date="2026",
            ),
            WorldDocument(
                title="Black Land Loss in America 1910–1970",
                content=(
                    "At its peak in 1910, African Americans owned approximately 16 million acres "
                    "of farmland in the United States. By 1997, that number had fallen to "
                    "approximately 2 million acres — a loss of over 85%. This decline was driven "
                    "by a combination of discriminatory lending, tax sales, fraud, intimidation, "
                    "and outright theft. Heirs' property laws made land particularly vulnerable. "
                    "Projects like the Greenville Sovereign community represent a deliberate effort "
                    "to rebuild Black land ownership as a foundation for economic sovereignty."
                ),
                source="USDA and ProPublica research composite",
                doc_type="secondary",
            ),
        ]
