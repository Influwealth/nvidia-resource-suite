"""
omniverse.worlds — World Interactive Origins world builders

Each module defines a world class with:
  - build(): create the full USD scene
  - get_spawn_points(): where students enter the world
  - get_quests(): available learning quests
  - get_guide(): AI guide character config
"""

from .base_world import BaseWorld, SpawnPoint, WorldQuest
from .east_flatbush import EastFlatbushWorld
from .greenville import GreenvilleWorld
from .silk_road import SilkRoadWorld
from .harlem_renaissance import HarlemRenaissanceWorld

__all__ = [
    "BaseWorld",
    "SpawnPoint",
    "WorldQuest",
    "EastFlatbushWorld",
    "GreenvilleWorld",
    "SilkRoadWorld",
    "HarlemRenaissanceWorld",
]
