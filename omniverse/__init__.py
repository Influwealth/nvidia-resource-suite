"""
nvidia-resource-suite.omniverse

NVIDIA Omniverse integration layer.
Supports: Kit, Nucleus, Farm, Replicator, Audio2Face, PhysX, DeepSearch.

All calls gracefully degrade when Omniverse Kit is not installed.
"""

from .kit_bridge import OmniverseKitBridge, OmniverseScene
from .nucleus import NucleusClient
from .farm import OmniverseFarm, RenderJob
from .replicator import ReplicatorClient
from .audio2face import Audio2FaceClient
from .physx import PhysXClient

__all__ = [
    "OmniverseKitBridge",
    "OmniverseScene",
    "NucleusClient",
    "OmniverseFarm",
    "RenderJob",
    "ReplicatorClient",
    "Audio2FaceClient",
    "PhysXClient",
]
