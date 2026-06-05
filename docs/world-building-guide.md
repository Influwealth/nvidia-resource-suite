# World Building Guide

Create your own educational world in 4 steps.

## 1. Define the World Class

```python
from omniverse.worlds.base_world import BaseWorld, SpawnPoint, WorldQuest, WorldDocument

class MyWorld(BaseWorld):
    THEME = "my-theme"        # matches WORLD_THEMES key in kit_bridge.py
    NAME = "My World"
    PERIOD = "Ancient / Modern / Future"
    SUBJECTS = ["history", "mathematics", "ecology"]
    LANGUAGES = ["en", "es"]

    def get_spawn_points(self) -> list[SpawnPoint]:
        return [
            SpawnPoint(location_id="entrance", name="Main Gate",
                       position=(0, 0, 0), description="Welcome to My World."),
        ]

    def get_quests(self) -> list[WorldQuest]:
        return [
            WorldQuest(
                quest_id="my_first_quest",
                title="The First Challenge",
                description="Learn something amazing.",
                difficulty="easy",
                duration_min=20,
                reward_ebtk=25,
                subjects=["history"],
                language="en",
            )
        ]

    def get_guide(self) -> dict:
        return {"name": "Sage", "description": "An ancient scholar.", "voice_style": "calm"}

    def get_knowledge_base(self) -> list[WorldDocument]:
        return [
            WorldDocument(
                doc_id="kb_001",
                title="The Origins",
                content="Long ago, this world was...",
                source="Community Archive",
            )
        ]
```

## 2. Add the World Theme

In `omniverse/kit_bridge.py`, add to `WORLD_THEMES`:

```python
"my-theme": {
    "name": "My World",
    "period": "Ancient",
    "biome": "forest",
    "sky": "dawn",
    "ambient_color": (0.9, 0.85, 0.7),
    "ground_material": "grass",
    "landmark": "the_great_tree",
    "population_density": "village",
    "sounds": ["birds", "wind", "distant_drums"],
}
```

## 3. Register in the API

In `api_server.py`, add to `_get_worlds()`:

```python
from omniverse.worlds.my_world import MyWorld
_worlds["my_world"] = MyWorld()
```

## 4. Add to MCP Server

The MCP server automatically picks up any world registered in `_worlds`.
Your world’s quests, knowledge base, and welcome message are immediately
available as Claude tools.

## Knowledge Base Tips

- Documents are embedded with `nvidia/nv-embed-v1` (4096 dimensions)
- RAG retrieval uses cosine similarity
- Two-stage reranking with `nvidia/nv-rerankqa-mistral-4b-v3`
- Keep documents under 1000 tokens for best retrieval
- Use `world` field to filter retrieval to your world

## Oral History Preservation

For recording community oral histories:

```python
from nemo.asr import NeMoASR

asr = NeMoASR()
result = asr.transcribe_oral_history(
    audio_path="elder_interview.wav",
    speaker_name="Elder Johnson",
    community="Greenville",
    language="en",
)
print(result["transcript"])
print(result["preservation_note"])
```

Transcripts can be added directly to your world’s RAG knowledge base.
