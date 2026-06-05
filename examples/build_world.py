#!/usr/bin/env python3
"""Build World: full Omniverse world creation pipeline with quests, RAG, and USD."""
import os
from loguru import logger

from nim.client import NIMClient
from nim.llm import LLMClient
from nim.embedding import EmbeddingClient
from nim.rag import RAGPipeline
from omniverse.kit_bridge import OmniverseKitBridge, WORLD_THEMES
from omniverse.worlds import EastFlatbushWorld, GreenvilleWorld
from education.world_quest import WorldQuestEngine

API_KEY = os.environ.get("NVIDIA_API_KEY", "")


def demo_list_themes():
    print("\n=== Available World Themes ===")
    for theme_id, theme in WORLD_THEMES.items():
        print(f"  {theme_id}: {theme['name']} ({theme['period']})")


def demo_build_usd_scene():
    print("\n=== Build USD Scene (East Flatbush) ===")
    bridge = OmniverseKitBridge()
    scene = bridge.create_scene(
        theme="brooklyn-90s",
        name="458 E 94th St",
    )
    print(f"Scene created: {scene.scene_id}")
    print(f"  Theme: {scene.theme}")
    print(f"  Status: {scene.status}")
    print(f"  USD path: {scene.usd_stage_path}")

    # Add a landmark
    bridge.add_asset(scene, asset_id="basketball_court", asset_type="prop")
    bridge.add_asset(scene, asset_id="record_shop", asset_type="building")
    print(f"  Added 2 assets to scene")


def demo_world_welcome():
    print("\n=== World Welcome Messages ===")
    for WorldClass in [EastFlatbushWorld, GreenvilleWorld]:
        world = WorldClass()
        print(f"\n-- {world.NAME} --")
        msg = world.welcome_message(language="en")
        print(msg[:400] + "..." if len(msg) > 400 else msg)


def demo_world_quests():
    print("\n=== World Quests ===")
    world = EastFlatbushWorld()
    quests = world.get_quests()
    for q in quests:
        print(f"  [{q.difficulty.upper()}] {q.title} — {q.reward_ebtk} EBTK ({q.duration_min} min)")
        print(f"    {q.description[:100]}")


def demo_rag_world_knowledge():
    print("\n=== RAG over World Knowledge Base ===")
    nim = NIMClient(api_key=API_KEY)
    embed = EmbeddingClient(nim)
    llm = LLMClient(nim)
    rag = RAGPipeline(embed, llm)

    # Load knowledge from both worlds
    for WorldClass in [EastFlatbushWorld, GreenvilleWorld]:
        world = WorldClass()
        kb = world.get_knowledge_base()
        rag.ingest([
            __import__('nim.rag', fromlist=['Document']).Document(
                id=f"{world.THEME}_{i}",
                title=doc.title,
                content=doc.content,
                source=doc.source,
                world=world.THEME,
            )
            for i, doc in enumerate(kb)
        ])

    print(f"Indexed {rag.document_count()} documents across {rag.worlds_indexed()} worlds")

    # Query with world filter
    result = rag.query(
        "What is the history of 458 E 94th Street?",
        top_k=3,
        world_filter="brooklyn-90s",
    )
    print(f"\nQuery: 'What is the history of 458 E 94th Street?'")
    print(f"Answer: {result.answer[:500]}")
    print(f"Sources: {[d.title for d in result.sources]}")


def demo_quest_engine():
    print("\n=== Quest Engine ===")
    nim = NIMClient(api_key=API_KEY)
    llm = LLMClient(nim)
    engine = WorldQuestEngine(llm)

    learner_id = "learner_brooklyn_001"
    available = engine.get_available_quests(learner_id, world="east_flatbush")
    print(f"Available quests for {learner_id}: {len(available)}")

    if available:
        quest = available[0]
        print(f"Starting: '{quest.title}'")
        progress = engine.start_quest(learner_id, quest.quest_id)
        print(f"  State: {progress.state}")

        # Complete all steps
        for step in range(quest.steps.__len__() if hasattr(quest.steps, '__len__') else 3):
            progress = engine.advance_step(learner_id, quest.quest_id, step_score=90.0)
            print(f"  Step {step+1}: {progress.state}")
            if progress.state == "completed":
                break

        if progress.state == "completed":
            reward = engine.claim_reward(learner_id, quest.quest_id)
            print(f"  Reward claimed: {reward.ebtk} EBTK, badge: {reward.badge}")


if __name__ == "__main__":
    demo_list_themes()
    demo_build_usd_scene()
    demo_world_welcome()
    demo_world_quests()
    demo_rag_world_knowledge()
    demo_quest_engine()
    print("\n=== World build demo complete! ===")
