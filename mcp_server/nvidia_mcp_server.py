"""Full MCP server exposing all NVIDIA Resource Suite tools to Claude and other LLMs."""
from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK not installed: pip install mcp")

from config import get_config
from nim.client import NIMClient, AVAILABLE_MODELS
from nim.llm import LLMClient
from nim.embedding import EmbeddingClient
from nim.rag import RAGPipeline, Document
from nim.guardrails import GuardrailsClient
from omniverse.worlds import EastFlatbushWorld, GreenvilleWorld, SilkRoadWorld, HarlemRenaissanceWorld
from education.curriculum import CurriculumEngine, LearnerProfile, AgeGroup
from education.world_quest import WorldQuestEngine
from education.curricula import K12STEMCurriculum, AILiteracyCurriculum, SovereignEconomicsCurriculum
from warp.physics import WarpPhysics
from earth2.weather import Earth2WeatherEngine
from earth2.climate_sim import FourCastNetClient, CorrDiffClient
from rapids.accelerator import RAPIDSAccelerator
from modulus.digital_twin import ModulusDigitalTwin, greenville_thermal_twin

# Initialize global singletons
_config = get_config()
_nim = NIMClient(
    api_key=os.environ.get("NVIDIA_API_KEY", ""),
    base_url=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
)
_llm = LLMClient(_nim)
_embed = EmbeddingClient(_nim)
_rag = RAGPipeline(_embed, _llm)
_guardrails = GuardrailsClient(_nim)
_warp = WarpPhysics()
_rapids = RAPIDSAccelerator()
_fcn = FourCastNetClient()
_cdf_client = CorrDiffClient()
_weather = Earth2WeatherEngine(_fcn, _cdf_client)
_curriculum = CurriculumEngine(_llm)
_quest_engine = WorldQuestEngine(_llm)
_k12 = K12STEMCurriculum()
_ai_lit = AILiteracyCurriculum()
_econ = SovereignEconomicsCurriculum()
_worlds = {
    "east_flatbush": EastFlatbushWorld(),
    "greenville": GreenvilleWorld(),
    "silk_road": SilkRoadWorld(),
    "harlem_renaissance": HarlemRenaissanceWorld(),
}


def _json(obj) -> str:
    return json.dumps(obj, default=str, indent=2)


# ------------------------------------------------------------------
# Tool handlers
# ------------------------------------------------------------------

async def handle_tool_call(tool_name: str, arguments: dict) -> str:
    try:
        return await _dispatch(tool_name, arguments)
    except Exception as e:
        logger.error(f"Tool {tool_name} error: {e}")
        return _json({"error": str(e), "tool": tool_name})


async def _dispatch(tool_name: str, args: dict) -> str:
    # ---- NIM / LLM tools ----
    if tool_name == "nim_chat":
        response = _llm.tutor(
            question=args["question"],
            world_context=args.get("world_context", ""),
            age_group=args.get("age_group", "adult"),
            language=args.get("language", "en"),
        )
        return response

    if tool_name == "nim_embed":
        vectors = _embed.embed_batch(args["texts"])
        return _json({"embeddings": vectors, "dim": len(vectors[0]) if vectors else 0})

    if tool_name == "nim_generate_quiz":
        quiz = _llm.generate_quiz(
            topic=args["topic"],
            num_questions=args.get("num_questions", 5),
            difficulty=args.get("difficulty", "medium"),
        )
        return quiz

    if tool_name == "nim_explain_concept":
        explanation = _llm.explain_concept(
            concept=args["concept"],
            domain=args.get("domain", "general"),
            examples=args.get("examples", True),
        )
        return explanation

    if tool_name == "nim_translate":
        translation = _llm.translate(
            text=args["text"],
            target_language=args["target_language"],
        )
        return translation

    if tool_name == "nim_code_assist":
        code = _llm.code_assist(
            task=args["task"],
            language=args.get("language", "python"),
        )
        return code

    if tool_name == "nim_list_models":
        return _json(AVAILABLE_MODELS)

    # ---- RAG tools ----
    if tool_name == "rag_ingest":
        docs = [Document(id=d["id"], title=d["title"], content=d["content"], source=d.get("source", ""), world=d.get("world", "")) for d in args["documents"]]
        _rag.ingest(docs)
        return _json({"ingested": len(docs), "total": _rag.document_count()})

    if tool_name == "rag_query":
        result = _rag.query(
            question=args["question"],
            top_k=args.get("top_k", 5),
            world_filter=args.get("world_filter"),
        )
        return _json({"answer": result.answer, "sources": [d.title for d in result.sources], "confidence": result.confidence})

    if tool_name == "rag_rerank_query":
        result = _rag.rerank_and_query(
            question=args["question"],
            retrieve_k=args.get("retrieve_k", 20),
            rerank_top_k=args.get("rerank_top_k", 5),
        )
        return _json({"answer": result.answer, "reranked": result.reranked, "citations": result.citations})

    # ---- World tools ----
    if tool_name == "world_list":
        return _json(list(_worlds.keys()))

    if tool_name == "world_welcome":
        world = _worlds.get(args["world_name"])
        if not world:
            return _json({"error": f"World '{args['world_name']}' not found"})
        msg = world.welcome_message(language=args.get("language", "en"))
        return msg

    if tool_name == "world_describe_location":
        world = _worlds.get(args["world_name"])
        if not world:
            return _json({"error": "World not found"})
        desc = world.describe_location(location_id=args["location_id"])
        return desc

    if tool_name == "world_get_quests":
        world = _worlds.get(args["world_name"])
        if not world:
            return _json({"error": "World not found"})
        quests = world.get_quests()
        return _json([{"id": q.quest_id, "title": q.title, "difficulty": q.difficulty, "reward_ebtk": q.reward_ebtk} for q in quests])

    if tool_name == "world_knowledge_base":
        world = _worlds.get(args["world_name"])
        if not world:
            return _json({"error": "World not found"})
        kb = world.get_knowledge_base()
        return _json([{"title": d.title, "content": d.content[:500]} for d in kb])

    # ---- Physics / Warp tools ----
    if tool_name == "warp_projectile":
        trajectory = _warp.simulate_projectile(
            initial_position=tuple(args.get("initial_position", [0, 0, 0])),
            initial_velocity=tuple(args.get("initial_velocity", [10, 20, 0])),
            gravity=args.get("gravity", -9.81),
        )
        return _json({"trajectory": trajectory[:20], "total_points": len(trajectory)})

    if tool_name == "warp_status":
        return _json(_warp.status())

    if tool_name == "warp_collision":
        result = _warp.rigid_body_collision(
            body_a_pos=tuple(args["body_a_pos"]),
            body_b_pos=tuple(args["body_b_pos"]),
            body_a_vel=tuple(args.get("body_a_vel", [5, 0, 0])),
            body_b_vel=tuple(args.get("body_b_vel", [-3, 0, 0])),
            mass_a=args.get("mass_a", 1.0),
            mass_b=args.get("mass_b", 1.0),
        )
        return _json(result)

    # ---- Earth-2 / Weather tools ----
    if tool_name == "weather_forecast":
        fc = _weather.forecast(
            location=args["location"],
            lat=args["lat"],
            lon=args["lon"],
            lead_hours=args.get("lead_hours", 72),
        )
        return _json({
            "location": fc.location,
            "alerts": fc.alerts,
            "daily": fc.daily_summary[:3],
            "model": fc.model,
        })

    if tool_name == "climate_greenville":
        scenario = args.get("scenario", "SSP2-4.5")
        result = _weather.greenville_climate_impact(scenario)
        return _json(result)

    if tool_name == "fourcastnet_forecast":
        fc = _fcn.forecast(
            lat=args["lat"],
            lon=args["lon"],
            lead_hours=args.get("lead_hours", 72),
        )
        return _json(fc)

    # ---- RAPIDS tools ----
    if tool_name == "rapids_status":
        return _json(_rapids.status())

    if tool_name == "rapids_benchmark":
        result = _rapids.benchmark_speedup(n_rows=args.get("n_rows", 100_000))
        return _json(result)

    # ---- Education / Curriculum tools ----
    if tool_name == "curriculum_generate_lesson":
        profile = LearnerProfile(
            age_group=AgeGroup(args.get("age_group", "g5_8")),
            language=args.get("language", "en"),
            world=args.get("world", "east_flatbush"),
        )
        lesson = _curriculum.generate_lesson(profile, subject=args["subject"], topic=args["topic"])
        return _json({
            "title": lesson.title,
            "objectives": lesson.objectives,
            "activities": lesson.activities,
            "ebtk_reward": lesson.ebtk_reward,
            "nvidia_tech": lesson.nvidia_tech_used,
        })

    if tool_name == "curriculum_k12_lessons":
        lessons = _k12.get_lessons(args.get("grade_band", "g5_8"))
        return _json(lessons)

    if tool_name == "curriculum_ai_literacy":
        modules = _ai_lit.by_age_group(args.get("age_group", "g9_12"))
        return _json(modules)

    if tool_name == "curriculum_sovereign_economics":
        token = args.get("token")
        if token:
            return _json(_econ.token_explainer(token))
        return _json(_econ.full_curriculum())

    # ---- Quest tools ----
    if tool_name == "quest_list":
        quests = _quest_engine.get_available_quests(
            learner_id=args.get("learner_id", "anon"),
            world=args.get("world"),
        )
        return _json([{"id": q.quest_id, "title": q.title, "world": q.world, "difficulty": q.difficulty, "ebtk": q.reward.ebtk} for q in quests])

    if tool_name == "quest_start":
        progress = _quest_engine.start_quest(args["learner_id"], args["quest_id"])
        return _json({"quest_id": progress.quest_id, "state": progress.state, "total_steps": progress.total_steps})

    if tool_name == "quest_advance":
        progress = _quest_engine.advance_step(args["learner_id"], args["quest_id"], args.get("step_score", 100))
        return _json({"state": progress.state, "current_step": progress.current_step, "score": progress.score})

    if tool_name == "quest_certificate":
        cert = _quest_engine.generate_certificate(args["learner_id"], args["quest_id"])
        return _json(cert)

    # ---- System tools ----
    if tool_name == "system_status":
        return _json(_config.availability_report())

    if tool_name == "guardrails_check":
        result = _guardrails.check_student_input(args["text"], world=args.get("world", "general"))
        return _json({"passed": result.passed, "flags": result.flags})

    return _json({"error": f"Unknown tool: {tool_name}"})


# ------------------------------------------------------------------
# MCP server definition
# ------------------------------------------------------------------

TOOLS: list[dict] = [
    {"name": "nim_chat", "description": "Chat with NIM LLM as educational tutor", "params": {"question": "str", "world_context": "str?", "age_group": "str?", "language": "str?"}},
    {"name": "nim_embed", "description": "Embed texts via NIM embedding model (4096-dim)", "params": {"texts": "list[str]"}},
    {"name": "nim_generate_quiz", "description": "Generate a quiz on any topic", "params": {"topic": "str", "num_questions": "int?", "difficulty": "str?"}},
    {"name": "nim_explain_concept", "description": "Explain any concept with examples", "params": {"concept": "str", "domain": "str?"}},
    {"name": "nim_translate", "description": "Translate text to any supported language", "params": {"text": "str", "target_language": "str"}},
    {"name": "nim_code_assist", "description": "AI code help powered by CodeLlama-70b", "params": {"task": "str", "language": "str?"}},
    {"name": "nim_list_models", "description": "List all available NIM models", "params": {}},
    {"name": "rag_ingest", "description": "Add documents to RAG knowledge base", "params": {"documents": "list[{id,title,content,source?,world?}]"}},
    {"name": "rag_query", "description": "Query RAG pipeline with citations", "params": {"question": "str", "top_k": "int?", "world_filter": "str?"}},
    {"name": "rag_rerank_query", "description": "Two-stage rerank RAG query", "params": {"question": "str", "retrieve_k": "int?", "rerank_top_k": "int?"}},
    {"name": "world_list", "description": "List available educational worlds", "params": {}},
    {"name": "world_welcome", "description": "Get world welcome message", "params": {"world_name": "str", "language": "str?"}},
    {"name": "world_describe_location", "description": "Describe a specific location in a world", "params": {"world_name": "str", "location_id": "str"}},
    {"name": "world_get_quests", "description": "Get available quests for a world", "params": {"world_name": "str"}},
    {"name": "world_knowledge_base", "description": "Get knowledge base documents for a world", "params": {"world_name": "str"}},
    {"name": "warp_projectile", "description": "Simulate projectile motion (educational physics)", "params": {"initial_position": "list?", "initial_velocity": "list?", "gravity": "float?"}},
    {"name": "warp_collision", "description": "Simulate elastic collision between two bodies", "params": {"body_a_pos": "list", "body_b_pos": "list", "body_a_vel": "list?", "body_b_vel": "list?", "mass_a": "float?", "mass_b": "float?"}},
    {"name": "warp_status", "description": "Check NVIDIA Warp availability and device", "params": {}},
    {"name": "weather_forecast", "description": "72-hour weather forecast via FourCastNet", "params": {"location": "str", "lat": "float", "lon": "float", "lead_hours": "int?"}},
    {"name": "climate_greenville", "description": "Climate impact projection for Greenville NC site", "params": {"scenario": "str?"}},
    {"name": "fourcastnet_forecast", "description": "Raw FourCastNet global atmospheric forecast", "params": {"lat": "float", "lon": "float", "lead_hours": "int?"}},
    {"name": "rapids_status", "description": "Check RAPIDS GPU acceleration availability", "params": {}},
    {"name": "rapids_benchmark", "description": "Benchmark GPU vs CPU speedup", "params": {"n_rows": "int?"}},
    {"name": "curriculum_generate_lesson", "description": "Generate adaptive lesson plan", "params": {"subject": "str", "topic": "str", "age_group": "str?", "language": "str?", "world": "str?"}},
    {"name": "curriculum_k12_lessons", "description": "Get K-12 STEM lessons for a grade band", "params": {"grade_band": "str?"}},
    {"name": "curriculum_ai_literacy", "description": "Get AI literacy curriculum modules", "params": {"age_group": "str?"}},
    {"name": "curriculum_sovereign_economics", "description": "Sovereign economics curriculum (GVC/GRN-USD/EBTK)", "params": {"token": "str?"}},
    {"name": "quest_list", "description": "List available quests for a learner", "params": {"learner_id": "str?", "world": "str?"}},
    {"name": "quest_start", "description": "Start a quest for a learner", "params": {"learner_id": "str", "quest_id": "str"}},
    {"name": "quest_advance", "description": "Advance quest by one step", "params": {"learner_id": "str", "quest_id": "str", "step_score": "float?"}},
    {"name": "quest_certificate", "description": "Generate completion certificate", "params": {"learner_id": "str", "quest_id": "str"}},
    {"name": "system_status", "description": "Full system availability report", "params": {}},
    {"name": "guardrails_check", "description": "Check text against educational content policy", "params": {"text": "str", "world": "str?"}},
]


async def run_mcp_server():
    if not MCP_AVAILABLE:
        raise ImportError("Install MCP SDK: pip install mcp")

    server = Server("nvidia-resource-suite")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema={
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in t.get("params", {}).keys()},
                },
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        result = await handle_tool_call(name, arguments)
        return [TextContent(type="text", text=result)]

    logger.info(f"NVIDIA Resource Suite MCP Server starting ({len(TOOLS)} tools)")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="nvidia-resource-suite",
                server_version="2.0.0",
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_mcp_server())
