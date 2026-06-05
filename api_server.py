"""Expanded FastAPI server: NIM / Omniverse / Triton / Physics / Climate / Education routes."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from config import get_config
from nim.client import NIMClient, AVAILABLE_MODELS
from nim.llm import LLMClient
from nim.embedding import EmbeddingClient
from nim.rag import RAGPipeline, Document
from nim.guardrails import GuardrailsClient
from omniverse.kit_bridge import OmniverseKitBridge, WORLD_THEMES
from omniverse.worlds import EastFlatbushWorld, GreenvilleWorld, SilkRoadWorld, HarlemRenaissanceWorld
from triton.client import TritonClient
from education.curriculum import CurriculumEngine, LearnerProfile, AgeGroup
from education.world_quest import WorldQuestEngine
from warp.physics import WarpPhysics
from earth2.weather import Earth2WeatherEngine
from earth2.climate_sim import FourCastNetClient, CorrDiffClient
from rapids.accelerator import RAPIDSAccelerator
from modulus.digital_twin import ModulusDigitalTwin, greenville_thermal_twin, east_flatbush_building_twin
from monad_node import get_handler as get_node_handler

# ------------------------------------------------------------------
# Lifespan + app setup
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    warnings = cfg.validate()
    for w in warnings:
        logger.warning(f"Config: {w}")
    # Register with MONAD Pentagon
    try:
        handler = get_node_handler()
        await handler.register()
        logger.info("NODE_DELTA registered with MONAD mesh")
    except Exception as e:
        logger.warning(f"MONAD registration skipped: {e}")
    yield
    logger.info("nvidia-resource-suite shutting down")


app = FastAPI(
    title="NVIDIA Resource Suite — Public Global Educational Platform",
    description=(
        "Open-source NVIDIA technology integration for world-building, interactive learning, "
        "Omniverse digital civilizations, physics simulation, and AI education for all ages."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SAP middleware
@app.middleware("http")
async def sap_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["x-sap-node-id"] = "NODE_DELTA"
    response.headers["x-sap-version"] = "3.7"
    return response

# ------------------------------------------------------------------
# Singletons (lazy init)
# ------------------------------------------------------------------

_nim: NIMClient | None = None
_llm: LLMClient | None = None
_embed: EmbeddingClient | None = None
_rag: RAGPipeline | None = None
_guardrails: GuardrailsClient | None = None
_kit: OmniverseKitBridge | None = None
_triton: TritonClient | None = None
_warp: WarpPhysics | None = None
_rapids: RAPIDSAccelerator | None = None
_weather_engine: Earth2WeatherEngine | None = None
_curriculum: CurriculumEngine | None = None
_quest_engine: WorldQuestEngine | None = None
_greenville_twin: ModulusDigitalTwin | None = None
_worlds: dict = {}


def _get_nim() -> NIMClient:
    global _nim
    if _nim is None:
        _nim = NIMClient(
            api_key=os.environ.get("NVIDIA_API_KEY", ""),
            base_url=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )
    return _nim


def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient(_get_nim())
    return _llm


def _get_embed() -> EmbeddingClient:
    global _embed
    if _embed is None:
        _embed = EmbeddingClient(_get_nim())
    return _embed


def _get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline(_get_embed(), _get_llm())
    return _rag


def _get_worlds() -> dict:
    global _worlds
    if not _worlds:
        _worlds = {
            "east_flatbush": EastFlatbushWorld(),
            "greenville": GreenvilleWorld(),
            "silk_road": SilkRoadWorld(),
            "harlem_renaissance": HarlemRenaissanceWorld(),
        }
    return _worlds


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    world_context: str = ""
    age_group: str = "adult"
    language: str = "en"
    model: str | None = None

class EmbedRequest(BaseModel):
    texts: list[str]
    input_type: str = "query"

class RAGIngestRequest(BaseModel):
    documents: list[dict]

class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 5
    world_filter: str | None = None
    use_rerank: bool = False

class QuizRequest(BaseModel):
    topic: str
    num_questions: int = 5
    difficulty: str = "medium"
    format: str = "multiple_choice"

class WorldRequest(BaseModel):
    world_name: str
    language: str = "en"

class ProjectileRequest(BaseModel):
    initial_position: list[float] = [0, 0, 0]
    initial_velocity: list[float] = [10, 20, 0]
    gravity: float = -9.81
    dt: float = 0.05
    max_time: float = 5.0

class WeatherRequest(BaseModel):
    location: str
    lat: float
    lon: float
    lead_hours: int = 72

class LessonRequest(BaseModel):
    subject: str
    topic: str
    age_group: str = "g5_8"
    language: str = "en"
    world: str = "east_flatbush"
    duration_min: int = 45

class QuestRequest(BaseModel):
    learner_id: str
    quest_id: str

class GuardrailsRequest(BaseModel):
    text: str
    world: str = "general"

# ------------------------------------------------------------------
# System routes
# ------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "NVIDIA Resource Suite",
        "version": "2.0.0",
        "edition": "Public Global Educational",
        "monad_node": "NODE_DELTA",
        "worlds": list(_get_worlds().keys()),
        "docs": "/docs",
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "node": "NODE_DELTA"}

@app.get("/status")
async def status():
    cfg = get_config()
    return cfg.availability_report()

@app.get("/models")
async def list_models():
    return {"available_models": AVAILABLE_MODELS}

# ------------------------------------------------------------------
# NIM routes
# ------------------------------------------------------------------

@app.post("/nim/chat")
async def nim_chat(req: ChatRequest):
    llm = _get_llm()
    response = llm.tutor(
        question=req.question,
        world_context=req.world_context,
        age_group=req.age_group,
        language=req.language,
        model=req.model,
    )
    return {"response": response, "model": req.model or "default"}

@app.post("/nim/stream")
async def nim_stream(req: ChatRequest):
    llm = _get_llm()
    def generator():
        for token in llm.stream_tutor(req.question, req.world_context):
            yield token
    return StreamingResponse(generator(), media_type="text/plain")

@app.post("/nim/embed")
async def nim_embed(req: EmbedRequest):
    embed = _get_embed()
    vectors = embed.embed_batch(req.texts, input_type=req.input_type)
    return {"embeddings": vectors, "dim": len(vectors[0]) if vectors else 0, "count": len(vectors)}

@app.post("/nim/quiz")
async def nim_quiz(req: QuizRequest):
    llm = _get_llm()
    quiz = llm.generate_quiz(req.topic, req.num_questions, req.difficulty, req.format)
    return {"quiz": quiz}

@app.post("/nim/explain")
async def nim_explain(concept: str, domain: str = "general"):
    llm = _get_llm()
    return {"explanation": llm.explain_concept(concept, domain)}

@app.post("/nim/code")
async def nim_code(task: str, language: str = "python"):
    llm = _get_llm()
    return {"code": llm.code_assist(task, language)}

@app.post("/nim/translate")
async def nim_translate(text: str, target_language: str):
    llm = _get_llm()
    return {"translation": llm.translate(text, target_language)}

# ------------------------------------------------------------------
# RAG routes
# ------------------------------------------------------------------

@app.post("/rag/ingest")
async def rag_ingest(req: RAGIngestRequest):
    rag = _get_rag()
    docs = [Document(id=d["id"], title=d["title"], content=d["content"],
                     source=d.get("source", ""), world=d.get("world", "")) for d in req.documents]
    rag.ingest(docs)
    return {"ingested": len(docs), "total": rag.document_count(), "worlds": rag.worlds_indexed()}

@app.post("/rag/query")
async def rag_query(req: RAGQueryRequest):
    rag = _get_rag()
    if req.use_rerank:
        result = rag.rerank_and_query(req.question, world_filter=req.world_filter)
    else:
        result = rag.query(req.question, top_k=req.top_k, world_filter=req.world_filter)
    return {
        "answer": result.answer,
        "sources": [d.title for d in result.sources],
        "confidence": result.confidence,
    }

# ------------------------------------------------------------------
# Omniverse / World routes
# ------------------------------------------------------------------

@app.get("/worlds")
async def list_worlds():
    return {"worlds": list(_get_worlds().keys()), "themes": list(WORLD_THEMES.keys())}

@app.post("/worlds/welcome")
async def world_welcome(req: WorldRequest):
    worlds = _get_worlds()
    world = worlds.get(req.world_name)
    if not world:
        raise HTTPException(status_code=404, detail=f"World '{req.world_name}' not found")
    return {"message": world.welcome_message(language=req.language)}

@app.get("/worlds/{world_name}/quests")
async def world_quests(world_name: str):
    worlds = _get_worlds()
    world = worlds.get(world_name)
    if not world:
        raise HTTPException(status_code=404, detail="World not found")
    quests = world.get_quests()
    return {"quests": [{"id": q.quest_id, "title": q.title, "difficulty": q.difficulty, "reward_ebtk": q.reward_ebtk, "duration_min": q.duration_min} for q in quests]}

@app.get("/worlds/{world_name}/knowledge")
async def world_knowledge(world_name: str):
    worlds = _get_worlds()
    world = worlds.get(world_name)
    if not world:
        raise HTTPException(status_code=404, detail="World not found")
    kb = world.get_knowledge_base()
    return {"documents": [{"title": d.title, "content": d.content[:500], "source": d.source} for d in kb]}

@app.post("/omniverse/scene")
async def create_scene(theme: str, name: str):
    global _kit
    if _kit is None:
        _kit = OmniverseKitBridge()
    scene = _kit.create_scene(theme=theme, name=name)
    return {"scene_id": scene.scene_id, "status": scene.status, "usd_path": scene.usd_stage_path}

# ------------------------------------------------------------------
# Physics / Warp routes
# ------------------------------------------------------------------

@app.post("/physics/projectile")
async def physics_projectile(req: ProjectileRequest):
    global _warp
    if _warp is None:
        _warp = WarpPhysics()
    trajectory = _warp.simulate_projectile(
        initial_position=tuple(req.initial_position),
        initial_velocity=tuple(req.initial_velocity),
        gravity=req.gravity,
        dt=req.dt,
        max_time=req.max_time,
    )
    peak = max(trajectory, key=lambda p: p["y"])
    return {"trajectory_points": len(trajectory), "peak": peak, "sample_points": trajectory[:10], "warp_available": _warp.available}

@app.get("/physics/status")
async def physics_status():
    global _warp
    if _warp is None:
        _warp = WarpPhysics()
    return _warp.status()

# ------------------------------------------------------------------
# Earth-2 / Climate routes
# ------------------------------------------------------------------

@app.post("/climate/forecast")
async def climate_forecast(req: WeatherRequest):
    global _weather_engine
    if _weather_engine is None:
        fcn = FourCastNetClient()
        _weather_engine = Earth2WeatherEngine(fcn)
    fc = _weather_engine.forecast(req.location, req.lat, req.lon, req.lead_hours)
    return {"location": fc.location, "daily": fc.daily_summary, "alerts": fc.alerts, "model": fc.model}

@app.get("/climate/greenville")
async def climate_greenville(scenario: str = "SSP2-4.5"):
    global _weather_engine
    if _weather_engine is None:
        _weather_engine = Earth2WeatherEngine()
    return _weather_engine.greenville_climate_impact(scenario)

# ------------------------------------------------------------------
# Education routes
# ------------------------------------------------------------------

@app.post("/education/lesson")
async def generate_lesson(req: LessonRequest):
    global _curriculum
    if _curriculum is None:
        _curriculum = CurriculumEngine(_get_llm())
    profile = LearnerProfile(age_group=AgeGroup(req.age_group), language=req.language, world=req.world)
    lesson = _curriculum.generate_lesson(profile, req.subject, req.topic, req.duration_min)
    return {"title": lesson.title, "objectives": lesson.objectives, "activities": lesson.activities, "ebtk_reward": lesson.ebtk_reward, "nvidia_tech": lesson.nvidia_tech_used}

@app.post("/education/quests/start")
async def start_quest(req: QuestRequest):
    global _quest_engine
    if _quest_engine is None:
        _quest_engine = WorldQuestEngine(_get_llm())
    progress = _quest_engine.start_quest(req.learner_id, req.quest_id)
    return {"quest_id": progress.quest_id, "state": progress.state, "total_steps": progress.total_steps}

@app.post("/education/quests/advance")
async def advance_quest(learner_id: str, quest_id: str, step_score: float = 100.0):
    global _quest_engine
    if _quest_engine is None:
        _quest_engine = WorldQuestEngine(_get_llm())
    progress = _quest_engine.advance_step(learner_id, quest_id, step_score)
    return {"state": progress.state, "current_step": progress.current_step, "score": progress.score}

@app.post("/education/quests/certificate")
async def quest_certificate(req: QuestRequest):
    global _quest_engine
    if _quest_engine is None:
        _quest_engine = WorldQuestEngine(_get_llm())
    return _quest_engine.generate_certificate(req.learner_id, req.quest_id)

# ------------------------------------------------------------------
# Guardrails
# ------------------------------------------------------------------

@app.post("/guardrails/check")
async def guardrails_check(req: GuardrailsRequest):
    global _guardrails
    if _guardrails is None:
        _guardrails = GuardrailsClient(_get_nim())
    result = _guardrails.check_student_input(req.text, world=req.world)
    return {"passed": result.passed, "flags": result.flags, "filtered": result.filtered_output}

# ------------------------------------------------------------------
# Digital Twin routes
# ------------------------------------------------------------------

@app.get("/twin/greenville")
async def greenville_twin_status():
    global _greenville_twin
    if _greenville_twin is None:
        _greenville_twin = greenville_thermal_twin()
    state = _greenville_twin.update_from_sensors({"temperature": 22.5, "heat_flux": 450.0})
    return {
        "twin_id": state.twin_id,
        "confidence": state.confidence,
        "anomalies": state.anomalies,
        "sensor_readings": state.sensor_readings,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", "7760"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
