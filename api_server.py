"""
NVIDIA Resource Suite — FastAPI HTTP Server
Port: 7760
SAP Node ID: nvidia-resource-suite

HTTP API for GPU scheduling, NIM inference, Omniverse, and education platform.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config
from nim.client import NIMClient
from scheduler import NvidiaScheduler, JobType

app = FastAPI(
    title="NVIDIA Resource Suite",
    version="1.0.0",
    description="GPU scheduling, NIM inference, and Omniverse integration for sovereign digital worlds",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

nim = NIMClient()
scheduler = NvidiaScheduler()


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    model: str = NIMClient.DEFAULT_CHAT_MODEL
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024


class EmbedRequest(BaseModel):
    texts: list[str]
    model: str = NIMClient.DEFAULT_EMBED_MODEL
    input_type: str = "query"


class VisionRequest(BaseModel):
    message: str
    image_url: str
    model: str = NIMClient.DEFAULT_VISION_MODEL


class JobRequest(BaseModel):
    task: str
    model: str = ""
    job_type: str = "inference"
    gpu_count: int = 1
    priority: int = 5


class TutorRequest(BaseModel):
    student_message: str
    world_context: str
    grade_level: str = "middle"


# ---------------------------------------------------------------------------
# SAP header helper
# ---------------------------------------------------------------------------

def sap_headers(trace_id: str) -> dict[str, str]:
    return {
        "x-sap-node-id": config.sap_node_id,
        "x-sap-trace-id": trace_id,
        "x-sap-version": "1.0",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "node": config.sap_node_id,
        "port": config.port,
        "nim_available": config.is_nim_available(),
        "gpu_pool": scheduler.list_resources()["total_gpus"],
    }


@app.post("/nim/chat")
async def nim_chat(req: ChatRequest, request: Request) -> dict[str, Any]:
    trace_id = request.headers.get("x-sap-trace-id", str(uuid.uuid4()))
    if not config.is_nim_available():
        raise HTTPException(status_code=503, detail="NVIDIA_API_KEY not configured")
    try:
        resp = nim.chat(req.message, req.model, req.system_prompt, req.temperature, req.max_tokens)
        return {"content": resp.content, "model": resp.model, "usage": resp.usage, "trace_id": trace_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/nim/embed")
async def nim_embed(req: EmbedRequest, request: Request) -> dict[str, Any]:
    trace_id = request.headers.get("x-sap-trace-id", str(uuid.uuid4()))
    if not config.is_nim_available():
        raise HTTPException(status_code=503, detail="NVIDIA_API_KEY not configured")
    try:
        vectors = nim.embed(req.texts, req.model, req.input_type)
        return {"embeddings": vectors, "count": len(vectors), "trace_id": trace_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/nim/vision")
async def nim_vision(req: VisionRequest, request: Request) -> dict[str, Any]:
    trace_id = request.headers.get("x-sap-trace-id", str(uuid.uuid4()))
    if not config.is_nim_available():
        raise HTTPException(status_code=503, detail="NVIDIA_API_KEY not configured")
    try:
        resp = nim.vision_chat(req.message, req.image_url, req.model)
        return {"content": resp.content, "model": resp.model, "trace_id": trace_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/nim/models")
async def list_models() -> dict[str, Any]:
    return {"models": nim.list_models()}


@app.post("/gpu/job")
async def submit_job(req: JobRequest, request: Request) -> dict[str, Any]:
    trace_id = request.headers.get("x-sap-trace-id", str(uuid.uuid4()))
    try:
        job_type = JobType(req.job_type)
    except ValueError:
        job_type = JobType.INFERENCE
    job = scheduler.submit(req.task, req.model, job_type, req.gpu_count, req.priority)
    return {"job_id": job.job_id, "status": job.status, "trace_id": trace_id}


@app.get("/gpu/job/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "task": job.task,
        "status": job.status,
        "job_type": job.job_type,
        "priority": job.priority,
        "submitted_at": job.submitted_at,
    }


@app.get("/gpu/resources")
async def gpu_resources() -> dict[str, Any]:
    return scheduler.list_resources()


@app.post("/education/tutor")
async def education_tutor(req: TutorRequest, request: Request) -> dict[str, Any]:
    trace_id = request.headers.get("x-sap-trace-id", str(uuid.uuid4()))
    if not config.is_nim_available():
        raise HTTPException(status_code=503, detail="NVIDIA_API_KEY not configured")
    system_prompt = f"""You are an AI tutor in World Interactive Origins.
World: {req.world_context}. Grade: {req.grade_level}.
Use Socratic method. Be encouraging and accurate. 2-4 sentences max."""
    try:
        resp = nim.chat(req.student_message, system_prompt=system_prompt, temperature=0.8, max_tokens=256)
        return {"response": resp.content, "world": req.world_context, "trace_id": trace_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    issues = config.validate()
    for issue in issues:
        print(f"WARNING: {issue}")
    uvicorn.run(app, host="0.0.0.0", port=config.port)
