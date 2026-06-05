"""
NVIDIA MCP Server — Model Context Protocol server for NVIDIA resources.

Exposes NVIDIA NIM, GPU scheduling, and Omniverse tools as MCP tools
that can be used by Claude, Claude Code, and other MCP clients.

Run: python -m mcp_server.nvidia_mcp_server
Protocol: MCP over stdio (JSON-RPC 2.0)

Add to Claude Code ~/.claude/settings.json:
{
  "mcpServers": {
    "nvidia-resource-suite": {
      "command": "python",
      "args": ["-m", "mcp_server.nvidia_mcp_server"],
      "cwd": "/path/to/nvidia-resource-suite",
      "env": {"NVIDIA_API_KEY": "nvapi-YOUR_KEY"}
    }
  }
}
"""
from __future__ import annotations

import json
import sys
import uuid
from typing import Any

from config import config
from nim.client import NIMClient
from scheduler import NvidiaScheduler, JobStatus

nim = NIMClient()
scheduler = NvidiaScheduler()


# ---------------------------------------------------------------------------
# MCP Protocol helpers
# ---------------------------------------------------------------------------

def _send(msg: dict[str, Any]) -> None:
    print(json.dumps(msg), flush=True)


def _error(id_: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def _result(id_: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "nvidia_chat",
        "description": "Generate text using NVIDIA NIM LLMs (Llama 3.1, Nemotron, Mixtral, Phi-3). Use for educational content, world-building narratives, tutoring dialogues.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The user message or prompt"},
                "model": {
                    "type": "string",
                    "description": "NIM model to use",
                    "enum": ["meta/llama-3.1-70b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct", "microsoft/phi-3-medium-128k-instruct", "mistralai/mixtral-8x7b-instruct-v0.1"],
                    "default": "meta/llama-3.1-70b-instruct",
                },
                "system_prompt": {"type": "string", "description": "Optional system prompt"},
                "temperature": {"type": "number", "default": 0.7},
                "max_tokens": {"type": "integer", "default": 1024},
            },
            "required": ["message"],
        },
    },
    {
        "name": "nvidia_embed",
        "description": "Generate text embeddings using NVIDIA NIM embedding models. Use for semantic search, RAG, and curriculum matching.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "texts": {"type": "array", "items": {"type": "string"}, "description": "List of texts to embed"},
                "model": {"type": "string", "default": "nvidia/nv-embedqa-e5-v5"},
                "input_type": {"type": "string", "enum": ["query", "passage"], "default": "query"},
            },
            "required": ["texts"],
        },
    },
    {
        "name": "nvidia_vision",
        "description": "Analyze images with NVIDIA NIM vision models. Use for analyzing 3D scene screenshots, educational diagrams, historical artifacts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Question or instruction about the image"},
                "image_url": {"type": "string", "description": "URL of the image to analyze"},
                "model": {"type": "string", "default": "microsoft/phi-3-vision-128k-instruct"},
            },
            "required": ["message", "image_url"],
        },
    },
    {
        "name": "nvidia_gpu_status",
        "description": "Get the current GPU job queue status and resource availability.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "nvidia_submit_job",
        "description": "Submit a GPU compute job to the NVIDIA job scheduler. Use for rendering, inference, or simulation workloads.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Description of the GPU task"},
                "model": {"type": "string", "description": "Model or workload name"},
                "gpu_count": {"type": "integer", "default": 1},
                "priority": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["task"],
        },
    },
    {
        "name": "nvidia_list_models",
        "description": "List all available NVIDIA NIM models by category.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "nvidia_education_tutor",
        "description": "Run an AI tutoring session for the World Interactive Origins platform. Returns a Socratic dialogue response for a student in a specific world/topic.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "student_message": {"type": "string"},
                "world_context": {
                    "type": "string",
                    "description": "The educational world/topic (e.g. 'ancient-silk-road', 'harlem-renaissance', 'brooklyn-1990s')",
                },
                "grade_level": {"type": "string", "enum": ["elementary", "middle", "high", "adult"], "default": "middle"},
            },
            "required": ["student_message", "world_context"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _execute_tool(name: str, args: dict[str, Any]) -> Any:
    if name == "nvidia_chat":
        resp = nim.chat(
            message=args["message"],
            model=args.get("model", NIMClient.DEFAULT_CHAT_MODEL),
            system_prompt=args.get("system_prompt"),
            temperature=args.get("temperature", 0.7),
            max_tokens=args.get("max_tokens", 1024),
        )
        return {"content": resp.content, "model": resp.model, "usage": resp.usage}

    if name == "nvidia_embed":
        vectors = nim.embed(
            texts=args["texts"],
            model=args.get("model", NIMClient.DEFAULT_EMBED_MODEL),
            input_type=args.get("input_type", "query"),
        )
        return {"embeddings": vectors, "count": len(vectors), "dimension": len(vectors[0]) if vectors else 0}

    if name == "nvidia_vision":
        resp = nim.vision_chat(
            message=args["message"],
            image_url=args["image_url"],
            model=args.get("model", NIMClient.DEFAULT_VISION_MODEL),
        )
        return {"content": resp.content, "model": resp.model}

    if name == "nvidia_gpu_status":
        return scheduler.list_resources()

    if name == "nvidia_submit_job":
        job = scheduler.submit(
            task=args["task"],
            model=args.get("model", ""),
            gpu_count=args.get("gpu_count", 1),
            priority=args.get("priority", 5),
        )
        return {"job_id": job.job_id, "status": job.status, "task": job.task}

    if name == "nvidia_list_models":
        return nim.list_models()

    if name == "nvidia_education_tutor":
        system_prompt = f"""You are an AI tutor inside the World Interactive Origins educational platform.
The student is currently exploring the '{args["world_context"]}' world.
Grade level: {args.get("grade_level", "middle")}.
Use the Socratic method — ask guiding questions, never just give answers.
Be encouraging, curious, and historically accurate.
Respond in 2-4 sentences."""
        resp = nim.chat(
            message=args["student_message"],
            system_prompt=system_prompt,
            temperature=0.8,
            max_tokens=256,
        )
        return {
            "response": resp.content,
            "world": args["world_context"],
            "grade_level": args.get("grade_level", "middle"),
        }

    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# MCP request handler
# ---------------------------------------------------------------------------

def handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return _result(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "nvidia-resource-suite", "version": "1.0.0"},
        })

    if method == "tools/list":
        return _result(req_id, {"tools": TOOLS})

    if method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        try:
            result = _execute_tool(tool_name, tool_args)
            return _result(req_id, {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                "isError": False,
            })
        except Exception as exc:
            return _result(req_id, {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            })

    if method == "notifications/initialized":
        return None  # No response for notifications

    if method == "ping":
        return _result(req_id, {})

    return _error(req_id, -32601, f"Method not found: {method}")


def main() -> None:
    """Run MCP server over stdio."""
    issues = config.validate()
    if issues:
        for issue in issues:
            print(f"[nvidia-mcp] WARNING: {issue}", file=sys.stderr)

    print("[nvidia-mcp] Server started. Waiting for MCP requests...", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            response = handle_request(req)
            if response is not None:
                _send(response)
        except json.JSONDecodeError as e:
            _send(_error(None, -32700, f"Parse error: {e}"))
        except Exception as e:
            _send(_error(None, -32603, f"Internal error: {e}"))


if __name__ == "__main__":
    main()
