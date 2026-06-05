# Setup Guide — NVIDIA Resource Suite

## Prerequisites

- Python 3.11+
- pip or uv
- NVIDIA API Key (free at [build.nvidia.com](https://build.nvidia.com/explore/discover))
- (Optional) NVIDIA GPU with CUDA 12.x for local inference
- (Optional) NVIDIA Omniverse for 3D world rendering

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/influwealth/nvidia-resource-suite
cd nvidia-resource-suite
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your NVIDIA_API_KEY
```

### 3. Start the HTTP API server

```bash
uvicorn api_server:app --port 7760 --reload
```

Open http://localhost:7760/docs for interactive API docs.

### 4. Add to Claude Code as an MCP server

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "nvidia-resource-suite": {
      "command": "python",
      "args": ["-m", "mcp_server.nvidia_mcp_server"],
      "cwd": "/absolute/path/to/nvidia-resource-suite",
      "env": {
        "NVIDIA_API_KEY": "nvapi-YOUR_KEY_HERE"
      }
    }
  }
}
```

Then in Claude Code you can use tools like:
- `nvidia_chat` — LLM inference
- `nvidia_embed` — Text embeddings
- `nvidia_gpu_status` — Job queue status
- `nvidia_education_tutor` — World Interactive Origins tutoring

### 5. Docker (recommended for production)

```bash
docker-compose up
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NVIDIA_API_KEY` | YES | — | NVIDIA NIM API key from build.nvidia.com |
| `NVIDIA_NIM_BASE_URL` | No | https://integrate.api.nvidia.com/v1 | NIM API endpoint |
| `OMNIVERSE_NUCLEUS_URL` | No | omniverse://localhost | Omniverse Nucleus URL |
| `OMNIVERSE_FARM_URL` | No | http://localhost:8222 | Omniverse Farm URL |
| `NVIDIA_SUITE_PORT` | No | 7760 | HTTP server port |
| `DEEPFLEX_BASE_URL` | No | http://localhost:8000 | DeepFlex Supervisor |

## Verification

```bash
# Check health
curl http://localhost:7760/health

# List available NIM models
curl http://localhost:7760/nim/models

# Test NIM inference
curl -X POST http://localhost:7760/nim/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What was the most traded good on the Silk Road?"}'
```
