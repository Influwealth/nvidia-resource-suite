# NVIDIA Resource Suite

Full NVIDIA integration suite for the Influwealth Sovereign Automation System and
digital world-building platform. Supports free and accessible education through
AI-powered interactive environments.

## Vision

We are building **World Interactive Origins** — an AI-powered, GPU-accelerated
educational platform where students explore history, science, and culture through
immersive 3D worlds rendered in real-time by NVIDIA Omniverse.

> "The best education is the one that meets every student where they are — for free."

NVIDIA's GPU infrastructure, NIM inference microservices, and Omniverse world-building
SDK give us the compute backbone to make this vision real.

## Architecture

```
NVIDIA Resource Suite (port 7760)
├── NIM Client          — AI inference via api.nvidia.com (LLM, vision, embedding)
├── GPU Scheduler       — Job queue for CUDA/rendering workloads
├── MCP Server          — Model Context Protocol server exposing NVIDIA tools
├── Omniverse Bridge    — USD scene building, Kit SDK, Nucleus integration
├── Education Platform  — Curriculum management, interactive world delivery
└── FastAPI HTTP API    — SAP-compliant HTTP interface (port 7760)
```

## SAP Integration
- **SAP Node ID**: `nvidia-resource-suite`
- **Port**: 7760
- **Managed by**: DeepFlex Supervisor (port 8000)
- **Feeds**: TurboQuant Core, qre-agent-platform, sovereign-mesh capsules

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set NVIDIA API key (get from build.nvidia.com)
export NVIDIA_API_KEY=nvapi-XXXX

# 3. Start the HTTP API server
uvicorn api_server:app --port 7760 --reload

# 4. Start the MCP server (for Claude integration)
python -m mcp_server.nvidia_mcp_server

# 5. (Optional) Start with Docker
docker-compose up
```

## NVIDIA API Key

Get your free API key at: https://build.nvidia.com/explore/discover

Set it as:
```bash
export NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE
# or add to .env file (never commit .env)
```

## Services

| Service | Description | Key |
|---------|-------------|-----|
| NIM (NVIDIA Inference Microservices) | LLM, vision, embedding inference | NVIDIA_API_KEY |
| Omniverse Nucleus | 3D asset server | OMNIVERSE_NUCLEUS_URL |
| Omniverse Farm | GPU render farm | OMNIVERSE_FARM_URL |

## Supported NIM Models

### Language Models
- `meta/llama-3.1-70b-instruct` — General instruction following
- `nvidia/llama-3.1-nemotron-70b-instruct` — NVIDIA-tuned Llama
- `microsoft/phi-3-medium-128k-instruct` — Long-context model
- `mistralai/mixtral-8x7b-instruct-v0.1` — Fast MoE model

### Embedding Models
- `nvidia/nv-embedqa-e5-v5` — Q&A optimized embeddings
- `snowflake/arctic-embed-l` — Large embedding model

### Vision Models
- `microsoft/phi-3-vision-128k-instruct` — Vision + language
- `nvidia/neva-22b` — NVIDIA vision-language model

### Specialized
- `nvidia/rerank-qa-mistral-4b` — Reranking for RAG
- `nvcr.io/nvidia/nemotron-nano-vl-8b-v1` — Compact vision-language

## Education Mission

The World Interactive Origins platform provides:

1. **Free access** — No paywalls. GPU costs covered by sovereign infrastructure.
2. **Interactive 3D worlds** — NVIDIA Omniverse scenes for every major historical period
3. **AI tutoring** — NIM-powered Socratic dialogue in every world
4. **Universal design** — Accessible on web, mobile, VR, and low-bandwidth connections
5. **Open curriculum** — Community-contributed lesson plans

## Documentation

- [Setup Guide](docs/SETUP.md)
- [API Reference](docs/API.md)
- [Omniverse Integration](docs/OMNIVERSE.md)
- [Education Platform](docs/EDUCATION.md)
- [MCP Server](docs/MCP_SERVER.md)
- [World Interactive Origins Plan](WORLD_INTERACTIVE_ORIGINS.md)

## License

MIT — Free to use, fork, and build upon.
