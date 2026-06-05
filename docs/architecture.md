# System Architecture

## Overview

```
nvidia-resource-suite/
├── nim/              # NVIDIA NIM API (LLM, embed, vision, audio, RAG, guardrails)
├── omniverse/        # Omniverse (Kit, Nucleus, Farm, Replicator, Audio2Face, PhysX, worlds)
├── triton/           # Triton Inference Server (client, model repository, ensemble)
├── tensorrt/         # TensorRT (optimizer, quantizer — FP16/INT8/INT4/FP8)
├── nemo/             # NeMo Framework (LLM, ASR, TTS, Guardrails, RAG pipeline)
├── isaac/            # Isaac Sim + Mission Dispatch
├── warp/             # NVIDIA Warp GPU physics kernels
├── modulus/          # Modulus digital twins and PDE solvers
├── earth2/           # Earth-2 climate (FourCastNet, CorrDiff)
├── rapids/           # RAPIDS cuDF, cuML, cuGraph
├── education/        # Curriculum engine, quest system, K-12/AI/economics
├── mcp_server/       # MCP server (32 tools for Claude and other LLMs)
├── api_server.py     # FastAPI REST API (port 7760)
├── config.py         # Environment configuration
└── monad_node.py     # MONAD v3.7 NODE_DELTA registration
```

## Module Dependencies

```
                         NIM API (cloud)
                              │
              ┌────────────┼────────────┐
              │             │            │
           nim/llm      nim/embed     nim/audio
              │             │            │
              └──────┬──────┼──────┘
                       │      │
                    nim/rag  education/curriculum
                       │      │
                    omniverse/worlds  ←── warp/physics
                                         modulus/digital_twin
                                         earth2/weather
                                         rapids/accelerator
```

## MONAD Pentagon Integration

This service runs as **NODE_DELTA** in the MONAD v3.7 Pentagon mesh:

| Node | Service | Port | Role |
|------|---------|------|------|
| NODE_ALPHA | DeepFlex | 8000 | Sovereign Control Hub |
| NODE_BETA | nvq-mesh-fabric | 9400 | Quantum Bridge |
| NODE_GAMMA | qre-agent-platform | 4943 | ICP Gateway |
| **NODE_DELTA** | **nvidia-resource-suite** | **7760** | **AI/6G Substrate** |
| NODE_EPSILON | vr-meeting-room | 7791 | Digital Twin/VR |

On startup, `monad_node.py` registers NODE_DELTA with NODE_ALPHA (DeepFlex) using:
- SAP headers: `x-sap-node-id`, `x-sap-version: 3.7`, `x-sap-trace-id`
- Heartbeat: every 30 seconds
- NIM model list and world themes reported as capabilities

## Graceful Degradation

Every module follows the pattern:

```python
try:
    import real_library
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    # Mock implementation
```

This ensures the platform runs anywhere: Raspberry Pi, browser WASM, Akash GPU node, or H100 cluster.

## Token Economy

| Token | Layer | Value | Backing |
|-------|-------|-------|---------|
| EBTK | Tier 1 | learning reward | community engagement |
| GVC | Tier 2 | 1 GVC = 1 USD | ICP canister |
| GRN-USD | Tier 2 | 1:1 USD | 53-acre land + bank deposits |
| WBST | Tier 3 | service unit | compute consumed |

Bridge rate: 100 EBTK = 1 GVC
