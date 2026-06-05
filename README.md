# NVIDIA Resource Suite — Public Global Edition

> Open-source · Educational · Sovereign Compute  
> Powered by the full NVIDIA technology stack.

**nvidia-resource-suite** is a free, open-source platform that brings the entire NVIDIA AI and
simulation stack to educators, world-builders, students, researchers, and sovereign communities
everywhere on Earth.

---

## What This Is

This repo is the foundation for:

- **World Interactive Origins** — AI-powered educational digital civilizations
- **Omniverse World-Building** — Photorealistic 3D worlds with Universal Scene Description (USD)
- **NIM AI Tutoring** — Free, GPU-accelerated AI for every student on Earth
- **Earth Simulation** — Climate, weather, and physics at planetary scale
- **Sovereign Compute** — Decentralized, community-owned GPU infrastructure
- **Open Learning** — Curricula, guides, and examples for all ages and all nations

---

## Full Technology Stack

| NVIDIA System | Module | Purpose |
|---------------|--------|---------|
| Omniverse Kit / Nucleus / Farm | `omniverse/` | 3D world-building, asset streaming, render dispatch |
| Omniverse Replicator | `omniverse/replicator.py` | Synthetic data generation |
| Omniverse Audio2Face / ACE | `omniverse/audio2face.py` | Avatar animation |
| PhysX | `omniverse/physx.py` | Real-time physics |
| NIM — LLM | `nim/llm.py` | Chat, reasoning, tutoring |
| NIM — Embedding | `nim/embedding.py` | Semantic search, RAG |
| NIM — Vision | `nim/vision.py` | Image understanding |
| NIM — Audio | `nim/audio.py` | Speech-to-text, text-to-speech |
| NIM — RAG | `nim/rag.py` | Retrieval-augmented generation |
| NeMo Guardrails NIM | `nim/guardrails.py` | Safe AI for student interactions |
| Triton Inference Server | `triton/` | Multi-model serving, dynamic batching |
| TensorRT / TensorRT-LLM | `tensorrt/` | Model optimization, FP8/INT4 quantization |
| NeMo Framework | `nemo/` | LLM, ASR, TTS, RAG fine-tuning |
| Isaac Sim / ROS | `isaac/` | Robotics simulation |
| NVIDIA Warp | `warp/` | GPU-accelerated Python physics kernels |
| NVIDIA Modulus | `modulus/` | Physics-ML, PDE solvers, digital twins |
| Earth-2 (CorrDiff / FourCastNet) | `earth2/` | Climate and weather simulation |
| RAPIDS (cuDF / cuML / cuGraph) | `rapids/` | GPU-accelerated data science |
| CUDA Runtime | `cuda/` | Low-level GPU memory and kernel management |

---

## Quick Start

```bash
git clone https://github.com/influwealth/nvidia-resource-suite
cd nvidia-resource-suite
cp .env.example .env
# Edit .env — add your NVIDIA_API_KEY
pip install -r requirements.txt
python examples/hello_nim.py
```

---

## Examples

```bash
# Say hello to a NIM LLM
python examples/hello_nim.py

# Build a 3D world scene
python examples/build_world.py --theme east-flatbush

# Run a climate simulation
python examples/climate_demo.py --region east-coast --days 7

# Launch an interactive education session
python examples/education_demo.py --world silk-road --age-group middle-school

# Start the full API server
uvicorn api_server:app --host 0.0.0.0 --port 7760 --reload
```

---

## World Themes

| World | Period | Anchor Site | Subjects |
|-------|--------|-------------|----------|
| East Flatbush Origins | 1960s–present | 458 E 94th St, Brooklyn | Community, Music, History |
| Greenville Sovereign | Pre-1865–present | 53-acre NC site | Land Rights, Economics, Governance |
| Ancient Silk Road | 100 BCE–1450 CE | Central Asia corridor | Trade, Culture, Science |
| Harlem Renaissance | 1920s–1930s | Harlem, New York | Art, Literature, Civil Rights |
| Great Migration | 1910–1970 | Southern US → Northern cities | History, Social Justice |
| Pre-Columbian Americas | Pre-1492 | Across the Americas | Archaeology, Science, Civilization |

---

## Free Access — Always

GPU-powered education must be free. This platform is:

- **Free** for all students and educators, globally
- **Open-source** — Apache 2.0 license
- **Deployable** on Akash Network decentralized compute
- **Grant-eligible** — NVIDIA Inception Program, NSF STEM, community funds
- **Multilingual** — 8+ languages in active development

---

## Architecture

```
Student / Researcher / Educator
        |
        v
nvidia-resource-suite API (port 7760)
        |
  +-----+-----+--------+----------+--------+
  |           |        |          |        |
 NIM       Omniverse Triton    Earth-2   NeMo
 (LLM/     (3D/USD)  (Serving) (Climate) (Framework)
  Embed/
  Vision)
        |
        v
DeepFlex Supervisor (port 8000) — MONAD NODE_ALPHA
        |
  +-----+-----+
  |           |
Argus       WealthBridge OS
(Infra)     (Business Logic)
```

---

## Documentation

- [Getting Started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Omniverse World-Building Guide](docs/omniverse-guide.md)
- [NIM Integration Guide](docs/nim-guide.md)
- [World-Building Templates](docs/world-building-guide.md)
- [Contributing](docs/contributing.md)

---

## Community

- **Educators**: education@influwealth.com — free accounts, curriculum tools
- **Researchers**: Open issues and PRs welcome
- **Community Partners**: See [WORLD_INTERACTIVE_ORIGINS.md](WORLD_INTERACTIVE_ORIGINS.md)
- **NVIDIA Partners**: bd@influwealth.com

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
