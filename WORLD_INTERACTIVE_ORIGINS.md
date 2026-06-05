# World Interactive Origins — Platform Plan

> **Note**: This document is a scaffold. The full plan from the Influwealth shared drive
> needs to be merged in here. Please share the document text and it will be incorporated.

## What is World Interactive Origins?

World Interactive Origins is an **AI-powered, GPU-accelerated educational platform**
where students explore history, culture, science, and social justice through
immersive 3D interactive worlds — for free.

Powered by NVIDIA Omniverse for real-time 3D rendering and NVIDIA NIM for AI tutoring,
it meets every student where they are: web browser, mobile, VR headset, or
low-bandwidth connection.

---

## The Problem We're Solving

1. **Geography of education is destiny** — A student in East Flatbush has less access
   to quality education than a student in Westchester. We refuse to accept that.

2. **History is not alive in textbooks** — Reading about the Silk Road is not the same
   as walking through a 3D reconstruction of a 10th-century bazaar and talking to
   an AI merchant who can answer any question.

3. **AI tutoring is locked behind paywalls** — The best personalized learning tools
   cost money. We are making them free.

---

## Platform Architecture

```
Student Device (Web / Mobile / VR / Low-bandwidth)
         ↓ HTTPS
World Interactive Origins Frontend (React/Next.js)
         ↓
DeepFlex Supervisor (port 8000) — routes requests
    ├── NVIDIA Resource Suite (port 7760)
    │       ├── NIM LLM Tutor (meta/llama-3.1-70b)
    │       ├── NIM Embeddings (curriculum matching)
    │       └── Omniverse Renderer (world scenes)
    ├── WealthBridge OS (port 8001) — grant/funding management
    └── Argus Prime (port 7700) — infrastructure ops
```

---

## World Themes (Current)

| World | Period | Subjects | Status |
|-------|--------|---------|--------|
| Ancient Silk Road | 100 BCE – 1450 CE | History, Economics, Geography | In development |
| Harlem Renaissance | 1920s–1930s | History, Art, Literature | In development |
| Brooklyn 1990s | 1990–2000 | Social Studies, Music, Community | Planned |
| The Great Migration | 1910–1970 | History, Social Justice, Geography | Planned |
| Pre-Columbian Americas | Pre-1492 | History, Science, Archaeology | Planned |
| East Flatbush Origins | 1960s–present | Local History, Community | **Priority** |

---

## East Flatbush Origins — Priority World

This world tells the story of the East Flatbush community in Brooklyn, NY —
its origins, its people, its culture, and its future.

**Why this first?**
- It's our home. We build what we know.
- Students in East Flatbush deserve to see their story told with dignity and power.
- This world connects directly to the Roblox and Unity game layer
  (see `roblox-wealthbridge-east-flatbush` and `eastflatbush-90s-unity`).

**Content needed from shared drive:**
- [ ] Community partner organizations list
- [ ] Oral history recordings to inform AI tutor dialogue
- [ ] Historical photos for 3D reference
- [ ] Curriculum review notes from community educators
- [ ] Accessibility requirements from disability advocates

---

## Education Model: Free and Accessible

### Access Tiers
| Tier | Access | Cost |
|------|--------|------|
| Open | Full world access, AI tutor (limited) | Free forever |
| Community | Full AI tutor, all worlds, progress tracking | Free with school/org account |
| Educator | Curriculum tools, class management | Free for verified educators |
| Sovereign Partner | White-label, custom worlds | WealthBridge OS agreement |

### No Paywalls. Ever.
The GPU compute cost is covered by:
1. Sovereign infrastructure (Akash Network decentralized compute)
2. NVIDIA Inception Program grants
3. WealthBridge OS business layer revenue
4. Community and institutional partnerships

---

## NVIDIA Integration

### Why NVIDIA?
- **Omniverse** is the best platform for photorealistic 3D world building
- **NIM** gives us enterprise-grade LLM inference with predictable costs
- **CUDA** lets us run everything on commodity GPU hardware (no proprietary lock-in)
- **NVIDIA Inception** provides startup support for education tech

### NVIDIA Tools Used
| Tool | Purpose |
|------|--------|
| NVIDIA NIM (LLM) | AI tutoring via Llama 3.1, Nemotron |
| NVIDIA NIM (Embedding) | Curriculum content matching, semantic search |
| NVIDIA NIM (Vision) | Analyzing student-generated artifacts |
| NVIDIA Omniverse Kit | 3D world scene building |
| NVIDIA Omniverse Nucleus | 3D asset storage and streaming |
| NVIDIA Omniverse Farm | GPU rendering of world scenes |
| NVIDIA CloudXR | VR/XR streaming for immersive access |
| NVIDIA CUDA | Local GPU inference for offline/edge |

---

## Content from Shared Drive

> **TODO**: Merge the following sections from the Influwealth shared drive:
>
> - [ ] Full World Interactive Origins vision document
> - [ ] Community partner agreements template
> - [ ] Curriculum development roadmap (Q3–Q4 2026)
> - [ ] Grant application drafts (NVIDIA Inception, NSF STEM)
> - [ ] Design mockups and wireframes
> - [ ] Accessibility audit results
> - [ ] Pilot school partner list
>
> Share the document or paste the content and this file will be updated immediately.

---

## Getting Involved

- **Educators**: Contact education@influwealth.com to become a curriculum contributor
- **Developers**: See `docs/SETUP.md` to run the platform locally
- **Community partners**: See `docs/PARTNERSHIPS.md` (coming soon)
- **NVIDIA**: We are applying for the NVIDIA Inception Program — contact bd@influwealth.com

---

## Development Roadmap

### Phase 1 (Current — Q3 2026)
- [x] NVIDIA NIM client integration
- [x] GPU job scheduler
- [x] MCP server for Claude integration
- [x] Omniverse bridge scaffold
- [x] Curriculum framework (Silk Road, Harlem Renaissance)
- [ ] East Flatbush Origins world — community content gathering
- [ ] FastAPI integration with DeepFlex Supervisor

### Phase 2 (Q4 2026)
- [ ] Full Omniverse world builds (5 worlds)
- [ ] Student progress tracking (privacy-first)
- [ ] Educator dashboard
- [ ] Pilot with 3 East Flatbush schools
- [ ] NVIDIA CloudXR VR streaming

### Phase 3 (2027)
- [ ] 20 worlds
- [ ] Multilingual AI tutoring (8+ languages)
- [ ] Community world-building tools
- [ ] Mobile app (iOS/Android)
- [ ] Roblox / Unity cross-platform token bridge
