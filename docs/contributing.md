# Contributing

Welcome! This is an open, global project. All skill levels welcome.

## Ways to Contribute

### Add a New World
Create an educational world for your culture, era, or community.
See [world-building-guide.md](world-building-guide.md).

### Add a Curriculum Module
Add a new lesson to `education/curricula/` in the style of `k12_stem.py`.
Include: learning objectives, NVIDIA tool used, world context tie-in, EBTK reward.

### Improve NIM Prompts
Better prompts = better education. The system prompts are in:
- `nim/llm.py` — `EDUCATION_SYSTEM_PROMPT`
- `nim/llm.py` — `WORLD_BUILDING_SYSTEM_PROMPT`

### Add Language Support
Translate welcome messages, quest descriptions, and curriculum content.
Current languages: English, Spanish, French, Arabic, Mandarin, Haitian Creole, Persian, Turkish.

### Report Issues
Open an issue on GitHub. Include:
- Which module failed
- Whether GPU is available
- Full error traceback

## Development Setup

```bash
git clone https://github.com/influwealth/nvidia-resource-suite
cd nvidia-resource-suite
pip install -r requirements.txt
cp .env.example .env  # add NVIDIA_API_KEY
pytest tests/           # run test suite
```

## Code Style

- Python 3.11+
- Type hints on all public functions
- Graceful degradation: every module must work without GPU
- No secrets in code — always load from environment variables
- Loguru for logging (`from loguru import logger`)

## Security

- Never commit `.env` or any file containing API keys
- `NVIDIA_API_KEY` and all secrets must be environment variables only
- See SECURITY.md for the full policy

## License

Apache 2.0. Free to use, modify, and distribute.
See [LICENSE](../LICENSE).

## Community

- Platform: Synapz (ICP-hosted, algorithm-free, cannot be deplatformed)
- Philosophy: Sovereign technology for all communities
- Founding vision: The Monadic Archive — 458 E 94th St, East Flatbush &amp; 53 acres, Greenville NC
