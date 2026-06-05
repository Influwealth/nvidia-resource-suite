"""AI literacy curriculum: what is AI, bias detection, prompt engineering, AI ethics."""
from __future__ import annotations


class AILiteracyCurriculum:
    """AI/ML literacy for all ages — from 'what is AI?' to bias mitigation and ethics."""

    MODULES = [
        {
            "id": "ai_001",
            "title": "What Is Artificial Intelligence?",
            "age_group": "all",
            "duration_min": 30,
            "concepts": ["AI definition", "machine learning", "pattern recognition", "data"],
            "activity": "Ask your NIM AI tutor 5 questions and notice HOW it answers.",
            "reflection": "What can AI do better than humans? What can humans do better than AI?",
            "ebtk_reward": 20,
            "nvidia_tool": "NIM LLM (meta/llama-3.1-70b)",
        },
        {
            "id": "ai_002",
            "title": "How Does a Neural Network Learn?",
            "age_group": "g5_8",
            "duration_min": 45,
            "concepts": ["neurons", "weights", "training data", "loss function", "gradient descent"],
            "activity": "Visualize training a 3-layer network on handwritten digits. Watch accuracy improve.",
            "reflection": "How is a neural network's training like a student studying for an exam?",
            "ebtk_reward": 35,
            "nvidia_tool": "cuML + NIM LLM",
        },
        {
            "id": "ai_003",
            "title": "Bias in AI Systems",
            "age_group": "g9_12",
            "duration_min": 60,
            "concepts": ["algorithmic bias", "training data bias", "fairness metrics", "disparate impact"],
            "activity": "Analyze a hiring algorithm dataset. Find where racial and gender bias appears.",
            "reflection": "How can we audit AI systems used by banks, courts, and schools?",
            "ebtk_reward": 50,
            "nvidia_tool": "NeMo Guardrails + NIM LLM + RAPIDS cuML",
            "resources": [
                "Obermeyer et al. (2019) - Dissecting racial bias in an algorithm",
                "Buolamwini & Gebru (2018) - Gender Shades",
            ],
        },
        {
            "id": "ai_004",
            "title": "Prompt Engineering Fundamentals",
            "age_group": "g9_12",
            "duration_min": 45,
            "concepts": ["prompts", "context", "system messages", "temperature", "few-shot learning"],
            "activity": "Write 5 different prompts for the same task. Compare quality. Learn chain-of-thought.",
            "reflection": "Why do the words you use with AI matter so much?",
            "ebtk_reward": 40,
            "nvidia_tool": "NIM LLM",
            "examples": [
                {"bad": "Write about history", "good": "In 3 bullet points, summarize how the Great Migration shaped Harlem between 1910-1930."},
                {"bad": "Help me code", "good": "Write a Python function that takes a list of crop yields and returns the top 3. Include docstring and type hints."},
            ],
        },
        {
            "id": "ai_005",
            "title": "AI Ethics and Sovereignty",
            "age_group": "adult",
            "duration_min": 90,
            "concepts": ["AI governance", "data sovereignty", "algorithmic accountability", "ICP decentralization"],
            "activity": "Design an AI policy for the Greenville sovereign community. What data can be used? Who decides?",
            "reflection": "How do communities protect their data sovereignty from corporate AI extraction?",
            "ebtk_reward": 100,
            "nvidia_tool": "NIM LLM + NeMo Guardrails + ICP Canister",
        },
        {
            "id": "ai_006",
            "title": "Building Your First AI Application",
            "age_group": "g9_12",
            "duration_min": 120,
            "concepts": ["APIs", "RAG", "embedding", "vector search", "full-stack AI"],
            "activity": "Build a simple RAG chatbot that answers questions about Harlem Renaissance history.",
            "reflection": "What AI applications could benefit your community?",
            "ebtk_reward": 150,
            "nvidia_tool": "NIM LLM + NIM Embedding + NeMo RAG + ChromaDB",
            "code_template": "examples/education_demo.py",
        },
    ]

    def get_module(self, module_id: str) -> dict | None:
        return next((m for m in self.MODULES if m["id"] == module_id), None)

    def by_age_group(self, age_group: str) -> list[dict]:
        return [m for m in self.MODULES if m["age_group"] in (age_group, "all")]

    def learning_path(self) -> list[str]:
        return [m["id"] for m in sorted(self.MODULES, key=lambda m: m.get("ebtk_reward", 0))]
