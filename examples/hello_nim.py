#!/usr/bin/env python3
"""Hello NIM: first NIM LLM call, streaming, embedding, vision, and code assist."""
import os
from pathlib import Path

from loguru import logger

# Ensure NVIDIA_API_KEY is set in environment (never hard-code)
API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not API_KEY:
    logger.warning("NVIDIA_API_KEY not set. Running in mock mode.")

from nim.client import NIMClient
from nim.llm import LLMClient
from nim.embedding import EmbeddingClient


def demo_basic_chat():
    print("\n=== 1. Basic Chat ===")
    client = NIMClient(api_key=API_KEY)
    llm = LLMClient(client)
    response = llm.tutor(
        question="What was the Silk Road and why was it important?",
        world_context="ancient Silk Road trading civilization",
        age_group="g5_8",
    )
    print(response)


def demo_streaming():
    print("\n=== 2. Streaming Chat ===")
    client = NIMClient(api_key=API_KEY)
    llm = LLMClient(client)
    print("Streaming response: ", end="", flush=True)
    for token in llm.stream_tutor(
        question="Tell me about the Harlem Renaissance in 3 sentences.",
        world_context="1920s Harlem, New York",
    ):
        print(token, end="", flush=True)
    print()


def demo_embedding():
    print("\n=== 3. Text Embedding ===")
    client = NIMClient(api_key=API_KEY)
    embed = EmbeddingClient(client)
    texts = [
        "The 53-acre Greenville site is a sovereign community space.",
        "East Flatbush Brooklyn in the 1990s was the center of Caribbean-American culture.",
        "Al-Khwarizmi invented algebra at the House of Wisdom in Baghdad.",
    ]
    vectors = embed.embed_batch(texts)
    print(f"Embedded {len(vectors)} texts. Dimension: {len(vectors[0])}")

    # Find most similar to a query
    query = "community land ownership and sovereignty"
    query_vec = embed.embed_single(query)
    scored = embed.find_most_relevant(query_vec, list(zip(texts, vectors)))
    print(f"Most relevant to '{query}':")
    for text, score in scored[:2]:
        print(f"  [{score:.3f}] {text}")


def demo_quiz_generation():
    print("\n=== 4. Quiz Generation ===")
    client = NIMClient(api_key=API_KEY)
    llm = LLMClient(client)
    quiz = llm.generate_quiz(
        topic="Newton's Laws of Motion",
        num_questions=3,
        difficulty="medium",
        format="multiple_choice",
    )
    print(quiz)


def demo_code_assist():
    print("\n=== 5. Code Assist (CodeLlama-70b) ===")
    client = NIMClient(api_key=API_KEY)
    llm = LLMClient(client)
    code = llm.code_assist(
        task="Write a Python function that calculates compound interest. Include docstring and type hints.",
        language="python",
    )
    print(code)


if __name__ == "__main__":
    demo_basic_chat()
    demo_embedding()
    demo_quiz_generation()
    demo_code_assist()
    # demo_streaming()  # Uncomment to see streaming tokens
    print("\n=== Hello NIM complete! ===")
