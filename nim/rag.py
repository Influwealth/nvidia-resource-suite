"""
NVIDIA NIM — RAG (Retrieval-Augmented Generation) Pipeline

Full RAG pipeline for educational content:
  - Ingest curriculum, oral histories, academic sources
  - Semantic search over world knowledge
  - Grounded AI tutoring with citations
  - Multi-world knowledge base management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .client import NIMClient
from .embedding import EmbeddingClient

log = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    title: str
    content: str
    source: str
    world: str = ""
    author: str = ""
    date: str = ""
    doc_type: str = "article"
    embedding: list[float] = field(default_factory=list)


@dataclass
class RAGResult:
    answer: str
    sources: list[Document]
    confidence: float
    query: str


class RAGPipeline:
    """Full RAG pipeline: ingest → embed → retrieve → generate."""

    SYSTEM_PROMPT = """
You are a knowledgeable educator with access to a curated knowledge base.
Answer the student's question using ONLY the provided context.
Always cite your sources. If the context doesn't contain the answer, say so clearly
rather than making something up.
Be encouraging, clear, and age-appropriate.
"""

    def __init__(
        self,
        nim_client: NIMClient | None = None,
        embedding_client: EmbeddingClient | None = None,
        vector_store: Any | None = None,
    ):
        self.nim = nim_client or NIMClient()
        self.embed = embedding_client or EmbeddingClient(self.nim)
        self._documents: list[Document] = []
        self._vector_store = vector_store

    def ingest(
        self,
        documents: list[Document],
        batch_size: int = 16,
    ) -> int:
        """Ingest documents into the knowledge base. Returns count ingested."""
        texts = [f"{doc.title}\n{doc.content}" for doc in documents]
        embeddings = self.embed.embed_batch(texts, input_type="passage", batch_size=batch_size)
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
            self._documents.append(doc)
        log.info("Ingested %d documents into RAG knowledge base", len(documents))
        return len(documents)

    def ingest_text(
        self,
        title: str,
        content: str,
        source: str,
        world: str = "",
        doc_type: str = "article",
    ) -> Document:
        """Convenience method: create and ingest a single document."""
        import uuid
        doc = Document(
            id=str(uuid.uuid4()),
            title=title,
            content=content,
            source=source,
            world=world,
            doc_type=doc_type,
        )
        self.ingest([doc])
        return doc

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        world_filter: str = "",
    ) -> list[Document]:
        """Retrieve the most relevant documents for a query."""
        corpus = [
            {**doc.__dict__, "description": f"{doc.title} {doc.content}"}
            for doc in self._documents
            if not world_filter or doc.world == world_filter
        ]
        if not corpus:
            return []
        results = self.embed.find_most_relevant(query, corpus, text_key="description", top_k=top_k)
        return [Document(**{k: v for k, v in r.items() if k != "_relevance_score" and k != "description"}) for r in results]

    def query(
        self,
        question: str,
        world: str = "",
        top_k: int = 5,
        model: str = "meta/llama-3.1-70b-instruct",
        age_group: str = "middle-school",
    ) -> RAGResult:
        """Full RAG query: retrieve + generate grounded answer with citations."""
        relevant_docs = self.retrieve(question, top_k=top_k, world_filter=world)

        if not relevant_docs:
            return RAGResult(
                answer="I don't have information about that in my knowledge base yet. Ask your teacher for more resources!",
                sources=[],
                confidence=0.0,
                query=question,
            )

        context_parts = [
            f"[Source {i+1}: {doc.title} ({doc.source})]\n{doc.content[:500]}"
            for i, doc in enumerate(relevant_docs)
        ]
        context = "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        answer = self.nim.chat(messages, model=model)

        return RAGResult(
            answer=answer,
            sources=relevant_docs,
            confidence=min(1.0, len(relevant_docs) / top_k),
            query=question,
        )

    def rerank_and_query(
        self,
        question: str,
        world: str = "",
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
    ) -> RAGResult:
        """Two-stage RAG: retrieve many → rerank → generate (higher quality)."""
        candidates = self.retrieve(question, top_k=top_k_retrieve, world_filter=world)
        if not candidates:
            return RAGResult(answer="No relevant content found.", sources=[], confidence=0.0, query=question)

        passages = [f"{doc.title}: {doc.content[:300]}" for doc in candidates]
        reranked = self.nim.rerank(question, passages, top_n=top_k_rerank)

        top_docs = [candidates[r["index"]] for r in reranked]
        context = "\n\n".join(
            f"[Source: {doc.title}]\n{doc.content[:500]}" for doc in top_docs
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        answer = self.nim.chat(messages)
        return RAGResult(answer=answer, sources=top_docs, confidence=0.9, query=question)

    def document_count(self) -> int:
        return len(self._documents)

    def worlds_indexed(self) -> list[str]:
        return list({doc.world for doc in self._documents if doc.world})
