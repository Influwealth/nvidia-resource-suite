"""NeMo Retriever RAG pipeline: two-stage rerank, citation grounding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class NeMoDocument:
    id: str
    title: str
    content: str
    source: str
    world: str = ""
    doc_type: str = "knowledge"  # knowledge | quest | oral_history | transcript
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class NeMoRAGResult:
    answer: str
    sources: list[NeMoDocument]
    query: str
    confidence: float
    reranked: bool = False
    citations: list[str] = field(default_factory=list)


class NeMoRAGPipeline:
    """Two-stage RAG: embed → retrieve → rerank → generate with citations.

    Uses NeMo Retriever microservices when available, falls back to
    NIM embedding + cosine similarity for open access deployments.
    """

    def __init__(
        self,
        embedding_client=None,
        llm_client=None,
        reranker_client=None,
        retriever_url: str | None = None,
    ):
        self._embed = embedding_client
        self._llm = llm_client
        self._reranker = reranker_client
        self._retriever_url = retriever_url
        self._documents: list[NeMoDocument] = []
        logger.info("NeMo RAG pipeline initialized")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def ingest(self, documents: list[NeMoDocument], batch_size: int = 16):
        if not self._embed:
            for doc in documents:
                doc.embedding = [0.0] * 4096
                self._documents.append(doc)
            logger.warning("No embedding client: stored documents without embeddings")
            return
        texts = [f"{d.title}. {d.content}" for d in documents]
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            embeddings = self._embed.embed_batch(batch)
            for j, doc in enumerate(documents[i: i + batch_size]):
                doc.embedding = embeddings[j]
                self._documents.append(doc)
        logger.info(f"Ingested {len(documents)} documents. Total: {len(self._documents)}")

    def ingest_text(
        self,
        doc_id: str,
        title: str,
        content: str,
        source: str = "",
        world: str = "",
        doc_type: str = "knowledge",
    ):
        doc = NeMoDocument(id=doc_id, title=title, content=content, source=source, world=world, doc_type=doc_type)
        self.ingest([doc])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        world_filter: str | None = None,
    ) -> list[NeMoDocument]:
        if not self._documents:
            return []
        if not self._embed:
            pool = [d for d in self._documents if not world_filter or d.world == world_filter]
            return pool[:top_k]

        query_emb = self._embed.embed_single(query)
        pool = [d for d in self._documents if not world_filter or d.world == world_filter]

        def _cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b + 1e-9)

        scored = [(d, _cosine(query_emb, d.embedding)) for d in pool if d.embedding]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:top_k]]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        world_filter: str | None = None,
    ) -> NeMoRAGResult:
        docs = self.retrieve(question, top_k=top_k, world_filter=world_filter)
        context = "\n\n".join(
            f"[{i+1}] {d.title}\n{d.content[:600]}" for i, d in enumerate(docs)
        )
        prompt = (
            f"Answer the question using only the context below. "
            f"Cite sources as [1], [2], etc.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        if self._llm:
            answer = self._llm.generate(prompt, max_new_tokens=512)
        else:
            answer = f"[No LLM configured] Context retrieved: {context[:200]}"

        citations = [f"[{i+1}] {d.title} — {d.source}" for i, d in enumerate(docs)]
        confidence = min(1.0, len(docs) / max(top_k, 1))
        return NeMoRAGResult(
            answer=answer, sources=docs, query=question,
            confidence=confidence, citations=citations,
        )

    def rerank_and_query(
        self,
        question: str,
        retrieve_k: int = 20,
        rerank_top_k: int = 5,
        world_filter: str | None = None,
    ) -> NeMoRAGResult:
        """Two-stage: retrieve many → rerank → generate from top-k."""
        candidates = self.retrieve(question, top_k=retrieve_k, world_filter=world_filter)
        if self._reranker and candidates:
            try:
                reranked = self._reranker.rerank(
                    query=question,
                    passages=[d.content for d in candidates],
                    top_n=rerank_top_k,
                )
                top_docs = [candidates[r["index"]] for r in reranked[:rerank_top_k]]
            except Exception as e:
                logger.warning(f"Reranker failed: {e}. Using cosine ranking.")
                top_docs = candidates[:rerank_top_k]
        else:
            top_docs = candidates[:rerank_top_k]

        context = "\n\n".join(
            f"[{i+1}] {d.title}\n{d.content[:600]}" for i, d in enumerate(top_docs)
        )
        prompt = (
            f"Answer using only the context. Cite as [1],[2], etc.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        answer = self._llm.generate(prompt, max_new_tokens=512) if self._llm else context[:300]
        citations = [f"[{i+1}] {d.title} — {d.source}" for i, d in enumerate(top_docs)]
        return NeMoRAGResult(
            answer=answer, sources=top_docs, query=question,
            confidence=0.9, reranked=True, citations=citations,
        )

    def document_count(self) -> int:
        return len(self._documents)

    def worlds_indexed(self) -> list[str]:
        return list({d.world for d in self._documents if d.world})
