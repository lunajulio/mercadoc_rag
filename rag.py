"""Orquestador del pipeline RAG completo de Mercado Central 24h.

Capa de recuperación (retrieval) + capa de generación, siguiendo las 5 fases:
  1. Pregunta -> embedding (mismo modelo que los documentos)
  2. Búsqueda semántica en Pinecone (con filtro opcional de metadatos)
  3. Filtrado por metadados (categoría)
  4. Reranking con cross-encoder sobre los candidatos
  5. Ensamblaje del contexto -> LLM (Gemini) -> respuesta con fuentes citadas

Incluye control de alucinación por umbral de confianza y un fallback que
intenta ofrecer el contacto del área responsable cuando no hay respuesta.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pipeline import llm, vector_store
from pipeline.config import settings
from pipeline.embeddings import EmbeddingModel
from pipeline.reranker import Reranker

_AREA_CONTACT_QUERIES = {
    "RH": "contacto correo teléfono responsable de recursos humanos",
    "Legal": "contacto correo teléfono área legal o jurídica",
    "Financiero": "contacto correo teléfono área financiera",
    "Operaciones": "contacto correo teléfono responsable de operaciones",
}


@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    has_answer: bool = True
    best_score: float = 0.0


def _find_area_contact(category_hint: str | None) -> dict | None:
    """Busca datos de contacto de un área en el propio corpus indexado, antes de sugerirla."""
    query_text = _AREA_CONTACT_QUERIES.get(category_hint or "", "contacto responsable del área")
    embedder = EmbeddingModel()
    candidates = vector_store.query(
        query_text, top_k=3, embedder=embedder, category_filter=[category_hint] if category_hint else None
    )
    if candidates and candidates[0][1] >= 0.45:
        document, score = candidates[0]
        return {"text": document.page_content, "score": score, "metadata": document.metadata}
    return None


def answer_question(
    question: str,
    category_filter: list[str] | None = None,
) -> RAGAnswer:
    embedder = EmbeddingModel()
    reranker = Reranker()

    # 1-3. Embedding de la pregunta + búsqueda semántica + filtro de metadatos
    candidates = vector_store.query(
        question, top_k=settings.top_k_retrieve, embedder=embedder, category_filter=category_filter
    )

    if not candidates:
        return _fallback_answer(category_filter)

    # 4. Reranking sobre los candidatos
    top_chunks = reranker.rerank(question, candidates, top_n=settings.top_k_rerank)
    best_score = top_chunks[0].metadata["rerank_score"] if top_chunks else 0.0

    # Control de alucinación: si ni el mejor candidato supera el umbral, no se arriesga respuesta.
    if best_score < settings.min_rerank_score:
        return _fallback_answer(category_filter)

    # 5. Ensamblaje del contexto + generación
    generated_text = llm.generate_answer(question, top_chunks)

    sources = [
        {
            "filename": c.metadata["filename"],
            "page": c.metadata["page"],
            "category": c.metadata["category"],
            "web_view_link": c.metadata.get("web_view_link", ""),
            "relevance": round(c.metadata["rerank_score"], 3),
        }
        for c in top_chunks
    ]
    return RAGAnswer(answer=generated_text, sources=sources, has_answer=True, best_score=best_score)


def _fallback_answer(category_filter: list[str] | None) -> RAGAnswer:
    category_hint = category_filter[0] if category_filter and len(category_filter) == 1 else None
    contact = _find_area_contact(category_hint)

    message = "No encontré esta información en los documentos disponibles."
    if contact:
        message += (
            f"\n\nEsto parece un tema para el área de **{contact['metadata']['category']}**. "
            f"Encontré este contacto en la documentación:\n\n> {contact['text']}"
        )
    else:
        message += " Esta pregunta parece estar fuera del alcance actual de la base de conocimiento."

    return RAGAnswer(answer=message, sources=[], has_answer=False, best_score=0.0)


def log_feedback(question: str, result: RAGAnswer, rating: str | None) -> None:
    """Registra la interacción para monitoreo de calidad (tasa de sin-respuesta, feedback negativo)."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "has_answer": result.has_answer,
        "best_score": result.best_score,
        "rating": rating,
    }
    with open(settings.feedback_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
