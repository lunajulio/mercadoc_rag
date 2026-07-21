"""Reclasificación (reranking): reordena los candidatos de la búsqueda vectorial
por relevancia real frente a la pregunta completa, usando un cross-encoder.

Es más lento que la búsqueda vectorial pero mucho más preciso, por eso se
aplica solo sobre los top_k_retrieve candidatos iniciales (no sobre todo el
índice), quedándose con los top_k_rerank mejores.
"""
from __future__ import annotations

import math

import streamlit as st
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document

from pipeline.config import settings


@st.cache_resource(show_spinner="Cargando modelo de reranking...")
def _load_reranker(model_name: str) -> HuggingFaceCrossEncoder:
    return HuggingFaceCrossEncoder(model_name=model_name)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model = _load_reranker(model_name or settings.reranker_model_name)

    def rerank(self, query: str, candidates: list[tuple[Document, float]], top_n: int) -> list[Document]:
        """Reordena candidatos por relevancia real frente a la pregunta.

        El cross-encoder devuelve logits sin acotar (pueden ser negativos o
        mayores a 1), así que se normalizan con sigmoide a un rango [0, 1]
        interpretable como "confianza", contra el cual se compara
        `MIN_RERANK_SCORE` para el control de alucinación.
        """
        if not candidates:
            return []
        documents = [doc for doc, _score in candidates]
        pairs = [(query, doc.page_content) for doc in documents]
        raw_scores = self.model.score(pairs)
        for document, raw_score in zip(documents, raw_scores):
            document.metadata["rerank_score"] = _sigmoid(float(raw_score))
        ranked = sorted(documents, key=lambda doc: doc.metadata["rerank_score"], reverse=True)
        return ranked[:top_n]
