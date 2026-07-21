"""Modelo de embeddings — el MISMO modelo se usa para documentos y para preguntas.

Se usa `intfloat/multilingual-e5-small` (local, gratuito, vía sentence-transformers).
Los modelos de la familia e5 requieren anteponer un prefijo distinto para
documentos ("passage: ") y para consultas ("query: "); mezclarlos degrada la
calidad de la búsqueda por similitud.
"""
from __future__ import annotations

import streamlit as st
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from pipeline.config import settings


@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class EmbeddingModel(Embeddings):
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model_name
        self.model = _load_model(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()
