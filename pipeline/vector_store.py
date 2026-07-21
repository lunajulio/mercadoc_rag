"""Base de datos vectorial (Pinecone): almacenamiento, indexación (HNSW gestionado
por Pinecone) e indexación paralela de metadados para filtrado.

Pinecone se eligió porque persiste entre redeploys de Streamlit Community
Cloud (el disco de la app es efímero), a diferencia de una solución local
como Chroma/FAISS.

La capa de acceso usa `langchain_pinecone.PineconeVectorStore` para
embeber/guardar/recuperar, pero el borrado por archivo sigue sobre el
cliente crudo de Pinecone porque los índices serverless no soportan
`delete(filter=...)` por metadatos.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from pipeline.config import settings
from pipeline.document_processor import Chunk
from pipeline.embeddings import EmbeddingModel


def get_pinecone_client() -> Pinecone:
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY no está configurado.")
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index_exists(pc: Pinecone) -> None:
    existing = {idx["name"] for idx in pc.list_indexes()}
    if settings.pinecone_index_name in existing:
        return
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )


def get_index():
    pc = get_pinecone_client()
    ensure_index_exists(pc)
    return pc.Index(settings.pinecone_index_name)


def get_vector_store(embedder: EmbeddingModel) -> PineconeVectorStore:
    return PineconeVectorStore(index=get_index(), embedding=embedder, text_key="text")


def upsert_chunks(chunks: list[Chunk], embedder: EmbeddingModel) -> int:
    """Vectoriza y guarda los chunks en Pinecone junto a su texto y metadatos."""
    if not chunks:
        return 0
    vector_store = get_vector_store(embedder)
    documents = [Document(page_content=chunk.text, metadata=chunk.metadata) for chunk in chunks]
    ids = [chunk.chunk_id for chunk in chunks]
    vector_store.add_documents(documents, ids=ids, batch_size=100)
    return len(chunks)


def delete_by_file_id(file_id: str) -> None:
    """Elimina todos los vectores de un archivo (usado antes de reindexar una versión modificada).

    Los índices serverless de Pinecone no soportan `delete(filter=...)` por
    metadatos, así que se listan los IDs por prefijo (los chunk_id se generan
    como f"{file_id}_p{pagina}_c{indice}") y se eliminan explícitamente.
    """
    index = get_index()
    ids_to_delete: list[str] = []
    for id_batch in index.list(prefix=f"{file_id}_"):
        ids_to_delete.extend(id_batch)
    if ids_to_delete:
        index.delete(ids=ids_to_delete)


def query(
    question: str,
    top_k: int,
    embedder: EmbeddingModel,
    category_filter: list[str] | None = None,
) -> list[tuple[Document, float]]:
    """Búsqueda semántica con filtrado opcional por categoría (metadatos).

    Usa `similarity_search_with_score` (score crudo de Pinecone) y NO
    `similarity_search_with_relevance_scores`: este último tiene un bug
    conocido de LangChain que invierte/escala mal el score para índices
    Pinecone con métrica coseno, lo que rompería en silencio los umbrales
    `min_rerank_score` y el `0.45` de `_find_area_contact` en rag.py.
    """
    vector_store = get_vector_store(embedder)
    pinecone_filter = {"category": {"$in": category_filter}} if category_filter else None
    return vector_store.similarity_search_with_score(question, k=top_k, filter=pinecone_filter)
