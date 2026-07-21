"""Configuración centralizada del agente RAG de Mercado Central 24h.

Lee variables de entorno desde `.env` en local y desde `st.secrets` cuando
corre en Streamlit Community Cloud, sin duplicar lógica en cada módulo.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _get(key: str, default: str | None = None) -> str | None:
    """Busca una clave primero en st.secrets (Streamlit Cloud) y luego en el entorno (.env local)."""
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


def get_drive_credentials_info() -> dict:
    """Devuelve el JSON de la service account de Google, desde secrets o desde archivo local."""
    inline_json = _get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if inline_json:
        return json.loads(inline_json)

    file_path = _get("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials/service_account.json")
    full_path = BASE_DIR / file_path if not os.path.isabs(file_path) else Path(file_path)
    if not full_path.exists():
        raise FileNotFoundError(
            f"No se encontró la credencial de Google en {full_path}. "
            "Define GOOGLE_SERVICE_ACCOUNT_JSON (secrets) o coloca el archivo en credentials/."
        )
    return json.loads(full_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Settings:
    # Empresa / branding
    company_name: str = "Mercado Central 24h"
    loyalty_program: str = "Cliente VIP Central"

    # Google Drive
    drive_folder_id: str = field(default_factory=lambda: _get("GOOGLE_DRIVE_FOLDER_ID", ""))

    # Gemini (LLM de generación)
    gemini_api_key: str = field(default_factory=lambda: _get("GEMINI_API_KEY", ""))
    gemini_model_name: str = field(default_factory=lambda: _get("GEMINI_MODEL_NAME", "gemini-2.5-flash"))

    # Pinecone (base de datos vectorial)
    pinecone_api_key: str = field(default_factory=lambda: _get("PINECONE_API_KEY", ""))
    pinecone_index_name: str = field(default_factory=lambda: _get("PINECONE_INDEX_NAME", "mercado-central-rag"))
    pinecone_cloud: str = field(default_factory=lambda: _get("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(default_factory=lambda: _get("PINECONE_REGION", "us-east-1"))

    # Embeddings (local, gratis)
    embedding_model_name: str = field(
        default_factory=lambda: _get("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
    )
    embedding_dimension: int = 384

    # Reranker (local, gratis)
    reranker_model_name: str = field(
        default_factory=lambda: _get("RERANKER_MODEL_NAME", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    )

    # Chunking
    chunk_size: int = field(default_factory=lambda: int(_get("CHUNK_SIZE", "1200")))
    chunk_overlap: int = field(default_factory=lambda: int(_get("CHUNK_OVERLAP", "150")))

    # Recuperación
    top_k_retrieve: int = field(default_factory=lambda: int(_get("TOP_K_RETRIEVE", "20")))
    top_k_rerank: int = field(default_factory=lambda: int(_get("TOP_K_RERANK", "5")))
    min_rerank_score: float = field(default_factory=lambda: float(_get("MIN_RERANK_SCORE", "0.15")))

    # Categorías válidas de documentos (ajustar según carpetas reales en Drive)
    categories: tuple[str, ...] = ("RH", "Legal", "Financiero", "Operaciones", "Fidelizacion", "General")

    # Admin
    admin_password: str = field(default_factory=lambda: _get("ADMIN_PASSWORD", ""))

    # Rutas locales
    data_dir: Path = BASE_DIR / "data"
    downloads_dir: Path = BASE_DIR / "data" / "downloads"
    manifest_path: Path = BASE_DIR / "data" / "manifest.json"
    feedback_path: Path = BASE_DIR / "data" / "feedback.jsonl"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.downloads_dir.mkdir(parents=True, exist_ok=True)
