"""Etapa 7: pipeline de actualización de documentos.

Compara el estado actual de la carpeta de Google Drive contra un manifiesto
local (data/manifest.json) para detectar archivos nuevos, modificados o
eliminados, y reindexa solo lo que cambió (no todo el corpus cada vez).

Se puede ejecutar:
  - manualmente: `python scripts/ingest.py`
  - desde el panel de administración de la app (botón "Sincronizar ahora")
  - de forma automática y periódica vía GitHub Actions (ver .github/workflows/sync_documents.yml)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from pipeline import vector_store
from pipeline.config import settings
from pipeline.document_processor import process_pdf
from pipeline.drive_loader import DriveFile, download_file, list_pdf_files
from pipeline.embeddings import EmbeddingModel


def _load_manifest() -> dict:
    if settings.manifest_path.exists():
        return json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    return {"files": {}, "last_sync": None}


def _save_manifest(manifest: dict) -> None:
    manifest["last_sync"] = datetime.now(timezone.utc).isoformat()
    settings.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _has_changed(drive_file: DriveFile, manifest_entry: dict | None) -> bool:
    if manifest_entry is None:
        return True
    return (
        manifest_entry.get("md5_checksum") != drive_file.md5_checksum
        or manifest_entry.get("modified_time") != drive_file.modified_time
    )


def sync_documents(progress_callback=None) -> dict:
    """Sincroniza el índice vectorial con el estado actual de Drive.

    Devuelve un resumen: {nuevos, modificados, eliminados, sin_cambios, total_chunks}.
    """
    def _report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    manifest = _load_manifest()
    embedder = EmbeddingModel()

    drive_files = list_pdf_files()
    current_ids = {f.file_id for f in drive_files}
    known_ids = set(manifest["files"].keys())

    deleted_ids = known_ids - current_ids
    summary = {"nuevos": 0, "modificados": 0, "eliminados": 0, "sin_cambios": 0, "total_chunks": 0}

    for file_id in deleted_ids:
        _report(f"Eliminando del índice: {manifest['files'][file_id]['name']}")
        vector_store.delete_by_file_id(file_id)
        del manifest["files"][file_id]
        summary["eliminados"] += 1

    for drive_file in drive_files:
        manifest_entry = manifest["files"].get(drive_file.file_id)
        if not _has_changed(drive_file, manifest_entry):
            summary["sin_cambios"] += 1
            continue

        is_update = manifest_entry is not None
        _report(f"{'Actualizando' if is_update else 'Indexando'}: {drive_file.name}")

        if is_update:
            vector_store.delete_by_file_id(drive_file.file_id)

        local_path = settings.downloads_dir / f"{drive_file.file_id}.pdf"
        download_file(drive_file.file_id, local_path)
        chunks = process_pdf(local_path, drive_file, settings.chunk_size, settings.chunk_overlap)
        chunk_count = vector_store.upsert_chunks(chunks, embedder)
        local_path.unlink(missing_ok=True)

        manifest["files"][drive_file.file_id] = {
            "name": drive_file.name,
            "category": drive_file.category,
            "md5_checksum": drive_file.md5_checksum,
            "modified_time": drive_file.modified_time,
            "chunk_count": chunk_count,
        }
        summary["total_chunks"] += chunk_count
        summary["modificados" if is_update else "nuevos"] += 1

    _save_manifest(manifest)
    return summary


def get_sync_status() -> dict:
    manifest = _load_manifest()
    return {
        "last_sync": manifest.get("last_sync"),
        "total_files": len(manifest["files"]),
        "total_chunks": sum(f.get("chunk_count", 0) for f in manifest["files"].values()),
        "files": manifest["files"],
    }
