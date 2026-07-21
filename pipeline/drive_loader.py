"""Integración con la API de Google Drive.

Usa una cuenta de servicio (service account) con acceso de solo lectura a la
carpeta compartida donde Mercado Central 24h aloja sus PDFs (políticas de RH,
manuales de operación, contratos, etc.). Solo se necesita compartir la
carpeta de Drive con el email de la cuenta de servicio (rol "Lector").
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from pipeline.config import get_drive_credentials_info, settings

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_DRIVE_FIELDS = "files(id, name, mimeType, modifiedTime, md5Checksum, parents, webViewLink, owners)"


@dataclass
class DriveFile:
    file_id: str
    name: str
    modified_time: str
    md5_checksum: str
    web_view_link: str
    category: str
    author: str


def get_drive_service():
    creds_info = get_drive_credentials_info()
    credentials = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _infer_category(folder_name: str) -> str:
    normalized = folder_name.strip().lower()
    for category in settings.categories:
        if category.lower() in normalized:
            return category
    return "General"


def list_pdf_files(folder_id: str | None = None) -> list[DriveFile]:
    """Lista recursivamente los PDFs dentro de la carpeta de Drive.

    La categoría de cada documento se infiere del nombre de la subcarpeta que
    lo contiene (ej. "RH/politica_vacaciones.pdf" -> categoría "RH"), lo que
    habilita el filtrado por metadados en la recuperación.
    """
    service = get_drive_service()
    root_folder_id = folder_id or settings.drive_folder_id
    if not root_folder_id:
        raise ValueError("GOOGLE_DRIVE_FOLDER_ID no está configurado.")

    all_files: list[DriveFile] = []
    folders_to_scan = [(root_folder_id, "General")]

    while folders_to_scan:
        current_folder_id, current_category = folders_to_scan.pop()
        page_token = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{current_folder_id}' in parents and trashed = false",
                    fields=f"nextPageToken, {_DRIVE_FIELDS}",
                    pageToken=page_token,
                )
                .execute()
            )
            for item in response.get("files", []):
                if item["mimeType"] == "application/vnd.google-apps.folder":
                    folders_to_scan.append((item["id"], _infer_category(item["name"])))
                    continue
                if item["mimeType"] != "application/pdf":
                    continue
                owners = item.get("owners", [])
                author = owners[0]["displayName"] if owners else "Desconocido"
                all_files.append(
                    DriveFile(
                        file_id=item["id"],
                        name=item["name"],
                        modified_time=item["modifiedTime"],
                        md5_checksum=item.get("md5Checksum", ""),
                        web_view_link=item.get("webViewLink", ""),
                        category=current_category,
                        author=author,
                    )
                )
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return all_files


def download_file(file_id: str, destination: Path) -> Path:
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with io.FileIO(destination, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return destination
