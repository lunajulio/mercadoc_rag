"""Etapa 2: extracción, limpieza, metadatos y fragmentación (chunking) de PDFs."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from pipeline.drive_loader import DriveFile


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _clean_text(text: str) -> str:
    text = text.replace("\xad", "")  # guiones de partición de palabra
    text = re.sub(r"-\n(?=\w)", "", text)  # une palabras cortadas al final de línea
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[str]:
    """Devuelve el texto limpio de cada página del PDF."""
    reader = PdfReader(str(pdf_path))
    return [_clean_text(page.extract_text() or "") for page in reader.pages]


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Fragmenta por párrafos/oraciones con solapamiento, vía LangChain."""
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def process_pdf(local_path: Path, drive_file: DriveFile, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Convierte un PDF descargado en una lista de Chunks listos para vectorizar.

    Cada chunk conserva metadatos de trazabilidad: archivo, página, categoría,
    autor y fecha de modificación en Drive — necesarios para el filtrado por
    metadados y para citar la fuente en la respuesta final.
    """
    pages = extract_pages(local_path)
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for idx, fragment in enumerate(_split_text(page_text, chunk_size, chunk_overlap)):
            if not fragment.strip():
                continue
            chunk_id = f"{drive_file.file_id}_p{page_number}_c{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=fragment.strip(),
                    metadata={
                        "drive_file_id": drive_file.file_id,
                        "filename": drive_file.name,
                        "category": drive_file.category,
                        "author": drive_file.author,
                        "modified_time": drive_file.modified_time,
                        "page": page_number,
                        "web_view_link": drive_file.web_view_link,
                    },
                )
            )
    return chunks
