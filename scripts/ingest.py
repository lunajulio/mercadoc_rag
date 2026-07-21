"""CLI para ejecutar la sincronización/ingesta de documentos manualmente.

Uso:
    python scripts/ingest.py

También lo ejecuta el workflow de GitHub Actions (.github/workflows/sync_documents.yml)
en una rutina diaria automática.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.sync import sync_documents  # noqa: E402


def main() -> None:
    print(f"Sincronizando documentos desde Google Drive...")
    summary = sync_documents(progress_callback=print)
    print("\nResumen de sincronización:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
