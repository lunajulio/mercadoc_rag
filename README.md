# Asistente de Documentación Interna — Mercado Central 24h

Agente conversacional (RAG) construido con Streamlit que responde preguntas de
los colaboradores sobre políticas y manuales internos (PDFs alojados en
Google Drive), citando siempre la fuente y evitando alucinaciones.

## Arquitectura

```
Google Drive (PDFs)
      │  Google Drive API (service account, solo lectura)
      ▼
pipeline/drive_loader.py      → lista y descarga PDFs, infiere categoría por carpeta
pipeline/document_processor.py→ extrae texto, limpia, fragmenta (chunks) y arma metadatos
pipeline/embeddings.py        → intfloat/multilingual-e5-small (local, mismo modelo doc/query)
pipeline/vector_store.py      → Pinecone (índice HNSW gestionado + filtrado por metadatos)
pipeline/reranker.py          → cross-encoder multilingüe (reordena top_k_retrieve → top_k_rerank)
pipeline/llm.py                → Gemini genera la respuesta solo con el contexto entregado
rag.py                         → orquesta las 5 fases de recuperación + generación + fallback
pipeline/sync.py               → detecta altas/bajas/modificaciones y reindexa incrementalmente
app.py                          → interfaz de chat en Streamlit
```

### Por qué estas decisiones

- **Embeddings locales (sentence-transformers)**: sin costo por uso ni API key adicional.
  Se usa el mismo modelo para documentos (`passage: `) y preguntas (`query: `) — condición
  obligatoria para que los vectores sean comparables.
- **Pinecone como base vectorial**: a diferencia de una solución local (Chroma/FAISS), el
  índice persiste aunque Streamlit Community Cloud reinicie o redepliegue la app (su disco es
  efímero). Esto también permite que la sincronización automática por GitHub Actions y la
  sincronización manual desde la app compartan el mismo índice.
- **Reranking con cross-encoder**: la búsqueda vectorial trae 20 candidatos amplios (rápido,
  aproximado); el cross-encoder los reordena comparando la pregunta completa contra cada
  fragmento (lento pero preciso) y se queda con los 5 mejores.
- **Umbral de confianza (`MIN_RERANK_SCORE`)**: si el mejor candidato tras el rerank no supera
  el umbral, el agente NO genera una respuesta con el LLM — responde el fallback explícito.

## Flujo de una pregunta (capa de recuperación + generación)

1. La pregunta se vectoriza con el mismo modelo de embeddings que los documentos.
2. Se buscan los `TOP_K_RETRIEVE` fragmentos más cercanos en Pinecone, opcionalmente
   filtrando por categoría (metadatos) desde el sidebar.
3. Un cross-encoder reordena esos candidatos y se retienen los `TOP_K_RERANK` mejores.
4. Si la mejor puntuación no supera `MIN_RERANK_SCORE`, se activa el fallback: el agente
   busca si existe información de contacto del área responsable (RH, Legal, Financiero,
   Operaciones) **dentro del propio corpus indexado** antes de sugerirla; si no encuentra
   nada relevante, indica que la pregunta está fuera del alcance de la base de conocimiento.
5. Si hay suficiente contexto, se arma el prompt (contexto + pregunta) y Gemini genera la
   respuesta, con instrucciones estrictas de no usar conocimiento externo y de citar
   `[archivo, página]` en cada afirmación.
6. La app muestra la respuesta junto con un panel expandible de "Fuentes" (archivo, página,
   categoría, relevancia).

## Mantenimiento continuo (requerimiento 7)

- **Pipeline de actualización** (`pipeline/sync.py`): compara Drive contra
  `data/manifest.json` (hash MD5 + fecha de modificación) y solo reprocesa lo que cambió.
  Se ejecuta manualmente (`scripts/ingest.py`), desde el panel admin de la app, o
  automáticamente todos los días vía `.github/workflows/sync_documents.yml`.
- **Curaduría de contenido**: se recomienda asignar un responsable por categoría (RH,
  Legal, Financiero, Operaciones) que revise trimestralmente si los PDFs en Drive siguen
  vigentes, moviendo/eliminando versiones obsoletas (el pipeline detecta la eliminación y
  limpia el índice automáticamente).
- **Monitoreo de calidad**: cada interacción se registra en `data/feedback.jsonl`
  (pregunta, si hubo respuesta, score de confianza, feedback 👍/👎). El panel admin de la
  app muestra tasa de preguntas sin respuesta y feedback negativo.
  ⚠️ Este log vive en el disco de la app: en Streamlit Cloud se reinicia con cada redeploy.
  Para monitoreo productivo real, migrar este log a un almacén externo (hoja de cálculo,
  base de datos, etc.).
- **Ciclo de mejora**: preguntas recurrentes sin buena respuesta señalan documentos
  faltantes; feedback negativo repetido en una categoría sugiere revisar el prompt o los
  parámetros de recuperación (`TOP_K_*`, `MIN_RERANK_SCORE`).
- **Actualización del modelo**: antes de cambiar `GEMINI_MODEL_NAME` en producción, probar
  el nuevo modelo localmente con el mismo set de preguntas de referencia.

## Pruebas locales y despliegue

Ver [`DEPLOY.md`](DEPLOY.md) para la guía paso a paso (Google Cloud, Pinecone, Gemini,
prueba local y despliegue en Streamlit Community Cloud).
