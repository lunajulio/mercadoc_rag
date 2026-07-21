# Asistente de Documentación Interna — Mercado Central 24h

Agente conversacional (RAG) construido con Streamlit que responde preguntas de
los colaboradores sobre políticas y manuales internos (PDFs alojados en
Google Drive), citando siempre la fuente y evitando alucinaciones.

## Descripción general

Mercado Central 24h es un supermercado de operación continua (tienda física,
delivery, app propia y programa de fidelidad "Cliente VIP Central"). Sus
colaboradores necesitan consultar rápidamente reglamentos, políticas de
atención al cliente, procedimientos operativos y manuales de proveedores, hoy
dispersos en PDFs dentro de una carpeta de Google Drive.

Este proyecto implementa un agente RAG (*Retrieval-Augmented Generation*) que:

- Indexa automáticamente los PDFs de Drive, organizados por categoría
  (RH, Legal, Financiero, Operaciones, Fidelización, General).
- Responde en lenguaje natural **únicamente con información encontrada en esos
  documentos**, citando siempre `[archivo, página]`.
- Declara explícitamente cuándo una pregunta está fuera de su base de
  conocimiento en vez de inventar una respuesta, y en ese caso intenta ofrecer
  el contacto del área responsable si existe en el corpus.
- Se mantiene actualizado solo: detecta altas, bajas y modificaciones en Drive
  y reindexa de forma incremental (manual o automática vía GitHub Actions).
- Registra cada interacción (pregunta, si hubo respuesta, confianza, feedback
  👍/👎) para monitorear la calidad del asistente.

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

## Tecnologías y herramientas

| Capa | Herramienta | Uso |
|---|---|---|
| Interfaz | [Streamlit](https://streamlit.io/) | Chat, sidebar de filtros, panel de administración |
| Origen de documentos | Google Drive API (`google-api-python-client`, cuenta de servicio) | Listado y descarga de PDFs, solo lectura |
| Procesamiento de PDFs | `pypdf`, `langchain-text-splitters` | Extracción de texto y chunking con overlap |
| Embeddings | `sentence-transformers` — `intfloat/multilingual-e5-small` (local, gratuito) | Vectoriza documentos y preguntas con el mismo modelo |
| Base vectorial | [Pinecone](https://www.pinecone.io/) (serverless, dimensión 384, coseno) | Almacena y busca los embeddings, persistente entre redeploys |
| Reranking | `sentence-transformers` cross-encoder — `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Reordena candidatos por relevancia real a la pregunta |
| Generación (LLM) | [Google Gemini](https://aistudio.google.com/) vía `langchain-google-genai` | Redacta la respuesta final solo con el contexto entregado |
| Orquestación | `langchain-core` | Prompt templates y cadena retrieval → generación |
| Automatización | GitHub Actions (`.github/workflows/sync_documents.yml`) | Reindexación diaria incremental |
| Lenguaje / runtime | Python 3.13 | — |

## Instrucciones para ejecutar el proyecto

> Guía detallada paso a paso (credenciales, troubleshooting y despliegue en
> Streamlit Community Cloud) en [`DEPLOY.md`](DEPLOY.md). Resumen rápido para
> correr en local:

### 1. Requisitos previos

- Python 3.13 (o 3.11/3.12).
- Cuenta de servicio de Google Cloud con la **Google Drive API** habilitada,
  con acceso de lectura a la carpeta de PDFs.
- Cuenta gratuita en [Pinecone](https://www.pinecone.io/) (API key).
- API key de [Google AI Studio](https://aistudio.google.com/apikey) para Gemini.

### 2. Instalación

```bash
git clone <url-del-repo>
cd mercado_central_rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuración

```bash
cp .env.example .env
```

Completa en `.env`: `GOOGLE_DRIVE_FOLDER_ID`, `GEMINI_API_KEY`,
`PINECONE_API_KEY`, y coloca el JSON de la cuenta de servicio en
`credentials/service_account.json` (ruta configurable con
`GOOGLE_SERVICE_ACCOUNT_FILE`).

### 4. Indexar los documentos

```bash
python scripts/ingest.py
```

### 5. Levantar la app

```bash
streamlit run app.py
```

Abre `http://localhost:8501`. Desde el sidebar puedes filtrar por categoría de
documento y, si configuraste `ADMIN_PASSWORD`, acceder al panel de
administración para resincronizar y ver métricas de calidad.

## Ejemplos de preguntas que el agente puede responder

Basado en los documentos actualmente indexados (política de atención al
cliente y devoluciones, reglamento interno, FAQ y manual de proveedores):

- "¿Cómo se procesa una devolución?"
- "¿Cuál es el plazo máximo para devolver un producto con defecto de fábrica?"
- "¿Qué documentos necesito para registrarme como proveedor?"
- "¿Cuáles son los horarios de recepción de mercadería para proveedores?"
- "¿Qué hago si un cliente quiere cambiar un producto sin ticket de compra?"
- "¿Cuál es el procedimiento ante una falla en el sistema de cobro?"
- "¿Cómo funciona el programa de fidelidad Cliente VIP Central?"
- "¿A quién contacto si tengo una duda legal sobre un contrato de proveedor?"
  (fuera del alcance directo de los documentos, pero el agente ofrece el
  contacto del área Legal si está indexado)
- "¿Cuál es la política de vacaciones del personal?" (ejemplo de pregunta
  fuera del alcance actual de la base de conocimiento, ya que ese documento de
  RH aún no está indexado)


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
