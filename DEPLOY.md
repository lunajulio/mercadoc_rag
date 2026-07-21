# Guía de despliegue — Mercado Central 24h

Sigue este orden: primero credenciales, luego prueba local, y solo al final el despliegue
en Streamlit Community Cloud.

## 1. Google Drive API (cuenta de servicio)

1. En [Google Cloud Console](https://console.cloud.google.com/), crea un proyecto (o usa
   uno existente).
2. Habilita la **Google Drive API** (APIs & Services → Library → buscar "Google Drive API").
3. Crea una cuenta de servicio (APIs & Services → Credentials → Create Credentials →
   Service Account).
4. Dentro de la cuenta de servicio, genera una clave JSON (Keys → Add Key → JSON) y
   descárgala.
5. Guarda ese archivo como `credentials/service_account.json` en este proyecto (esta ruta
   ya está en `.gitignore`, nunca se sube al repo).
6. En Google Drive, comparte la carpeta que contiene los PDFs de la empresa con el email
   de la cuenta de servicio (campo `client_email` dentro del JSON), con permiso de
   **Lector**.
7. Organiza los PDFs en subcarpetas por categoría (`RH/`, `Legal/`, `Financiero/`,
   `Operaciones/`, etc.) — el pipeline infiere la categoría del nombre de la subcarpeta.
8. Copia el ID de la carpeta raíz (está en la URL de Drive:
   `drive.google.com/drive/folders/<ESTE_ID>`).

## 2. Pinecone (base de datos vectorial)

1. Crea una cuenta gratuita en [pinecone.io](https://www.pinecone.io/).
2. Genera una API key (Project → API Keys).
3. No es necesario crear el índice manualmente: `pipeline/vector_store.py` lo crea
   automáticamente en el primer uso (serverless, dimensión 384, métrica coseno). Solo
   confirma que la región elegida (`PINECONE_CLOUD` / `PINECONE_REGION`) esté disponible en
   el plan gratuito de tu cuenta.

## 3. Gemini (LLM de generación)

1. Genera una API key en [Google AI Studio](https://aistudio.google.com/apikey).

## 4. Configurar variables de entorno

1. Ya existe un archivo `.env` en este proyecto (copiado de `.env.example`). Complétalo con
   los valores obtenidos en los pasos 1-3:
   - `GOOGLE_DRIVE_FOLDER_ID`
   - `GEMINI_API_KEY`
   - `PINECONE_API_KEY`
2. Verifica que `credentials/service_account.json` exista y `GOOGLE_SERVICE_ACCOUNT_FILE`
   apunte a esa ruta (ya viene así por defecto).

## 5. Probar localmente (obligatorio antes de desplegar)

```bash
cd mercado_central_rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ejecuta la primera ingesta para poblar el índice vectorial en Pinecone:

```bash
python scripts/ingest.py
```

Deberías ver un resumen con la cantidad de documentos nuevos indexados. Si falla, revisa:
- Que la carpeta de Drive esté compartida con el email de la cuenta de servicio.
- Que `PINECONE_API_KEY` y `GEMINI_API_KEY` sean válidas.

Levanta la app localmente:

```bash
streamlit run app.py
```

Abre `http://localhost:8501`, prueba preguntas que sí estén en tus documentos (deben
responder citando archivo/página) y preguntas que NO estén (debe responder el fallback
explícito, y sugerir el área responsable si hay contacto indexado). Prueba también los
filtros de categoría del sidebar y, si configuraste `ADMIN_PASSWORD`, el botón
"Sincronizar ahora" y el panel de monitoreo.

Solo continúa al despliegue cuando el flujo local funcione como esperas.

## 6. Subir el proyecto a GitHub

Este directorio aún no es un repositorio git. Desde la raíz de `mercado_central_rag/`:

```bash
git init
git add .
git commit -m "Agente RAG de documentación interna - Mercado Central 24h"
git branch -M main
git remote add origin https://github.com/<tu_usuario>/<tu_repo>.git
git push -u origin main
```

`credentials/service_account.json`, `.env` y `data/feedback.jsonl` quedan fuera del commit
gracias a `.gitignore` — nunca se suben.

## 7. Desplegar en Streamlit Community Cloud

1. Entra a [share.streamlit.io](https://share.streamlit.io/) con tu cuenta de GitHub.
2. **New app** → selecciona el repositorio, la rama `main` y el archivo principal:
   `mercado_central_rag/app.py` (ajusta la ruta si el repo raíz ya es `mercado_central_rag`).
3. En **Advanced settings → Python version**, elegí explícitamente **3.13** (o 3.11/3.12).
   El repo incluye un `runtime.txt` con `python-3.13`, pero hay reportes recientes de que
   Community Cloud lo ignora en algunos casos — el dropdown de Advanced settings es la vía
   confiable. Esto importa porque `langchain-pinecone` (y otras deps de LangChain) todavía
   no publican wheels para Python 3.14+ (la versión que Cloud usa por defecto hoy), lo que
   rompe la instalación con "No matching distribution found for langchain-pinecone".
4. En **Advanced settings → Secrets**, pega el contenido de
   `.streamlit/secrets.toml.example` con tus valores reales, incluyendo el JSON completo de
   la cuenta de servicio como string en `GOOGLE_SERVICE_ACCOUNT_JSON` (una sola línea).
5. Click en **Deploy**.

> Si tu app ya está desplegada y falló con Python 3.14 (como en el log de error), cambiar
> `runtime.txt` y hacer push **no alcanza** para una app existente: Community Cloud solo
> aplica la versión de Python en el deploy inicial. Hay que borrar la app desde el
> dashboard y volver a desplegarla desde cero, eligiendo la versión de Python correcta en
> el paso 3 de arriba.

Como el índice vive en Pinecone (no en el disco de Streamlit Cloud), **no hace falta**
volver a correr la ingesta después de cada redeploy: los datos ya están indexados desde tu
prueba local o desde el workflow de GitHub Actions.

## 8. Activar la sincronización automática (opcional pero recomendado)

El workflow `.github/workflows/sync_documents.yml` corre diariamente y reindexa solo los
cambios detectados en Drive. Para activarlo:

1. En GitHub: Settings → Secrets and variables → Actions → New repository secret, y
   agrega los mismos valores que en `secrets.toml` (`GOOGLE_SERVICE_ACCOUNT_JSON`,
   `GOOGLE_DRIVE_FOLDER_ID`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`,
   `PINECONE_REGION`).
2. El workflow necesita permiso de escritura para confirmar el `data/manifest.json`
   actualizado: Settings → Actions → General → Workflow permissions → **Read and write
   permissions**.
3. Puedes disparar una corrida manual desde la pestaña **Actions** del repo
   ("Run workflow") para probarlo sin esperar al cron diario.

## Notas sobre recursos en el plan gratuito de Streamlit Cloud

Los modelos de embeddings y reranking (sentence-transformers) se cargan en memoria con
`@st.cache_resource` para no recargarlos en cada pregunta. En el plan gratuito (~1 GB de
RAM) esto puede ser ajustado si además cargas modelos muy grandes: si notas errores de
memoria, considera modelos aún más pequeños o solicitar más recursos en tu plan de
Streamlit Cloud.
