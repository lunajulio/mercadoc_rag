"""Interfaz Streamlit del agente de documentación interna de Mercado Central 24h."""
from __future__ import annotations

import json

import streamlit as st

import rag
from pipeline import sync
from pipeline.config import settings

st.set_page_config(
    page_title=f"{settings.company_name}",
    page_icon="🛒",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = {}

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title(f"🛒 {settings.company_name}")
    st.caption(f"Programa {settings.loyalty_program} · Asistente de documentación interna")

    st.subheader("Filtros")
    selected_categories = st.multiselect(
        "Categoría de documentos",
        options=list(settings.categories),
        default=[],
        help="Restringe la búsqueda a categorías específicas antes de calcular similitud semántica.",
    )

    st.divider()
    st.subheader("Panel de administración")
    is_admin = False
    if settings.admin_password:
        pwd = st.text_input("Contraseña admin", type="password")
        is_admin = pwd == settings.admin_password
    else:
        st.caption("Define ADMIN_PASSWORD para habilitar el panel de administración.")

    if is_admin:
        status = sync.get_sync_status()
        st.metric("Documentos indexados", status["total_files"])
        st.metric("Fragmentos (chunks)", status["total_chunks"])
        st.caption(f"Última sincronización: {status['last_sync'] or 'nunca'}")

        if st.button("🔄 Sincronizar ahora", use_container_width=True):
            progress_box = st.empty()
            summary = sync.sync_documents(progress_callback=lambda msg: progress_box.info(msg))
            st.success(
                f"Nuevos: {summary['nuevos']} · Modificados: {summary['modificados']} · "
                f"Eliminados: {summary['eliminados']} · Sin cambios: {summary['sin_cambios']}"
            )

        if settings.feedback_path.exists():
            lines = settings.feedback_path.read_text(encoding="utf-8").strip().splitlines()
            entries = [json.loads(line) for line in lines] if lines else []
            if entries:
                st.divider()
                st.subheader("Monitoreo de calidad")
                total = len(entries)
                sin_respuesta = sum(1 for e in entries if not e["has_answer"])
                negativos = sum(1 for e in entries if e.get("rating") == "down")
                st.metric("Preguntas totales", total)
                st.metric("% sin respuesta", f"{sin_respuesta / total * 100:.0f}%")
                st.metric("Feedback negativo", negativos)

# --------------------------------------------------------------------------
# Chat principal
# --------------------------------------------------------------------------
st.header("Asistente de documentación interna")
st.caption(
    "Responde preguntas sobre políticas, manuales y procedimientos internos a partir de los "
    "PDFs oficiales alojados en Google Drive. Cuando no encuentra la respuesta, lo indica "
    "claramente en vez de inventarla."
)

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📎 Fuentes"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['filename']}** · página {source['page']} · "
                        f"categoría _{source['category']}_ · relevancia {source['relevance']}"
                    )

question = st.chat_input("Escribe tu pregunta sobre políticas o procedimientos internos...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la documentación..."):
            result = rag.answer_question(question, category_filter=selected_categories or None)
        st.markdown(result.answer)
        if result.sources:
            with st.expander("📎 Fuentes"):
                for source in result.sources:
                    st.markdown(
                        f"- **{source['filename']}** · página {source['page']} · "
                        f"categoría _{source['category']}_ · relevancia {source['relevance']}"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )
    st.session_state.last_result = {"question": question, "result": result}

if st.session_state.last_result:
    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        if st.button("👍", key="feedback_up"):
            rag.log_feedback(
                st.session_state.last_result["question"], st.session_state.last_result["result"], "up"
            )
            st.toast("¡Gracias por tu feedback!")
    with col2:
        if st.button("👎", key="feedback_down"):
            rag.log_feedback(
                st.session_state.last_result["question"], st.session_state.last_result["result"], "down"
            )
            st.toast("Gracias, usaremos esto para mejorar la base de conocimiento.")
