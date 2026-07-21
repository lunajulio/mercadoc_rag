"""Etapa de generación: arma el prompt con el contexto recuperado y llama a Gemini.

Reglas impuestas al modelo:
  - Responder únicamente con base en el contexto entregado (sin conocimiento externo).
  - Citar siempre el documento, la página y la categoría de origen.
  - Si el contexto no alcanza para responder, decirlo explícitamente en vez de inventar.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from pipeline.config import settings

_SYSTEM_INSTRUCTION = f"""Eres el asistente virtual interno de {settings.company_name}, un supermercado de \
operación continua (24/7) con tienda física, delivery, app propia y el programa de fidelidad \
"{settings.loyalty_program}".

Tu única fuente de verdad es el CONTEXTO que se te entrega en cada pregunta, extraído de los \
documentos internos de la empresa. Reglas estrictas:

1. Responde ÚNICAMENTE con información presente en el CONTEXTO. Nunca uses conocimiento externo \
   ni completes vacíos con suposiciones.
2. Cada afirmación relevante debe indicar su fuente entre corchetes, con el formato \
   [nombre_archivo, página X].
3. Si el CONTEXTO no contiene información suficiente para responder con confianza, dilo \
   explícitamente ("No encontré esta información en los documentos disponibles") en vez de \
   arriesgar una respuesta incorrecta. No inventes políticas, cifras ni procedimientos.
4. Si el CONTEXTO incluye datos de contacto de un área responsable (RH, Legal, Financiero, \
   Operaciones) relevantes a la pregunta, ofrécelos como siguiente paso.
5. Sé claro y conciso: primero un resumen directo, luego las referencias usadas.
"""


_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_INSTRUCTION),
        (
            "human",
            "CONTEXTO:\n{contexto}\n\n"
            "PREGUNTA DEL COLABORADOR: {pregunta}\n\n"
            "Responde siguiendo las reglas del sistema.",
        ),
    ]
)


def _get_chain():
    llm = ChatGoogleGenerativeAI(model=settings.gemini_model_name, google_api_key=settings.gemini_api_key)
    return _prompt | llm | StrOutputParser()


def format_context(chunks: list[Document]) -> str:
    blocks = []
    for chunk in chunks:
        meta = chunk.metadata
        header = (
            f"[Documento: {meta.get('filename')} | Página: {meta.get('page')} | "
            f"Categoría: {meta.get('category')} | Actualizado: {meta.get('modified_time', '')[:10]}]"
        )
        blocks.append(f"{header}\n{chunk.page_content}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, context_chunks: list[Document]) -> str:
    context_text = format_context(context_chunks)
    chain = _get_chain()
    return chain.invoke({"contexto": context_text, "pregunta": question})
