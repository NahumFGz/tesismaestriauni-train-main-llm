from dotenv import load_dotenv

load_dotenv()

import os
import re
import time
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range


# ── Embeddings y vector store con reintentos ─────────────────────────────────
def init_vector_store_with_retries(max_retries: int = 5, delay: int = 3) -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔁 Intento {attempt} de conexión a Qdrant...")
            qdrant_client = QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
            )
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name="attendance_docs",
                embedding=embeddings,
            )
            print("✅ Conexión a Qdrant exitosa")
            return vector_store
        except Exception as e:
            print(f"⚠️ Fallo al conectar a Qdrant: {e}")
            last_exception = e
            if attempt < max_retries:
                time.sleep(delay)

    raise RuntimeError(
        f"No se pudo conectar a Qdrant después de {max_retries} intentos"
    ) from last_exception


vector_store = init_vector_store_with_retries()


class RetrieverState(TypedDict):
    question: str
    filtro: Filter | None
    context: List[Document]


# ── Utilidades para fechas en español ──────────────────────
_MESES_ES: dict[str, int] = {
    # completos
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
    # abreviados
    "ene": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "sep": 9,
    "set": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12,
}


# ── Nodo parse_query para fechas ───────────────────────────


def parse_query(state: RetrieverState) -> RetrieverState:
    q = state["question"].lower()
    filtro: Filter | None = None

    # 1) 21 de octubre del 2022
    m = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+del?\s+(\d{4})", q)
    if m and m.group(2) in _MESES_ES:
        d, mes_txt, a = int(m.group(1)), m.group(2), int(m.group(3))
        filtro = Filter(
            must=[
                FieldCondition(key="metadata.anio", match=MatchValue(value=a)),
                FieldCondition(key="metadata.mes", match=MatchValue(value=_MESES_ES[mes_txt])),
                FieldCondition(key="metadata.dia", match=MatchValue(value=d)),
            ]
        )

    # 2) 21/10/2022
    if filtro is None:
        m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", q)
        if m:
            d, mth, a = map(int, m.groups())
            filtro = Filter(
                must=[
                    FieldCondition(key="metadata.anio", match=MatchValue(value=a)),
                    FieldCondition(key="metadata.mes", match=MatchValue(value=mth)),
                    FieldCondition(key="metadata.dia", match=MatchValue(value=d)),
                ]
            )

    # 2b) 21-10-2022 o 21.10.2022
    if filtro is None:
        m = re.search(r"(\d{1,2})[\-.](\d{1,2})[\-.](\d{4})", q)
        if m:
            d, mth, a = map(int, m.groups())
            filtro = Filter(
                must=[
                    FieldCondition(key="metadata.anio", match=MatchValue(value=a)),
                    FieldCondition(key="metadata.mes", match=MatchValue(value=mth)),
                    FieldCondition(key="metadata.dia", match=MatchValue(value=d)),
                ]
            )

    # 3) octubre 2022 / oct 2022
    if filtro is None:
        m = re.search(r"(\w+)\s+del?\s+(\d{4})", q)
        if m and m.group(1) in _MESES_ES:
            mes_txt, a = m.group(1), int(m.group(2))
            filtro = Filter(
                must=[
                    FieldCondition(key="metadata.anio", match=MatchValue(value=a)),
                    FieldCondition(key="metadata.mes", match=MatchValue(value=_MESES_ES[mes_txt])),
                ]
            )

    # 3b) 10/2022 o 10-2022
    if filtro is None:
        m = re.search(r"\b(\d{1,2})[\-/](\d{4})\b", q)
        if m:
            mth, a = map(int, m.groups())
            if 1 <= mth <= 12:
                filtro = Filter(
                    must=[
                        FieldCondition(key="metadata.anio", match=MatchValue(value=a)),
                        FieldCondition(key="metadata.mes", match=MatchValue(value=mth)),
                    ]
                )

    # 4) solo año
    if filtro is None:
        m = re.search(r"\b(20\d{2})\b", q)
        if m:
            a = int(m.group(1))
            filtro = Filter(must=[FieldCondition(key="metadata.anio", match=MatchValue(value=a))])

    # 5) rango de años
    if filtro is None:
        m = re.search(r"(20\d{2})\s*(?:-|a|al|hasta)\s*(20\d{2})", q)
        if m:
            a1, a2 = sorted(map(int, m.groups()))
            filtro = Filter(must=[FieldCondition(key="metadata.anio", range=Range(gte=a1, lte=a2))])

    return {"question": state["question"], "filtro": filtro}


def retrieve(state: RetrieverState) -> RetrieverState:
    filtro = state.get("filtro")
    if filtro:
        docs = vector_store.similarity_search(state["question"], filter=filtro)
    else:
        docs = vector_store.similarity_search(state["question"])
    return {"question": state["question"], "context": docs}


# Definir el grafo de solo retrieval
retriever_graph = (
    StateGraph(RetrieverState)
    .add_node("parse", parse_query)
    .add_node("retrieve", retrieve)
    .add_edge(START, "parse")
    .add_edge("parse", "retrieve")
    .set_entry_point("parse")
    .set_finish_point("retrieve")
    .compile()
)

# ── Tools LangChain ----------------------------------------------------------


class DocumentosResponse(TypedDict):
    documentos: List[str]


@tool("buscar_documentos_asistencia")
async def buscar_documentos_asistencia(pregunta: str) -> DocumentosResponse:
    """Busca y retorna documentos relacionados con la asistencia parlamentaria según la consulta proporcionada.
    Esta herramienta permite buscar documentos de asistencia parlamentaria, devolviendo:
    - El contenido de los documentos más relevantes

    Ejemplo de uso:
    - "¿Cuál fue la asistencia del 15 de marzo de 2024?"
    """
    try:
        result = await retriever_graph.ainvoke({"question": pregunta})
        documentos = []
        for doc in result["context"]:
            documentos.append(doc.page_content)
            # Metadatos disponibles para futuro uso: doc.metadata
        return {"documentos": documentos}
    except Exception as e:
        return {"documentos": [f"Error al buscar documentos: {str(e)}"]}


class RangoFechasResponse(TypedDict):
    rango: str


@tool("obtener_rango_asistencia")
async def obtener_rango_asistencia() -> RangoFechasResponse:
    """Retorna el rango de fechas disponible para consultar asistencias parlamentarias."""
    return {
        "rango": "La información de asistencias está disponible desde enero de 2009 hasta marzo de 2025"
    }


# Lista pública de herramientas

tools_list = [buscar_documentos_asistencia, obtener_rango_asistencia]
