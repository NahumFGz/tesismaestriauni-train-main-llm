from dotenv import load_dotenv

load_dotenv()

import datetime
import decimal
import json
import os
import time
from typing import Any, List, TypedDict

from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from sqlalchemy import text

# ── DB ───────────────────────────────────────────────────────────────────────


def init_sql_database_with_retries(max_retries: int = 5, delay: int = 3) -> SQLDatabase:
    uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔁 Intento {attempt} de conexión a PostgreSQL...")
            db = SQLDatabase.from_uri(uri)
            # Validamos la conexión ejecutando un query simple
            db.run("SELECT 1")
            print("✅ Conexión a PostgreSQL exitosa")
            return db
        except Exception as e:
            print(f"⚠️ Error al conectar a PostgreSQL: {e}")
            last_exception = e
            if attempt < max_retries:
                time.sleep(delay)

    raise RuntimeError(
        f"No se pudo conectar a PostgreSQL después de {max_retries} intentos"
    ) from last_exception


host = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")
db_name = os.getenv("POSTGRES_DB")
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

db = init_sql_database_with_retries()

# ── LLM y herramientas SQL ──────────────────────────────────────────────────

llm = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0.0)

_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
_tools = _toolkit.get_tools()

_get_schema_tool = next(t for t in _tools if t.name == "sql_db_schema")
_list_tables_tool = next(t for t in _tools if t.name == "sql_db_list_tables")
_run_query_tool = next(t for t in _tools if t.name == "sql_db_query")

_get_schema_node = ToolNode([_get_schema_tool], name="get_schema")
_run_query_node = ToolNode([_run_query_tool], name="run_query")

# ── Definición de nodos de función ──────────────────────────────────────────


async def list_tables(state: MessagesState):
    """Nodo inicial: obtiene las tablas disponibles y las añade a la conversación."""
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_msg = AIMessage(content="", tool_calls=[tool_call])
    tool_msg = await _list_tables_tool.ainvoke(tool_call)  # type: ignore
    response = AIMessage(f"Tablas disponibles: {tool_msg.content}")
    return {"messages": [tool_call_msg, tool_msg, response], "attempt": 0}


async def call_get_schema(state: MessagesState):
    """Fuerza al modelo a llamar a sql_db_schema sobre una tabla elegida por él."""
    llm_with_tools = llm.bind_tools([_get_schema_tool], tool_choice="any")
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


async def generate_query(state: MessagesState):
    """Genera la consulta SQL a partir de la pregunta del usuario."""
    system_message = {
        "role": "system",
        "content": f"""
            Eres un agente diseñado para interactuar con una base de datos SQL.
            Dada una pregunta de entrada, crea una consulta {db.dialect} sintácticamente correcta para ejecutar,
            luego observa los resultados de la consulta y devuelve la respuesta. A menos que el usuario
            especifique un número específico de ejemplos que desee obtener, siempre limita tu
            consulta a un máximo de 5 resultados.

            Puedes ordenar los resultados por una columna relevante para devolver los ejemplos
            más interesantes de la base de datos. Nunca consultes todas las columnas de una tabla específica,
            solo solicita las columnas relevantes dada la pregunta.

            IMPORTANTE: Siempre incluye en tus consultas los campos de 'ruc_proveedor','nombre_proveedor','fecha_de_inicio_de_actividades', 'monto_girado', 'monto' si están disponibles en las tablas consultadas.

            IMPORTANTE: La tabla contratos_orden_servicio tiene la columna tipo_contrato que puede ser 'CONTRATO' o 'ORDEN SERVICIO' y debes usarla si te piden detalles de contratos o ordenes de servicio. tambien tiene la columna 'fecha_inicio' y 'fecha_fin' que son las fechas de inicio y fin del contrato o orden de servicio.

            NO hagas ninguna declaración DML (INSERT, UPDATE, DELETE, DROP, etc.) en la base de datos.
        """,
    }
    llm_with_tools = llm.bind_tools([_run_query_tool])
    response = await llm_with_tools.ainvoke([system_message] + state["messages"])
    return {"messages": [response]}


async def check_query(state: MessagesState):
    """
    Valida (y, si fuera necesario, reescribe) la consulta SQL más reciente
    antes de ejecutarla.
    """
    # ── 1. Localizar el último AIMessage que realmente contenga una llamada a sql_db_query
    last_ai_with_tool = next(
        (
            msg
            for msg in reversed(state["messages"])
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)
        ),
        None,
    )

    # Si no encontramos la tool-call, abortamos con un mensaje controlado
    if last_ai_with_tool is None:
        err = AIMessage(content="Error: no se encontró una consulta que revisar.")
        return {"messages": [err]}

    sql_query = last_ai_with_tool.tool_calls[0]["args"]["query"]

    # ── 2. Construimos el prompt de revisión
    system_message = {
        "role": "system",
        "content": f"""
            Eres un experto en SQL con gran atención al detalle.
            Revisa dos veces la consulta {db.dialect} para detectar errores comunes como:
            - NOT IN con valores NULL
            - UNION vs UNION ALL
            - BETWEEN en rangos exclusivos
            - Incompatibilidad de tipos
            - Citar identificadores
            - Nº de argumentos en funciones
            - Conversión de tipos
            - Columnas correctas en JOINs

            Si detectas alguno de estos problemas, reescribe la consulta corregida.
            Si no hay errores, reproduce la consulta original.

            Luego llama a la herramienta apropiada para ejecutarla.
        """,
    }

    user_message = {"role": "user", "content": sql_query}

    # ── 3. Forzamos al LLM a elegir únicamente sql_db_query como herramienta
    llm_with_tools = llm.bind_tools([_run_query_tool], tool_choice="any")
    response = await llm_with_tools.ainvoke([system_message, user_message])

    # Mantener el mismo ID para que LangGraph haga tracking correcto
    response.id = last_ai_with_tool.id

    return {"messages": [response]}


def should_continue(state: MessagesState):
    """Decide si el grafo debe continuar o finalizar."""
    return END if not state["messages"][-1].tool_calls else "check_query"


# ── Construcción del grafo ───────────────────────────────────────────────────

builder = StateGraph(MessagesState)

builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(_get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(_run_query_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("check_query", "run_query")

# Flujo: si generate_query produce tool_call -> check_query loop; después de run_query, decidimos reintentar o finalizar.


def _needs_retry(state):  # type: ignore
    """Limita los reintentos a un máximo de 3."""
    last_msg = state["messages"][-1]

    # Sin error ⇒ terminar
    if "Error:" not in str(last_msg.content):
        return END

    # Incrementamos contador de intentos
    state["attempt"] = state.get("attempt", 0) + 1

    # Si aún no superamos el límite de 3, reintentamos
    if state["attempt"] <= 3:
        # Volvemos a generate_query para construir una nueva consulta desde cero
        return "generate_query"

    # Demasiados intentos: finalizar con error
    return END


builder.add_conditional_edges("run_query", _needs_retry)

agent = builder.compile()

# ── Utilidades ──────────────────────────────────────────────────────────────


def _serialize(value: Any):
    """Convierte valores a tipos serializables por JSON."""
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return value


# ── Definición de herramientas MCP ──────────────────────────────────────────
class SQLRowsResponse(TypedDict):
    query: str
    columns: List[str]
    rows: List[List[Any]]


@tool("obtener_tablas_contrataciones")
async def consultar_contrataciones_json(pregunta: str) -> SQLRowsResponse:  # noqa: C901
    """Devuelve el resultado crudo (encabezados y filas) de una consulta sobre contrataciones públicas."""
    try:
        # Ejecutamos el grafo para obtener la consulta SQL generada
        result = await agent.ainvoke({"messages": [{"role": "user", "content": pregunta}]})

        # Localizamos la última llamada a la herramienta sql_db_query para extraer la consulta
        sql_query: str | None = None
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and message.tool_calls:
                for tc in message.tool_calls:
                    if tc.get("name") == "sql_db_query":
                        sql_query = tc["args"]["query"]
                        break
            if sql_query:
                break

        if not sql_query:
            return {"query": "", "columns": [], "rows": []}

        try:
            # Ejecutamos la consulta directamente con SQLAlchemy para obtener encabezados
            with db._engine.connect() as conn:
                result_proxy = conn.execute(text(sql_query))
                columns = list(result_proxy.keys())
                rows = [[_serialize(v) for v in row] for row in result_proxy.fetchall()]

            return {"query": sql_query, "columns": columns, "rows": rows}
        except Exception as exc:
            # Fallback: intentamos usar el contenido ya devuelto por la tool sql_db_query
            import ast

            for message in reversed(result["messages"]):
                if isinstance(message, ToolMessage) and message.tool in ["sql_db_query", None]:
                    raw_content = message.content.strip()
                    try:
                        parsed_rows = ast.literal_eval(raw_content)
                        if isinstance(parsed_rows, list):
                            return {
                                "query": sql_query,
                                "columns": [],  # No tenemos encabezados fiables
                                "rows": [[_serialize(v) for v in row] for row in parsed_rows],
                            }
                    except Exception:  # noqa: BLE001
                        continue

            # Si todo falla, informamos del error preservando el contrato
            return {"query": sql_query, "columns": [], "rows": [[f"Error: {str(exc)}"]]}
    except Exception as exc:  # pylint: disable=broad-except
        # Capturamos cualquier otro error inesperado
        return {"query": "", "columns": [], "rows": [[f"Error: {str(exc)}"]]}


class TablasResponse(TypedDict):
    tablas: List[str]


@tool("obtener_tablas_contrataciones")
async def obtener_tablas_contrataciones() -> TablasResponse:
    """Retorna las tablas disponibles en la base de datos de contrataciones públicas."""
    try:
        return {"tablas": db.get_usable_table_names()}
    except Exception as e:
        return {"tablas": [f"Error al obtener tablas: {str(e)}"]}


tools_list = [consultar_contrataciones_json, obtener_tablas_contrataciones]
