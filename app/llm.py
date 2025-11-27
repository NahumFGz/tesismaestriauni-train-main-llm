"""
Módulo principal del agente conversacional LLM.

Este módulo contiene la lógica principal del agente conversacional basado en LangGraph,
incluyendo los nodos del grafo, la clasificación de consultas, y las funciones de
ejecución tanto síncronas como con streaming.

Responsabilidades:
- Definir y construir el grafo conversacional
- Procesar consultas de transparencia gubernamental
- Manejar el streaming de respuestas
- Gestionar el estado de la conversación
"""

# %%
import asyncio
from typing import Annotated
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

# Utilidades y prompts propios de la aplicación
from app.prompts import fallback_system_msg, main_system_msg, rewriter_msg

# Importar módulos de herramientas individuales
from app.tools import tools_attendance, tools_local, tools_procurement, tools_voting
from app.utils import format_history_context, get_last_question

# ============================================================================
# ESTADO DEL CHAT
# ============================================================================


class ChatState(TypedDict):
    """
    Estado del chat que se pasa entre nodos del grafo.

    Attributes:
        raw_messages: Mensajes originales del usuario sin procesar
        messages: Mensajes procesados y reescritos
        topic_decision: Decisión del clasificador ('YES' o 'NO')
    """

    raw_messages: Annotated[list, add_messages]
    messages: Annotated[list, add_messages]
    topic_decision: str


# ============================================================================
# CONFIGURACIÓN DE MODELOS LLM
# ============================================================================

# Modelo para reescritura de consultas
model_rewriter = init_chat_model("openai:gpt-4o", temperature=0.0)
model_rewriter_fallback = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0.0)
llm_rewriter = model_rewriter.with_fallbacks([model_rewriter_fallback])

# Modelo para clasificación de consultas
model_classifier = init_chat_model("openai:gpt-4o", temperature=0.0)
model_classifier_fallback = init_chat_model("anthropic:claude-3-haiku-20240307", temperature=0.0)
llm_classifier = model_classifier.with_fallbacks([model_classifier_fallback])

# Modelo principal para consultas de transparencia
model_main = init_chat_model("openai:gpt-4o", temperature=0.5)
model_main_fallback = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0.5)
llm_main = model_main.with_fallbacks([model_main_fallback])

# Modelo de respaldo para consultas generales
model_fallback = init_chat_model("openai:gpt-4o", temperature=0.0)
model_fallback_fallback = init_chat_model("anthropic:claude-3-haiku-20240307", temperature=0.0)
llm_fallback = model_fallback.with_fallbacks([model_fallback_fallback])


# ============================================================================
# NODOS DEL GRAFO
# ============================================================================


async def rewrite_node(state: ChatState) -> ChatState:
    """
    Nodo que reescribe la última pregunta del usuario para mejorar claridad.

    Args:
        state: Estado actual del chat

    Returns:
        ChatState: Estado actualizado con mensaje reescrito
    """
    last_user_msg: HumanMessage = state["raw_messages"][-1]

    # Reescribir la consulta con el LLM especializado
    rewritten = await llm_rewriter.ainvoke([rewriter_msg, last_user_msg])

    # Validación: evitar respuestas excesivamente largas
    rewritten_content = rewritten.content.strip()
    if len(rewritten_content) > 3 * len(last_user_msg.content):
        rewritten_content = last_user_msg.content

    return {
        "raw_messages": state["raw_messages"],
        "messages": state["messages"] + [HumanMessage(content=rewritten_content)],
        "topic_decision": state.get("topic_decision", ""),
    }


async def classifier_node(state: ChatState) -> ChatState:
    """
    Nodo que clasifica si la consulta está relacionada con transparencia gubernamental.

    Args:
        state: Estado actual del chat

    Returns:
        ChatState: Estado actualizado con decisión de clasificación
    """
    msgs_for_context = state["messages"]

    # Obtener contexto histórico
    history_context = format_history_context(
        msgs_for_context, max_chars=150, exclude_last=True, last_n=4
    )

    # Obtener la última pregunta
    last_question = get_last_question(msgs_for_context)

    # Prompt de clasificación
    prompt = f"""
    Eres un verificador que decide si la última pregunta del usuario puede ser respondida en el contexto de transparencia gubernamental del Estado peruano.

    TEMAS RELEVANTES:
    - Contrataciones públicas (montos, órdenes de servicio, contratos, proveedores)
    - Empresas que han contratado con el Estado peruano
    - Asistencia y votaciones de congresistas
    - Información relacionada a congresistas (identidad, región, actividad legislativa)
    - Transparencia y fiscalización gubernamental en general

    INSTRUCCIONES:
    - Si el contexto histórico muestra que el usuario estuvo hablando de los TEMAS RELEVANTES que se muestran arriba, responde 'YES'.
    - Si la última pregunta es sobre un tema totalmente distinto al contexto histórico, responde 'NO'.
    - Si no hay contexto histórico, evalúa solo si la última pregunta es sobre los TEMAS RELEVANTES.

    Solo responde con 'YES' o 'NO' (sin explicaciones ni comentarios adicionales).

    CONTEXTO HISTÓRICO:
    {history_context}

    ÚLTIMA PREGUNTA:
    {last_question}

    ¿Se puede responder la última pregunta en el contexto de los TEMAS RELEVANTES?
    """

    # Clasificar la consulta
    classification = await llm_classifier.ainvoke([HumanMessage(content=prompt)])
    decision = classification.content.strip().upper()
    if decision not in ("YES", "NO"):
        decision = "NO"

    return {
        "raw_messages": state["raw_messages"],
        "messages": state["messages"],
        "topic_decision": decision,
    }


async def chatbot_node(state: ChatState) -> ChatState:
    """
    Nodo principal que maneja consultas de transparencia gubernamental.

    Args:
        state: Estado actual del chat

    Returns:
        ChatState: Estado actualizado con respuesta del chatbot
    """
    # Obtener herramientas de transparencia
    tools = get_transparency_tools()

    # Preparar mensajes para el LLM
    conversation_history = state["messages"]
    messages_for_llm = [main_system_msg] + conversation_history

    # Invocar LLM con herramientas disponibles
    llm_with_tools = llm_main.bind_tools(tools)
    response = await llm_with_tools.ainvoke(messages_for_llm)

    return {
        "raw_messages": state["raw_messages"],
        "messages": state["messages"] + [response],
        "topic_decision": state["topic_decision"],
    }


async def fallback_node(state: ChatState) -> ChatState:
    """
    Nodo de respaldo para consultas que no son de transparencia gubernamental.

    Args:
        state: Estado actual del chat

    Returns:
        ChatState: Estado actualizado con respuesta de fallback
    """
    # Usar solo el último mensaje para el fallback
    messages_for_llm = [fallback_system_msg] + [state["messages"][-1]]

    # Respuesta con LLM de fallback
    response = await llm_fallback.ainvoke(messages_for_llm)

    return {
        "raw_messages": state["raw_messages"],
        "messages": state["messages"] + [response],
        "topic_decision": state["topic_decision"],
    }


def route_after_classifier(state: ChatState) -> str:
    """
    Función de enrutamiento que decide el flujo después de la clasificación.

    Args:
        state: Estado actual del chat

    Returns:
        str: Nombre del siguiente nodo ('chatbot' o 'fallback')
    """
    if state["topic_decision"] == "YES":
        return "chatbot"
    else:
        return "fallback"


# ============================================================================
# CONSTRUCCIÓN Y GESTIÓN DEL GRAFO
# ============================================================================

# Variable global para el grafo compilado
compiled_graph = None


async def build_graph():
    """
    Construye y compila el grafo del agente conversacional.

    Estructura del grafo:
    START → rewriter → classifier → [chatbot | fallback] → END
                                      ↓ ↑
                                    tools

    Returns:
        CompiledGraph: Grafo compilado listo para procesar consultas
    """
    # Inicializar memoria en memoria (puede reemplazarse por otro checkpointer si se desea persistencia en BD)
    memory_saver = MemorySaver()

    # Obtener las herramientas de transparencia
    tools = get_transparency_tools()

    # Crear el grafo
    graph = StateGraph(ChatState)

    # Agregar nodos
    graph.add_node("rewriter", rewrite_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("fallback", fallback_node)

    # Definir flujo principal
    graph.add_edge(START, "rewriter")
    graph.add_edge("rewriter", "classifier")

    # Enrutamiento condicional después del clasificador
    graph.add_conditional_edges(
        "classifier", route_after_classifier, {"chatbot": "chatbot", "fallback": "fallback"}
    )

    # Flujo del chatbot con herramientas
    graph.add_conditional_edges("chatbot", tools_condition)
    graph.add_edge("tools", "chatbot")

    # Flujo del fallback termina directamente
    graph.add_edge("fallback", END)

    # Compilar con memoria en memoria
    return graph.compile(checkpointer=memory_saver)


async def get_graph():
    """
    Obtiene o inicializa el grafo compilado (patrón singleton).

    Returns:
        CompiledGraph: Instancia del grafo compilado
    """
    global compiled_graph
    if compiled_graph is None:
        compiled_graph = await build_graph()
    return compiled_graph


# ============================================================================
# FUNCIONES PÚBLICAS DE EJECUCIÓN
# ============================================================================


async def run(query: str, thread_id: str = None):
    """
    Ejecuta una consulta en el grafo con memoria persistente.

    Args:
        query: La consulta del usuario
        thread_id: ID del hilo para la memoria. Si es None, se genera uno nuevo

    Returns:
        dict: Diccionario con la respuesta y metadatos
            - response: Respuesta del agente
            - thread_id: ID del hilo de conversación
    """
    if thread_id is None:
        thread_id = str(uuid4())

    # Configuración para la memoria
    config = {"configurable": {"thread_id": thread_id}}

    # Preparar estado inicial
    input_state = {
        "raw_messages": [HumanMessage(content=query)],
        "messages": [],
        "topic_decision": "",
    }

    # Ejecutar el grafo
    graph = await get_graph()
    result = await graph.ainvoke(input_state, config)

    # Extraer la respuesta del último mensaje del asistente
    response_message = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            response_message = msg.content
            break

    return {
        "response": response_message,
        "thread_id": thread_id,
    }


async def run_stream(query: str, thread_id: str = None):
    """
    Ejecuta una consulta con streaming de la respuesta.

    Args:
        query: La consulta del usuario
        thread_id: ID del hilo para la memoria. Si es None, se genera uno nuevo

    Yields:
        dict: Diccionarios con información del streaming:
            - thread_id: ID del hilo para la memoria
            - token: Token individual (cadena vacía cuando is_complete=True)
            - is_complete: True cuando el streaming ha terminado
            - full_message: Mensaje completo (solo cuando is_complete=True)
            - node: Nodo que generó la respuesta ("chatbot" o "fallback")
    """
    if thread_id is None:
        thread_id = str(uuid4())

    # Configuración para la memoria
    config = {"configurable": {"thread_id": thread_id}}

    # Preparar estado inicial
    input_state = {
        "raw_messages": [HumanMessage(content=query)],
        "messages": [],
        "topic_decision": "",
    }

    full_message = ""
    response_node = ""

    # Hacer streaming del grafo
    graph = await get_graph()
    async for result in graph.astream(input_state, config, stream_mode="messages"):
        if isinstance(result, tuple):
            message_chunk, metadata = result
            nodo = metadata.get("langgraph_node")

            # Solo hacer streaming de los nodos finales
            if nodo in ["chatbot", "fallback"]:
                # Procesar contenido del mensaje
                token = ""
                if isinstance(message_chunk.content, str):
                    token = message_chunk.content
                elif isinstance(message_chunk.content, list):
                    # Si es una lista, extraer solo el contenido de texto, ignorando tool_use
                    for item in message_chunk.content:
                        if isinstance(item, dict):
                            # Solo extraer contenido de tipo 'text'
                            if item.get("type") == "text":
                                token += item.get("text", "")
                        elif isinstance(item, str):
                            token += item
                        # Ignorar elementos de tipo 'tool_use' o similares
                else:
                    token = str(message_chunk.content)

                # Solo hacer yield si hay contenido de texto real
                if token:
                    full_message += token
                    response_node = nodo

                    # Yield del token individual
                    yield {
                        "thread_id": thread_id,
                        "token": token,
                        "is_complete": False,
                        "full_message": "",
                        "node": nodo,
                    }

    # Yield final con el mensaje completo
    yield {
        "thread_id": thread_id,
        "token": "",
        "is_complete": True,
        "full_message": full_message,
        "node": response_node,
    }


# ============================================================================
# LIMPIEZA DE RECURSOS
# ============================================================================


async def cleanup_llm_resources():
    """
    Limpia los recursos específicos del módulo LLM (grafo compilado).

    Esta función libera la referencia al grafo compilado, permitiendo
    que sea recolectado por el garbage collector.
    """
    global compiled_graph

    try:
        if compiled_graph is not None:
            compiled_graph = None
            print("🧠 Grafo compilado liberado")
    except Exception as e:
        print(f"⚠️  Error al limpiar recursos del LLM: {e}")


def get_transparency_tools():
    """Devuelve la lista de herramientas disponibles de transparencia.

    Se compone de la unión de las listas `tools_list` definidas en cada módulo
    `tools_*` más las utilidades locales.
    """

    return (
        tools_voting.tools_list
        + tools_attendance.tools_list
        + tools_procurement.tools_list
        + tools_local.tools_local_list
    )
