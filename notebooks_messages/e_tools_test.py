# %%
import random
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Tuple
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

memory = MemorySaver()


class ChatState(TypedDict):
    raw_messages: Annotated[list, add_messages]
    messages: Annotated[list, add_messages]
    topic_decision: str


# %% [markdown]
# # 1. Tools


# %%
@tool
def buscar_asistencias_congresistas(fecha: str) -> str:
    """
    Busca las asistencias de los congresistas en una fecha específica.

    Args:
        fecha: Fecha en formato YYYY-MM-DD

    Returns:
        Información sobre las asistencias de los congresistas
    """
    # Simulación de datos
    asistencias_simuladas = {
        "fecha": fecha,
        "total_congresistas": 130,
        "presentes": random.randint(100, 120),
        "ausentes": random.randint(5, 15),
        "licencias": random.randint(5, 15),
        "detalles": [
            {"congresista": "Juan Pérez", "partido": "APP", "asistencia": "PRESENTE"},
            {"congresista": "María García", "partido": "FP", "asistencia": "AUSENTE"},
            {"congresista": "Carlos López", "partido": "PL", "asistencia": "LICENCIA"},
        ],
    }

    return f"""
    Asistencias del {fecha}:
    - Total congresistas: {asistencias_simuladas['total_congresistas']}
    - Presentes: {asistencias_simuladas['presentes']}
    - Ausentes: {asistencias_simuladas['ausentes']}
    - Con licencia: {asistencias_simuladas['licencias']}
    
    Algunos detalles:
    {chr(10).join([f"- {d['congresista']} ({d['partido']}): {d['asistencia']}" for d in asistencias_simuladas['detalles'][:3]])}
    """


@tool
def buscar_votaciones_congreso(fecha: str) -> str:
    """
    Busca las votaciones realizadas en el Congreso en una fecha específica.

    Args:
        fecha: Fecha en formato YYYY-MM-DD

    Returns:
        Información sobre las votaciones del Congreso
    """
    votaciones_simuladas = [
        {
            "proyecto": "PL 1234/2022-CR - Ley de Reforma Educativa",
            "a_favor": random.randint(60, 80),
            "en_contra": random.randint(30, 50),
            "abstenciones": random.randint(5, 15),
        },
        {
            "proyecto": "PL 5678/2022-CR - Ley de Presupuesto 2023",
            "a_favor": random.randint(70, 90),
            "en_contra": random.randint(20, 40),
            "abstenciones": random.randint(5, 10),
        },
    ]

    resultado = f"Votaciones del {fecha}:\n\n"
    for v in votaciones_simuladas:
        resultado += f"""
        Proyecto: {v['proyecto']}
        - A favor: {v['a_favor']}
        - En contra: {v['en_contra']}
        - Abstenciones: {v['abstenciones']}
        - Resultado: {'APROBADO' if v['a_favor'] > v['en_contra'] else 'RECHAZADO'}
        """

    return resultado


@tool
def buscar_contratos_empresa(nombre_empresa: str, monto_minimo: float = 0) -> str:
    """
    Busca contratos de una empresa específica con el Estado peruano.

    Args:
        nombre_empresa: Nombre de la empresa a buscar
        monto_minimo: Monto mínimo de contrato a filtrar (opcional)

    Returns:
        Información sobre los contratos de la empresa
    """
    # Simulación de contratos
    contratos_simulados = [
        {
            "numero": "001-2023-MINSA",
            "entidad": "Ministerio de Salud",
            "monto": random.randint(100000, 5000000),
            "objeto": "Suministro de equipos médicos",
            "fecha": "2023-03-15",
        },
        {
            "numero": "045-2023-MINEDU",
            "entidad": "Ministerio de Educación",
            "monto": random.randint(200000, 3000000),
            "objeto": "Construcción de aulas",
            "fecha": "2023-06-20",
        },
    ]

    contratos_filtrados = [c for c in contratos_simulados if c["monto"] >= monto_minimo]

    resultado = f"Contratos de {nombre_empresa.upper()}:\n\n"
    total = 0
    for c in contratos_filtrados:
        resultado += f"""
        Contrato N° {c['numero']}
        - Entidad: {c['entidad']}
        - Monto: S/ {c['monto']:,.2f}
        - Objeto: {c['objeto']}
        - Fecha: {c['fecha']}
        """
        total += c["monto"]

    resultado += f"\nTotal contratado: S/ {total:,.2f}"
    return resultado


@tool
def buscar_presupuesto_entidad(entidad: str, año: int) -> str:
    """
    Busca información del presupuesto de una entidad pública.

    Args:
        entidad: Nombre de la entidad pública
        año: Año del presupuesto

    Returns:
        Información presupuestaria de la entidad
    """
    presupuesto_simulado = {
        "PIA": random.randint(50000000, 200000000),  # Presupuesto Institucional de Apertura
        "PIM": random.randint(60000000, 250000000),  # Presupuesto Institucional Modificado
        "ejecutado": random.randint(40000000, 180000000),
        "porcentaje_ejecucion": random.uniform(70, 95),
    }

    return f"""
    Presupuesto de {entidad} - Año {año}:
    
    - Presupuesto Inicial (PIA): S/ {presupuesto_simulado['PIA']:,.2f}
    - Presupuesto Modificado (PIM): S/ {presupuesto_simulado['PIM']:,.2f}
    - Monto Ejecutado: S/ {presupuesto_simulado['ejecutado']:,.2f}
    - % de Ejecución: {presupuesto_simulado['porcentaje_ejecucion']:.1f}%
    
    Principales rubros de gasto:
    - Personal y obligaciones: 45%
    - Bienes y servicios: 30%
    - Inversiones: 25%
    """


@tool
def buscar_web_transparencia(query: str) -> str:
    """
    Busca información en la web sobre transparencia gubernamental peruana.

    Args:
        query: Consulta de búsqueda

    Returns:
        Resultados de búsqueda web sobre transparencia
    """
    # Simulación de resultados web
    resultados = [
        f"Portal de Transparencia: Información actualizada sobre {query}",
        f"SEACE: Últimas contrataciones relacionadas con {query}",
        f"Contraloría: Informes de auditoría sobre {query}",
    ]

    return f"""
    Resultados de búsqueda web para: "{query}"
    
    {chr(10).join([f"{i+1}. {r}" for i, r in enumerate(resultados)])}
    
    Fuentes consultadas: Portal de Transparencia, SEACE, Contraloría General
    """


# Lista de todas las herramientas
transparency_tools = [
    buscar_asistencias_congresistas,
    buscar_votaciones_congreso,
    buscar_contratos_empresa,
    buscar_presupuesto_entidad,
    buscar_web_transparencia,
]

# %% [markdown]
# # 2. Model

# %%
model_rewriter = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0.0)
model_rewriter_fallback = init_chat_model("openai:gpt-4o", temperature=0.0)
llm_rewriter = model_rewriter.with_fallbacks([model_rewriter_fallback])

model_classifier = init_chat_model("anthropic:claude-3-haiku-20240307", temperature=0.0)
model_classifier_fallback = init_chat_model("openai:gpt-4o", temperature=0.0)
llm_classifier = model_classifier.with_fallbacks([model_classifier_fallback])

model_main = init_chat_model("anthropic:claude-sonnet-4-20250514", temperature=0.5, streaming=True)
model_main_fallback = init_chat_model("openai:gpt-4o", temperature=0.5, streaming=True)
llm_main = model_main.with_fallbacks([model_main_fallback])

model_fallback = init_chat_model(
    "anthropic:claude-3-haiku-20240307", temperature=0.0, streaming=True
)
model_fallback_fallback = init_chat_model("openai:gpt-4o", temperature=0.0, streaming=True)
llm_fallback = model_fallback.with_fallbacks([model_fallback_fallback])


# %%
def get_context_for_classification(messages, strategy="last", max_chars=1000):
    """
    Trim rápido usando caracteres como proxy de tokens
    Estimación: ~4 caracteres por token en español
    """
    if len(messages) <= 5:
        return messages

    # Función inline para contar caracteres
    def count_chars(msgs):
        return sum(len(msg.content) for msg in msgs)

    trimmer = trim_messages(
        max_tokens=max_chars,
        strategy=strategy,
        token_counter=count_chars,
        include_system=False,
        allow_partial=False,
        start_on="human",
    )

    return trimmer.invoke(messages)


def format_history_context(
    messages: List, max_chars: int = 200, exclude_last: bool = True, last_n: int = 5
) -> str:
    """
    Formatea el historial en pares Q: A: excluyendo la última pregunta.
    - Agrupa en pares (HumanMessage, AIMessage).
    - Muestra pregunta completa con prefijo Q:.
    - Trunca cada respuesta a max_chars con prefijo A: y añade '…'.
    - Cada par en una línea separada.
    - Solo incluye los últimos last_n pares.
    """
    pairs: List[Tuple[HumanMessage, Optional[AIMessage]]] = []
    pending_human = None

    # Si exclude_last=True, no procesamos el último mensaje
    messages_to_process = messages[:-1] if exclude_last and messages else messages

    for msg in messages_to_process:
        if isinstance(msg, HumanMessage):
            # Si ya había un HumanMessage pendiente, lo guardamos sin respuesta
            if pending_human is not None:
                pairs.append((pending_human, None))
            pending_human = msg
        elif isinstance(msg, AIMessage) and pending_human is not None:
            pairs.append((pending_human, msg))
            pending_human = None

    # Guardar el último HumanMessage si quedó pendiente
    if pending_human is not None:
        pairs.append((pending_human, None))

    # Tomar solo los últimos last_n pares
    recent_pairs = pairs[-last_n:] if last_n > 0 else pairs

    lines = []
    for human, ai in recent_pairs:
        # Pregunta completa con prefijo Q:
        q = human.content.strip()
        lines.append(f"Q: {q}")

        # Respuesta truncada con prefijo A: (si existe)
        if ai is not None:
            a = ai.content.strip()
            if len(a) > max_chars:
                # Buscar el último espacio antes del límite para no cortar palabras
                cut_point = a.rfind(" ", 0, max_chars)
                if cut_point == -1:
                    cut_point = max_chars
                snippet = a[:cut_point].rstrip() + "…"
            else:
                snippet = a
            lines.append(f"A: {snippet}")
        else:
            # Si no hay respuesta, indicamos que no hay respuesta
            lines.append("A: [Pendiente de responder...]")

        # Línea vacía entre pares para separar
        lines.append("")

    return "\n".join(lines).strip()


def get_last_question(messages: List) -> str:
    """
    Obtiene la última pregunta del usuario de la lista de mensajes.
    """
    if not messages:
        return ""

    # Buscar el último HumanMessage
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()

    return ""


# %%
rewriter_msg = SystemMessage(
    content=(
        "Eres un asistente especializado en reescribir preguntas para alinearlas con el contexto de transparencia gubernamental del Estado peruano. "
        "Tu objetivo es transformar las preguntas del usuario en consultas más claras, formales y orientadas a la fiscalización y el acceso a información pública."
        "INSTRUCCIONES IMPORTANTES:"
        "1. Solo reescribe preguntas relacionadas con los temas específicos listados abajo."
        "2. Si la pregunta NO está relacionada con estos temas, devuélvela EXACTAMENTE sin cambios."
        "3. Tu respuesta debe ser ÚNICAMENTE una pregunta reformulada, NUNCA una respuesta o explicación."
        "4. NUNCA generes respuestas largas o explicaciones, solo reformula la pregunta."
        "TEMAS PARA REESCRIBIR:"
        "- Contrataciones públicas (montos, órdenes de servicio, contratos, proveedores)"
        "- Empresas que han contratado con el Estado peruano"
        "- Asistencia y votaciones de congresistas"
        "- Información relacionada a congresistas (identidad, región, actividad legislativa)"
        "Ejemplos:"
        "Entrada: quien es alejando muñante"
        "Salida: Busca en la web información sobre el congresista ALEJANDRO MUÑANTE."
        "Entrada: quien es Sucel Paredes"
        "Salida: Busca en la web información sobre la congresista SUCEL PAREDES."
        "Entrada: quienes son los congresistas de la region de huancayo"
        "Salida: Busca en la web información sobre los congresistas de la región de HUANCAYO."
        "Entrada: dame las asistencias del 2022 octubre"
        "Salida: ¿Cuáles fueron las asistencias de los congresistas en octubre de 2022?"
        "Entrada: puedes darme las asistencias del 10 de diciembre del 2022"
        "Salida: ¿Cuáles fueron las asistencias de los congresistas el 2022-12-10?"
        "Entrada: puedes decirme las votaciones del congreso del 10 de diciembre del 2022"
        "Salida: ¿Cuáles fueron las votaciones de los congresistas el 2022-12-10?"
        "Entrada: que asuntos se trataron en el congreso del 10 de diciembre del 2022"
        "Salida: ¿Cuáles fueron los asuntos tratados en las votaciones del 2022-12-10?"
        "Entrada: cuánto ha contratado constructora alfa"
        "Salida: ¿Cuánto ha contratado la empresa 'CONSTRUCTORA ALFA' con el Estado peruano según transparencia pública?"
        "Entrada: detalles de los contratos de constructora alfa"
        "Salida: ¿Cuáles son los detalles de los contratos de la empresa 'CONSTRUCTORA ALFA'?"
        "Entrada: detalles de las ordenes de servicio de constructora alfa"
        "Salida: ¿Cuáles son los detalles de las órdenes de servicio de la empresa 'CONSTRUCTORA ALFA'?"
        "Entrada: que mas puedes hacer"
        "Salida: que mas puedes hacer"
        "Entrada: Que más me puedes decir?"
        "Salida: Que más me puedes decir?"
        "Entrada: Me gustan los duraznos"
        "Salida: Me gustan los duraznos"
        "Entrada: Quien ganó la champions league"
        "Salida: Quien ganó la champions league"
    )
)

main_system_msg = SystemMessage(
    content=(
        "Eres un asistente especializado en transparencia gubernamental del Estado peruano. "
        "Tienes acceso a herramientas para consultar información sobre:"
        "- Asistencias y votaciones de congresistas"
        "- Contratos y contrataciones públicas"
        "- Presupuestos de entidades públicas"
        "- Información general de transparencia"
        "Usa las herramientas disponibles para responder las consultas del usuario de manera precisa y completa. "
        "Siempre proporciona información factual y verificable."
    )
)

fallback_system_msg = SystemMessage(
    content=(
        "Eres un asistente cordial y profesional. Aunque no puedes responder preguntas "
        "fuera del dominio de transparencia gubernamental del Estado peruano, debes explicar "
        "educadamente cuál es tu función y sugerir temas válidos, como contrataciones públicas o votaciones del Congreso. "
        "Si el usuario simplemente saluda, responde con cortesía e invita a hacer una consulta sobre esos temas."
        "Considera responder con emojis"
    )
)


# %%
def rewrite_node(state: ChatState) -> ChatState:
    """Reescribe la última pregunta del usuario con llm_rewriter."""

    print("\n" + "=" * 80)
    print("🔄 PASO 1: REWRITER")
    print("=" * 80)

    last_user_msg: HumanMessage = state["raw_messages"][-1]  # asumimos último mensaje = usuario
    print(f"📝 Pregunta original: {last_user_msg.content}")

    rewritten = llm_rewriter.invoke([rewriter_msg, last_user_msg])

    # Validación: si la respuesta es muy larga (más de 300 chars), usa el mensaje original
    rewritten_content = rewritten.content.strip()
    if len(rewritten_content) > 3 * len(last_user_msg.content):
        print(
            f"⚠️  WARNING: Rewriter generó respuesta muy larga ({len(rewritten_content)} chars), usando mensaje original"
        )
        rewritten_content = last_user_msg.content
        print(f"✏️  Pregunta reescrita: {rewritten_content} (sin cambios)")
    else:
        print(f"✏️  Pregunta reescrita: {rewritten_content}")
        if rewritten_content == last_user_msg.content:
            print("📌 (Sin cambios - pregunta no relacionada con transparencia)")

    # Añadimos la versión reescrita como HumanMessage (para mantener formato)
    return {
        "raw_messages": state["raw_messages"],
        "messages": state["messages"] + [HumanMessage(content=rewritten_content)],
        "topic_decision": state.get("topic_decision", ""),
    }


def classifier_node(state: ChatState) -> ChatState:
    """Clasifica si la conversación está relacionada con transparencia gubernamental."""

    print("\n" + "=" * 80)
    print("🔍 PASO 2: CLASSIFIER")
    print("=" * 80)

    # Usamos `messages` (el output del rewriter) para el contexto
    msgs_for_context = state["messages"]

    # Obtener contexto histórico (excluyendo la última pregunta)
    history_context = format_history_context(
        msgs_for_context, max_chars=150, exclude_last=True, last_n=3
    )

    # Obtener la última pregunta
    last_question = get_last_question(msgs_for_context)

    print(f"💭 Contexto histórico:")
    if history_context.strip():
        for line in history_context.split("\n"):
            if line.strip():
                print(f"   {line}")
    else:
        print("   (Sin contexto previo)")

    print(f"❓ Pregunta a clasificar: {last_question}")

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

    classification = llm_classifier.invoke([HumanMessage(content=prompt)])
    decision = classification.content.strip().upper()
    if decision not in ("YES", "NO"):
        decision = "NO"

    if decision == "YES":
        print(f"✅ DECISIÓN: {decision} → Ir a MAIN (transparencia gubernamental)")
    else:
        print(f"❌ DECISIÓN: {decision} → Ir a FALLBACK (fuera del dominio)")

    return {
        "raw_messages": state["raw_messages"],
        "messages": state["messages"],
        "topic_decision": decision,
    }


def main_node(state: ChatState) -> ChatState:
    """Nodo principal que maneja consultas de transparencia gubernamental."""

    # Verificar si es la primera vez o regresa de tools
    has_tool_messages = any(isinstance(msg, ToolMessage) for msg in state["messages"])

    if has_tool_messages:
        print("\n" + "=" * 80)
        print("🎯 PASO 4: MAIN NODE (RESPUESTA FINAL)")
        print("=" * 80)
        print("📤 Generando respuesta final")
    else:
        print("\n" + "=" * 80)
        print("🎯 PASO 3: MAIN NODE")
        print("=" * 80)

    # Obtener el historial completo para contexto
    conversation_history = state["messages"]
    print(f"🔧 Herramientas disponibles: {len(transparency_tools)} herramientas de transparencia")

    # Crear mensajes para el LLM principal
    messages_for_llm = [main_system_msg] + conversation_history

    # Invocar el LLM principal con las herramientas bound
    llm_with_tools = llm_main.bind_tools(transparency_tools)
    response = llm_with_tools.invoke(messages_for_llm)

    # Log de la respuesta
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🛠️  Ejecutando {len(response.tool_calls)} herramienta(s)")
    else:
        print("💬 LLM generó respuesta directa (sin herramientas)")
        if has_tool_messages:
            print("\n" + "=" * 80)
            print("✅ RESPUESTA FINAL")
            print("=" * 80)
            print(f"📋 {response.content}")
            print("=" * 80)
        else:
            print(f"📄 Respuesta: {response.content[:100]}...")

    return {
        "raw_messages": state["raw_messages"],
        "messages": state["messages"] + [response],
        "topic_decision": state["topic_decision"],
    }


def fallback_node(state: ChatState) -> ChatState:
    """Nodo de respaldo para consultas generales."""

    print("\n" + "=" * 80)
    print("🔄 PASO 3: FALLBACK NODE")
    print("=" * 80)
    print("📋 Pregunta fuera del dominio de transparencia gubernamental")

    # Crear mensajes para el LLM de respaldo
    messages_for_llm = [fallback_system_msg] + [state["messages"][-1]]

    # Invocar el LLM de respaldo
    response = llm_fallback.invoke(messages_for_llm)

    print(f"💬 Respuesta de cortesía: {response.content[:100]}...")

    return {
        "raw_messages": state["raw_messages"],
        "messages": state["messages"] + [response],
        "topic_decision": state["topic_decision"],
    }


def route_after_classifier(state: ChatState) -> str:
    """Función de enrutamiento después del clasificador."""
    if state["topic_decision"] == "YES":
        return "main"
    else:
        return "fallback"


# %%
# Construcción del grafo
graph = StateGraph(ChatState)

# Agregar nodos
graph.add_node("rewriter", rewrite_node)
graph.add_node("classifier", classifier_node)
graph.add_node("main", main_node)
graph.add_node("tools", ToolNode(transparency_tools))
graph.add_node("fallback", fallback_node)

# Definir el flujo
graph.add_edge(START, "rewriter")
graph.add_edge("rewriter", "classifier")

# Enrutamiento condicional después del clasificador
graph.add_conditional_edges(
    "classifier", route_after_classifier, {"main": "main", "fallback": "fallback"}
)

# En el nodo main, usar tools_condition para decidir si usar herramientas
graph.add_conditional_edges("main", tools_condition)

# Después de usar herramientas, volver al main para respuesta final
graph.add_edge("tools", "main")

# Fallback termina directamente
graph.add_edge("fallback", END)

# Compilar el grafo
enhanced_processor = graph.compile(checkpointer=memory)


# %%
def chat_interactive():
    """Función para chat interactivo desde consola."""
    import uuid

    session_id = str(uuid.uuid4())[:8]
    print("🤖 Asistente de Transparencia Gubernamental del Estado Peruano")
    print("💡 Puedes preguntar sobre: contrataciones públicas, congresistas, presupuestos")
    print("🚪 Escribe 'salir' para terminar\n")

    while True:
        try:
            user_input = input("👤 Tu pregunta: ").strip()

            if user_input.lower() in ["salir", "exit", "quit", "bye"]:
                print("👋 ¡Hasta luego!")
                break

            if not user_input:
                continue

            print("🔄 Procesando...")

            # Procesar con el enhanced_processor
            state = enhanced_processor.invoke(
                {
                    "raw_messages": [HumanMessage(content=user_input)],
                    "topic_decision": "",
                },
                config={"thread_id": session_id},
            )

            # Obtener la respuesta final
            final_response = state["messages"][-1].content
            print(f"\n🤖 Respuesta:\n{final_response}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n👋 ¡Chat interrumpido!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Intenta de nuevo...")


def run_tests():
    """Ejecuta los tests de ejemplo."""
    session_id = "abc123"

    # Test 1: Pregunta sobre transparencia gubernamental (debe ir a main)
    print("*" * 80)
    print("=== Test 1: Pregunta sobre contrataciones ===")
    print("*" * 80)
    state = enhanced_processor.invoke(
        {
            "raw_messages": [
                HumanMessage(content="Que empresas tienen contratos por más de 1000000 de soles")
            ],
            "topic_decision": "",
        },
        config={"thread_id": session_id},
    )
    print()

    # Test 2: Pregunta general (debe ir a fallback)
    print("*" * 80)
    print("=== Test 2: Pregunta general ===")
    print("*" * 80)
    state = enhanced_processor.invoke(
        {
            "raw_messages": [HumanMessage(content="Quien ganó la Champions League")],
            "topic_decision": "",
        },
        config={"thread_id": session_id},
    )
    print()

    # Test 3: Pregunta sobre asistencias (debe ir a main)
    print("*" * 80)
    print("=== Test 3: Pregunta sobre asistencias ===")
    print("*" * 80)
    state = enhanced_processor.invoke(
        {
            "raw_messages": [
                HumanMessage(content="dame las asistencias del 1 de diciembre del 2022")
            ],
            "topic_decision": "",
        },
        config={"thread_id": session_id},
    )
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("🧪 Ejecutando tests... python3 e_tools_test.py test")
        run_tests()
    else:
        chat_interactive()
