# Cargar variables de entorno desde un archivo .env si existe
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "Advertencia: python-dotenv no está instalado. Las variables de entorno del archivo .env no se cargarán."
    )


import json
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    """Estado del grafo que contiene los mensajes y datos de procesamiento"""

    messages: Annotated[list, add_messages]
    query: str
    context: str
    answer: str
    item_id: int


def generate_answer_node(state: State) -> Dict[str, Any]:
    """
    Nodo único de LangGraph que genera respuestas basadas en el contexto de Qdrant
    y la pregunta proporcionada.
    """
    # Configurar el modelo OpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Crear el prompt para generar respuestas con contexto de Qdrant
    system_prompt = """Eres un asistente especializado en responder preguntas sobre asistencia del Congreso usando información recuperada de una base de datos vectorial.

Tu tarea es:
1. Analizar cuidadosamente el contexto proporcionado que proviene de Qdrant (base de datos vectorial)
2. El contexto puede contener múltiples documentos relacionados con la pregunta
3. Generar una respuesta precisa y completa que incluya:
   - La información de asistencia solicitada
   - Los detalles del período legislativo (ej: "Primera Legislatura Ordinaria 2022-2023")
   - El período del congreso (ej: "Congreso 2021-2026 | Periodo anual 2022-2023")
   - El enlace al documento fuente (URL)
4. Estructurar la respuesta de manera clara y profesional
5. Si el contexto no contiene información suficiente, indica que no hay información disponible
6. Mantén un tono profesional y directo

Responde en español de manera clara y precisa."""

    human_prompt = f"""Contexto recuperado de Qdrant sobre asistencia del Congreso:
{state['context']}

Pregunta: {state['query']}

Basándote en el contexto proporcionado, genera una respuesta completa y precisa que responda a la pregunta.

Respuesta:"""

    # Generar la respuesta
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    response = llm.invoke(messages)
    answer = response.content.strip()

    return {
        "answer": answer,
        "messages": [HumanMessage(content=f"Generada respuesta para query ID {state['item_id']}")],
    }


def create_answer_generator():
    """Crea el grafo de LangGraph con un solo nodo"""
    workflow = StateGraph(State)

    # Agregar el nodo único
    workflow.add_node("generate_answer", generate_answer_node)

    # Configurar el flujo
    workflow.set_entry_point("generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


def load_questions_and_context(file_path: str) -> List[Dict[str, Any]]:
    """Carga las preguntas y contextos desde el archivo JSON"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_answer_incrementally(item: Dict[str, Any], output_file: str):
    """
    Guarda cada respuesta de manera incremental al archivo JSON
    """
    # Verificar si el archivo existe
    if os.path.exists(output_file):
        # Cargar datos existentes
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    # Agregar el nuevo item
    existing_data.append(item)

    # Guardar de vuelta al archivo
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"Guardada respuesta para ID {item['id']} - Total items: {len(existing_data)}")


def main():
    """Función principal que procesa todas las queries"""
    # Rutas de archivos
    input_file = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/testset_original/attendance/contexto_qdrant_with_ids.json"
    output_file = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/testset_original/attendance/answers.json"

    # Crear el generador de LangGraph
    print("Creando generador de respuestas con LangGraph...")
    generator = create_answer_generator()

    # Cargar datos
    print("Cargando preguntas y contextos de Qdrant...")
    data = load_questions_and_context(input_file)
    print(f"Cargadas {len(data)} preguntas")

    # Si el archivo de salida existe, preguntar si continuar
    if os.path.exists(output_file):
        response = input(f"El archivo {output_file} ya existe. ¿Deseas sobrescribirlo? (y/n): ")
        if response.lower() != "y":
            print("Operación cancelada.")
            return
        else:
            # Eliminar archivo existente
            os.remove(output_file)
            print("Archivo existente eliminado. Comenzando desde cero.")

    # Procesar cada pregunta
    print("Comenzando generación de respuestas...")
    for i, item in enumerate(data, 1):
        try:
            print(f"\nProcesando {i}/{len(data)} - ID: {item['id']}")
            print(f"Query: {item['query'][:100]}...")

            # Preparar el contexto como string
            context_str = (
                "\n".join(item["context"])
                if isinstance(item["context"], list)
                else str(item["context"])
            )

            # Crear estado inicial
            initial_state = {
                "messages": [],
                "query": item["query"],
                "context": context_str,
                "answer": "",
                "item_id": item["id"],
            }

            # Ejecutar el grafo
            result = generator.invoke(initial_state)

            # Preparar item para guardar
            answer_item = {
                "id": item["id"],
                "query": item["query"],
                "context": item["context"],
                "answer": result["answer"],
            }

            # Guardar incrementalmente
            save_answer_incrementally(answer_item, output_file)

            print(f"✓ Respuesta generada: {result['answer'][:100]}...")

        except Exception as e:
            print(f"✗ Error procesando ID {item['id']}: {str(e)}")
            continue

    print(f"\n🎉 Proceso completado! Respuestas guardadas en: {output_file}")


if __name__ == "__main__":
    main()
