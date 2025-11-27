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
    ground_truth: str
    item_id: int


def generate_ground_truth_node(state: State) -> Dict[str, Any]:
    """
    Nodo único de LangGraph que genera la respuesta de verdad fundamental
    basada en la pregunta y el contexto proporcionado.
    """
    # Configurar el modelo OpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Crear el prompt sencillo
    system_prompt = """Eres un asistente especializado en generar respuestas precisas sobre votaciones del Congreso basadas en el contexto proporcionado.

Tu tarea es:
1. Leer cuidadosamente el contexto proporcionado que contiene información sobre votaciones del Congreso
2. Extraer todos los datos relevantes: fecha, legislatura, período del congreso, asunto votado, presidente de sesión y URL del documento
3. Generar una respuesta completa que incluya:
   - La información de las votaciones realizadas en la fecha solicitada
   - Los detalles del período legislativo (ej: "Primera Legislatura Ordinaria 2022-2023")
   - El período del congreso (ej: "Congreso 2021-2026 | Periodo anual 2022-2023")
   - Los asuntos o proyectos de ley votados
   - El presidente de la sesión cuando esté disponible
   - Los enlaces a los documentos fuente (URLs)
4. Estructurar la respuesta de manera clara y profesional, organizando las votaciones por orden cronológico o temático
5. Si el contexto no contiene información suficiente, indica que no hay información disponible

Responde en español y de manera clara y directa."""

    human_prompt = f"""Contexto con información de votaciones del Congreso:
{state['context']}

Pregunta: {state['query']}

Por favor, genera una respuesta completa que incluya:
1. Las votaciones realizadas en la fecha solicitada
2. Los asuntos o proyectos de ley votados
3. Los detalles del período legislativo y congreso
4. El presidente de la sesión (cuando esté disponible)
5. Los enlaces a los documentos fuente

Respuesta:"""

    # Generar la respuesta
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    response = llm.invoke(messages)
    ground_truth = response.content.strip()

    return {
        "ground_truth": ground_truth,
        "messages": [HumanMessage(content=f"Generada respuesta para query ID {state['item_id']}")],
    }


def create_ground_truth_generator():
    """Crea el grafo de LangGraph con un solo nodo"""
    workflow = StateGraph(State)

    # Agregar el nodo único
    workflow.add_node("generate_ground_truth", generate_ground_truth_node)

    # Configurar el flujo
    workflow.set_entry_point("generate_ground_truth")
    workflow.add_edge("generate_ground_truth", END)

    return workflow.compile()


def load_questions_and_context(file_path: str) -> List[Dict[str, Any]]:
    """Carga las preguntas y contextos desde el archivo JSON"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_ground_truth_incrementally(item: Dict[str, Any], output_file: str):
    """
    Guarda cada ground truth de manera incremental al archivo JSON
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

    print(f"Guardado ground truth para ID {item['id']} - Total items: {len(existing_data)}")


def main():
    """Función principal que procesa todas las queries"""
    # Rutas de archivos
    input_file = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/testset_original/voting/preguntas_contexto_esperado_with_ids.json"
    output_file = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/testset_original/voting/ground_truths.json"

    # Crear el generador de LangGraph
    print("Creando generador de ground truth con LangGraph...")
    generator = create_ground_truth_generator()

    # Cargar datos
    print("Cargando preguntas y contextos...")
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
    print("Comenzando generación de ground truths...")
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
                "ground_truth": "",
                "item_id": item["id"],
            }

            # Ejecutar el grafo
            result = generator.invoke(initial_state)

            # Preparar item para guardar
            ground_truth_item = {
                "id": item["id"],
                "query": item["query"],
                "context": item["context"],
                "ground_truth": result["ground_truth"],
            }

            # Guardar incrementalmente
            save_ground_truth_incrementally(ground_truth_item, output_file)

            print(f"✓ Ground truth generado: {result['ground_truth'][:100]}...")

        except Exception as e:
            print(f"✗ Error procesando ID {item['id']}: {str(e)}")
            continue

    print(f"\n🎉 Proceso completado! Ground truths guardados en: {output_file}")


if __name__ == "__main__":
    main()
