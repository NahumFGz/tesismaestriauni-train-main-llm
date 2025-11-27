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
import time
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    """Estado del grafo que contiene los mensajes y datos de procesamiento"""

    messages: Annotated[list, add_messages]
    question: str
    expected: List[List[str]]
    ground_truth: str
    item_id: int
    id_original: str


def retry_with_backoff(func, max_retries=10, delay=20):
    """
    Función para reintentar una operación con pausas en caso de rate limit.

    Args:
        func: Función a ejecutar
        max_retries: Número máximo de reintentos (default: 10)
        delay: Tiempo de pausa entre reintentos en segundos (default: 20)

    Returns:
        Resultado de la función si es exitosa

    Raises:
        Exception: La última excepción si se agotan los reintentos
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            # Verificar si es un error de rate limit
            if "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
                if attempt < max_retries:
                    print(
                        f"⚠️ Rate limit detectado (intento {attempt + 1}/{max_retries + 1}). Pausando {delay} segundos..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Rate limit: Se agotaron los {max_retries} reintentos")
                    raise e
            else:
                # Si no es rate limit, lanzar la excepción inmediatamente
                raise e

    # Este punto no debería alcanzarse nunca
    raise Exception("Error inesperado en retry_with_backoff")


def generate_ground_truth_node(state: State) -> Dict[str, Any]:
    """
    Nodo único de LangGraph que genera la respuesta de verdad fundamental
    basada en la pregunta y el contexto proporcionado.
    Incluye manejo de rate limits con reintentos automáticos.
    """

    def _generate_ground_truth():
        # Configurar el modelo OpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Crear el prompt sencillo
        system_prompt = """Eres un asistente especializado en generar respuestas precisas para consultas SQL sobre datos de contrataciones públicas.

Tu tarea es:
1. Revisar el contexto proporcionado que contiene los resultados de la consulta SQL ejecutada por nuestro agente
2. Generar una respuesta en lenguaje natural que explique claramente:
   - Los resultados obtenidos de la base de datos
   - Interpretación de los datos cuando sea relevante segun la pregunta
   - Formateo claro de números, nombres y otros datos

Responde en español y de manera clara y directa."""

        human_prompt = f"""Pregunta sobre contrataciones públicas:
{state['question']}

Respuesta esperada de la consulta SQL (formato: [columnas], [fila1], [fila2], ...):
{state['expected']}

Por favor, genera una respuesta en lenguaje natural que:
1. Presente los resultados de manera clara y comprensible
2. Sea precisa y profesional

Respuesta:"""

        # Generar la respuesta
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

        response = llm.invoke(messages)
        return response.content.strip()

    # Usar la función de reintentos
    ground_truth = retry_with_backoff(_generate_ground_truth)

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


def load_questions_and_expected(file_path: str) -> List[Dict[str, Any]]:
    """Carga las preguntas y respuestas esperadas desde el archivo JSON unificado"""
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
    input_file = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/testset_original/procurement/unified_procurement_data.json"
    output_file = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/testset_original/procurement/ground_truths.json"

    # Crear el generador de LangGraph
    print("Creando generador de ground truth con LangGraph...")
    generator = create_ground_truth_generator()

    # Cargar datos
    print("Cargando preguntas y respuestas esperadas...")
    data = load_questions_and_expected(input_file)
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
            print(f"Question: {item['question'][:100]}...")

            # Crear estado inicial
            initial_state = {
                "messages": [],
                "question": item["question"],
                "expected": item["expected"],
                "ground_truth": "",
                "item_id": item["id"],
                "id_original": item["id_original"],
            }

            # Ejecutar el grafo
            result = generator.invoke(initial_state)

            # Preparar item para guardar
            ground_truth_item = {
                "id": item["id"],
                "id_original": item["id_original"],
                "question": item["question"],
                "context": item["expected"],
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
