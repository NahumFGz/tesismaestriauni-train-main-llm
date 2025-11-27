"""
Script de ejemplo para usar RAGAS (Retrieval-Augmented Generation Assessment)
Este script demuestra cómo evaluar sistemas RAG usando diferentes métricas de RAGAS.

RAGAS es un framework para evaluar aplicaciones de Retrieval-Augmented Generation (RAG).
Proporciona métricas específicas para evaluar tanto la calidad de la recuperación de información
como la calidad de la generación de respuestas.

=== MÉTRICAS DE RECUPERACIÓN (Retrieval) ===
Estas métricas evalúan qué tan bien el sistema recupera información relevante:

1. Context Precision (context_precision):
   - Mide qué tan precisos son los contextos recuperados
   - Evalúa si los fragmentos recuperados son relevantes para la pregunta
   - Valores más altos indican mejor precisión en la recuperación

2. Context Recall (context_recall):
   - Mide qué tan completa es la recuperación de información relevante
   - Evalúa si se recuperó toda la información necesaria para responder
   - Compara el ground_truth con los contexts para ver si toda la info está presente
   - IMPORTANTE: Si sale 0, significa que el ground_truth no coincide con los contexts
   - Valores más altos indican mejor cobertura de información relevante

3. Context Relevance (ContextRelevance):
   - Evalúa la relevancia general de los contextos recuperados
   - Mide qué tan relacionados están los contextos con la pregunta
   - Combina aspectos de precisión y utilidad del contexto

=== MÉTRICAS DE GENERACIÓN (Generation) ===
Estas métricas evalúan la calidad de las respuestas generadas:

4. Answer Relevancy (answer_relevancy):
   - Mide qué tan relevante es la respuesta generada para la pregunta
   - Evalúa si la respuesta aborda directamente lo que se pregunta
   - No considera la corrección factual, solo la relevancia

5. Answer Similarity (answer_similarity):
   - Compara la similitud semántica entre la respuesta generada y la respuesta ideal
   - Usa embeddings para medir similitud conceptual
   - Útil cuando hay múltiples formas correctas de responder

6. Answer Correctness (answer_correctness):
   - Evalúa la corrección factual de la respuesta generada
   - Combina aspectos semánticos y factuales
   - Considera tanto la exactitud como la completitud de la información

7. Faithfulness (faithfulness):
   - Mide qué tan fiel es la respuesta al contexto proporcionado
   - Evalúa si la respuesta se basa únicamente en la información recuperada
   - Detecta alucinaciones o información no respaldada por el contexto

=== INTERPRETACIÓN DE RESULTADOS ===
- Valores cercanos a 1.0: Excelente rendimiento
- Valores entre 0.7-0.9: Buen rendimiento
- Valores entre 0.5-0.7: Rendimiento moderado que requiere mejoras
- Valores menores a 0.5: Rendimiento pobre que requiere revisión significativa

=== USO DEL SCRIPT ===
1. Configura tu OPENAI_API_KEY en el archivo .env
2. Ejecuta: python ragas_personalizado.py
3. Revisa los resultados en pantalla y en el archivo resultados_ragas.csv
"""

import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Importaciones de RAGAS
from ragas import evaluate
from ragas.metrics import (
    ContextRelevance,
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)


def cargar_variables_entorno():
    """Cargar variables de entorno desde el archivo .env en la raíz del proyecto"""
    # Buscar el archivo .env en la raíz del proyecto
    proyecto_raiz = Path(__file__).parent.parent.parent
    env_path = proyecto_raiz / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Variables de entorno cargadas desde: {env_path}")
    else:
        print(f"⚠️  No se encontró archivo .env en: {env_path}")
        print("Por favor, crea un archivo .env con tu OPENAI_API_KEY")
        return False

    # Verificar que la API key esté disponible
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        print("❌ OPENAI_API_KEY no está configurada correctamente en el archivo .env")
        return False

    print("✅ OPENAI_API_KEY cargada correctamente")
    return True


def crear_datos_ejemplo():
    """Crear datos de ejemplo para evaluar con RAGAS"""
    # IMPORTANTE para context_recall:
    # - ground_truth = respuesta IDEAL que se puede construir con los contexts
    # - NO debe ser copia exacta de un context
    # - SÍ puede combinar información de múltiples contexts
    # - context_recall mide si los contexts contienen toda la info del ground_truth

    datos_ejemplo = {
        "question": [
            "¿Cuál es la capital de Francia y dónde se encuentra?",
            "¿Cómo funciona la fotosíntesis y qué produce?",
            "¿Qué es el machine learning y para qué sirve?",
        ],
        "answer": [
            "La capital de Francia es París, una ciudad ubicada en el norte del país y es la más poblada.",
            "La fotosíntesis es el proceso por el cual las plantas usan luz solar, dióxido de carbono y agua para producir glucosa y oxígeno.",
            "Machine learning es un método de análisis de datos que automatiza la construcción de modelos analíticos para permitir a las máquinas aprender patrones.",
        ],
        "contexts": [
            [
                "París es la capital y ciudad más poblada de Francia. Se encuentra en el norte del país.",
                "Francia es un país europeo con una rica historia cultural.",
                "París tiene una población de más de 2 millones de habitantes en la ciudad.",
            ],
            [
                "La fotosíntesis es el proceso biológico donde las plantas usan luz solar, dióxido de carbono y agua para producir glucosa y oxígeno.",
                "Este proceso ocurre principalmente en las hojas de las plantas.",
                "La clorofila es el pigmento verde que captura la luz solar para la fotosíntesis.",
            ],
            [
                "El machine learning es un método de análisis de datos que automatiza la construcción de modelos analíticos.",
                "Permite a las máquinas aprender patrones de los datos sin ser programadas explícitamente.",
                "Se utiliza en aplicaciones como reconocimiento de imágenes y procesamiento de lenguaje natural.",
            ],
        ],
        "ground_truth": [
            "La capital de Francia es París, ubicada en el norte del país y es su ciudad más poblada.",
            "La fotosíntesis es el proceso donde las plantas usan luz solar, CO2 y agua para crear glucosa y oxígeno, principalmente en las hojas usando clorofila.",
            "El machine learning automatiza la construcción de modelos analíticos para que las máquinas aprendan patrones de datos, usado en reconocimiento de imágenes y NLP.",
        ],
    }

    return Dataset.from_dict(datos_ejemplo)


def ejecutar_evaluacion_ragas():
    """Ejecutar evaluación usando métricas de RAGAS"""

    print("🚀 Iniciando evaluación con RAGAS...")

    # Crear dataset de ejemplo
    dataset = crear_datos_ejemplo()
    print(f"📊 Dataset creado con {len(dataset)} ejemplos")

    # Definir métricas a evaluar
    metricas = [
        answer_relevancy,
        answer_similarity,
        answer_correctness,
        faithfulness,
        context_precision,
        context_recall,
        ContextRelevance(),
    ]

    print("📋 Métricas a evaluar:")
    for metrica in metricas:
        print(f"  - {metrica.name}")

    try:
        # Ejecutar evaluación
        print("\n⏳ Ejecutando evaluación (esto puede tomar unos minutos)...")
        resultado = evaluate(
            dataset=dataset,
            metrics=metricas,
        )

        print("\n✅ Evaluación completada!")
        return resultado

    except Exception as e:
        print(f"❌ Error durante la evaluación: {str(e)}")
        return None


def mostrar_resultados(resultado):
    """Mostrar los resultados de la evaluación"""

    if resultado is None:
        print("❌ No hay resultados para mostrar")
        return

    print("\n" + "=" * 60)
    print("📊 RESULTADOS DE LA EVALUACIÓN RAGAS")
    print("=" * 60)

    # Convertir a DataFrame para mejor visualización
    df_resultados = resultado.to_pandas()

    print(f"\n📋 Columnas disponibles en los resultados: {list(df_resultados.columns)}")

    # Mostrar estadísticas generales
    print("\n📈 Puntuaciones promedio por métrica:")
    print("-" * 40)

    metricas_numericas = df_resultados.select_dtypes(include=["float64", "int64"]).columns

    for metrica in metricas_numericas:
        if metrica in df_resultados.columns:
            promedio = df_resultados[metrica].mean()
            print(f"{metrica:25s}: {promedio:.3f}")

    print("\n📋 Resultados detallados por fila:")
    print("-" * 40)

    for i, row in df_resultados.iterrows():
        print(f"\nFila {i+1}:")
        for metrica in metricas_numericas:
            if metrica in row:
                print(f"  {metrica:20s}: {row[metrica]:.3f}")

    # Mostrar todas las columnas y valores
    print("\n📋 DataFrame completo:")
    print("-" * 40)
    print(df_resultados.to_string())

    # Guardar resultados en CSV
    archivo_resultados = "resultados_ragas.csv"
    df_resultados.to_csv(archivo_resultados, index=False, encoding="utf-8")
    print(f"\n💾 Resultados guardados en: {archivo_resultados}")


def main():
    """Función principal"""
    print("🔧 RAGAS - Evaluación de Sistemas RAG")
    print("=" * 50)

    # Cargar variables de entorno
    if not cargar_variables_entorno():
        print("\n❌ No se pueden cargar las variables de entorno necesarias")
        print("Por favor:")
        print("1. Crea un archivo .env en la raíz del proyecto")
        print("2. Agrega tu OPENAI_API_KEY=tu_clave_real")
        sys.exit(1)

    # Ejecutar evaluación
    resultado = ejecutar_evaluacion_ragas()

    # Mostrar resultados
    mostrar_resultados(resultado)

    print("\n✅ Evaluación completada exitosamente!")


if __name__ == "__main__":
    main()
