"""
Script para evaluar el dataset de votaciones del Congreso usando RAGAS.
Este script carga los datos de answers.json y ground_truths.json para evaluar
las respuestas generadas usando métricas de RAGAS.

ENFOQUE: Evaluación métrica por métrica para evitar sobrecargar la API de OpenAI.
Cada métrica se evalúa individualmente con reintentos automáticos por rate limits.

=== MÉTRICAS DE RECUPERACIÓN (Retrieval) ===
1. Context Precision: Precisión de los contextos recuperados
2. Context Recall: Completitud de la información recuperada

=== MÉTRICAS DE GENERACIÓN (Generation) ===
3. Answer Relevancy: Relevancia de la respuesta para la pregunta
4. Answer Similarity: Similitud semántica con la respuesta ideal
5. Answer Correctness: Corrección factual de la respuesta
6. Faithfulness: Fidelidad de la respuesta al contexto

=== USO DEL SCRIPT ===
1. Configura tu OPENAI_API_KEY en el archivo .env
2. Ejecuta: python ragas_voting.py
3. Revisa los archivos en ./ragas/voting/ y el archivo consolidado
"""

import asyncio
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Importaciones de RAGAS
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)


def configurar_supresion_warnings():
    """Configurar la supresión de warnings molestos de asyncio"""
    # Suprimir warnings de asyncio sobre event loop cerrado
    warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
    warnings.filterwarnings("ignore", message=".*Task exception was never retrieved.*")


async def cerrar_recursos_asyncio():
    """Cerrar recursos asincrónicos pendientes de forma limpia"""
    try:
        # Obtener todas las tareas pendientes
        pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]

        if pending_tasks:
            print(f"🔄 Cerrando {len(pending_tasks)} tareas asincrónicas pendientes...")

            # Cancelar todas las tareas pendientes
            for task in pending_tasks:
                task.cancel()

            # Esperar a que se cancelen con timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True), timeout=5.0
                )
            except asyncio.TimeoutError:
                print("⚠️ Algunas tareas no se pudieron cancelar en el tiempo límite")
            except Exception:
                # Ignorar errores de cancelación
                pass

    except Exception as e:
        # Ignorar errores durante el cierre
        pass


def limpiar_event_loop():
    """Limpiar el event loop y recursos asincrónicos"""
    try:
        # Configurar supresión de warnings
        configurar_supresion_warnings()

        # Obtener el event loop actual si existe
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                # Ejecutar el cierre de recursos
                loop.run_until_complete(cerrar_recursos_asyncio())
        except RuntimeError:
            # No hay event loop activo, está bien
            pass

    except Exception:
        # Ignorar cualquier error durante la limpieza
        pass


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


def cargar_datos_answers():
    """Cargar datos del archivo answers.json"""
    try:
        answers_path = Path(__file__).parent.parent / "testset_original" / "voting" / "answers.json"

        if not answers_path.exists():
            print(f"❌ No se encontró el archivo: {answers_path}")
            return None

        with open(answers_path, "r", encoding="utf-8") as f:
            answers_data = json.load(f)

        print(f"✅ Cargados {len(answers_data)} registros de answers.json")
        return answers_data

    except Exception as e:
        print(f"❌ Error al cargar answers.json: {str(e)}")
        return None


def cargar_datos_ground_truths():
    """Cargar datos del archivo ground_truths.json"""
    try:
        ground_truths_path = (
            Path(__file__).parent.parent / "testset_original" / "voting" / "ground_truths.json"
        )

        if not ground_truths_path.exists():
            print(f"❌ No se encontró el archivo: {ground_truths_path}")
            return None

        with open(ground_truths_path, "r", encoding="utf-8") as f:
            ground_truths_data = json.load(f)

        print(f"✅ Cargados {len(ground_truths_data)} registros de ground_truths.json")
        return ground_truths_data

    except Exception as e:
        print(f"❌ Error al cargar ground_truths.json: {str(e)}")
        return None


def validar_datos_item(answer_item, ground_truth_item):
    """Validar que un item tenga todos los datos necesarios y en el formato correcto"""
    try:
        # Validar campos requeridos
        required_fields = ["id", "query", "answer", "context"]
        for field in required_fields:
            if field not in answer_item:
                return False, f"Campo '{field}' faltante en answer_item"
            if not answer_item[field]:
                return False, f"Campo '{field}' está vacío en answer_item"

        # Validar ground_truth
        if "ground_truth" not in ground_truth_item:
            return False, "Campo 'ground_truth' faltante"
        if not ground_truth_item["ground_truth"]:
            return False, "Campo 'ground_truth' está vacío"

        # Validar context es lista no vacía con strings
        context = answer_item["context"]
        if not isinstance(context, list):
            return False, f"Context debe ser lista, es {type(context)}"
        if len(context) == 0:
            return False, "Context está vacío"
        if not all(isinstance(c, str) and c.strip() for c in context):
            return False, "Context contiene elementos no válidos"

        # Validar que query, answer y ground_truth sean strings no vacíos
        for field, item in [("query", answer_item), ("answer", answer_item)]:
            if not isinstance(item[field], str) or not item[field].strip():
                return False, f"Campo '{field}' debe ser string no vacío"

        if (
            not isinstance(ground_truth_item["ground_truth"], str)
            or not ground_truth_item["ground_truth"].strip()
        ):
            return False, "Ground_truth debe ser string no vacío"

        return True, "OK"

    except Exception as e:
        return False, f"Error en validación: {str(e)}"


def crear_dataset_voting():
    """Crear dataset combinando answers y ground_truths por ID"""

    # Cargar datos
    answers_data = cargar_datos_answers()
    ground_truths_data = cargar_datos_ground_truths()

    if answers_data is None or ground_truths_data is None:
        print("❌ No se pudieron cargar los datos necesarios")
        return None

    # Crear diccionario de ground_truths indexado por ID para búsqueda rápida
    ground_truths_dict = {item["id"]: item for item in ground_truths_data}

    # Preparar listas para el dataset
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    datos_combinados = 0
    datos_faltantes = 0
    datos_invalidos = 0

    print("🔍 Validando y procesando datos...")

    for i, answer_item in enumerate(answers_data):
        answer_id = answer_item.get("id")

        if answer_id is None:
            print(f"⚠️  Item {i} sin ID, saltando...")
            datos_invalidos += 1
            continue

        # Buscar el ground_truth correspondiente
        if answer_id in ground_truths_dict:
            ground_truth_item = ground_truths_dict[answer_id]

            # Validar datos antes de agregar
            is_valid, error_msg = validar_datos_item(answer_item, ground_truth_item)

            if is_valid:
                # Extraer datos
                questions.append(answer_item["query"])
                answers.append(answer_item["answer"])
                contexts.append(answer_item["context"])
                ground_truths.append(ground_truth_item["ground_truth"])

                datos_combinados += 1
            else:
                print(f"⚠️  ID {answer_id} - Datos inválidos: {error_msg}")
                datos_invalidos += 1
        else:
            print(f"⚠️  No se encontró ground_truth para ID: {answer_id}")
            datos_faltantes += 1

    print(f"📊 Datos combinados exitosamente: {datos_combinados}")
    if datos_faltantes > 0:
        print(f"⚠️  Datos sin ground_truth: {datos_faltantes}")
    if datos_invalidos > 0:
        print(f"❌ Datos inválidos saltados: {datos_invalidos}")

    if datos_combinados == 0:
        print("❌ No hay datos válidos para crear el dataset")
        return None

    # Crear dataset
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    # Verificar estructura final del dataset
    print("🔍 Verificando estructura final del dataset...")
    for key, values in dataset_dict.items():
        print(f"  - {key}: {len(values)} elementos")
        if len(values) > 0:
            print(f"    Ejemplo: {str(values[0])[:100]}...")

    return Dataset.from_dict(dataset_dict)


def obtener_directorio_resultados():
    """Obtener el directorio donde se guardan los resultados por métrica"""
    directorio = Path(__file__).parent / "voting"
    directorio.mkdir(exist_ok=True)
    return directorio


def obtener_archivo_metrica(metrica_nombre):
    """Obtener la ruta del archivo JSON para una métrica específica"""
    directorio = obtener_directorio_resultados()
    return directorio / f"{metrica_nombre}.json"


def guardar_resultado_metrica(elemento_id, metrica_nombre, valor_metrica, datos_elemento):
    """Guardar el resultado de una métrica específica"""
    try:
        archivo_metrica = obtener_archivo_metrica(metrica_nombre)

        # Cargar resultados existentes
        if archivo_metrica.exists():
            with open(archivo_metrica, "r", encoding="utf-8") as f:
                resultados_existentes = json.load(f)
        else:
            resultados_existentes = []

        # Buscar si ya existe un resultado para este elemento
        encontrado = False
        for i, resultado in enumerate(resultados_existentes):
            if resultado.get("elemento_id") == elemento_id:
                resultados_existentes[i][metrica_nombre] = valor_metrica
                encontrado = True
                break

        # Si no existe, crear nuevo registro
        if not encontrado:
            nuevo_resultado = {
                "elemento_id": elemento_id,
                "question": datos_elemento["question"],
                "answer": datos_elemento["answer"],
                "ground_truth": datos_elemento["ground_truth"],
                "contexts": datos_elemento["contexts"],
                metrica_nombre: valor_metrica,
            }
            resultados_existentes.append(nuevo_resultado)

        # Guardar de vuelta al archivo
        with open(archivo_metrica, "w", encoding="utf-8") as f:
            json.dump(resultados_existentes, f, ensure_ascii=False, indent=2)

        print(f"💾 {metrica_nombre}: guardado elemento {elemento_id} = {valor_metrica:.3f}")

    except Exception as e:
        print(f"❌ Error al guardar {metrica_nombre}: {str(e)}")


def cargar_resultado_metrica(elemento_id, metrica_nombre):
    """Cargar el resultado de una métrica específica para un elemento"""
    try:
        archivo_metrica = obtener_archivo_metrica(metrica_nombre)

        if not archivo_metrica.exists():
            return None

        with open(archivo_metrica, "r", encoding="utf-8") as f:
            resultados = json.load(f)

        for resultado in resultados:
            if resultado.get("elemento_id") == elemento_id:
                return resultado.get(metrica_nombre)

        return None

    except Exception as e:
        print(f"❌ Error al cargar {metrica_nombre}: {str(e)}")
        return None


def guardar_resultado_individual(resultado_fila, archivo_json="voting_ragas.json"):
    """Guardar una fila individual de resultados en el archivo JSON (DEPRECATED)"""
    # Esta función se mantiene por compatibilidad pero ya no se usa
    pass


def obtener_metricas_esperadas():
    """Obtener la lista de nombres de métricas esperadas"""
    return list(obtener_mapa_metricas().keys())


def verificar_metricas_completas(resultado_dict):
    """Verificar si todas las métricas tienen valores válidos"""
    metricas_esperadas = obtener_metricas_esperadas()
    metricas_faltantes = []

    # Las métricas están agrupadas bajo 'metrics'
    metrics_section = resultado_dict.get("metrics", {})

    for metrica in metricas_esperadas:
        if (
            metrica not in metrics_section
            or metrics_section[metrica] is None
            or pd.isna(metrics_section[metrica])
        ):
            metricas_faltantes.append(metrica)

    return metricas_faltantes


def actualizar_resultado_en_json(elemento_id, nuevos_valores, archivo_json="voting_ragas.json"):
    """Actualizar un resultado específico en el archivo JSON"""
    try:
        if not os.path.exists(archivo_json):
            print(f"❌ Archivo {archivo_json} no existe")
            return False

        with open(archivo_json, "r", encoding="utf-8") as f:
            resultados = json.load(f)

        # Buscar el elemento por ID y actualizar
        for i, resultado in enumerate(resultados):
            if resultado.get("elemento_id") == elemento_id:
                # Actualizar solo las métricas (los nuevos valores van en la sección metrics)
                if "metrics" not in resultados[i]:
                    resultados[i]["metrics"] = {}

                for key, value in nuevos_valores.items():
                    resultados[i]["metrics"][key] = value

                # Guardar de vuelta
                with open(archivo_json, "w", encoding="utf-8") as f:
                    json.dump(resultados, f, ensure_ascii=False, indent=2)

                print(f"✅ Resultado actualizado para elemento {elemento_id}")
                return True

        print(f"❌ No se encontró elemento con ID {elemento_id}")
        return False

    except Exception as e:
        print(f"❌ Error al actualizar resultado: {str(e)}")
        return False


def obtener_mapa_metricas():
    """Obtener el mapeo de nombres de métricas a objetos de métricas"""
    return {
        "answer_relevancy": answer_relevancy,
        "answer_similarity": answer_similarity,
        "answer_correctness": answer_correctness,
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }


def obtener_lista_metricas():
    """Obtener la lista de objetos de métricas para evaluación"""
    mapa = obtener_mapa_metricas()
    return list(mapa.values())


def evaluar_metrica_individual(
    dataset_elemento, metrica_nombre, metrica_obj, max_reintentos=10, espera_segundos=20
):
    """Evaluar una métrica individual con reintentos"""

    for intento in range(max_reintentos):
        try:
            print(f"  Intento {intento + 1}/{max_reintentos} para {metrica_nombre}")

            # Evaluar solo esta métrica
            resultado_metrica = evaluate(dataset=dataset_elemento, metrics=[metrica_obj])

            # Extraer el valor de la métrica
            df_metrica = resultado_metrica.to_pandas()
            valor_metrica = df_metrica.iloc[0][metrica_nombre]

            if valor_metrica is not None and not pd.isna(valor_metrica):
                print(f"  ✅ {metrica_nombre} = {valor_metrica:.3f}")
                return valor_metrica
            else:
                print(f"  ⚠️ {metrica_nombre} devolvió valor nulo")

        except Exception as e:
            print(f"  ❌ Error en intento {intento + 1}: {str(e)}")

        if intento < max_reintentos - 1:
            print(f"  ⏳ Esperando {espera_segundos} segundos antes del siguiente intento...")
            time.sleep(espera_segundos)

    print(f"  ❌ Falló definitivamente la métrica: {metrica_nombre}")
    return None


def procesar_metrica_para_todos_elementos(dataset, metrica_nombre, metrica_obj):
    """Procesar una métrica específica para todos los elementos del dataset"""

    print(f"\n{'='*60}")
    print(f"📊 PROCESANDO MÉTRICA: {metrica_nombre}")
    print(f"{'='*60}")

    elementos_procesados = 0
    elementos_saltados = 0

    for i in range(len(dataset)):
        elemento_id = i + 1

        # Verificar si ya existe el resultado para esta métrica
        valor_existente = cargar_resultado_metrica(elemento_id, metrica_nombre)
        if valor_existente is not None:
            print(f"  ✅ Elemento {elemento_id}: Ya existe = {valor_existente:.3f}")
            elementos_saltados += 1
            continue

        print(f"  🔄 Elemento {elemento_id}/{len(dataset)}: Evaluando {metrica_nombre}")

        # Crear dataset con solo este elemento
        elemento_dataset = Dataset.from_dict(
            {
                "question": [dataset[i]["question"]],
                "answer": [dataset[i]["answer"]],
                "contexts": [dataset[i]["contexts"]],
                "ground_truth": [dataset[i]["ground_truth"]],
            }
        )

        # Datos del elemento para guardar
        datos_elemento = {
            "question": dataset[i]["question"],
            "answer": dataset[i]["answer"],
            "ground_truth": dataset[i]["ground_truth"],
            "contexts": dataset[i]["contexts"],
        }

        # Evaluar la métrica
        valor_metrica = evaluar_metrica_individual(
            elemento_dataset, metrica_nombre, metrica_obj, max_reintentos=10, espera_segundos=20
        )

        if valor_metrica is not None:
            # Guardar resultado inmediatamente
            guardar_resultado_metrica(elemento_id, metrica_nombre, valor_metrica, datos_elemento)
            elementos_procesados += 1
            print(f"  ✅ Elemento {elemento_id}: {metrica_nombre} = {valor_metrica:.3f}")
        else:
            print(f"  ❌ Elemento {elemento_id}: No se pudo obtener {metrica_nombre}")

    print(f"\n📊 {metrica_nombre} completada:")
    print(f"  ✅ Procesados: {elementos_procesados}")
    print(f"  ⏭️  Saltados (ya existían): {elementos_saltados}")
    print(f"  📈 Total: {elementos_procesados + elementos_saltados}/{len(dataset)}")

    return elementos_procesados


def consolidar_resultados_metricas():
    """Consolidar todos los resultados de métricas en un archivo final"""
    try:
        directorio = obtener_directorio_resultados()
        archivo_consolidado = directorio.parent / "consolidated_voting_ragas.json"

        metricas_esperadas = obtener_metricas_esperadas()
        resultados_consolidados = []

        print("🔄 Consolidando resultados de todas las métricas...")

        # Obtener todos los elementos únicos
        elementos_ids = set()
        for metrica_nombre in metricas_esperadas:
            archivo_metrica = obtener_archivo_metrica(metrica_nombre)
            if archivo_metrica.exists():
                with open(archivo_metrica, "r", encoding="utf-8") as f:
                    datos = json.load(f)
                    for item in datos:
                        elementos_ids.add(item.get("elemento_id"))

        # Consolidar por elemento
        for elemento_id in sorted(elementos_ids):
            resultado_elemento = {
                "elemento_id": elemento_id,
                "question": None,
                "answer": None,
                "ground_truth": None,
                "contexts": None,
                "metrics": {},
            }

            # Recopilar datos de cada métrica
            for metrica_nombre in metricas_esperadas:
                archivo_metrica = obtener_archivo_metrica(metrica_nombre)
                if archivo_metrica.exists():
                    with open(archivo_metrica, "r", encoding="utf-8") as f:
                        datos = json.load(f)

                    for item in datos:
                        if item.get("elemento_id") == elemento_id:
                            # Llenar datos básicos la primera vez
                            if resultado_elemento["question"] is None:
                                resultado_elemento["question"] = item.get("question")
                                resultado_elemento["answer"] = item.get("answer")
                                resultado_elemento["ground_truth"] = item.get("ground_truth")
                                resultado_elemento["contexts"] = item.get("contexts")

                            # Agregar métrica
                            if metrica_nombre in item:
                                resultado_elemento["metrics"][metrica_nombre] = item[metrica_nombre]
                            break

            resultados_consolidados.append(resultado_elemento)

        # Guardar archivo consolidado
        with open(archivo_consolidado, "w", encoding="utf-8") as f:
            json.dump(resultados_consolidados, f, ensure_ascii=False, indent=2)

        print(f"✅ Resultados consolidados guardados en: {archivo_consolidado}")
        print(f"📊 Total elementos consolidados: {len(resultados_consolidados)}")

        return resultados_consolidados, archivo_consolidado

    except Exception as e:
        print(f"❌ Error consolidando resultados: {str(e)}")
        return None, None


def estructurar_resultado_json(df_resultado, dataset_item, elemento_id):
    """Estructurar el resultado en un formato JSON organizado (DEPRECATED)"""
    # Esta función ya no se usa con el nuevo flujo por métricas
    pass


def ejecutar_evaluacion_ragas(max_samples=None):
    """Ejecutar evaluación usando métricas de RAGAS procesando métrica por métrica"""

    print("🚀 Iniciando evaluación con RAGAS para dataset de votaciones...")
    print("📋 Nuevo enfoque: Procesar TODOS los elementos para cada métrica")

    # Crear dataset
    dataset = crear_dataset_voting()
    if dataset is None:
        print("❌ No se pudo crear el dataset")
        return None

    print(f"📊 Dataset creado con {len(dataset)} ejemplos")

    # Limitar el número de muestras si se especifica (útil para debug)
    if max_samples and max_samples < len(dataset):
        print(f"🔧 Limitando evaluación a {max_samples} muestras para prueba")
        # Crear un subset del dataset
        indices = list(range(min(max_samples, len(dataset))))
        dataset = dataset.select(indices)
        print(f"📊 Dataset reducido a {len(dataset)} ejemplos")

    # Obtener métricas
    mapa_metricas = obtener_mapa_metricas()
    print(f"📋 Métricas a evaluar: {len(mapa_metricas)}")
    for metrica in mapa_metricas.keys():
        print(f"  - {metrica}")

    # Mostrar algunos ejemplos del dataset para debug
    print("\n🔍 Ejemplos del dataset:")
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        print(f"  Ejemplo {i+1}:")
        print(f"    Question: {item['question'][:80]}...")
        print(f"    Answer: {item['answer'][:80]}...")
        print(f"    Contexts: {len(item['contexts'])} elementos")
        print(f"    Ground truth: {item['ground_truth'][:80]}...")

    # Crear directorio de resultados
    directorio = obtener_directorio_resultados()
    print(f"📁 Directorio de resultados: {directorio}")

    try:
        print(f"\n⏳ Ejecutando evaluación métrica por métrica...")
        print(
            f"💡 Se procesarán {len(dataset)} elementos para cada una de las {len(mapa_metricas)} métricas"
        )

        total_elementos_procesados = 0
        metricas_completadas = 0

        # Procesar cada métrica para todos los elementos
        for i, (metrica_nombre, metrica_obj) in enumerate(mapa_metricas.items(), 1):
            print(f"\n🎯 MÉTRICA {i}/{len(mapa_metricas)}: {metrica_nombre}")

            try:
                elementos_procesados = procesar_metrica_para_todos_elementos(
                    dataset, metrica_nombre, metrica_obj
                )

                total_elementos_procesados += elementos_procesados
                metricas_completadas += 1

                print(f"✅ Métrica {metrica_nombre} completada exitosamente")

            except Exception as e:
                print(f"❌ Error procesando métrica {metrica_nombre}: {str(e)}")
                traceback.print_exc()
                continue

        print(f"\n{'='*80}")
        print(f"🎉 EVALUACIÓN GLOBAL COMPLETADA")
        print(f"{'='*80}")
        print(f"📊 Métricas completadas: {metricas_completadas}/{len(mapa_metricas)}")
        print(f"📊 Total elementos procesados: {total_elementos_procesados}")
        print(
            f"📊 Promedio por métrica: {total_elementos_procesados // max(metricas_completadas, 1)}"
        )

        # Consolidar resultados
        print("\n🔄 Consolidando resultados de todas las métricas...")
        resultados_consolidados, archivo_consolidado = consolidar_resultados_metricas()

        return resultados_consolidados

    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {str(e)}")
        print(f"Tipo de error: {type(e).__name__}")

        # Información adicional para debug
        print("\n🔍 Traceback completo:")
        traceback.print_exc()

        return None


def mostrar_y_guardar_resultados(resultado):
    """Mostrar los resultados de la evaluación desde el archivo consolidado"""

    # Si el resultado es None, intentar cargar desde el archivo consolidado
    if resultado is None:
        directorio = obtener_directorio_resultados()
        archivo_consolidado = directorio.parent / "voting_ragas_consolidated.json"

        if archivo_consolidado.exists():
            try:
                with open(archivo_consolidado, "r", encoding="utf-8") as f:
                    resultado = json.load(f)
                print(f"📂 Cargados resultados desde {archivo_consolidado}")
            except Exception as e:
                print(f"❌ Error al cargar {archivo_consolidado}: {str(e)}")
                return
        else:
            print("❌ No hay resultados consolidados para mostrar")
            print("💡 Ejecuta primero la evaluación para generar resultados")
            return

    if not resultado or len(resultado) == 0:
        print("❌ No hay resultados para mostrar")
        return

    print("\n" + "=" * 60)
    print("📊 RESULTADOS DE LA EVALUACIÓN RAGAS - DATASET VOTING")
    print("=" * 60)

    # Si el resultado es una lista de diccionarios, convertir a DataFrame
    if isinstance(resultado, list):
        df_resultados = pd.DataFrame(resultado)
    else:
        # Si es el resultado directo de RAGAS
        df_resultados = resultado.to_pandas()

    print(f"\n📋 Columnas disponibles: {list(df_resultados.columns)}")
    print(f"📊 Total de elementos evaluados: {len(df_resultados)}")

    # Extraer métricas de la estructura organizada
    df_metricas = None
    metricas_numericas = []

    if "metrics" in df_resultados.columns and not df_resultados["metrics"].isna().all():
        # Nueva estructura: métricas están en columna 'metrics'
        metricas_data = [m for m in df_resultados["metrics"] if m and isinstance(m, dict)]
        if metricas_data:
            df_metricas = pd.DataFrame(metricas_data)
            metricas_numericas = df_metricas.select_dtypes(include=["float64", "int64"]).columns

    # Fallback: buscar métricas como columnas directas (estructura antigua)
    if df_metricas is None or len(metricas_numericas) == 0:
        columnas_excluir = [
            "elemento_id",
            "question",
            "answer",
            "contexts",
            "ground_truth",
            "metrics",
        ]
        metricas_numericas = df_resultados.select_dtypes(include=["float64", "int64"]).columns
        metricas_numericas = [col for col in metricas_numericas if col not in columnas_excluir]
        df_metricas = df_resultados

    if len(metricas_numericas) > 0:
        # Mostrar estadísticas generales
        print("\n📈 Puntuaciones promedio por métrica:")
        print("-" * 40)

        for metrica in metricas_numericas:
            if metrica in df_metricas.columns:
                promedio = df_metricas[metrica].mean()
                print(f"{metrica:25s}: {promedio:.3f}")

        # Mostrar algunas estadísticas adicionales
        print("\n📊 Estadísticas detalladas:")
        print("-" * 40)
        for metrica in metricas_numericas:
            if metrica in df_metricas.columns:
                serie = df_metricas[metrica]
                print(f"\n{metrica}:")
                print(f"  Promedio: {serie.mean():.3f}")
                print(f"  Mediana:  {serie.median():.3f}")
                print(f"  Mín:      {serie.min():.3f}")
                print(f"  Máx:      {serie.max():.3f}")
                print(f"  Std:      {serie.std():.3f}")
    else:
        print("⚠️ No se encontraron métricas numéricas para mostrar estadísticas")

    # Los resultados están en archivos individuales y consolidado
    directorio = obtener_directorio_resultados()
    archivo_consolidado = directorio.parent / "consolidated_voting_ragas.json"

    print(f"\n💾 Resultados disponibles:")
    print(f"  📁 Archivos individuales: {directorio}")
    if archivo_consolidado.exists():
        print(f"  📄 Archivo consolidado: {archivo_consolidado}")
        print(f"  📊 Total elementos: {len(df_resultados)}")
    else:
        print(f"  ⚠️ Archivo consolidado no encontrado")


def main():
    """Función principal"""
    print("🔧 RAGAS - Evaluación del Dataset de Votaciones del Congreso")
    print("=" * 60)

    # Configurar supresión de warnings al inicio
    configurar_supresion_warnings()

    # Cargar variables de entorno
    if not cargar_variables_entorno():
        print("\n❌ No se pueden cargar las variables de entorno necesarias")
        print("Por favor:")
        print("1. Crea un archivo .env en la raíz del proyecto")
        print("2. Agrega tu OPENAI_API_KEY=tu_clave_real")
        limpiar_event_loop()  # Limpiar antes de salir
        sys.exit(1)

    # Preguntar al usuario si quiere hacer una prueba pequeña primero
    print("\n🤔 ¿Qué tipo de evaluación deseas ejecutar?")
    print("1. Prueba pequeña (5 muestras) - Recomendado para primera ejecución")
    print("2. Evaluación completa (todos los datos)")

    try:
        opcion = input("\nSelecciona una opción (1 o 2, default=1): ").strip()
        if not opcion:
            opcion = "1"
    except KeyboardInterrupt:
        print("\n\n👋 Evaluación cancelada por el usuario")
        limpiar_event_loop()  # Limpiar antes de salir
        sys.exit(0)

    # Ejecutar evaluación según la opción
    if opcion == "1":
        print("\n🧪 Ejecutando prueba con 5 muestras...")
        resultado = ejecutar_evaluacion_ragas(max_samples=5)
    elif opcion == "2":
        print("\n🚀 Ejecutando evaluación completa...")
        resultado = ejecutar_evaluacion_ragas()
    else:
        print(f"\n⚠️  Opción '{opcion}' no válida, usando prueba pequeña por defecto...")
        resultado = ejecutar_evaluacion_ragas(max_samples=5)

    # Mostrar y guardar resultados
    mostrar_y_guardar_resultados(resultado)

    if resultado is not None:
        print("\n✅ Evaluación del dataset de votaciones completada exitosamente!")
        print("📋 Revisa los archivos en ./ragas/voting/ y el archivo consolidado")
        print("💡 Cada métrica se guardó por separado para evitar problemas de API")
    else:
        print("\n❌ La evaluación no se completó correctamente")
        print("💡 Intenta ejecutar primero con la opción de prueba pequeña")

    # Limpiar recursos asincrónicos al final
    limpiar_event_loop()


if __name__ == "__main__":
    main()
