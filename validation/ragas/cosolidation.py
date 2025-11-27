#!/usr/bin/env python3
"""
Script para consolidar archivos JSON de métricas RAGAS de las carpetas attendance, procurement y voting.
Combina todos los archivos JSON en un formato unificado con las métricas requeridas.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Carga un archivo JSON y retorna su contenido como lista de diccionarios."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error cargando archivo {file_path}: {e}")
        return []


def consolidate_metrics(data: List[Dict[str, Any]], metric_name: str) -> Dict[int, float]:
    """Consolida las métricas de un archivo JSON específico."""
    metrics_dict = {}
    for item in data:
        elemento_id = item.get("elemento_id")
        if elemento_id is not None:
            # Buscar la métrica específica en el item
            metric_value = item.get(metric_name, 0.0)
            metrics_dict[elemento_id] = metric_value
    return metrics_dict


def consolidate_all_metrics(base_path: str, folder_name: str) -> Dict[int, Dict[str, float]]:
    """Consolida todas las métricas de una carpeta específica."""
    folder_path = os.path.join(base_path, folder_name)
    all_metrics = {}

    # Lista de métricas requeridas
    required_metrics = [
        "answer_relevancy",
        "answer_similarity",
        "answer_correctness",
        "faithfulness",
        "context_precision",
        "context_recall",
    ]

    # Cargar cada archivo de métrica
    for metric in required_metrics:
        file_path = os.path.join(folder_path, f"{metric}.json")
        if os.path.exists(file_path):
            data = load_json_file(file_path)
            metric_dict = consolidate_metrics(data, metric)

            # Agregar métricas al diccionario principal
            for elemento_id, value in metric_dict.items():
                if elemento_id not in all_metrics:
                    all_metrics[elemento_id] = {}
                all_metrics[elemento_id][metric] = value
        else:
            print(f"Archivo no encontrado: {file_path}")

    return all_metrics


def get_base_data_from_all_files(base_path: str, folder_name: str) -> Dict[int, Dict[str, Any]]:
    """Extrae los datos base de todos los archivos JSON de una carpeta, priorizando valores no vacíos."""
    folder_path = os.path.join(base_path, folder_name)
    base_data = {}

    # Lista de archivos JSON a revisar
    json_files = [
        "answer_correctness.json",
        "answer_relevancy.json",
        "answer_similarity.json",
        "faithfulness.json",
        "context_precision.json",
        "context_recall.json",
    ]

    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        if os.path.exists(file_path):
            data = load_json_file(file_path)
            for item in data:
                elemento_id = item.get("elemento_id")
                if elemento_id is not None:
                    if elemento_id not in base_data:
                        base_data[elemento_id] = {
                            "elemento_id": elemento_id,
                            "question": "",
                            "answer": "",
                            "ground_truth": "",
                            "contexts": [],
                        }

                    # Actualizar solo si el valor actual está vacío y el nuevo no
                    current = base_data[elemento_id]
                    if not current["question"] and item.get("question"):
                        current["question"] = item.get("question", "")
                    if not current["answer"] and item.get("answer"):
                        current["answer"] = item.get("answer", "")
                    if not current["ground_truth"] and item.get("ground_truth"):
                        current["ground_truth"] = item.get("ground_truth", "")
                    if not current["contexts"] and item.get("contexts"):
                        current["contexts"] = item.get("contexts", [])

    return base_data


def create_consolidated_data(
    base_data: Dict[int, Dict[str, Any]], metrics_data: Dict[int, Dict[str, float]]
) -> List[Dict[str, Any]]:
    """Crea la estructura consolidada final con datos base y métricas."""
    consolidated = []

    # Obtener todos los elemento_id únicos
    all_elemento_ids = set(base_data.keys()) | set(metrics_data.keys())

    for elemento_id in sorted(all_elemento_ids):
        # Obtener datos base
        base_item = base_data.get(elemento_id, {})

        # Obtener métricas
        metrics = metrics_data.get(elemento_id, {})

        # Crear estructura final
        consolidated_item = {
            "elemento_id": elemento_id,
            "question": base_item.get("question", ""),
            "answer": base_item.get("answer", ""),
            "ground_truth": base_item.get("ground_truth", ""),
            "contexts": base_item.get("contexts", []),
            "metrics": {
                "answer_relevancy": metrics.get("answer_relevancy", 0.0),
                "answer_similarity": metrics.get("answer_similarity", 0.0),
                "answer_correctness": metrics.get("answer_correctness", 0.0),
                "faithfulness": metrics.get("faithfulness", 0.0),
                "context_precision": metrics.get("context_precision", 0.0),
                "context_recall": metrics.get("context_recall", 0.0),
            },
        }
        consolidated.append(consolidated_item)

    return consolidated


def main():
    """Función principal que ejecuta la consolidación."""
    # Rutas base
    base_path = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/ragas"
    folders = ["attendance", "procurement", "voting"]

    folder_data = {}  # Para almacenar datos de cada carpeta individualmente
    total_elements = 0

    for folder in folders:
        print(f"Procesando carpeta: {folder}")

        # Obtener datos base de todos los archivos JSON de la carpeta
        base_data = get_base_data_from_all_files(base_path, folder)
        if not base_data:
            print(f"No se encontraron datos en {folder}")
            continue

        # Obtener todas las métricas
        metrics_data = consolidate_all_metrics(base_path, folder)

        # Crear datos consolidados para esta carpeta
        folder_consolidated = create_consolidated_data(base_data, metrics_data)

        # Guardar datos de esta carpeta para archivo individual
        folder_data[folder] = folder_consolidated.copy()
        total_elements += len(folder_consolidated)
        print(f"Procesados {len(folder_consolidated)} elementos de {folder}")

    # Guardar archivos individuales por carpeta
    for folder, data in folder_data.items():
        individual_file = os.path.join(base_path, f"consolidated_{folder}_ragas.json")
        with open(individual_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Archivo individual generado: {individual_file}")

    print(f"\nConsolidación completada!")
    print(f"Total de elementos procesados: {total_elements}")

    # Mostrar estadísticas por carpeta
    for folder, data in folder_data.items():
        print(f"- {folder}: {len(data)} elementos")


if __name__ == "__main__":
    main()
