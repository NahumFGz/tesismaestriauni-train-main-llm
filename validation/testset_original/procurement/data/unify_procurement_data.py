#!/usr/bin/env python3
"""
Script para unificar todos los archivos JSON de procurement
Combina archivos de las carpetas basic, intermediate y advanced
Solo toma los campos expected y our_agent (columns y rows)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List


def extract_relevant_data(data: Dict[str, Any], new_id: int) -> Dict[str, Any]:
    """
    Extrae solo los campos relevantes de expected y our_agent
    Convierte la estructura a arrays simples [columnas, filas]
    """
    expected_data = data.get("expected", {})
    our_agent_data = data.get("our_agent", {})

    # Convertir a estructura de array simple: [columnas, filas]
    expected_columns = expected_data.get("columns", [])
    expected_rows = expected_data.get("rows", [])
    our_agent_columns = our_agent_data.get("columns", [])
    our_agent_rows = our_agent_data.get("rows", [])

    result = {
        "id": new_id,
        "id_original": data.get("id", ""),
        "question": data.get("question", ""),
        "expected": [expected_columns] + expected_rows,
        "our_agent": [our_agent_columns] + our_agent_rows,
    }
    return result


def process_json_file(file_path: Path, new_id: int) -> List[Dict[str, Any]]:
    """
    Procesa un archivo JSON y extrae los datos relevantes
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Los archivos contienen una lista con un elemento
        if isinstance(data, list) and len(data) > 0:
            return [extract_relevant_data(data[0], new_id)]
        elif isinstance(data, dict):
            return [extract_relevant_data(data, new_id)]
        else:
            print(f"Formato inesperado en {file_path}")
            return []

    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
        return []


def main():
    # Directorio base
    base_dir = Path(
        "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/testset_original/procurement/scripts"
    )

    # Carpetas a procesar
    folders = ["basic", "intermediate", "advanced"]

    # Lista para almacenar todos los datos unificados
    unified_data = []

    # Contador para IDs incrementales
    current_id = 1

    print("Iniciando procesamiento de archivos JSON...")

    # Procesar cada carpeta
    for folder in folders:
        folder_path = base_dir / folder
        if not folder_path.exists():
            print(f"Carpeta no encontrada: {folder_path}")
            continue

        print(f"Procesando carpeta: {folder}")

        # Buscar todos los archivos JSON en la carpeta
        json_files = list(folder_path.glob("*.json"))
        json_files.sort()  # Ordenar por nombre

        print(f"  Encontrados {len(json_files)} archivos JSON")

        # Procesar cada archivo
        for json_file in json_files:
            print(f"  Procesando: {json_file.name}")
            extracted_data = process_json_file(json_file, current_id)
            unified_data.extend(extracted_data)
            current_id += 1

    print(f"\nTotal de registros procesados: {len(unified_data)}")

    # Guardar el archivo unificado
    output_file = base_dir / "unified_procurement_data.json"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(unified_data, f, indent=2, ensure_ascii=False)

        print(f"Archivo unificado guardado en: {output_file}")
        print(f"Tamaño del archivo: {output_file.stat().st_size / 1024:.2f} KB")

    except Exception as e:
        print(f"Error guardando el archivo unificado: {e}")
        return False

    # Mostrar estadísticas
    basic_count = len([item for item in unified_data if item["id_original"].startswith("basic_")])
    intermediate_count = len(
        [item for item in unified_data if item["id_original"].startswith("intermediate_")]
    )
    advanced_count = len(
        [item for item in unified_data if item["id_original"].startswith("advanced_")]
    )

    print(f"\nEstadísticas:")
    print(f"  Basic: {basic_count} registros")
    print(f"  Intermediate: {intermediate_count} registros")
    print(f"  Advanced: {advanced_count} registros")
    print(f"  Total: {len(unified_data)} registros")

    # Mostrar un ejemplo del primer registro
    if unified_data:
        print(f"\nEjemplo del primer registro:")
        print(json.dumps(unified_data[0], indent=2, ensure_ascii=False))

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Script ejecutado exitosamente!")
    else:
        print("\n❌ Error ejecutando el script!")
