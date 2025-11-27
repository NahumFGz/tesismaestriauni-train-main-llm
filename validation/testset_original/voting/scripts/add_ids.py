#!/usr/bin/env python3
"""
Script para agregar IDs consecutivos a los archivos JSON y validar que las queries sean iguales.
Combina todas las preguntas de las secciones 'fecha', 'mes' y 'legislatura' en una sola lista.
"""

import json
import os
from collections import OrderedDict


def load_json_file(filepath):
    """Carga un archivo JSON y retorna su contenido."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error al cargar {filepath}: {e}")
        return None


def extract_all_items(data):
    """Extrae todos los elementos de las secciones fecha, mes y legislatura."""
    all_items = []
    sections = ["fecha", "mes", "legislatura"]

    for section in sections:
        if section in data:
            for item in data[section]:
                all_items.append(item)

    return all_items


def add_ids_and_reorder(items, start_id=1):
    """Agrega IDs consecutivos y reordena los elementos (id, query, context, ...)."""
    processed_items = []

    for i, item in enumerate(items):
        # Crear nuevo diccionario ordenado
        new_item = OrderedDict()
        new_item["id"] = start_id + i
        new_item["query"] = item.get("query", "")
        new_item["context"] = item.get("context", [])

        # Agregar cualquier otro campo que pueda existir
        for key, value in item.items():
            if key not in ["query", "context"]:
                new_item[key] = value

        processed_items.append(new_item)

    return processed_items


def validate_queries(items1, items2):
    """Valida que las queries sean iguales entre ambos archivos."""
    if len(items1) != len(items2):
        print(f"⚠️  ADVERTENCIA: Diferente número de elementos: {len(items1)} vs {len(items2)}")
        return False

    mismatches = []
    for i, (item1, item2) in enumerate(zip(items1, items2)):
        query1 = item1.get("query", "").strip()
        query2 = item2.get("query", "").strip()

        if query1 != query2:
            mismatches.append({"index": i + 1, "query1": query1, "query2": query2})

    if mismatches:
        print(f"❌ Encontradas {len(mismatches)} diferencias en las queries:")
        for mismatch in mismatches[:5]:  # Mostrar solo las primeras 5
            print(f"  ID {mismatch['index']}:")
            print(f"    Archivo 1: {mismatch['query1'][:100]}...")
            print(f"    Archivo 2: {mismatch['query2'][:100]}...")
        if len(mismatches) > 5:
            print(f"    ... y {len(mismatches) - 5} más")
        return False

    print("✅ Todas las queries son iguales entre ambos archivos")
    return True


def save_json_file(data, filepath):
    """Guarda los datos en un archivo JSON con formato bonito."""
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        print(f"✅ Archivo guardado: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error al guardar {filepath}: {e}")
        return False


def main():
    # Rutas de los archivos
    base_path = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/testset_original/voting"

    contexto_file = os.path.join(base_path, "contexto_qdrant.json")
    preguntas_file = os.path.join(base_path, "preguntas_contexto_esperado.json")

    # Nuevos archivos de salida
    contexto_output = os.path.join(base_path, "contexto_qdrant_with_ids.json")
    preguntas_output = os.path.join(base_path, "preguntas_contexto_esperado_with_ids.json")

    print("🔄 Cargando archivos JSON...")

    # Cargar archivos
    contexto_data = load_json_file(contexto_file)
    preguntas_data = load_json_file(preguntas_file)

    if not contexto_data or not preguntas_data:
        print("❌ Error al cargar los archivos")
        return

    print("📊 Extrayendo elementos de todas las secciones...")

    # Extraer todos los elementos
    contexto_items = extract_all_items(contexto_data)
    preguntas_items = extract_all_items(preguntas_data)

    print(f"   Contexto: {len(contexto_items)} elementos")
    print(f"   Preguntas: {len(preguntas_items)} elementos")

    print("🔍 Validando que las queries sean iguales...")

    # Validar queries
    queries_match = validate_queries(contexto_items, preguntas_items)

    if not queries_match:
        print("❌ Las queries no coinciden. Revisa los archivos originales.")
        # Continuar de todas formas para generar los archivos

    print("🔢 Agregando IDs consecutivos...")

    # Agregar IDs y reordenar
    contexto_with_ids = add_ids_and_reorder(contexto_items)
    preguntas_with_ids = add_ids_and_reorder(preguntas_items)

    print("💾 Guardando archivos con IDs...")

    # Guardar archivos procesados
    contexto_saved = save_json_file(contexto_with_ids, contexto_output)
    preguntas_saved = save_json_file(preguntas_with_ids, preguntas_output)

    if contexto_saved and preguntas_saved:
        print("🎉 Proceso completado exitosamente!")
        print(f"📁 Archivos generados:")
        print(f"   - {contexto_output}")
        print(f"   - {preguntas_output}")
        print(f"📊 Total de elementos procesados: {len(contexto_with_ids)}")
        print(f"🔢 IDs asignados: 1 a {len(contexto_with_ids)}")
    else:
        print("❌ Error al guardar algunos archivos")


if __name__ == "__main__":
    main()
