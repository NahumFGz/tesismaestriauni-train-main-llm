#!/usr/bin/env python3
"""
Script para comparar las queries de ambos archivos JSON con IDs y verificar que sean iguales.
"""

import json
import os
from typing import Dict, List, Tuple


def load_json_file(filepath: str) -> List[Dict]:
    """Carga un archivo JSON y retorna su contenido."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"❌ Error al cargar {filepath}: {e}")
        return []


def compare_queries_by_id(data1: List[Dict], data2: List[Dict]) -> Tuple[bool, List[Dict]]:
    """Compara las queries por ID entre ambos archivos."""
    # Crear diccionarios indexados por ID para búsqueda rápida
    dict1 = {item["id"]: item for item in data1}
    dict2 = {item["id"]: item for item in data2}
    
    # Obtener todos los IDs únicos
    all_ids = set(dict1.keys()) | set(dict2.keys())
    
    differences = []
    matches = 0
    
    for id_num in sorted(all_ids):
        item1 = dict1.get(id_num)
        item2 = dict2.get(id_num)
        
        if not item1:
            differences.append({
                "id": id_num,
                "error": "ID faltante en archivo 1 (contexto_qdrant_with_ids.json)",
                "query1": None,
                "query2": item2.get("query", "") if item2 else None
            })
        elif not item2:
            differences.append({
                "id": id_num,
                "error": "ID faltante en archivo 2 (preguntas_contexto_esperado_with_ids.json)",
                "query1": item1.get("query", ""),
                "query2": None
            })
        else:
            query1 = item1.get("query", "").strip()
            query2 = item2.get("query", "").strip()
            
            if query1 != query2:
                differences.append({
                    "id": id_num,
                    "error": "Queries diferentes",
                    "query1": query1,
                    "query2": query2
                })
            else:
                matches += 1
    
    return len(differences) == 0, differences, matches


def print_comparison_report(all_match: bool, differences: List[Dict], matches: int, total_items1: int, total_items2: int):
    """Imprime el reporte de comparación."""
    print("=" * 80)
    print("📋 REPORTE DE COMPARACIÓN DE ARCHIVOS JSON")
    print("=" * 80)
    print()
    
    print(f"📊 Estadísticas generales:")
    print(f"   • Total elementos archivo 1: {total_items1}")
    print(f"   • Total elementos archivo 2: {total_items2}")
    print(f"   • Coincidencias exactas: {matches}")
    print(f"   • Diferencias encontradas: {len(differences)}")
    print()
    
    if all_match:
        print("✅ RESULTADO: TODOS LOS IDs TIENEN QUERIES IGUALES")
        print("🎉 Los archivos son completamente consistentes!")
    else:
        print("❌ RESULTADO: SE ENCONTRARON DIFERENCIAS")
        print(f"⚠️  {len(differences)} elementos tienen queries diferentes o faltantes")
        print()
        
        print("🔍 DETALLES DE LAS DIFERENCIAS:")
        print("-" * 60)
        
        for i, diff in enumerate(differences[:10], 1):  # Mostrar solo las primeras 10
            print(f"\n{i}. ID: {diff['id']}")
            print(f"   Error: {diff['error']}")
            
            if diff['query1'] is not None:
                print(f"   Query 1: {diff['query1'][:100]}{'...' if len(diff['query1']) > 100 else ''}")
            else:
                print(f"   Query 1: [FALTANTE]")
                
            if diff['query2'] is not None:
                print(f"   Query 2: {diff['query2'][:100]}{'...' if len(diff['query2']) > 100 else ''}")
            else:
                print(f"   Query 2: [FALTANTE]")
        
        if len(differences) > 10:
            print(f"\n   ... y {len(differences) - 10} diferencias más")
    
    print()
    print("=" * 80)


def main():
    # Rutas de los archivos
    base_path = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/testset_original/attendance"
    
    file1 = os.path.join(base_path, "contexto_qdrant_with_ids.json")
    file2 = os.path.join(base_path, "preguntas_contexto_esperado_with_ids.json")
    
    print("🔄 Cargando archivos JSON con IDs...")
    
    # Cargar archivos
    data1 = load_json_file(file1)
    data2 = load_json_file(file2)
    
    if not data1 or not data2:
        print("❌ Error al cargar uno o ambos archivos")
        return
    
    print(f"✅ Archivos cargados exitosamente")
    print(f"   • Archivo 1: {len(data1)} elementos")
    print(f"   • Archivo 2: {len(data2)} elementos")
    print()
    
    print("🔍 Comparando queries por ID...")
    
    # Comparar archivos
    all_match, differences, matches = compare_queries_by_id(data1, data2)
    
    # Mostrar reporte
    print_comparison_report(all_match, differences, matches, len(data1), len(data2))
    
    # Verificación adicional: comprobar que los IDs son consecutivos
    print("🔢 Verificando que los IDs sean consecutivos...")
    
    ids1 = sorted([item["id"] for item in data1])
    ids2 = sorted([item["id"] for item in data2])
    
    expected_ids = list(range(1, max(len(data1), len(data2)) + 1))
    
    if ids1 == expected_ids and ids2 == expected_ids:
        print("✅ Los IDs son consecutivos del 1 al", len(expected_ids))
    else:
        print("❌ Los IDs no son consecutivos:")
        if ids1 != expected_ids:
            print(f"   • Archivo 1 - IDs faltantes/incorrectos: {set(expected_ids) - set(ids1)}")
        if ids2 != expected_ids:
            print(f"   • Archivo 2 - IDs faltantes/incorrectos: {set(expected_ids) - set(ids2)}")


if __name__ == "__main__":
    main()
