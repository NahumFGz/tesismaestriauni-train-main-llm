#!/usr/bin/env python3
"""
Script para contar la cantidad de IDs en los archivos answers.json y ground_truths.json
"""

import json
import os


def contar_ids_en_archivo(ruta_archivo):
    """
    Cuenta la cantidad de IDs únicos en un archivo JSON.

    Args:
        ruta_archivo (str): Ruta al archivo JSON

    Returns:
        tuple: (cantidad_total, cantidad_ids_unicos, lista_ids)
    """
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            datos = json.load(archivo)

        if not isinstance(datos, list):
            print(f"Error: El archivo {ruta_archivo} no contiene una lista JSON")
            return 0, 0, []

        ids = []
        for item in datos:
            if "id" in item:
                ids.append(item["id"])

        ids_unicos = list(set(ids))
        ids_unicos.sort()

        return len(ids), len(ids_unicos), ids_unicos

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_archivo}")
        return 0, 0, []
    except json.JSONDecodeError:
        print(f"Error: El archivo {ruta_archivo} no es un JSON válido")
        return 0, 0, []
    except Exception as e:
        print(f"Error inesperado al procesar {ruta_archivo}: {e}")
        return 0, 0, []


def main():
    # Rutas de los archivos
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    archivo_answers = os.path.join(directorio_actual, "answers.json")
    archivo_ground_truths = os.path.join(directorio_actual, "ground_truths.json")

    print("=" * 60)
    print("CONTADOR DE IDs EN ARCHIVOS JSON")
    print("=" * 60)

    # Contar IDs en answers.json
    print(f"\nProcesando: {archivo_answers}")
    total_answers, unicos_answers, lista_answers = contar_ids_en_archivo(archivo_answers)

    if total_answers > 0:
        print(f"✓ Total de registros: {total_answers}")
        print(f"✓ IDs únicos: {unicos_answers}")
        if total_answers != unicos_answers:
            print(f"⚠️  Hay IDs duplicados!")
        print(f"✓ Rango de IDs: {min(lista_answers)} - {max(lista_answers)}")

    # Contar IDs en ground_truths.json
    print(f"\nProcesando: {archivo_ground_truths}")
    total_ground_truths, unicos_ground_truths, lista_ground_truths = contar_ids_en_archivo(
        archivo_ground_truths
    )

    if total_ground_truths > 0:
        print(f"✓ Total de registros: {total_ground_truths}")
        print(f"✓ IDs únicos: {unicos_ground_truths}")
        if total_ground_truths != unicos_ground_truths:
            print(f"⚠️  Hay IDs duplicados!")
        print(f"✓ Rango de IDs: {min(lista_ground_truths)} - {max(lista_ground_truths)}")

    # Comparación entre archivos
    print("\n" + "=" * 60)
    print("COMPARACIÓN ENTRE ARCHIVOS")
    print("=" * 60)

    if total_answers > 0 and total_ground_truths > 0:
        set_answers = set(lista_answers)
        set_ground_truths = set(lista_ground_truths)

        ids_comunes = set_answers.intersection(set_ground_truths)
        ids_solo_answers = set_answers - set_ground_truths
        ids_solo_ground_truths = set_ground_truths - set_answers

        print(f"IDs en común: {len(ids_comunes)}")
        print(f"IDs solo en answers.json: {len(ids_solo_answers)}")
        print(f"IDs solo en ground_truths.json: {len(ids_solo_ground_truths)}")

        if ids_solo_answers:
            print(f"IDs únicos de answers.json: {sorted(list(ids_solo_answers))}")

        if ids_solo_ground_truths:
            print(f"IDs únicos de ground_truths.json: {sorted(list(ids_solo_ground_truths))}")

        if len(ids_comunes) == len(set_answers) == len(set_ground_truths):
            print("✓ Ambos archivos tienen exactamente los mismos IDs")
        else:
            print("⚠️  Los archivos tienen diferentes conjuntos de IDs")

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Total de IDs en answers.json: {total_answers}")
    print(f"Total de IDs en ground_truths.json: {total_ground_truths}")
    print(f"Total combinado: {total_answers + total_ground_truths}")


if __name__ == "__main__":
    main()
