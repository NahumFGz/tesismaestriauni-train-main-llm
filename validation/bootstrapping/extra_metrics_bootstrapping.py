#!/usr/bin/env python3
"""
Script para realizar bootstrapping de métricas extra (BERTScore, BLEU, ROUGE)
Procesa los archivos consolidados de attendance, procurement y voting
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Configuración global
N_BOOTSTRAP_SAMPLES = 2000  # Número de muestras bootstrap por defecto


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Carga datos desde un archivo JSON con estructura de métricas extra

    Args:
        file_path: Ruta al archivo JSON

    Returns:
        Lista de diccionarios con los datos
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extraer la lista de resultados
        if "results" in data:
            results = data["results"]
        else:
            results = data

        print(f"✓ Cargados {len(results)} elementos desde {file_path}")
        return results
    except Exception as e:
        print(f"✗ Error cargando {file_path}: {e}")
        return []


def extract_extra_metrics(data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Extrae las métricas extra de los datos

    Args:
        data: Lista de diccionarios con los datos

    Returns:
        Diccionario con listas de valores para cada métrica
    """
    metrics = {
        "bertscore_f1": [],
        "bleu": [],
        "rouge1_f1": [],
        "rouge2_f1": [],
        "rougeL_f1": [],
    }

    for item in data:
        if "metrics" in item:
            metrics_data = item["metrics"]

            # BERTScore F1
            if "bertscore" in metrics_data and "f1" in metrics_data["bertscore"]:
                metrics["bertscore_f1"].append(metrics_data["bertscore"]["f1"])

            # BLEU
            if "bleu" in metrics_data and "bleu" in metrics_data["bleu"]:
                metrics["bleu"].append(metrics_data["bleu"]["bleu"])

            # ROUGE F1 scores
            if "rouge" in metrics_data:
                rouge_data = metrics_data["rouge"]

                if "rouge1" in rouge_data and "f1" in rouge_data["rouge1"]:
                    metrics["rouge1_f1"].append(rouge_data["rouge1"]["f1"])

                if "rouge2" in rouge_data and "f1" in rouge_data["rouge2"]:
                    metrics["rouge2_f1"].append(rouge_data["rouge2"]["f1"])

                if "rougeL" in rouge_data and "f1" in rouge_data["rougeL"]:
                    metrics["rougeL_f1"].append(rouge_data["rougeL"]["f1"])

    return metrics


def bootstrap_metric(
    values: List[float], n_bootstrap: int = None, confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Realiza bootstrapping para una métrica específica

    Args:
        values: Lista de valores de la métrica
        n_bootstrap: Número de muestras bootstrap (usa valor global si es None)
        confidence_level: Nivel de confianza para los intervalos

    Returns:
        Diccionario con estadísticas bootstrap
    """
    if n_bootstrap is None:
        n_bootstrap = N_BOOTSTRAP_SAMPLES

    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_samples": 0,
        }

    values = np.array(values)
    n_samples = len(values)

    # Calcular estadísticas originales
    mean_original = np.mean(values)
    std_original = np.std(values, ddof=1)
    median_original = np.median(values)
    q25_original = np.percentile(values, 25)
    q75_original = np.percentile(values, 75)

    # Bootstrapping
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(values, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)

    # Calcular intervalos de confianza
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return {
        "mean": float(mean_original),
        "std": float(std_original),
        "median": float(median_original),
        "q25": float(q25_original),
        "q75": float(q75_original),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_samples": int(n_samples),
    }


def bootstrap_all_metrics(
    data: List[Dict[str, Any]], n_bootstrap: int = None
) -> Dict[str, Dict[str, float]]:
    """
    Realiza bootstrapping para todas las métricas extra

    Args:
        data: Lista de diccionarios con los datos
        n_bootstrap: Número de muestras bootstrap (usa valor global si es None)

    Returns:
        Diccionario con estadísticas bootstrap para cada métrica
    """
    if n_bootstrap is None:
        n_bootstrap = N_BOOTSTRAP_SAMPLES

    print("Extrayendo métricas extra de los datos...")
    metrics_data = extract_extra_metrics(data)

    print("Realizando bootstrapping para cada métrica...")
    bootstrap_results = {}

    for metric_name, values in metrics_data.items():
        print(f"  - Procesando {metric_name}: {len(values)} muestras")
        bootstrap_results[metric_name] = bootstrap_metric(values, n_bootstrap)

    return bootstrap_results


def process_single_file(
    file_path: str, file_name: str, output_dir: str, n_bootstrap: int = None
) -> None:
    """
    Procesa un archivo individual y guarda los resultados

    Args:
        file_path: Ruta al archivo JSON
        file_name: Nombre del archivo para identificar la fuente
        output_dir: Directorio de salida
        n_bootstrap: Número de muestras bootstrap
    """
    print(f"\n=== PROCESANDO {file_name.upper()} ===")

    # Cargar datos
    data = load_json_data(file_path)
    if not data:
        print(f"✗ No se pudieron cargar datos de {file_path}")
        return

    # Realizar bootstrapping
    print(f"Realizando bootstrapping para {file_name}...")
    bootstrap_results = bootstrap_all_metrics(data, n_bootstrap)

    # Preparar resultados
    results = {
        "metadata": {
            "source_file": file_name,
            "n_bootstrap_samples": n_bootstrap or N_BOOTSTRAP_SAMPLES,
            "total_elements": len(data),
            "timestamp": str(np.datetime64("now")),
        },
        "bootstrap_statistics": bootstrap_results,
    }

    # Guardar resultados
    output_path = os.path.join(output_dir, f"extra_metrics_{file_name}_bootstraping.json")
    save_results(results, output_path)

    # Mostrar resumen
    print(f"\n=== RESUMEN {file_name.upper()} ===")
    for metric_name, stats in bootstrap_results.items():
        print(
            f"{metric_name}: Media={stats['mean']:.4f}, IC95%=[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
        )


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Guarda los resultados en un archivo JSON

    Args:
        results: Diccionario con los resultados
        output_path: Ruta del archivo de salida
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✓ Resultados guardados en {output_path}")


def main():
    """
    Función principal del script
    """
    parser = argparse.ArgumentParser(description="Bootstrapping de métricas extra")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=None,
        help=f"Número de muestras bootstrap (default: {N_BOOTSTRAP_SAMPLES})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/bootstrapping/output",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--skip-individual", action="store_true", help="Saltar procesamiento individual de archivos"
    )
    parser.add_argument(
        "--skip-combined", action="store_true", help="Saltar procesamiento combinado"
    )
    args = parser.parse_args()

    # Usar valor global si no se especifica
    n_bootstrap = args.n_bootstrap or N_BOOTSTRAP_SAMPLES

    # Configuración de archivos
    file_configs = [
        {
            "path": "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/extra_metrics/consolidated_attendance_extra_metrics.json",
            "name": "attendance",
        },
        {
            "path": "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/extra_metrics/consolidated_procurement_extra_metrics.json",
            "name": "procurement",
        },
        {
            "path": "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/extra_metrics/consolidated_voting_extra_metrics.json",
            "name": "voting",
        },
    ]

    print("=== BOOTSTRAPPING DE MÉTRICAS EXTRA ===")
    print(f"Número de muestras bootstrap: {n_bootstrap}")
    print(f"Directorio de salida: {args.output_dir}")
    print()

    # Procesar archivos individuales
    if not args.skip_individual:
        print("=== PROCESAMIENTO INDIVIDUAL ===")
        for config in file_configs:
            process_single_file(config["path"], config["name"], args.output_dir, n_bootstrap)

    # Procesar archivos combinados
    if not args.skip_combined:
        print("\n=== PROCESAMIENTO COMBINADO ===")

        # Cargar todos los datos
        all_data = []
        file_names = []

        for config in file_configs:
            print(f"Cargando datos de {config['name']}...")
            data = load_json_data(config["path"])
            if data:
                # Agregar información del archivo origen
                for item in data:
                    item["source_file"] = config["name"]
                all_data.extend(data)
                file_names.append(config["name"])

        if not all_data:
            print("✗ No se pudieron cargar datos de ningún archivo")
            return

        print(f"\nTotal de elementos procesados: {len(all_data)}")

        # Realizar bootstrapping
        print("\nRealizando bootstrapping...")
        bootstrap_results = bootstrap_all_metrics(all_data, n_bootstrap)

        # Preparar resultados finales
        results = {
            "metadata": {
                "n_bootstrap_samples": n_bootstrap,
                "total_elements": len(all_data),
                "source_files": file_names,
                "timestamp": str(np.datetime64("now")),
            },
            "bootstrap_statistics": bootstrap_results,
        }

        # Guardar resultados
        output_path = os.path.join(args.output_dir, "extra_metrics_combined_bootstraping.json")
        save_results(results, output_path)

        # Mostrar resumen
        print("\n=== RESUMEN COMBINADO ===")
        for metric_name, stats in bootstrap_results.items():
            print(f"\n{metric_name.upper()}:")
            print(f"  Muestra: {stats['n_samples']} elementos")
            print(f"  Media: {stats['mean']:.4f}")
            print(f"  Desv. Est.: {stats['std']:.4f}")
            print(f"  Mediana: {stats['median']:.4f}")
            print(f"  Q25: {stats['q25']:.4f}")
            print(f"  Q75: {stats['q75']:.4f}")
            print(f"  IC 95%: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")

    print(f"\n✓ Proceso completado exitosamente")


if __name__ == "__main__":
    main()
