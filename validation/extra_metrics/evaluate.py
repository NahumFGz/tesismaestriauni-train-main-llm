"""
Evalúa respuestas con métricas clásicas de generación (BLEU, ROUGE, BERTScore).

ENTRADA:
- Archivo JSON con lista de items, cada uno con:
  {
    "elemento_id": ...,
    "question": "...",
    "answer": "...",
    "ground_truth": "..."
  }

SALIDA:
- Archivo JSON con la misma lista, añadiendo un bloque "metrics" con:
  {
    "bleu": {
      "bleu": ...,            # BLEU-4 normalizado (0–1) → este es el valor a reportar
      "precisions": [...],    # Precisión por n-grama (1-gram, 2-gram, 3-gram, 4-gram)
      "brevity_penalty": ..., # Penalización por respuestas demasiado cortas (<1.0 si aplica)
      "length_ratio": ...,    # Longitud relativa (resp/ref)
      "hyp_len": ...,         # Longitud de la respuesta
      "ref_len": ...          # Longitud del ground truth
    },
    "rouge": {
      "rouge1": {"precision": ..., "recall": ..., "f1": ...}, # usar rouge1["f1"]
      "rouge2": {"precision": ..., "recall": ..., "f1": ...}, # usar rouge2["f1"]
      "rougeL": {"precision": ..., "recall": ..., "f1": ...}  # usar rougeL["f1"]
    },
    "bertscore": {
      "precision": ...,       # Cuánto del texto generado es semánticamente correcto
      "recall": ...,          # Cuánto de la referencia está cubierto semánticamente
      "f1": ...,              # ESTE es el valor a reportar en el paper
      "lang": "es"            # Idioma usado para embeddings (XLM-R por defecto)
    }
  }

CÓMO INTERPRETAR LOS RESULTADOS:
- BLEU ("bleu"): mide coincidencia literal de n-gramas. Un valor ~0.5 indica coincidencia parcial
  con diferencias de redacción o longitud. Muy útil como baseline histórico,
  pero poco robusto a sinónimos/paráfrasis.
- ROUGE (tomar los f1 de "rouge1", "rouge2" y "rougeL"): mide recuperación de información.
  ROUGE-1 refleja palabras, ROUGE-2 refleja pares, ROUGE-L refleja subsecuencias largas.
  Recall alto (>0.7) indica que la respuesta cubre gran parte de la referencia.
- BERTScore ("f1"): mide similitud semántica usando embeddings. Valores en 0.6–0.8 se consideran
  moderados, >0.85 altos. Es más robusto que BLEU/ROUGE para LLMs, porque reconoce
  sinónimos y paráfrasis.

QUÉ MÉTRICAS INCLUIR EN EL PAPER:
- Principales (para tu caso con agentes RAG y generación abierta):
  - Métricas RAGAS: Faithfulness, Answer Relevancy, Context Recall, Context Precision, Answer Correctness.
  - BERTScore (f1) como métrica semántica complementaria, más cercana a juicios humanos.
- Complementarias (para comparación con estudios previos):
  - BLEU ("bleu") y ROUGE (rouge1.f1, rouge2.f1, rougeL.f1) como baseline clásico.
    Útiles para mostrar consistencia con literatura histórica, pero aclarando sus limitaciones.

En resumen:
- Usa RAGAS + BERTScore (f1) como métricas centrales en resultados.
- Reporta BLEU ("bleu") y ROUGE (f1 de 1/2/L) en una tabla comparativa o anexo como métricas de referencia.
"""

import json
import logging
import os
from datetime import datetime

# BLEU
import sacrebleu

# BERTScore
from bert_score import score as bertscore

# ROUGE
from rouge_score import rouge_scorer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("evaluation.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)

# ==============================
# RUTAS POR DEFECTO
# ==============================
INPUT_FILES = {
    "attendance": "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/bootstrapping/input/consolidated_attendance_ragas.json",
    "procurement": "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/bootstrapping/input/consolidated_procurement_ragas.json",
    "voting": "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/bootstrapping/input/consolidated_voting_ragas.json",
}

OUTPUT_DIR = "/home/nahumfg/Projects/GithubProjects/tesismaestriauni-launcher/train-main-llm/validation_boostrap/extra_metrics/"


def compute_bleu(hyp: str, ref: str):
    logger.debug(f"Calculando BLEU para hipótesis: '{hyp[:50]}...' y referencia: '{ref[:50]}...'")
    sb = sacrebleu.sentence_bleu(hyp, [ref], smooth_method="exp")
    result = {
        "bleu": sb.score / 100.0,
        "precisions": [p / 100.0 for p in sb.precisions],
        "brevity_penalty": sb.bp,
        "length_ratio": sb.sys_len / max(sb.ref_len, 1),
        "hyp_len": sb.sys_len,
        "ref_len": sb.ref_len,
    }
    logger.debug(f"BLEU calculado: {result['bleu']:.4f}")
    return result


def compute_rouge(hyp: str, ref: str):
    logger.debug(f"Calculando ROUGE para hipótesis: '{hyp[:50]}...' y referencia: '{ref[:50]}...'")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, hyp)

    def pack(s):
        return {"precision": s.precision, "recall": s.recall, "f1": s.fmeasure}

    result = {
        "rouge1": pack(scores["rouge1"]),
        "rouge2": pack(scores["rouge2"]),
        "rougeL": pack(scores["rougeL"]),
    }
    logger.debug(f"ROUGE calculado - ROUGE-1 F1: {result['rouge1']['f1']:.4f}")
    return result


def compute_bertscore(hyp: str, ref: str, lang="es"):
    logger.debug(
        f"Calculando BERTScore para hipótesis: '{hyp[:50]}...' y referencia: '{ref[:50]}...'"
    )
    P, R, F1 = bertscore([hyp], [ref], lang=lang, rescale_with_baseline=True)
    result = {"precision": float(P[0]), "recall": float(R[0]), "f1": float(F1[0]), "lang": lang}
    logger.debug(f"BERTScore calculado - F1: {result['f1']:.4f}")
    return result


def process_single_file(input_path: str, output_path: str, dataset_name: str):
    """Procesa un solo archivo de entrada y genera el archivo de salida correspondiente."""
    logger.info(f"Procesando dataset: {dataset_name}")
    logger.info(f"Archivo de entrada: {input_path}")
    logger.info(f"Archivo de salida: {output_path}")

    try:
        logger.info("Cargando datos de entrada...")
        with open(input_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        logger.info(f"✓ Datos cargados exitosamente. Total de items: {len(items)}")
    except FileNotFoundError:
        logger.error(f"✗ Error: No se encontró el archivo {input_path}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"✗ Error al decodificar JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error inesperado al cargar datos: {e}")
        return False

    evaluated = []
    items_con_metricas = 0
    items_sin_metricas = 0

    logger.info("Iniciando procesamiento de items...")
    for i, it in enumerate(items, 1):
        question = it.get("question", "")
        answer = it.get("answer", "")
        ground_truth = it.get("ground_truth", "")
        elemento_id = it.get("elemento_id")

        logger.info(f"Procesando item {i}/{len(items)} - ID: {elemento_id}")
        logger.debug(f"Pregunta: {question[:100]}...")
        logger.debug(f"Respuesta: {answer[:100]}...")
        logger.debug(f"Verdad de referencia: {ground_truth[:100]}...")

        if not answer.strip() or not ground_truth.strip():
            logger.warning(f"Item {i} omitido - respuesta o verdad de referencia vacía")
            metrics = {"bleu": None, "rouge": None, "bertscore": None}
            items_sin_metricas += 1
        else:
            logger.info(f"Calculando métricas para item {i}...")
            try:
                metrics = {
                    "bleu": compute_bleu(answer, ground_truth),
                    "rouge": compute_rouge(answer, ground_truth),
                    "bertscore": compute_bertscore(answer, ground_truth, lang="es"),
                }
                items_con_metricas += 1
                logger.info(f"✓ Métricas calculadas para item {i}")
            except Exception as e:
                logger.error(f"✗ Error calculando métricas para item {i}: {e}")
                metrics = {"bleu": None, "rouge": None, "bertscore": None}
                items_sin_metricas += 1

        evaluated.append(
            {
                "elemento_id": elemento_id,
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "metrics": metrics,
            }
        )

        # Log de progreso cada 10 items
        if i % 10 == 0:
            logger.info(f"Progreso: {i}/{len(items)} items procesados ({i/len(items)*100:.1f}%)")

    logger.info("=" * 40)
    logger.info(f"RESUMEN DE PROCESAMIENTO - {dataset_name.upper()}")
    logger.info("=" * 40)
    logger.info(f"Total de items procesados: {len(evaluated)}")
    logger.info(f"Items con métricas calculadas: {items_con_metricas}")
    logger.info(f"Items sin métricas (datos vacíos): {items_sin_metricas}")

    out = {"results": evaluated}

    try:
        logger.info("Guardando resultados...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Archivo guardado exitosamente en: {output_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Error al guardar archivo: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("INICIANDO EVALUACIÓN DE MÉTRICAS EXTRA")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Directorio de salida: {OUTPUT_DIR}")

    # Procesar cada archivo
    results = {}
    for dataset_name, input_path in INPUT_FILES.items():
        logger.info("=" * 60)
        logger.info(f"PROCESANDO DATASET: {dataset_name.upper()}")
        logger.info("=" * 60)

        output_path = os.path.join(OUTPUT_DIR, f"consolidated_{dataset_name}_extra_metrics.json")
        success = process_single_file(input_path, output_path, dataset_name)
        results[dataset_name] = {
            "success": success,
            "input_path": input_path,
            "output_path": output_path,
        }

    # Resumen final
    logger.info("=" * 60)
    logger.info("RESUMEN FINAL DE TODOS LOS DATASETS")
    logger.info("=" * 60)

    successful_datasets = 0
    for dataset_name, result in results.items():
        status = "✓ EXITOSO" if result["success"] else "✗ FALLÓ"
        logger.info(f"{dataset_name.upper()}: {status}")
        if result["success"]:
            successful_datasets += 1
            print(f"[OK] {dataset_name}: Evaluación completada - {result['output_path']}")
        else:
            print(f"[ERROR] {dataset_name}: Falló el procesamiento - {result['input_path']}")

    logger.info("=" * 60)
    logger.info("EVALUACIÓN COMPLETADA")
    logger.info("=" * 60)
    logger.info(f"Datasets procesados exitosamente: {successful_datasets}/{len(INPUT_FILES)}")
    print(f"[INFO] Logs detallados guardados en: evaluation.log")


if __name__ == "__main__":
    main()
