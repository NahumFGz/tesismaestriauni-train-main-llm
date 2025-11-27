# Análisis de Métricas de Bootstrapping - Evaluación del Modelo LLM

Este documento presenta un análisis comparativo de las métricas obtenidas mediante bootstrapping (2000 muestras) para evaluar el rendimiento del modelo LLM en diferentes dominios: asistencia (attendance), votación (voting), adquisiciones (procurement) y el conjunto combinado (all).

## Resumen de Datos

- **Método**: Bootstrap con 2000 muestras
- **Fecha de evaluación**: 24 de septiembre de 2025

## Descripción de Métricas

### Métricas RAGAS

**Métricas de Generación:**

- **Answer Relevancy**: Evalúa qué tan pertinente es la respuesta generada respecto a la pregunta formulada

  - _Fortaleza_: No requiere ground truth, útil en producción
  - _Debilidad_: Puede considerar relevantes respuestas factualmente incorrectas

- **Answer Similarity**: Mide la similitud semántica entre la respuesta generada y la respuesta de referencia (ground truth)

  - _Fortaleza_: Captura similitud semántica profunda usando embeddings
  - _Debilidad_: Penaliza respuestas correctas expresadas de forma diferente

- **Answer Correctness**: Combina similitud semántica y exactitud factual comparando con el ground truth

  - _Fortaleza_: Métrica más completa que considera tanto forma como contenido
  - _Debilidad_: Altamente dependiente de la calidad del ground truth de referencia

- **Faithfulness**: Verifica si la respuesta generada se basa fielmente en el contexto proporcionado, sin alucinaciones
  - _Fortaleza_: Detecta alucinaciones sin necesidad de ground truth
  - _Debilidad_: No evalúa si el contexto mismo es correcto o completo

**Métricas de Recuperación:**

- **Context Precision**: Mide la proporción de contextos recuperados que son realmente relevantes para la pregunta

  - _Fortaleza_: Identifica ruido informativo en la recuperación
  - _Debilidad_: Requiere anotación manual de relevancia de contextos

- **Context Recall**: Evalúa qué fracción de todos los contextos relevantes fue efectivamente recuperada por el sistema
  - _Fortaleza_: Detecta problemas de cobertura en la recuperación
  - _Debilidad_: Difícil determinar el conjunto completo de contextos "relevantes"

### Métricas de Similitud Textual

- **BERTScore F1**: Utiliza embeddings contextuales para evaluar similitud semántica entre textos generados y de referencia

  - _Fortaleza_: Captura similitud semántica profunda, robusto a paráfrasis
  - _Debilidad_: Puede no detectar errores factuales sutiles pero semánticamente coherentes

- **BLEU**: Mide coincidencias de n-gramas entre texto generado y referencia, comúnmente usado en traducción automática

  - _Fortaleza_: Métrica estándar, fácil interpretación, rápido cálculo
  - _Debilidad_: Se enfoca en coincidencias exactas, ignora similitud semántica

- **ROUGE-1 F1**: Evalúa solapamiento de unigramas (palabras individuales) entre texto generado y referencia

  - _Fortaleza_: Mide cobertura de conceptos clave, menos sensible al orden
  - _Debilidad_: No considera relaciones entre palabras ni coherencia

- **ROUGE-2 F1**: Mide solapamiento de bigramas (pares de palabras consecutivas) para capturar fluidez local

  - _Fortaleza_: Captura algo de estructura local y fluidez del texto
  - _Debilidad_: Muy restrictivo, penaliza sinónimos y paráfrasis válidas

- **ROUGE-L F1**: Analiza la subsecuencia común más larga, capturando similitudes estructurales y de orden
  - _Fortaleza_: Considera orden y estructura del texto completo
  - _Debilidad_: Muy sensible a reorganización textual, incluso si es válida

## Cuadros Comparativos de Métricas

### Métricas RAGAS (Retrieval Augmented Generation Assessment)

| Métrica                                 | Attendance                      | Voting                          | Procurement                     | All (Combined)                  |
| --------------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| **Answer Relevancy** _(sin GT • Gen)_   | 0.907 ± 0.190<br>[0.867, 0.940] | 0.937 ± 0.117<br>[0.904, 0.955] | 0.890 ± 0.122<br>[0.865, 0.908] | 0.907 ± 0.149<br>[0.889, 0.923] |
| **Answer Similarity** _(con GT • Gen)_  | 0.987 ± 0.018<br>[0.983, 0.990] | 0.975 ± 0.010<br>[0.972, 0.977] | 0.986 ± 0.014<br>[0.983, 0.988] | 0.984 ± 0.016<br>[0.982, 0.985] |
| **Answer Correctness** _(con GT • Gen)_ | 0.905 ± 0.187<br>[0.867, 0.939] | 0.858 ± 0.221<br>[0.800, 0.905] | 0.899 ± 0.141<br>[0.874, 0.922] | 0.891 ± 0.179<br>[0.871, 0.911] |
| **Faithfulness** _(sin GT • Gen)_       | 0.957 ± 0.114<br>[0.932, 0.977] | 0.977 ± 0.089<br>[0.955, 0.994] | 0.910 ± 0.092<br>[0.894, 0.925] | 0.942 ± 0.103<br>[0.930, 0.954] |
| **Context Precision** _(con GT • Ret)_  | 0.986 ± 0.102<br>[0.962, 0.999] | 1.000 ± 0.000<br>[1.000, 1.000] | 0.711 ± 0.300<br>[0.658, 0.762] | 0.876 ± 0.245<br>[0.849, 0.901] |
| **Context Recall** _(con GT • Ret)_     | 0.748 ± 0.190<br>[0.711, 0.785] | 0.857 ± 0.226<br>[0.798, 0.905] | 0.826 ± 0.266<br>[0.780, 0.869] | 0.807 ± 0.236<br>[0.780, 0.833] |

**Leyenda:** _GT = Ground Truth | Gen = Generación | Ret = Recuperación_

### Métricas de Similitud Textual (Extra Metrics)

| Métrica          | Attendance                      | Voting                          | Procurement                     | All (Combined)                  |
| ---------------- | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| **BERTScore F1** | 0.773 ± 0.222<br>[0.727, 0.812] | 0.661 ± 0.094<br>[0.637, 0.681] | 0.764 ± 0.180<br>[0.733, 0.795] | 0.742 ± 0.186<br>[0.721, 0.763] |
| **BLEU**         | 0.679 ± 0.220<br>[0.633, 0.721] | 0.602 ± 0.162<br>[0.564, 0.640] | 0.634 ± 0.247<br>[0.588, 0.679] | 0.642 ± 0.221<br>[0.617, 0.667] |
| **ROUGE-1 F1**   | 0.800 ± 0.162<br>[0.766, 0.829] | 0.831 ± 0.080<br>[0.812, 0.850] | 0.813 ± 0.140<br>[0.788, 0.837] | 0.813 ± 0.137<br>[0.797, 0.828] |
| **ROUGE-2 F1**   | 0.692 ± 0.178<br>[0.655, 0.725] | 0.734 ± 0.090<br>[0.713, 0.754] | 0.701 ± 0.213<br>[0.661, 0.738] | 0.706 ± 0.178<br>[0.686, 0.726] |
| **ROUGE-L F1**   | 0.762 ± 0.177<br>[0.726, 0.796] | 0.607 ± 0.088<br>[0.587, 0.629] | 0.748 ± 0.191<br>[0.713, 0.783] | 0.719 ± 0.178<br>[0.698, 0.738] |

## Interpretación de Resultados por Dominio

### Dominio Voting (70 elementos)

**Fortalezas:**

- **Context Precision perfecta (1.000)**: El sistema recupera únicamente contextos relevantes, sin ruido informativo
- **Answer Relevancy alta (0.937)**: Las respuestas son altamente pertinentes a las preguntas formuladas
- **Faithfulness excelente (0.977)**: Las respuestas se basan fielmente en el contexto proporcionado
- **Menor variabilidad general**: Indica consistencia y predictibilidad del modelo en este dominio

**Limitaciones identificadas:**

- **ROUGE-L F1 más bajo (0.607)**: Sugiere diferencias en la estructura de secuencias largas comparado con ground truth
- **BERTScore F1 moderado (0.661)**: Posibles diferencias semánticas sutiles en las respuestas generadas

### Dominio Attendance (100 elementos)

**Fortalezas:**

- **Answer Similarity excepcional (0.987)**: Las respuestas son muy similares al ground truth esperado
- **Faithfulness alta (0.957)**: Excelente adherencia al contexto proporcionado
- **Context Precision alta (0.986)**: Recuperación precisa de contextos relevantes

**Áreas de mejora:**

- **Context Recall moderado (0.748)**: El sistema no recupera todos los contextos relevantes disponibles
- **Mayor variabilidad en métricas textuales**: Inconsistencia en la generación de texto específico del dominio

### Dominio Procurement (120 elementos)

**Desafíos significativos:**

- **Faithfulness baja (0.713 ± 0.313)**: Alta variabilidad indica problemas de consistencia en adherencia al contexto
- **Context Precision problemática (0.711 ± 0.300)**: Recuperación imprecisa con alta variabilidad
- **Mayor desviación estándar general**: Sugiere que el modelo tiene dificultades para manejar la complejidad de este dominio

**Fortalezas relativas:**

- **Answer Similarity alta (0.986)**: Cuando genera respuestas, mantiene similitud semántica con el ground truth
- **Context Recall aceptable (0.826)**: Capacidad razonable para recuperar contextos relevantes

### Conjunto Combinado (All - 290 elementos)

**Rendimiento balanceado:**

- Representa el promedio ponderado de todos los dominios
- **Métricas estables**: Answer Similarity (0.984), ROUGE-1 F1 (0.813) muestran consistencia cross-domain
- **Variabilidad controlada**: Los intervalos de confianza indican robustez estadística general

## Limitaciones de las Métricas por Tipo

### Métricas sin Ground Truth _(sin GT)_

**Answer Relevancy:**

- _Limitación_: No puede detectar respuestas factualmente incorrectas pero relevantes
- _Sesgo_: Favorece respuestas verbosas que parecen relevantes

**Faithfulness:**

- _Limitación_: Solo evalúa consistencia con contexto, no veracidad absoluta
- _Problema_: Puede ser alta incluso con contextos incorrectos o incompletos

### Métricas con Ground Truth _(con GT)_

**Answer Similarity & Correctness:**

- _Limitación_: Dependientes de la calidad del ground truth de referencia
- _Sesgo_: Penalizan respuestas correctas pero expresadas diferentemente

**Context Precision & Recall:**

- _Limitación_: Requieren anotación manual de contextos relevantes
- _Problema_: Pueden no capturar contextos implícitamente relevantes

### Métricas de Similitud Textual

**BERTScore F1:**

- _Fortaleza_: Captura similitud semántica profunda
- _Limitación_: Puede no detectar errores factuales sutiles

**BLEU & ROUGE:**

- _Limitación_: Enfoque en coincidencias n-gram, no en significado
- _Sesgo_: Favorecen respuestas que usan vocabulario similar al ground truth

**ROUGE-L F1:**

- _Limitación específica_: Sensible al orden de palabras y estructura de secuencias
- _Problema_: Puede penalizar respuestas correctas con diferente organización textual

## Recomendaciones por Dominio

1. **Voting**: Mantener el enfoque actual, optimizar para ROUGE-L
2. **Attendance**: Mejorar estrategias de recuperación de contexto (Context Recall)
3. **Procurement**: Requiere intervención prioritaria en calidad de contexto y consistencia
4. **General**: Implementar métricas complementarias que no dependan exclusivamente de ground truth

---

_Análisis generado automáticamente el 24 de septiembre de 2025_
