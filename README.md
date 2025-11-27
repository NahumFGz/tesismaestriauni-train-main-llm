# tesismaestriauni-train-main-llm

Sistema de agente conversacional LLM para transparencia gubernamental del Estado peruano, con capacidades de consulta sobre votaciones, asistencias de congresistas, contrataciones públicas y búsqueda web.

## Ejecución del Sistema Principal

### Menú interactivo (por defecto)

```bash
python main.py
```

### Modo rápido - prueba una pregunta de cada categoría

```bash
python main.py --rapido
```

### Pregunta directa

```bash
python main.py --pregunta "¿Cuáles fueron las votaciones del 15 de marzo de 2024?"
```

### Pregunta con streaming

```bash
python main.py --streaming "Dame información sobre contratos públicos"
```

## Sistema de Validación y Evaluación

El sistema incluye un módulo completo de validación que evalúa la calidad de las respuestas del agente LLM usando múltiples métricas.

### Requisitos Previos para Validación

1. **Servicios externos funcionando:**

   - Base de datos PostgreSQL con datos de transparencia
   - Qdrant vector database para búsquedas semánticas
   - APIs de OpenAI y Anthropic configuradas

2. **Variables de entorno configuradas:**

   ```bash
   # APIs LLM
   OPENAI_API_KEY=tu_clave_openai
   ANTHROPIC_API_KEY=tu_clave_anthropic

   # Base de datos
   DATABASE_URL=postgresql://usuario:password@localhost:5432/transparency_db

   # Qdrant
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=tu_clave_qdrant  # opcional
   ```

### Instalación de Dependencias de Validación

```bash
# Instalar dependencias específicas para validación
pip install -r validation/requirements.txt
```

### Configuración del PYTHONPATH

Para que el sistema de validación funcione correctamente, configura el PYTHONPATH:

```bash
# Linux/Mac
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Windows
set PYTHONPATH=%PYTHONPATH%;%cd%
```

### Ejecución de Validaciones

#### 1. Validación Completa

Ejecuta todas las métricas de evaluación (RAGAS + LLM Score + métricas personalizadas):

```bash
python validation/run_validation.py full
```

**Qué incluye:**

- Creación automática de dataset de prueba
- Evaluación con métricas RAGAS (faithfulness, answer_relevancy, context_precision, etc.)
- Evaluación con LLM como juez (accuracy, relevance, completeness, clarity)
- Métricas personalizadas (similaridad semántica, extracción de entidades, etc.)
- Generación de reporte comprehensivo con visualizaciones
- Exportación de resultados en JSON y HTML

#### 2. Validación Rápida

Evalúa una pregunta específica:

```bash
python validation/run_validation.py quick "¿Cuáles fueron las votaciones del congreso en marzo 2024?"
```

#### 3. Creación de Dataset Personalizado

Genera un dataset de prueba personalizado:

```bash
python validation/run_validation.py create-dataset
```

#### 4. Opciones Adicionales

```bash
# Validación completa con dataset personalizado
python validation/run_validation.py full --dataset mi_dataset.json

# Validación rápida con métricas específicas
python validation/run_validation.py quick "pregunta" --metrics ragas llm_score

# Ver ayuda completa
python validation/run_validation.py --help
```

### Interpretación de Resultados

#### Métricas RAGAS (0-1, mayor es mejor):

- **Faithfulness**: Qué tan fiel es la respuesta al contexto
- **Answer Relevancy**: Relevancia de la respuesta a la pregunta
- **Context Precision**: Precisión del contexto recuperado
- **Context Recall**: Completitud del contexto recuperado
- **Answer Correctness**: Corrección factual de la respuesta
- **Answer Similarity**: Similaridad semántica con respuesta esperada

#### Métricas LLM Score (1-10, mayor es mejor):

- **Accuracy**: Precisión factual
- **Relevance**: Relevancia al tema
- **Completeness**: Completitud de la información
- **Clarity**: Claridad de la explicación

#### Métricas Personalizadas:

- **Semantic Similarity**: Similaridad semántica con respuesta esperada
- **Entity Extraction**: Extracción correcta de entidades nombradas
- **Domain Keywords**: Uso de palabras clave del dominio
- **Transparency Compliance**: Cumplimiento de principios de transparencia

### Archivos de Salida

Los resultados se guardan en:

- `validation/results/validation_report_YYYYMMDD_HHMMSS.json`: Resultados detallados
- `validation/results/validation_report_YYYYMMDD_HHMMSS.html`: Reporte visual
- `validation/results/plots/`: Gráficos y visualizaciones

### Troubleshooting

#### Error "ModuleNotFoundError: No module named 'validation'"

```bash
# Asegúrate de configurar PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Error de conexión a bases de datos

```bash
# Verifica que los servicios estén ejecutándose
docker ps  # si usas Docker
# o verifica conexiones directas a PostgreSQL y Qdrant
```

#### Error de APIs LLM

```bash
# Verifica que las claves API estén configuradas
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### Personalización

Para personalizar las validaciones:

1. **Modificar dataset**: Edita `validation/dataset_creator.py`
2. **Agregar métricas**: Extiende `validation/metrics.py`
3. **Personalizar reportes**: Modifica `validation/reports.py`
4. **Ajustar criterios LLM**: Edita prompts en `validation/llm_scorer.py`

## Estructura del Proyecto

```
├── app/                    # Código principal del agente
│   ├── tools/             # Herramientas de transparencia
│   └── llm.py            # Lógica principal del LLM
├── validation/            # Sistema de validación
│   ├── run_validation.py # Script principal de validación
│   ├── ragas_evaluator.py # Evaluación RAGAS
│   ├── llm_scorer.py     # Evaluación con LLM
│   └── results/          # Resultados de validaciones
├── main.py               # Interfaz principal
└── README.md             # Este archivo
```
