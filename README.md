# NLP Analyzer – Proyecto Final

Sistema modular para análisis de lenguaje natural orientado a textos en español e inglés.  
Incluye **preprocesamiento**, **n-gramas**, **wordclouds**, **modelado de tópicos con BERTopic**, **análisis de outliers**, **ablación de palabras**, **visualizaciones 3D de embeddings** y un generador de **reportes HTML**.

---

## 1. Estructura del proyecto

```bash
proyecto_final/
├── nlp_analyzer.py          # Script principal
├── reporte_nlp.html         # Ejemplo de reporte generado
│
├── config/                  # Configuración del proyecto
│   └── settings.py
│
├── data_input/              # Datos de entrada de ejemplo
│   └── test.csv
│
├── processing/              # Módulos del pipeline de NLP
│   ├── preprocess.py        # Preprocesamiento y limpieza de texto
│   ├── ngrams.py            # Cálculo y visualización de n-gramas
│   ├── wordcloud.py         # Generación de nubes de palabras
│   ├── topics.py            # Modelo de tópicos con BERTopic
│   ├── outliers.py          # Análisis de outliers (tópico -1)
│   ├── ablation.py          # Ablación de keywords por tópico
│   └── visualization.py     # Reducción de dimensionalidad y gráficas 2D/3D
│
├── utils/
│   └── color_palettes.py    # Paletas de color (incluye opciones para daltónicos)
│
└── web_report/
    └── generator.py         # Generación del reporte HTML final
```

---

## 2. Uso

Ejecutar desde la raíz del proyecto:

```bash
python nlp_analyzer.py \
    -f data_input/test.csv \
    -c texto \
    -l spanish \
    -p okabe_ito \
    -t "Reporte NLP"
```

### 2.1 Argumentos principales

| Opción | Nombre largo | Descripción | Ejemplo |
|--------|--------------|-------------|---------|
| `-f` | `--File` | Ruta del archivo CSV de entrada. | `-f data_input/test.csv` |
| `-c` | `--Column_name` | Nombre de la columna que contiene los textos. | `-c texto` |
| `-l` | `--Language` | Idioma del texto: `spanish` o `english`. | `-l spanish` |
| `-p` | `--palette` | Paleta de colores definida en `utils/color_palettes.py`. | `-p okabe_ito` |
| `-t` | `--Title` | Título del reporte HTML generado. | `-t "Reporte NLP"` |

El archivo HTML resultante resume, de forma integrada:

- Preprocesamiento de texto
- N-gramas (unigramas, bigramas, trigramas)
- Wordclouds
- Tópicos generados con BERTopic
- Visualizaciones 3D (UMAP y t-SNE)
- Análisis de outliers
- Ablación de keywords por tópico

---

## 3. Flujo del pipeline

El script principal `nlp_analyzer.py` orquesta el siguiente flujo:

1. Preprocesamiento de textos (limpieza, normalización, lematización).
2. Generación de n-gramas (unigramas, bigramas, trigramas) y sus frecuencias.
3. Creación de wordclouds con paletas personalizadas.
4. Modelado de tópicos con BERTopic y SentenceTransformers.
5. Ablación de keywords para encontrar términos exclusivos y representativos por tópico.
6. Análisis de outliers (tópico -1 de BERTopic).
7. Reducción de dimensionalidad (UMAP, t-SNE) y visualizaciones 3D de embeddings.
8. Generación del reporte HTML con tablas, imágenes y gráficas (potencialmente interactivas).

---

## 4. Descripción de módulos

### 4.1 `nlp_analyzer.py`

Script principal del proyecto.

- Lee argumentos de línea de comandos.
- Conecta los módulos de `processing/`, `utils/` y `web_report/`.
- Ejecuta el pipeline completo y genera el reporte HTML final.

### 4.2 `config/settings.py`

Centraliza y valida la configuración del proyecto:

- Ruta del archivo CSV.
- Nombre de la columna de texto.
- Título del reporte.
- Paleta de colores seleccionada.
- Evita repetir lógica de validación en el resto de módulos.

### 4.3 `processing/preprocess.py`

Encargado del preprocesamiento de texto, incluyendo:

- Conversión a minúsculas.
- Limpieza básica (signos, caracteres especiales, etc.).
- Eliminación de stopwords.
- Lematización con spaCy.

Soporta dos idiomas principales:

- `spanish`
- `english`

Adapta el pipeline de procesamiento dependiendo del idioma seleccionado.

### 4.4 `processing/ngrams.py`

Genera:

- Unigramas
- Bigramas
- Trigramas

Calcula frecuencias u otras métricas asociadas. Puede producir visualizaciones (por ejemplo, barras con Matplotlib) para mostrar los n-gramas más frecuentes.

### 4.5 `processing/wordcloud.py`

Crea nubes de palabras a partir de los tokens preprocesados.

- Utiliza las paletas definidas en `utils/color_palettes.py`.
- Permite utilizar paletas aptas para personas con daltonismo (por ejemplo, Okabe–Ito).
- Genera imágenes que pueden ser posteriormente embebidas en el reporte HTML.

### 4.6 `processing/topics.py`

Implementa el modelado de tópicos con BERTopic y SentenceTransformers.

**Responsabilidades principales:**

- Entrenar el modelo de BERTopic sobre el corpus.
- Obtener la asignación de tópico por documento.
- Extraer keywords representativas por tópico.
- Calcular y almacenar los embeddings de oraciones/documentos para usos posteriores (visualizaciones, outliers, etc.).

### 4.7 `processing/outliers.py`

Se enfoca en el análisis de los documentos asignados al tópico `-1` de BERTopic, considerados como outliers.

**Funcionalidades:**

- Resumen estadístico (por ejemplo, longitud de textos).
- Ejemplos de textos fuera de los tópicos principales.
- Información que ayuda a interpretar por qué ciertos textos no encajan en ningún tópico dominante.

### 4.8 `processing/ablation.py`

Realiza ablación de keywords por tópico, es decir:

- Identifica palabras especialmente representativas de un tópico.
- Compara la presencia de términos entre tópicos.
- Facilita la depuración de vocabulario y el entendimiento semántico de cada grupo temático.

**Útil para:**

- Interpretabilidad del modelo de tópicos.
- Selección de keywords clave para reportes o dashboards.

### 4.9 `processing/visualization.py`

Responsable de las visualizaciones 2D/3D de embeddings:

- Aplica algoritmos de reducción de dimensionalidad:
  - UMAP
  - t-SNE
- Genera gráficas (por ejemplo, con Plotly) para:
  - Explorar la distribución de documentos en el espacio de tópicos.
  - Colorear por tópico asignado o detectar outliers visualmente.

Estas visualizaciones pueden integrarse en el reporte HTML.

### 4.10 `utils/color_palettes.py`

Define paletas de color reutilizables en todo el proyecto:

- Para gráficas de n-gramas.
- Para wordclouds.
- Para visualizaciones 2D/3D.

Incluye paletas diseñadas para ser aptas para personas con daltonismo, como la paleta **Okabe–Ito**.

Permite cambiar de forma consistente el estilo visual de los reportes sin modificar cada módulo por separado.

### 4.11 `web_report/generator.py`

Genera el reporte HTML final, por ejemplo `reporte_nlp.html`.

**Funcionalidades típicas:**

- Uso de Bootstrap para un layout limpio y responsivo.
- Inclusión de imágenes (wordclouds, gráficos de n-gramas, visualizaciones) embebidas, por ejemplo con base64.
- Inserción de tablas de resultados (frecuencias de n-gramas, resumen de tópicos, outliers, etc.).
- Inclusión de gráficas interactivas (por ejemplo, Plotly) cuando sea necesario.

**El resultado es un reporte listo para:**

- Compartirse con usuarios no técnicos.
- Documentar resultados de análisis de texto.

---

## 5. Instalación

### 5.1 Requisitos

Instala las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

### 5.2 `requirements.txt`

Dependencias principales utilizadas en el proyecto:

```
pandas
numpy
matplotlib
plotly
wordcloud
scikit-learn
nltk
spacy
bertopic
sentence-transformers
umap-learn
torch
```

### 5.3 Modelos de spaCy

Descarga los modelos de idioma necesarios:

```bash
# Para español
python -m spacy download es_core_news_sm

# Para inglés
python -m spacy download en_core_web_sm
```

---

## 6. Ejemplo de uso completo

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar modelos de spaCy
python -m spacy download es_core_news_sm

# 3. Ejecutar el análisis
python nlp_analyzer.py \
    -f data_input/test.csv \
    -c texto \
    -l spanish \
    -p okabe_ito \
    -t "Análisis de Textos en Español"

# 4. Abrir el reporte generado
# El archivo reporte_nlp.html se generará en el directorio raíz
```

---

## 7. Licencia

[Especificar la licencia del proyecto]

---

## 8. Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para sugerencias y mejoras.

---

## 9. Contacto

[Información de contacto del autor o equipo]