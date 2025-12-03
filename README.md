NLP Analyzer – Proyecto Final

Sistema modular para análisis de lenguaje natural orientado a textos en español e inglés.
Incluye preprocesamiento, n-gramas, wordclouds, modelado de tópicos con BERTopic, análisis de outliers, ablación de palabras y visualizaciones 3D de embeddings, además de un generador de reportes HTML.

1. Estructura del proyecto
proyecto_final/
├── nlp_analyzer.py          # Script principal
├── reporte_nlp.html         # Ejemplo de reporte generado
│
├── config/
│   └── settings.py
│
├── data_input/
│   └── test.csv
│
├── processing/
│   ├── preprocess.py
│   ├── ngrams.py
│   ├── wordcloud.py
│   ├── topics.py
│   ├── outliers.py
│   ├── ablation.py
│   └── visualization.py
│
├── utils/
│   └── color_palettes.py
│
└── web_report/
    └── generator.py

2. Instalación y requisitos

Se recomienda usar un entorno virtual.

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# o
.\.venv\Scripts\activate       # Windows


Instala las dependencias:

pip install -r requirements.txt


Modelos de spaCy requeridos:

python -m spacy download es_core_news_lg
python -m spacy download en_core_web_sm

3. Uso

Ejecutar desde la raíz del proyecto:

python nlp_analyzer.py \
    -f data_input/test.csv \
    -c texto \
    -l spanish \
    -p okabe_ito \
    -t "Reporte NLP"


Argumentos principales:

-f, --File Ruta del archivo CSV.

-c, --Column_name Columna con textos.

-l, --Language Idioma del texto: spanish o english.

-p, --palette Paleta de colores definida en color_palettes.py.

-t, --Title Título del reporte.

El archivo HTML resultante resume:

Preprocesamiento

N-gramas

Wordcloud

Tópicos de BERTopic

Visualizaciones 3D (UMAP y t-SNE)

Análisis de outliers

Ablación de keywords por tópico

4. Descripción de módulos
4.1. nlp_analyzer.py

Orquesta el pipeline completo:
preprocesamiento → n-gramas → wordcloud → tópicos → ablación → outliers → visualizaciones → reporte HTML.

4.2. config/settings.py

Valida ruta del archivo, columna, título y paleta de colores.

4.3. processing/preprocess.py

Limpieza, normalización, eliminación de stopwords y lematización opcional con spaCy.

4.4. processing/ngrams.py

Genera unigramas, bigramas y trigramas. Produce gráficas con Matplotlib.

4.5. processing/wordcloud.py

Genera wordclouds utilizando paletas personalizadas desde color_palettes.py.

4.6. processing/topics.py

Implementa BERTopic y SentenceTransformers para extraer tópicos, keywords y embeddings.

4.7. processing/outliers.py

Analiza el tópico -1 (outliers) del modelo BERTopic. Da estadísticas, ejemplos y análisis de longitud.

4.8. processing/ablation.py

Realiza ablación de keywords para detectar palabras exclusivas por tópico.

4.9. processing/visualization.py

Reducción de dimensionalidad (UMAP y t-SNE) y visualizaciones 3D con Plotly.

4.10. utils/color_palettes.py

Define paletas de color, incluyendo opciones aptas para daltonismo (Okabe–Ito).

4.11. web_report/generator.py

Genera un reporte HTML con Bootstrap, imágenes base64, tablas y gráficas interactivas.

5. requirements.txt

Este archivo fue generado a partir de todos los imports presentes en tu proyecto:

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


Dependencias opcionales (si usas exportación o funciones adicionales):

typing_extensions


Modelos requeridos por spaCy:

es_core_news_lg @ https://github.com/explosion/spacy-models/releases/download/es_core_news_lg-3.7.0/es_core_news_lg-3.7.0-py3-none-any.whl
en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl