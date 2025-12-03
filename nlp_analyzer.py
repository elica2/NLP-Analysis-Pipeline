import argparse
import pandas as pd
import base64
import logging
import io

from processing.preprocess import TextPreprocessor
from processing.ngrams import NgramCreator
from processing.wordcloud import WordCloudWrapper
from processing.topics import TopicModeler
from processing.outliers import OutlierAnalyzer
from processing.visualization import Visualization
from processing.ablation import TopicAblation
from web_report.generator import WebReport
import matplotlib

matplotlib.use("Agg")

def fig_to_base64(fig):
    """Convierte figura matplotlib a base64 para insertarla al HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_pipeline(dataset_path: str,
                 text_column: str,
                 language: str,
                 palette: str,
                 title: str):
    """
    Ejecuta TODO el pipeline de NLP y genera un reporte HTML interactivo.
    """
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

    log = logging.getLogger("NLP-Pipeline")

    # --- Cargar dataset ---
    log.info("Cargando dataset desde %s", dataset_path)
    df = pd.read_csv(dataset_path)
    texts = df[text_column].astype(str).tolist()

    # --- PREPROCESAMIENTO ---
    log.info("Preprocesando texto...")
    pre = TextPreprocessor(texts, language=language, lemma=True)
    cleaned_texts, tokens = pre.process_all()


    # --- NGRAMS ---
    log.info("Generando N-grams...")
    ng = NgramCreator(tokens=tokens, palette=palette, top_k=10)
    bigrams = ng.compute(2)
    trigrams = ng.compute(3)

    bigram_b64 = ng.plot_to_base64(2)
    trigram_b64 = ng.plot_to_base64(3)

    # --- WORDCLOUD ---
    log.info("Creando WordCloud...")
    wcw = WordCloudWrapper(title="WordCloud", tokens=tokens, palette=palette)
    wc = wcw.create_cloud()
    fig_wc = wcw.plot(wc) 
    wc_b64 = fig_to_base64(fig_wc)

    # --- TOPIC MODELING ---
    log.info("Entrenando modelo BERTopic...")
    tm = TopicModeler(cleaned_texts, language=language)
    tm.fit()
    df_topics = tm.get_topic_info()
    df_docs = tm.get_documents_dataframe()
    embeddings = tm.get_embeddings()

    # --- ABLACIÓN DE TÓPICOS ---
    log.info("Haciendo ablación de tópicos...")
    ab = TopicAblation(tm)
    ablation_result = ab.run_all(top_n=None)
    ablated_keywords = ablation_result["ablated_keywords"]

    # Crear tabla de ablación con columnas completas
    df_topics_ablated = df_topics.copy()
    rep_docs = {
    topic_id: tm.get_representative_docs(topic_id, top_n=1)[0]
    for topic_id in df_topics["Topic"].tolist()
    if topic_id != -1}

    # Agregar las palabras exclusivas como nueva columna Representation
    df_topics_ablated["Representation"] = df_topics_ablated["Topic"].apply(
        lambda topic_id: ", ".join(ablated_keywords.get(topic_id, []))
                        if topic_id != -1 else "(outlier)")

    # Agregar el documento más representativo (ya calculado en rep_docs)
    df_topics_ablated["Representative_Docs"] = df_topics_ablated["Topic"].apply(
        lambda topic_id: rep_docs.get(topic_id, ""))
    
    # --- OUTLIERS ---
    log.info("Analizando el Tópico -1 (Outliers)...")
    outlier_analyzer = OutlierAnalyzer(df_docs, tm.topic_model)
    outlier_report = outlier_analyzer.run_outlier_analysis(top_n_keywords=15, top_n_docs=3)
    
    # Prepara un DataFrame para mostrar el resumen del outlier en el reporte
    outlier_summary_data = {
        "Métrica": [
            "Total de Outliers (Tópico -1)", 
            "Proporción (%)", 
            "Longitud Promedio (Outliers)", 
            "Longitud Promedio (Temáticos)",
            "Palabras Clave (Top 3)",
            "Documento Representativo (Ejemplo)"
        ],
        "Valor": [
            outlier_report["total_outliers"],
            f'{outlier_report["proportion"]:.2f}%',
            f'{outlier_report["length_analysis"]["avg_outlier_length_words"]} palabras',
            f'{outlier_report["length_analysis"]["avg_thematic_length_words"]} palabras',
            ", ".join([k[0] for k in outlier_report["keyword_summary"][:3]]),
            outlier_report["doc_examples"][0] if outlier_report["doc_examples"] else "N/A"
        ]
    }
    df_outlier_summary = pd.DataFrame(outlier_summary_data)


    # --- VISUALIZACIÓN (UMAP + TSNE 3D) ---
    log.info("Reduciendo dimensiones con UMAP y TSNE...")
    viz = Visualization(embeddings, df_docs, palette=palette)
    fig_umap, fig_tsne = viz.generate_both()

    # --- REPORTE ---
    log.info("Generando reporte HTML final...")
    report = WebReport(title=title, palette=palette)

    report.add_image("WordCloud general", wc_b64)

    report.add_image("Top 10 bigramas", bigram_b64)
    report.add_image("Top 10 trigramas", trigram_b64)

    # Agregar tabla de tópicos
    report.add_table("Resumen de tópicos", df_topics)
    report.add_table("Tópicos después de Ablación", df_topics_ablated)


    # Agregar visualizaciones Plotly
    report.add_plotly("UMAP 3D de Tópicos", fig_umap)
    report.add_plotly("t-SNE 3D de Tópicos", fig_tsne)

    report.add_table("Análisis Outliers", df_outlier_summary)

    # Guardar reporte
    output_path = report.generate("reporte_nlp.html")
    print(f"Reporte generado en: {output_path}")
    return output_path

def crear_parser():
    parser = argparse.ArgumentParser(
        description='Script de análisis de lenguaje natural (NLP) completamente automatizado, diseñado para ejecutarse desde consola y generar un reporte HTML',
        epilog='python parser.py -f datos.csv -c comentario -l spanish -p viridis -t "Reporte NLP"'
    )

    # Argumentos posicionales
    parser.add_argument(
        '-f','--File',
        help='Ruta al archivo de entrada'
    )

    parser.add_argument(
        '-c','--Column_name',
        help='Nombre de la columna con textos'
    )

    parser.add_argument(
        '-l','--Language',
        help='Idioma del texto'
    )

    parser.add_argument(
        '-p','--palette',
        choices=['okabe_ito', 'sunset', 'medium_earthy', 'viridis', 'magma'],
        default='okabe_ito',
        help='Paleta de colores'
    )

    parser.add_argument(
        '-t','--Title',
        help='Título del reporte'
    )

    return parser

def main(args):
    run_pipeline(
        dataset_path=args.File,
        text_column=args.Column_name,
        language=args.Language,
        palette=args.palette,
        title=args.Title
    )

if __name__ == "__main__":
    parser = crear_parser()
    args = parser.parse_args()
    main(args)