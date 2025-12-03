from typing import Dict, List
import pandas as pd
from bertopic import BERTopic


class OutlierAnalyzer:
    """
    Analiza y caracteriza los documentos asignados al Tópico -1 (outliers)
    generados por un modelo BERTopic.
    """

    def __init__(self, df_docs: pd.DataFrame, bertopic_model: BERTopic):
        """
        Inicializa con el DataFrame de documentos (que incluye la columna 'topic')
        y el modelo BERTopic entrenado.
        """
        assert "topic" in df_docs.columns, "df_docs debe contener la columna 'topic'"
        assert isinstance(bertopic_model, BERTopic), "bertopic_model debe ser una instancia de BERTopic"
        
        self.df_docs = df_docs
        self.topic_model = bertopic_model
        
        # Filtra solo los documentos outliers (Tópico -1)
        self.outliers = self.df_docs[self.df_docs["topic"] == -1].copy()
        print(f"[OutlierAnalyzer] → Encontrados {len(self.outliers)} documentos outliers (Tópico -1).")

    # 1. Caracterización General
    
    def get_outlier_count(self) -> int:
        """Devuelve el número total de documentos outliers."""
        return len(self.outliers)
    
    def get_outlier_proportion(self) -> float:
        """Devuelve la proporción de outliers respecto al total de documentos."""
        total_docs = len(self.df_docs)
        if total_docs == 0:
            return 0.0
        return len(self.outliers) / total_docs

    def get_outlier_docs(self) -> pd.DataFrame:
        """Devuelve el DataFrame solo con los documentos outliers."""
        return self.outliers

    ## 2. Palabras Clave de Outliers
    
    def summarize_outliers(self, top_n: int = 10) -> List[tuple]:
        """
        Calcula las palabras clave que son representativas del tópico -1 (outliers).
        Utiliza el método get_topic de BERTopic.
        """
        outlier_keywords = self.topic_model.get_topic(-1)
        if outlier_keywords is None:
            return []
        
        # Devuelve las N principales palabras clave y sus puntuaciones
        return outlier_keywords[:top_n]

    ## 3. Ejemplos de Documentos Outliers
    
    def get_representative_outliers(self, top_n: int = 5) -> List[str]:
        """
        Obtiene los documentos más representativos del Tópico -1.
        Esto puede ayudar a entender por qué son outliers (e.g., ruido, longitud).
        """
        rep_docs = self.topic_model.get_representative_docs()
        # BERTopic guarda automáticamente los documentos representativos para -1
        docs_for_outliers = rep_docs.get(-1, [])
        return docs_for_outliers[:top_n]

    ## 4. Análisis de Longitud (Característica Común de Outliers)
    
    def analyze_length(self) -> Dict[str, float]:
        """
        Analiza la longitud de los documentos outliers vs. el resto de documentos.
        A menudo, los outliers son documentos muy cortos o muy largos/ruidosos.
        """
        if self.outliers.empty:
            return {"promedio_outliers": 0, "promedio_temáticos": 0}

        # Longitud de los outliers
        outlier_lengths = self.outliers["text"].apply(lambda x: len(x.split()))
        avg_outlier_length = outlier_lengths.mean()

        # Longitud de los documentos temáticos (tópicos != -1)
        thematic_docs = self.df_docs[self.df_docs["topic"] != -1]
        if thematic_docs.empty:
             avg_thematic_length = 0
        else:
            thematic_lengths = thematic_docs["text"].apply(lambda x: len(x.split()))
            avg_thematic_length = thematic_lengths.mean()

        return {
            "avg_outlier_length_words": round(avg_outlier_length, 2),
            "avg_thematic_length_words": round(avg_thematic_length, 2)
        }

    def run_outlier_analysis(self, top_n_keywords: int = 10, top_n_docs: int = 5) -> Dict:
        """Ejecuta el análisis completo y devuelve un resumen."""
        
        keyword_summary = self.summarize_outliers(top_n_keywords)
        doc_examples = self.get_representative_outliers(top_n_docs)
        length_analysis = self.analyze_length()
        
        return {
            "total_outliers": self.get_outlier_count(),
            "proportion": round(self.get_outlier_proportion() * 100, 2),
            "keyword_summary": keyword_summary,
            "doc_examples": doc_examples,
            "length_analysis": length_analysis
        }