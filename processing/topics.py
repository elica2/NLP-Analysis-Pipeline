from typing import List, Optional
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import torch

class TopicModeler:
    """
    Envuelve BERTopic + SentenceTransformer.
    Usa automáticamente el modelo all-mpnet-base-v2 para español e inglés.
    """

    def __init__(
        self,
        docs: List[str],
        language: str = "spanish",
        embedding_model_name: Optional[str] = None,
        n_topics: str | int = "auto"
    ):
        assert isinstance(docs, list) and len(docs) > 0, "La lista de documentos no puede estar vacía"
        assert language in {"spanish", "english"}, "Idioma no soportado (usa 'spanish' o 'english')"

        self.docs = docs
        self.language = language
        self.n_topics = n_topics

        # Si el usuario no especifica nada, usar all-mpnet-base-v2 
        self.embedding_model_name = embedding_model_name or "sentence-transformers/all-mpnet-base-v2"

        # Se llenan durante fit()
        self.embedder: SentenceTransformer | None = None
        self.embeddings: np.ndarray | None = None
        self.topic_model: BERTopic | None = None
        self.topics: List[int] | None = None
        self.probs: np.ndarray | None = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TopicModeler] → Usando dispositivo: {self.device}")

    def fit(self):
        """Genera embeddings y entrena BERTopic."""
        self._load_embedding_model()
        self._compute_embeddings()
        self._fit_bertopic()
        return self

    def _load_embedding_model(self):
        """Carga el modelo de sentence-transformers."""
        self.embedder = SentenceTransformer(self.embedding_model_name, device=self.device)

    def _compute_embeddings(self):
        """Obtiene embeddings de cada documento."""
        assert self.embedder is not None, "El modelo de embeddings no está cargado"
        self.embeddings = self.embedder.encode(
            self.docs,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )

    def get_embeddings(self) -> np.ndarray:
        """Devuelve la matriz de embeddings utilizada en el modelo."""
        assert self.embeddings is not None, "Embeddings no calculados"
        return self.embeddings

    def _fit_bertopic(self):
        """Ajusta BERTopic usando los embeddings precalculados."""
        assert self.embeddings is not None, "Embeddings no calculados"

        self.topic_model = BERTopic(
            language=self.language,
            nr_topics=self.n_topics,
            calculate_probabilities=True,
            verbose=False
        )

        self.topics, self.probs = self.topic_model.fit_transform(self.docs, self.embeddings)

    def get_topic_info(self) -> pd.DataFrame:
        """Devuelve información global de todos los tópicos."""
        assert self.topic_model is not None, "El modelo de tópicos no está entrenado"
        return self.topic_model.get_topic_info()

    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> List[tuple]:
        """Devuelve lista de (keyword, peso) para un tópico."""
        assert self.topic_model is not None, "El modelo de tópicos no está entrenado"
        words = self.topic_model.get_topic(topic_id)
        if words is None:
            return []
        return words[:top_n]

    def get_representative_docs(self, topic_id: int, top_n: int = 1) -> List[str]:
        """Devuelve los documentos más representativos de un tópico."""
        assert self.topic_model is not None, "El modelo de tópicos no está entrenado"
        rep_docs = self.topic_model.get_representative_docs()
        docs_for_topic = rep_docs.get(topic_id, [])
        return docs_for_topic[:top_n]

    def get_documents_dataframe(self) -> pd.DataFrame:
        """Devuelve un DataFrame con doc_id, texto y tópico asignado."""
        assert self.topics is not None and self.embeddings is not None, "Modelo no entrenado"

        df = pd.DataFrame({
            "doc_id": range(len(self.docs)),
            "text": self.docs,
            "topic": self.topics
        })
        return df

    def run_pipeline(self, top_n_keywords: int = 10, top_n_docs: int = 3):
        """
        Ejecuta TODO el pipeline completo y devuelve:
        - modelo entrenado
        - embeddings
        - dataframe de documentos (con tópicos)
        - tabla de tópicos
        - keywords por tópico
        - docs representativos por tópico
        """

        # 1) Entrenar todo
        self.fit()

        # 2) Obtener dataframe de documentos
        df_docs = self.get_documents_dataframe()

        # 3) Info global de tópicos
        df_topics = self.get_topic_info()

        # 4) Keywords por cada tópico
        keywords = {
            topic_id: self.get_topic_keywords(topic_id, top_n_keywords)
            for topic_id in df_topics["Topic"].tolist()
            if topic_id != -1
        }

        # 5) Documentos representativos por tópico
        repr_docs = {
            topic_id: self.get_representative_docs(topic_id, top_n_docs)
            for topic_id in df_topics["Topic"].tolist()
            if topic_id != -1
        }

        # 6) Regresar todo en un dict ordenado
        return {
            "model": self.topic_model,
            "embeddings": self.embeddings,
            "df_docs": df_docs,
            "df_topics": df_topics,
            "keywords": keywords,
            "representative_docs": repr_docs,
        }

