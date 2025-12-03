import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.manifold import TSNE
import umap.umap_ as umap


class Visualization:
    """
    Visualización automática de embeddings en 3D usando UMAP y t-SNE
    con parámetros adaptativos.
    """

    def __init__(self, embeddings: np.ndarray, df_docs: pd.DataFrame, palette: str):
        assert isinstance(embeddings, np.ndarray), "Embeddings must be a numpy array"
        assert "topic" in df_docs.columns, "df_docs must contain a 'topic' column"

        self.embeddings = embeddings          # matriz de embeddings
        self.df = df_docs.copy()              # copia del dataframe con columna 'topic'
        self.palette = palette                # nombre de la paleta a usar

    def _get_palette(self):
        # Cargar paleta desde tu diccionario global
        from utils.color_palettes import COLOR_SCHEMES
        return COLOR_SCHEMES[self.palette]

    def reduce_umap_3d(self):
        # Número de muestras
        N = len(self.embeddings)
        if N < 3:
            raise ValueError("Se requieren al menos 3 muestras para UMAP 3D.")

        # n_neighbors adaptativo: sqrt(N), limitado entre 5 y 50 y no mayor a N-1
        n_neighbors = int(np.sqrt(N))
        n_neighbors = max(5, min(n_neighbors, 50, N - 1))

        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42
        )

        reduced = reducer.fit_transform(self.embeddings)

        self.df["umap_x"] = reduced[:, 0]
        self.df["umap_y"] = reduced[:, 1]
        self.df["umap_z"] = reduced[:, 2]

    def reduce_tsne_3d(self):
        # Número de muestras
        N = len(self.embeddings)
        if N < 3:
            raise ValueError("Se requieren al menos 3 muestras para t-SNE 3D.")

        # perplexity adaptativa: sqrt(N), limitada entre 5 y 50 y menor que N
        perplexity = int(np.sqrt(N))
        max_perplexity = max(2, N - 1)   # garantizar < N
        perplexity = max(5, min(perplexity, 50, max_perplexity))

        # n_iter adaptativo
        n_iter = max(750, int(250 * np.sqrt(N)))

        reducer = TSNE(
            n_components=3,
            perplexity=perplexity,
            max_iter=n_iter,
            learning_rate="auto",
            random_state=42
        )

        reduced = reducer.fit_transform(self.embeddings)

        self.df["tsne_x"] = reduced[:, 0]
        self.df["tsne_y"] = reduced[:, 1]
        self.df["tsne_z"] = reduced[:, 2]

    def plot_umap_3d(self):
        # Obtener paleta de colores
        palette = self._get_palette()
        self.df["topic"] = self.df["topic"].astype(str)

        # Gráfico UMAP 3D
        fig = px.scatter_3d(
            self.df,
            x="umap_x", y="umap_y", z="umap_z",
            color="topic",
            color_discrete_sequence=palette,
            title="UMAP 3D",
            hover_data=["topic"]
        )

        fig.update_traces(
            marker=dict(size=4), 
            selector=dict(mode='markers')
        )

        fig.update_layout(width=900, height=700)
        return fig

    def plot_tsne_3d(self):
        # Obtener paleta de colores
        palette = self._get_palette()
        self.df["topic"] = self.df["topic"].astype(str)

        # Gráfico t-SNE 3D
        fig = px.scatter_3d(
            self.df,
            x="tsne_x", y="tsne_y", z="tsne_z",
            color="topic",
            color_discrete_sequence=palette,
            title="t-SNE 3D",
            hover_data=["topic"]
        )

        fig.update_traces(
            marker=dict(size=4),
            selector=dict(mode='markers')
        )

        fig.update_layout(width=900, height=700)
        return fig

    def generate_both(self, show: bool = False):
        """
        Ejecuta todo el pipeline:
          - UMAP 3D
          - t-SNE 3D
          - Genera ambas figuras

        Devuelve:
          fig_umap, fig_tsne
        """
        self.reduce_umap_3d()
        self.reduce_tsne_3d()

        fig_umap = self.plot_umap_3d()
        fig_tsne = self.plot_tsne_3d()

        if show:
            fig_umap.show()
            fig_tsne.show()

        return fig_umap, fig_tsne