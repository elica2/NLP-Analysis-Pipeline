from collections import Counter
from typing import List, Tuple
import matplotlib.pyplot as plt
import io
import base64

class NgramCreator:
    def __init__(self, tokens: List[str], palette: str, top_k: int = 10):
        assert isinstance(tokens, list) and len(tokens) > 0, "Tokens list cannot be empty"
        self.tokens = tokens
        self.palette = palette
        self.top_k = top_k
        self.results = {}

    def compute(self, n: int) -> List[Tuple[tuple, int]]:
        assert n >= 1, "n must be >= 1"

        # Generar los n-gramas usando zip
        ngrams = zip(*[self.tokens[i:] for i in range(n)])
        counts = Counter(ngrams).most_common(self.top_k)

        self.results[n] = counts
        return counts

    def plot(self, n: int, angle: int = 60):
        assert n in self.results, f"No {n}-grams computed yet. Call compute({n}) first."

        labels = [" ".join(g) for g, _ in self.results[n]]
        values = [c for _, c in self.results[n]]

        plt.figure(figsize=(10, 5))
        plt.bar(labels, values, color=self._get_palette(self.palette))
        plt.xticks(rotation=angle)
        plt.title(f"Top {len(values)} {n}-grams")
        plt.xlabel("N-gramas")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.show()

    def _get_palette(self, name):
        from utils.color_palettes import COLOR_SCHEMES
        return COLOR_SCHEMES[name]
    
    def plot_to_base64(self, n: int, angle: int = 60):
        assert n in self.results, f"No {n}-grams computed yet. Call compute({n}) first."

        labels = [" ".join(g) for g, _ in self.results[n]]
        values = [c for _, c in self.results[n]]

        palette = self._get_palette(self.palette)
        color = palette[len(palette) // 2]

        fig = plt.figure(figsize=(10, 5))
        plt.bar(labels, values, color=color, hatch="//")
        plt.xticks(rotation=angle)
        plt.title(f"Top {len(values)} {n}-grams")
        plt.xlabel("N-gramas")
        plt.ylabel("Frecuencia")
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)

        return b64
    
    def run_all(self, angle: int = 60):
        """
        Ejecuta TODO el pipeline:
        - unigramas
        - bigramas
        - trigramas
        - genera gráficas
        - devuelve un dict con los 3 resultados
        """

        # 1) calcular n-gramas
        unigrams = self.compute(1)
        bigrams = self.compute(2)
        trigrams = self.compute(3)

        # 2) graficar
        self.plot(1, angle)
        self.plot(2, angle)
        self.plot(3, angle)

        # 3) retornar resultados
        return {
            "unigrams": unigrams,
            "bigrams": bigrams,
            "trigrams": trigrams
        }


