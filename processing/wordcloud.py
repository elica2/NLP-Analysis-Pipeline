import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap

class WordCloudWrapper:
    def __init__(self, title: str, tokens: list[str], palette: str):
        self.title =  title
        self.tokens = tokens
        self.palette = palette

    def _get_palette(self, name):
        from utils.color_palettes import COLOR_SCHEMES
        return COLOR_SCHEMES[name]

    def create_cloud(self):
        assert self.tokens, "No tokens found. Try again."
        colors = self._get_palette(self.palette)  # lista de HEX
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        wc = WordCloud(
            background_color="white",
            colormap=cmap,
            width=1000,
            height=500
        ).generate(" ".join(self.tokens))

        return wc

    def plot(self, wc):
        fig = plt.figure(figsize=(12,6))
        plt.imshow(wc, interpolation="bilinear")
        plt.title(self.title)
        plt.axis("off")
        plt.show()

        return fig

    
    def generate(self, show: bool = False):
        """
        Pipeline completo:
        1) Crear WordCloud
        2) Crear figura matplotlib
        3) Devolver (fig, wc)
        4) Mostrarlo si show=True

        Útil para integrarlo al HTML (codificar a base64).
        """
        wc = self.create_cloud()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(self.title)
        ax.axis("off")

        if show:
            plt.show()

        return fig, wc