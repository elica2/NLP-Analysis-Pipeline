import re
from typing import List, Dict, Any

import pandas as pd


class WebReport:
    """
    Genera un reporte HTML tipo dashboard, interactivo, estático.
    - Usa Bootstrap desde CDN
    - Navegación lateral (navbar superior)
    - Secciones colapsables
    - Soporta:
        * HTML arbitrario
        * Imágenes base64
        * Tablas (pandas.DataFrame)
        * Gráficas Plotly (fig.to_html)
    """

    def __init__(self, title: str, palette: str):
        self.title = title
        self.palette = palette
        self.sections: List[Dict[str, Any]] = []

    # ----------------- Helpers -----------------
    def _slugify(self, text: str) -> str:
        """Genera un id válido para anchors HTML a partir del subtítulo."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        text = re.sub(r"-+", "-", text).strip("-")
        return text or "section"

    # ----------------- API pública -----------------
    def add_section(self, subtitle: str, content_html: str):
        """Agrega una sección genérica al reporte."""
        sec_id = self._slugify(subtitle)
        self.sections.append(
            {
                "id": sec_id,
                "subtitle": subtitle,
                "content": content_html,
            }
        )

    def add_image(self, subtitle: str, base64_img: str, width: str = "100%"):
        """Agrega una sección con una imagen en base64."""
        img_html = f"""
        <div class="text-center">
            <img src="data:image/png;base64,{base64_img}" style="max-width:{width}; height:auto;" class="img-fluid rounded shadow-sm">
        </div>
        """
        self.add_section(subtitle, img_html)

    def add_table(self, subtitle: str, df: pd.DataFrame):
        """Agrega una sección con una tabla generada desde un DataFrame."""
        table_html = df.to_html(
            index=False,
            classes="table table-striped table-bordered table-hover align-middle",
            border=0,
            escape=False
        )
        wrapped = f"""
        <div class="table-responsive">
            {table_html}
        </div>
        """
        self.add_section(subtitle, wrapped)

    def add_plotly(self, subtitle: str, fig):
        """
        Agrega una sección con una gráfica Plotly interactiva.
        'fig' debe ser un objeto plotly.graph_objects.Figure o similar.
        """
        plot_html = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False
        )
        self.add_section(subtitle, plot_html)

    # ----------------- Generador HTML -----------------
    def generate(self, path: str = "report.html") -> str:
        """Genera el archivo HTML final en 'path' y devuelve la ruta."""

        # Navbar links
        nav_links = "\n".join(
            [
                f'<a class="nav-link" href="#{sec["id"]}">{sec["subtitle"]}</a>'
                for sec in self.sections
            ]
        )

        # Secciones
        sections_html = ""
        for sec in self.sections:
            sections_html += f"""
            <section id="{sec["id"]}" class="mb-5">
                <details open class="card shadow-sm">
                    <summary class="card-header bg-light fw-semibold">
                        {sec["subtitle"]}
                    </summary>
                    <div class="card-body">
                        {sec["content"]}
                    </div>
                </details>
            </section>
            """

        # HTML completo
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="utf-8">
            <title>{self.title}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">

            <!-- Bootstrap CSS -->
            <link
              href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
              rel="stylesheet"
            >

            <style>
                body {{
                    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
                    background-color: #f5f5f7;
                }}
                .navbar-custom {{
                    background: linear-gradient(90deg, #111827, #1f2933);
                }}
                .navbar-brand, .navbar-nav .nav-link {{
                    color: #f9fafb !important;
                }}
                .navbar-nav .nav-link:hover {{
                    color: #e5e7eb !important;
                }}
                main {{
                    padding-top: 80px;
                }}
                summary {{
                    cursor: pointer;
                    list-style: none;
                }}
                summary::-webkit-details-marker {{
                    display: none;
                }}
            </style>
        </head>
        <body>
            <!-- Navbar -->
            <nav class="navbar navbar-expand-lg navbar-dark navbar-custom fixed-top shadow-sm">
              <div class="container-fluid">
                <a class="navbar-brand fw-bold" href="#top">{self.title}</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarContent">
                  <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarContent">
                  <ul class="navbar-nav ms-auto mb-2 mb-lg-0">
                    {nav_links}
                  </ul>
                </div>
              </div>
            </nav>

            <!-- Contenido -->
            <main class="container" id="top">
              <div class="py-4">
                <h1 class="mb-4">{self.title}</h1>
                {sections_html}
              </div>
            </main>

            <!-- Bootstrap JS -->
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        return path
