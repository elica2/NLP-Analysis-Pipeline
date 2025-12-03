from utils.color_palettes import COLOR_SCHEMES
class Config:
    ALLOWED_PALETTES = set(COLOR_SCHEMES.keys())

    def __init__(self, file_path, column, title, palette):
        self.file_path = file_path
        self.column = column
        self.title = title
        self.palette = palette
        self._validate()

    def _validate(self):
        self._validate_title()
        self._validate_file()
        self._validate_column()
        self._validate_palette()

    def _validate_title(self):
        assert isinstance(self.title, str) and self.title.strip(), "Título inválido"

    def _validate_file(self):
        import os
        assert isinstance(self.file_path, str) and self.file_path.strip(), "La ruta del archivo debe de ser un string"
        assert os.path.isfile(self.file_path), f"Archivo no encontrado: {self.file_path}"
        assert self.file_path.lower().endswith((".csv",)), "Error. Formato soportado: .csv"

    def _validate_column(self):
        import pandas as pd
        df = pd.read_csv(self.file_path)
        assert self.column in df.columns, f'No se encontró la columna. Columnas disponibles: {list(df.columns)}'

    def _validate_palette(self):
        assert self.palette in self.ALLOWED_PALETTES, f"Paleta inválida. Opciones: {self.ALLOWED_PALETTES}"