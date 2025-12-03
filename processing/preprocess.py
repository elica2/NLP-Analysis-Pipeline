import re
import unicodedata
from typing import List

class TextPreprocessor:
    SUPPORTED_LANGS = {"spanish", "english"}

    def __init__(self, texts: List[str], language: str = "spanish", lemma: bool = False):
        assert isinstance(texts, list) and len(texts) > 0, "La lista de textos no puede estar vacía"
        assert language in self.SUPPORTED_LANGS, f"Idioma no soportado. Disponible: {self.SUPPORTED_LANGS}"

        self.raw_texts = texts
        self.cleaned = []
        self.language = language
        self.lemma = lemma
        self.nlp = None

        if self.lemma:
            self._load_spacy_model()

    # ------ MAIN CLEANING ------
    def clean(self):
        cleaned_list = []
        for text in self.raw_texts:
            t = text.lower()

            # Quitar acentos SIEMPRE antes de lematizar
            t = self._remove_accents(t)

            # Normalización específica del idioma
            t = self._normalize_contractions(t)

            # Quitar símbolos, números y puntuación
            t = self._remove_symbols(t)

            # Normalizar espacios
            t = self._normalize_spaces(t)

            cleaned_list.append(t)

        self.cleaned = cleaned_list
        return self

    # ------ STOPWORDS ------
    def remove_stopwords(self):
        from nltk.corpus import stopwords
        sw = set(stopwords.words(self.language))

        filtered = []
        for sentence in self.cleaned:
            tokens = [w for w in sentence.split() if w not in sw]
            filtered.append(" ".join(tokens))

        self.cleaned = filtered
        return self

    # ------ LEMMATIZATION ------
    def lemmatize(self):
        if not self.lemma:
            return self

        lemmatized = []
        for sentence in self.cleaned:
            doc = self.nlp(sentence)
            lemmas = [
                token.lemma_
                for token in doc
                if token.lemma_ != "" and len(token.lemma_) > 2
            ]
            lemmatized.append(" ".join(lemmas))

        self.cleaned = lemmatized
        return self

    # ------ TOKENIZE ------
    def tokenize(self):
        tokens = []
        for sentence in self.cleaned:
            for t in sentence.split():
                if len(t) > 2:      # descartar tokens muy cortos
                    tokens.extend([t])
        return tokens

    # ------ UTILS ------
    def _load_spacy_model(self):
        import spacy
        if self.language == "spanish":
            self.nlp = spacy.load("es_core_news_lg")
        else:
            self.nlp = spacy.load("en_core_web_lg")

    def _remove_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        return text.encode("ascii", "ignore").decode("utf-8")

    def _normalize_contractions(self, text):
        if self.language == "english":
            text = re.sub(r"n't\b", " not", text)
            text = re.sub(r"'re\b", " are", text)
            text = re.sub(r"'m\b", " am", text)
            text = re.sub(r"'ll\b", " will", text)
        return text

    def _remove_symbols(self, text):
        return re.sub(r"[^a-zñáéíóúü\s]", " ", text)

    def _normalize_spaces(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def process_all(self):
        """
        Ejecuta TODA la limpieza:
            1. clean()
            2. remove_stopwords()
            3. lemmatize() (si aplica)
            4. tokenize()

        Devuelve:
            cleaned_texts: lista de textos procesados
            tokens: lista de tokens finales
        """
        self.clean()
        self.remove_stopwords()

        if self.lemma:
            self.lemmatize()

        tokens = self.tokenize()

        return self.cleaned, tokens