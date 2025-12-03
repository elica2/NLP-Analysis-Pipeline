from typing import Dict, List, Tuple

class TopicAblation:
    def __init__(self, topic_modeler):
        self.tm = topic_modeler  

    def get_topic_words(self, top_n: int | None = None) -> Dict[int, List[str]]:
        """
        Obtiene todas las palabras por tópico, o las top_n si se indica.
        """
        topic_words = {}
        topic_info = self.tm.get_topic_info()

        for topic_id in topic_info["Topic"]:
            if topic_id == -1:
                continue

            # obtener todas las palabras del tópico
            words = self.tm.get_topic_keywords(topic_id, top_n if top_n else 9999)

            # quitar pesos
            only_words = [w for w, _ in words]

            topic_words[topic_id] = only_words

        return topic_words


    def ablate(self, top_n: int = 10) -> Dict[int, List[str]]:
        """Elimina palabras compartidas entre tópicos."""
        topic_words = self.get_topic_words(top_n)

        # Conjunto de palabras repetidas entre temas
        all_words = [word for words in topic_words.values() for word in words]
        duplicates = {w for w in all_words if all_words.count(w) > 1}

        # Crear diccionario con palabras exclusivas por tema
        exclusive = {
            topic_id: [w for w in words if w not in duplicates]
            for topic_id, words in topic_words.items()
        }

        return exclusive
    
    def run_all(self, top_n: int | None = None) -> Dict[str, Dict]:
        """
        Ejecuta todo el proceso de ablación:
        - keywords por tópico (sin límite si top_n=None)
        - palabras duplicadas globales
        - palabras exclusivas por tópico
        """

        # obtener palabras
        words_per_topic = self.get_topic_words(top_n)

        # duplicados globales
        all_words = [w for words in words_per_topic.values() for w in words]
        duplicates = {w for w in all_words if all_words.count(w) > 1}

        # ablación
        exclusive = {
            topic_id: [w for w in words if w not in duplicates]
            for topic_id, words in words_per_topic.items()
        }

        return {
            "keywords_per_topic": words_per_topic,
            "duplicate_words": list(duplicates),
            "ablated_keywords": exclusive
        }

