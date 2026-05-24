import torch
from transformers import pipeline
from typing import List, Dict
from src.nlp.emotion_dictionary import get_all_keywords_flat
from src.nlp.emotion_dictionary import get_all_phrases
from src.config import SENTIMENT_MODEL_NAME, MIN_EMOTION_SCORE


class SentimentAnalyzer:
    """
    Анализ тональности писем с помощью RuBERT + эмоций по ключевым словам.
    """

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.model_name = SENTIMENT_MODEL_NAME

        print(f"Загружаю модель {self.model_name}...")
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            device=device,
            truncation=True,
            max_length=512
        )

        # Загружаем словари эмоций
        self.emotion_keywords = get_all_keywords_flat()
        self.emotion_phrases = get_all_phrases()

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        results = self.pipe(texts)
        return results

    def detect_emotions(self, text: str, min_score: int = MIN_EMOTION_SCORE) -> tuple[List[str], List[str]]:
        """
        Определяет эмоции в тексте по ключевым словам и фразам.

        Args:
            text: Текст письма
            min_score: Минимальное количество совпадений для эмоции

        Returns:
            Список обнаруженных эмоций
        """
        text_lower = text.lower()
        detected = {}
        triggers = []

        for emotion in self.emotion_keywords:
            score = 0

            # 1. Сначала ищем целые фразы (точное совпадение)
            for phrase in self.emotion_phrases.get(emotion, []):
                if phrase.lower() in text_lower:
                    score += 2  # Фразы «весят» больше, чем отдельные слова
                    triggers.append(phrase)

            # 2. Потом ищем отдельные слова
            for keyword in self.emotion_keywords.get(emotion, []):
                if keyword.lower() in text_lower:
                    score += 1
                    triggers.append(keyword)

            if score >= min_score:
                detected[emotion] = score

        # Сортируем по количеству совпадений
        sorted_emotions = sorted(detected.items(), key=lambda x: x[1], reverse=True)
        return [emotion for emotion, score in sorted_emotions], triggers

    @staticmethod
    def map_to_historical_categories(label: str) -> str:
        label_upper = label.upper()
        mapping = {
            "POSITIVE": "воодушевление",
            "NEGATIVE": "подавленность",
            "NEUTRAL": "равнодушие"
        }
        return mapping.get(label_upper, "равнодушие")
