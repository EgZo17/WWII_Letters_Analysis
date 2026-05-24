import pandas as pd
from typing import List, Dict
from collections import defaultdict


class TemporalAnalyzer:
    """
    Анализ динамики писем по времени (1941-1945).
    """

    def __init__(self, letters: List[Dict]):
        """
        Args:
            letters: Список обогащённых писем из letters_enriched.json
        """
        self.df = pd.DataFrame(letters)
        self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
        # Фильтруем только письма с корректным годом
        self.df = self.df[self.df['year'].between(1941, 1945)].copy()

    def get_yearly_stats(self) -> pd.DataFrame:
        """
        Считает базовую статистику по каждому году.

        Returns:
            DataFrame с колонками:
            - year: год
            - letters_count: количество писем
            - avg_word_count: средняя длина письма
            - positive_ratio: доля позитивных писем
            - negative_ratio: доля негативных писем
            - neutral_ratio: доля нейтральных писем
            - top_emotion: самая частая эмоция
        """
        stats = []

        for year in range(1941, 1946):
            year_df = self.df[self.df['year'] == year]

            if len(year_df) == 0:
                continue

            # Подсчёт тональности
            sentiment_counts = year_df['sentiment_raw'].value_counts(normalize=True)

            # Подсчёт эмоций (берём первую из списка как основную)
            all_emotions = []
            for emotions in year_df['emotions'].dropna():
                if emotions:
                    all_emotions.append(emotions[0])

            top_emotion = max(set(all_emotions), key=all_emotions.count) if all_emotions else None

            stats.append({
                'year': year,
                'letters_count': len(year_df),
                'avg_word_count': round(year_df['word_count'].mean(), 1),
                'std_word_count': round(year_df['word_count'].std(), 1),
                'positive_ratio': round(sentiment_counts.get('positive', 0) * 100, 1),
                'negative_ratio': round(sentiment_counts.get('negative', 0) * 100, 1),
                'neutral_ratio': round(sentiment_counts.get('neutral', 0) * 100, 1),
                'top_emotion': top_emotion
            })

        return pd.DataFrame(stats)

    def get_emotion_dynamics(self) -> Dict[int, Dict[str, float]]:
        """
        Возвращает распределение эмоций по годам.

        Returns:
            {year: {emotion: percentage, ...}, ...}
        """
        result = {}

        for year in range(1941, 1946):
            year_df = self.df[self.df['year'] == year]

            if len(year_df) == 0:
                continue

            emotion_counts = defaultdict(int)
            total = 0

            for emotions in year_df['emotions'].dropna():
                for emotion in emotions:
                    emotion_counts[emotion] += 1
                    total += 1

            if total > 0:
                result[year] = {
                    emotion: round(count / total * 100, 1)
                    for emotion, count in emotion_counts.items()
                }

        return result

    def calculate_morale_index(self) -> pd.DataFrame:
        """
        Рассчитывает интегральный "индекс морального духа".

        Формула: (позитив% * 1.0) + (нейтрал% * 0.3) - (негатив% * 0.8)
        Чем выше — тем лучше настроение.

        Returns:
            DataFrame с year и morale_index
        """
        stats = self.get_yearly_stats()

        stats['morale_index'] = (
                stats['positive_ratio'] * 1.0 +
                stats['neutral_ratio'] * 0.3 -
                stats['negative_ratio'] * 0.8
        )

        return stats[['year', 'morale_index']]
