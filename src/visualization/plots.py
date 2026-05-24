import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Optional
import os

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'  # Поддержка кириллицы


class PlotGenerator:
    """
    Генерация графиков для отчёта.
    """

    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_sentiment_over_time(self, stats_df: pd.DataFrame) -> str:
        """
        График: Доля позитив/негатив/нейтрал по годам.

        Returns:
            Путь к сохранённому файлу
        """
        plt.figure(figsize=(10, 6))

        x = stats_df['year']
        plt.plot(x, stats_df['positive_ratio'], label='Позитив', marker='o', linewidth=2)
        plt.plot(x, stats_df['negative_ratio'], label='Негатив', marker='s', linewidth=2)
        plt.plot(x, stats_df['neutral_ratio'], label='Нейтрал', marker='^', linewidth=2)

        plt.xlabel('Год', fontsize=12)
        plt.ylabel('Доля писем, %', fontsize=12)
        plt.title('Динамика тональности писем (1941-1945)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.xticks(x)
        plt.grid(alpha=0.3)

        filepath = f"{self.output_dir}/sentiment_over_time.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_word_count_over_time(self, stats_df: pd.DataFrame) -> str:
        """
        График: Средняя длина письма по годам + ошибка.
        """
        plt.figure(figsize=(8, 6))

        # Bar plot с error bars
        plt.bar(
            stats_df['year'],
            stats_df['avg_word_count'],
            yerr=stats_df['std_word_count'] / 2,  # Упрощённая оценка ошибки
            capsize=5,
            color='skyblue',
            edgecolor='navy'
        )

        plt.xlabel('Год', fontsize=12)
        plt.ylabel('Среднее количество слов', fontsize=12)
        plt.title('Динамика длины писем (1941-1945)', fontsize=14, fontweight='bold')
        plt.xticks(stats_df['year'])
        plt.grid(axis='y', alpha=0.3)

        filepath = f"{self.output_dir}/word_count_over_time.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_morale_index(self, stats_df: pd.DataFrame) -> str:
        """
        График: Интегральный индекс морального духа.
        """
        plt.figure(figsize=(8, 6))

        plt.plot(
            stats_df['year'],
            stats_df['morale_index'],
            marker='o',
            linewidth=3,
            markersize=8,
            color='darkgreen'
        )
        plt.fill_between(
            stats_df['year'],
            stats_df['morale_index'],
            alpha=0.3,
            color='green'
        )

        # Линия нуля
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('Год', fontsize=12)
        plt.ylabel('Индекс морального духа', fontsize=12)
        plt.title('Динамика морального духа солдат (1941-1945)', fontsize=14, fontweight='bold')
        plt.xticks(stats_df['year'])
        plt.grid(alpha=0.3)

        filepath = f"{self.output_dir}/morale_index.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_emotion_cloud(self, emotion_stats: Dict[int, Dict[str, float]], year: int) -> Optional[str]:
        """
        Визуализация доминирующих эмоций за конкретный год.
        Простая альтернатива wordcloud.
        """
        if year not in emotion_stats:
            return None

        emotions = emotion_stats[year]

        # Сортируем по убыванию
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]
        labels, values = zip(*sorted_emotions)

        plt.figure(figsize=(8, 6))
        bars = plt.barh(labels, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#a29bfe'])

        # Подписываем значения на барах
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height() / 2,
                     f'{width}%', va='center', fontsize=10)

        plt.xlabel('Доля упоминаний, %', fontsize=11)
        plt.title(f'Эмоции в письмах {year} года', fontsize=14, fontweight='bold')
        plt.xlim(0, max(values) + 10)

        filepath = f"{self.output_dir}/emotions_{year}.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath
