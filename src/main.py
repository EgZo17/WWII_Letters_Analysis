import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    ENRICHED_LETTERS_DB,
    PLOTS_DIR,
    ANALYSIS_REPORT,
    ensure_dirs
)
from src.analysis.temporal_analysis import TemporalAnalyzer
from src.analysis.hypothesis_test import HypothesisTester
from src.visualization.plots import PlotGenerator


class NumpyEncoder(json.JSONEncoder):
    """
    Конвертирует типы numpy в нативные типы Python для JSON
    """
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def main():
    ensure_dirs()

    print("WWII Letters Analysis")
    print("=" * 50)

    # 1. Загрузка данных
    if not ENRICHED_LETTERS_DB.exists():
        print(f"Ошибка: Файл {ENRICHED_LETTERS_DB.name} не найден!")
        print("Подсказка: Сначала запустите python scripts/process_letters.py")
        return

    print(f"Загрузка данных из {ENRICHED_LETTERS_DB.name}...")
    with open(ENRICHED_LETTERS_DB, 'r', encoding='utf-8') as f:
        letters = json.load(f)

    print(f"Загружено {len(letters)} писем")

    # 2. Анализ по времени
    print("\nАнализ динамики...")
    temporal = TemporalAnalyzer(letters)
    stats = temporal.get_yearly_stats()
    morale_df = temporal.calculate_morale_index()
    stats = stats.merge(morale_df, on='year')

    print("\nСтатистика по годам:")
    print(stats.to_string(index=False))

    # 3. Проверка гипотез
    print("\nПроверка гипотез...")
    df = pd.DataFrame(letters)
    tester = HypothesisTester(df)

    print("\n--- Гипотеза 1: Длина писем ---")
    length_result = tester.test_letters_length_hypothesis()
    if 'error' not in length_result:
        print(f"Ранний период (1941-42): {length_result['early_period_avg']} слов")
        print(f"Поздний период (1944-45): {length_result['late_period_avg']} слов")
        p_val = length_result['p_value']
        if p_val < 0.055:
            significance = "Значимо"
        elif p_val < 0.11:
            direction = "к сокращению" if length_result['direction'] == 'shorter' else "к удлинению"
            significance = f"Пограничная значимость (тенденция {direction})"
        else:
            significance = "Не значимо"
        print(f"p-value: {p_val} -> {significance}")

    print("\n--- Гипотеза 2: Тональность ---")
    sentiment_result = tester.test_sentiment_hypothesis()
    if 'error' not in sentiment_result:
        print(f"Доля негатива 1941-42: {sentiment_result['early_negative_ratio']}%")
        print(f"Доля негатива 1944-45: {sentiment_result['late_negative_ratio']}%")
        p_val = sentiment_result['p_value']
        if p_val < 0.055 and sentiment_result['confirmed']:
            significance = "Подтверждено"
        elif p_val < 0.11:
            if sentiment_result['late_negative_ratio'] < sentiment_result['early_negative_ratio']:
                significance = "Пограничная значимость (тенденция к снижению негатива)"
            else:
                significance = "Пограничная значимость (тенденция к росту негатива)"
        else:
            significance = "Не подтверждено"
        print(f"p-value: {p_val} -> {significance}")

    # Итоговый вывод
    print("\n" + "=" * 50)
    print("ВЫВОД:")
    print(tester.get_conclusion())
    print("=" * 50)

    # 4. Визуализация
    print("\nГенерация графиков...")
    plotter = PlotGenerator(output_dir=str(PLOTS_DIR))

    plots_paths = {
        'sentiment': plotter.plot_sentiment_over_time(stats),
        'word_count': plotter.plot_word_count_over_time(stats),
        'morale': plotter.plot_morale_index(stats)
    }

    # Графики эмоций для каждого года
    emotion_dynamics = temporal.get_emotion_dynamics()
    for year in emotion_dynamics:
        plotter.plot_emotion_cloud(emotion_dynamics, year)

    print(f"Графики сохранены в: {PLOTS_DIR}")

    # 5. Сохранение отчета
    print("\nСохранение отчета...")
    report = {
        'summary': {
            'total_letters': len(letters),
            'years_covered': list(range(1941, 1946)),
            'hypothesis_length': length_result if 'error' not in length_result else None,
            'hypothesis_sentiment': sentiment_result if 'error' not in sentiment_result else None,
            'conclusion': tester.get_conclusion()
        },
        'yearly_stats': stats.to_dict('records'),
        'plots': plots_paths
    }

    with open(ANALYSIS_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"Отчет сохранен: {ANALYSIS_REPORT}")
    print("\nАнализ завершен.")


if __name__ == "__main__":
    main()
