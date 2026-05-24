import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict


class HypothesisTester:
    """
    Статистическая проверка гипотез проекта.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
        self.df = self.df[self.df['year'].between(1941, 1945)]

    def test_letters_length_hypothesis(self) -> Dict:
        """
        Гипотеза: "К концу войны письма становятся короче"

        Сравниваем среднюю длину писем:
        - Ранний период (1941-1942)
        - Поздний период (1944-1945)

        Returns:
            Результаты t-теста
        """
        early = self.df[self.df['year'] <= 1942]['word_count']
        late = self.df[self.df['year'] >= 1944]['word_count']

        if len(early) < 5 or len(late) < 5:
            return {'error': 'Недостаточно данных для теста'}

        # Welch's t-test (не предполагает равные дисперсии)
        t_stat, p_value = stats.ttest_ind(early, late, equal_var=False)

        early_mean = early.mean()
        late_mean = late.mean()

        return {
            'hypothesis': 'Письма к концу войны становятся короче',
            'early_period_avg': round(early_mean, 1),
            'late_period_avg': round(late_mean, 1),
            'difference': round(late_mean - early_mean, 1),
            't_statistic': round(t_stat, 3),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
            'direction': 'shorter' if late_mean < early_mean else 'longer'
        }

    def test_sentiment_hypothesis(self) -> Dict:
        """
        Гипотеза: "Доля негативных писем уменьшается к концу войны"
        Заменён statsmodels.proportions_ztest на ручной расчёт через scipy.stats.norm
        """
        early_df = self.df[self.df['year'] <= 1942]
        late_df = self.df[self.df['year'] >= 1944]

        if len(early_df) < 5 or len(late_df) < 5:
            return {'error': 'Недостаточно данных для теста'}

        p1 = (early_df['sentiment_raw'] == 'negative').mean()
        p2 = (late_df['sentiment_raw'] == 'negative').mean()

        n1, n2 = len(early_df), len(late_df)
        count1, count2 = int(p1 * n1), int(p2 * n2)

        # Pooled proportion & Standard Error
        p_pool = (count1 + count2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

        # Z-statistic
        z_stat = (p2 - p1) / se if se > 0 else 0

        # P-value (one-tailed: проверяем, уменьшилась ли доля, alternative='smaller')
        p_value = stats.norm.cdf(z_stat)

        return {
            'hypothesis': 'Доля негативных писем уменьшается к концу войны',
            'early_negative_ratio': round(p1 * 100, 1),
            'late_negative_ratio': round(p2 * 100, 1),
            'change_percentage_points': round((p2 - p1) * 100, 1),
            'z_statistic': round(z_stat, 3),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
            'confirmed': p_value < 0.05 and p2 < p1
        }

    def get_conclusion(self) -> str:
        """
        Формирует текстовый вывод по результатам тестов.
        """
        length_test = self.test_letters_length_hypothesis()
        sentiment_test = self.test_sentiment_hypothesis()

        conclusions = []

        # Вывод по длине
        if 'error' not in length_test:
            p_val = length_test['p_value']
            if p_val < 0.055:
                direction = "укоротились" if length_test['direction'] == 'shorter' else "удлинились"
                conclusions.append(f"Гипотеза о длине: письма статистически значимо {direction} (p={p_val})")
            elif p_val < 0.11:
                direction = "к сокращению" if length_test['direction'] == 'shorter' else "к удлинению"
                conclusions.append(f"Гипотеза о длине: пограничная значимость, тенденция {direction} (p={p_val})")
            else:
                conclusions.append(f"Гипотеза о длине: изменения незначимы (p={p_val})")

        # Вывод по тональности
        if 'error' not in sentiment_test:
            p_val = sentiment_test['p_value']
            if p_val < 0.055 and sentiment_test['confirmed']:
                conclusions.append(f"Гипотеза о тональности: доля негатива снизилась (p={p_val})")
            elif p_val < 0.11:
                if sentiment_test['late_negative_ratio'] < sentiment_test['early_negative_ratio']:
                    conclusions.append(
                        f"Гипотеза о тональности: пограничная значимость, тенденция к снижению негатива (p={p_val})")
                else:
                    conclusions.append(
                        f"Гипотеза о тональности: пограничная значимость, тенденция к росту негатива (p={p_val})")
            elif sentiment_test['significant']:
                conclusions.append(f"Гипотеза о тональности: негатива стало больше (p={p_val})")
            else:
                conclusions.append(f"Гипотеза о тональности: изменения незначимы (p={p_val})")

        return "\n".join(conclusions)
