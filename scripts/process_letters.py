import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import INPUT_LETTERS_DB, ENRICHED_LETTERS_DB
from src.nlp.sentiment_analyzer import SentimentAnalyzer

DATA_INPUT = INPUT_LETTERS_DB
DATA_OUTPUT = ENRICHED_LETTERS_DB


def process_letters():
    if not os.path.exists(DATA_INPUT):
        print(f"Файл {DATA_INPUT} не найден!")
        return

    print("Загружаю базу писем...")
    with open(DATA_INPUT, 'r', encoding='utf-8') as f:
        letters = json.load(f)

    if not letters:
        print("База пуста.")
        return

    # Базовая валидация
    valid_letters = []
    for letter in letters:
        if letter.get("year") not in range(1941, 1946):
            print(f"Пропускаю #{letter.get('id')} — год вне диапазона 1941-1945")
            continue
        valid_letters.append(letter)

    if len(valid_letters) < len(letters):
        print(f"Отфильтровано {len(letters) - len(valid_letters)} некорректных записей")

    letters = valid_letters
    print(f"Загружено {len(letters)} писем.")
    print("Запускаю NLP-анализ...")

    analyzer = SentimentAnalyzer()
    texts = [letter["text"] for letter in letters]

    start_time = time.time()
    predictions = analyzer.analyze_batch(texts)
    duration = round(time.time() - start_time, 2)

    print(f"Анализ тональности завершен за {duration} сек.")
    print("Анализирую эмоции по ключевым словам...")

    # Обогащаем данные
    emotion_stats = {}

    for i, letter in enumerate(letters):
        pred = predictions[i]

        # Базовая тональность
        letter["sentiment_raw"] = pred["label"]
        letter["sentiment_confidence"] = round(pred["score"], 4)
        letter["sentiment_category"] = analyzer.map_to_historical_categories(pred["label"])

        # Эмоции по ключевым словам
        emotions, triggers = analyzer.detect_emotions(letter["text"], min_score=2)
        letter["emotions"] = emotions
        letter["triggers"] = triggers
        letter["word_count"] = len(letter["text"].split())

        # Считаем статистику эмоций
        for emotion in emotions:
            emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1

    # Сохраняем
    with open(DATA_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(letters, f, ensure_ascii=False, indent=4)

    print(f"Обогащённая база сохранена в {DATA_OUTPUT}")

    # Показываем статистику эмоций
    print("\nСТАТИСТИКА ЭМОЦИЙ:")
    for emotion, count in sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * count
        print(f"  {emotion:12} : {count:3} писем {bar}")

    print("\nГотово!")


if __name__ == "__main__":
    process_letters()
