import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import INPUT_LETTERS_DB, MIN_TEXT_LENGTH

DATA_FILE = INPUT_LETTERS_DB


def load_letters():
    """
    Загружает существующие письма из файла
    """
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return []


def save_letters(letters):
    """
    Сохраняет список писем в файл
    """
    os.makedirs("../data", exist_ok=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(letters, f, ensure_ascii=False, indent=4)
    print(f"Сохранено {len(letters)} писем в {DATA_FILE}")


def get_multiline_input(prompt: str) -> str:
    """
    Читает многострочный текст до команды 'х' на отдельной строке
    """
    print(prompt)
    print("Вставьте текст письма. Введите 'х' (русская), чтобы завершить ввод.")

    lines = []
    while True:
        line = input()
        if line.strip().lower() == 'х':
            break
        lines.append(line)
        if line.lower() in ['q', 'quit', 'exit']:
            print("Выход из программы.")
            exit(0)

    return '\n'.join(lines)


def extract_year_from_date(date_str):
    """
    Пытается вытащить год из строки даты
    """
    if date_str:
        parts = date_str.split('.')
        try:
            year = int(parts[-1])
            print("Год определён автоматически:", year)
            return year
        except ValueError:
            print("Ошибка при обработке. Введите год вручную.")
            return None
    return None


def print_stats(letters):
    """
    Печатает статистику по годам
    """
    if not letters:
        print("Писем пока нет.")
        return

    print("\n" + "=" * 40)
    print(" СТАТИСТИКА БАЗЫ ДАННЫХ")
    print("=" * 40)

    years_count = {}
    for letter in letters:
        year = letter.get('year')
        if year:
            years_count[year] = years_count.get(year, 0) + 1

    for year in sorted(years_count.keys()):
        count = years_count[year]
        bar = "█" * count
        print(f"{year} год: {count} писем {bar}")

    print(f"ВСЕГО: {len(letters)} писем")
    print("=" * 40 + "\n")


def main():
    print("ДОБАВЛЕНИЕ ПИСЕМ В БАЗУ (WWII Letters Analysis)")
    print("Для выхода введите 'q' или 'quit' в поле текста.")

    letters = load_letters()
    next_id = len(letters) + 1
    print_stats(letters)

    source = input("Введите источник добавляемых писем (ссылка/название). Если письма личные/родственника/знакомого, то нажмите Enter: \n")
    if not source: source = "Личный архив"

    while True:
        print(f"--- Письмо #{next_id} ---")

        # 1. Текст письма
        text = get_multiline_input("Текст письма (вставить): ")

        if len(text) < MIN_TEXT_LENGTH:
            print(f"Текст слишком короткий (минимум {MIN_TEXT_LENGTH} символов). Пропускаем.")
            continue

        # 2. Автор (необязательно)
        author = input("От кого (Enter, если неизвестно): ").strip()
        if not author: author = "Неизвестный солдат"

        # 3. Дата (необязательно)
        date_str = input("Дата (дд.мм.гггг или год, Enter если нет): ").strip()

        # 4. Год
        year = extract_year_from_date(date_str)
        if not year:
            year_input = input("Введите год (1941-1945): ").strip()
            try:
                year = int(year_input)
            except ValueError:
                year = 0  # Если не ввели число

        # 5. Кому
        recipient = input("Кому (Enter, если нет): ").strip()

        # Формируем объект
        new_letter = {
            "id": next_id,
            "text": text,
            "from": author,
            "date": date_str if date_str else None,
            "year": year,
            "to": recipient if recipient else None,
            "source": source
        }

        letters.append(new_letter)
        save_letters(letters)

        print(f"Письмо #{next_id} успешно добавлено!")
        next_id += 1
        print_stats(letters)


if __name__ == "__main__":
    main()
