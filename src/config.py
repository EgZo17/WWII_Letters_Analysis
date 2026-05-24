from pathlib import Path

# Корневая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Пути к данным
DATA_DIR = BASE_DIR / "data"
INPUT_LETTERS_DB = DATA_DIR / "letters_database.json"
ENRICHED_LETTERS_DB = DATA_DIR / "letters_enriched.json"

# Пути к результатам
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
ANALYSIS_REPORT = RESULTS_DIR / "analysis_report.json"

# Настройки NLP
SENTIMENT_MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"
MIN_TEXT_LENGTH = 15
MIN_EMOTION_SCORE = 2

# Утилита для создания папок
def ensure_dirs():
    """Создаёт необходимые директории, если их нет"""
    for d in [DATA_DIR, RESULTS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
