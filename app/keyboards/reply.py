from aiogram.types import ReplyKeyboardMarkup, KeyboardButton


def main_menu_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📚 Урок-тест"), KeyboardButton(text="🎧 Аудио-урок")],
            [KeyboardButton(text="⚡ Быстрый вопрос"),
            KeyboardButton(text="📊 Статистика")], [KeyboardButton(text="🧪 Тестирование")],
        ],
        resize_keyboard=True,
    )



def tests_menu_kb() -> ReplyKeyboardMarkup:
    keyboard = [
        [
            KeyboardButton(text="Тест"),
            KeyboardButton(text="Анализ текста"),
        ],
        [
            KeyboardButton(text="В главное меню"),
        ],
    ]
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def lesson_in_progress_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="В главное меню")],
        ],
        resize_keyboard=True,
    )


def next_question_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Следующий вопрос")],
            [KeyboardButton(text="В главное меню")],
        ],
        resize_keyboard=True,
    )


def quick_question_kb() -> ReplyKeyboardMarkup:
    """Клавиатура выбора типа быстрого вопроса."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Слова"), KeyboardButton(text="Грамматика")],
            [KeyboardButton(text="Аудирование"), KeyboardButton(text="Произношение")],
            [KeyboardButton(text="В главное меню")],
        ],
        resize_keyboard=True,
    )


def quick_question_after_answer_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Ещё вопрос")],
            [KeyboardButton(text="В главное меню")],
        ],
        resize_keyboard=True,
    )