from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def options_kb(
    options: list[str],
    question_id: int,
    prefix: str = "answer",  # <-- добавили префикс
) -> InlineKeyboardMarkup:
    buttons = []
    for idx, opt in enumerate(options):
        buttons.append(
            [
                InlineKeyboardButton(
                    text=opt,
                    callback_data=f"{prefix}:{question_id}:{idx}",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def level_test_options_kb(options: list[str], question_index: int) -> InlineKeyboardMarkup:
    """
    Клавиатура для вопросов теста уровня.
    callback_data: 'lvlanswer:{q_index}:{answer_index}'
    """
    keyboard = [
        [
            InlineKeyboardButton(
                text=option,
                callback_data=f"lvlanswer:{question_index}:{idx}",
            )
        ]
        for idx, option in enumerate(options)
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)
