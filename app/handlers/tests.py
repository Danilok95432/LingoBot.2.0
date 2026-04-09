from __future__ import annotations

from typing import Any, Dict, List

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from app.core.ml.question_generator import QuestionGenerator
from app.core.services.difficulty_regression_service import DifficultyRegressionService
from app.core.services.test_service import TestService, LevelTestQuestion
from app.db.session import async_session_maker
from app.keyboards import main_menu_kb, tests_menu_kb
from app.keyboards.inline import level_test_options_kb
from app.states import LevelTestStates

tests_router = Router(name="tests")


# ---------------------------------------------------------------------------
# Меню тестирования
# ---------------------------------------------------------------------------


@tests_router.message(F.text == "🧪 Тестирование")
async def show_tests_menu(message: Message, state: FSMContext) -> None:
    """Показываем подменю с выбором способа тестирования."""
    await state.clear()
    await message.answer(
        "Выберите формат определения уровня:",
        reply_markup=tests_menu_kb(),
    )


# ---------------------------------------------------------------------------
# Квиз-тест уровня (варианты ответа)
# ---------------------------------------------------------------------------


@tests_router.message(F.text == "Тест")
async def start_level_quiz(message: Message, state: FSMContext, user: Any) -> None:
    """Запуск квиз-теста уровня."""
    await state.clear()

    async with async_session_maker() as session:
        service = TestService(
            session=session,
            diff_regressor=DifficultyRegressionService(session),
            question_generator=QuestionGenerator(),
        )

        # предполагаемый уровень берём из профиля пользователя, если он задан
        approx_level = getattr(user, "level", None)
        questions: List[LevelTestQuestion] = await service.build_level_quiz(
            approx_level=approx_level,
            questions_per_level=2,  # итого 12 вопросов (6 уровней * 2)
        )

    # Сохраняем вопросы в FSM как обычные словари
    questions_data: List[Dict[str, Any]] = [
        {
            "index": q.index,
            "level": q.level,
            "type": q.type,
            "text": q.text,
            "options": q.options,
            "correct_index": q.correct_index,
        }
        for q in questions
    ]

    await state.update_data(
        quiz_questions=questions_data,
        answers_correct=[False] * len(questions_data),
        current_index=0,
    )
    await state.set_state(LevelTestStates.QUIZ_IN_PROGRESS)

    await _send_current_quiz_question(message, questions_data, 0)


async def _send_current_quiz_question(
    message: Message,
    questions: List[Dict[str, Any]],
    index: int,
) -> None:
    question = questions[index]
    text = question["text"]
    options = question["options"]
    await message.answer(
        text,
        reply_markup=level_test_options_kb(options=options, question_index=index),
    )


@tests_router.callback_query(
    LevelTestStates.QUIZ_IN_PROGRESS,
    F.data.startswith("lvlanswer:"),
)
async def handle_level_quiz_answer(
    callback: CallbackQuery,
    state: FSMContext,
    user: Any,
) -> None:
    await callback.answer()

    try:
        _, q_idx_str, ans_idx_str = callback.data.split(":", maxsplit=2)  # type: ignore[arg-type]
        q_idx = int(q_idx_str)
        ans_idx = int(ans_idx_str)
    except Exception:
        await callback.message.answer("Не получилось распознать ответ, попробуйте ещё раз.")
        return

    data = await state.get_data()
    questions: List[Dict[str, Any]] = data.get("quiz_questions", [])
    answers_correct: List[bool] = data.get("answers_correct", [])
    current_index: int = data.get("current_index", 0)

    if not questions or q_idx >= len(questions):
        await callback.message.answer(
            "Сессия тестирования устарела. Запустите тест ещё раз из главного меню.",
            reply_markup=main_menu_kb(),
        )
        await state.clear()
        return

    question = questions[q_idx]
    correct_index = int(question["correct_index"])

    is_correct = ans_idx == correct_index
    if q_idx < len(answers_correct):
        answers_correct[q_idx] = is_correct

    # Краткая обратная связь
    if is_correct:
        feedback = "✅ Верно!"
    else:
        correct_option = question["options"][correct_index]
        feedback = f"❌ Неверно. Правильный ответ: {correct_option}"

    await callback.message.answer(feedback)

    # Переходим к следующему вопросу или завершаем тест
    next_index = current_index + 1
    total = len(questions)

    if next_index >= total:
        # Завершение теста, оценка уровня
        from app.db.repositories import UserRepository

        async with async_session_maker() as session:
            service = TestService(
                session=session,
                diff_regressor=DifficultyRegressionService(session),
                question_generator=QuestionGenerator(),
            )

            level_questions = [
                LevelTestQuestion(
                    index=q["index"],
                    level=q["level"],
                    type=q["type"],
                    text=q["text"],
                    options=list(q["options"]),
                    correct_index=int(q["correct_index"]),
                )
                for q in questions
            ]
            level = service.estimate_level_from_quiz(
                level_questions,
                answers_correct,
            )

            # Обновляем уровень пользователя в БД
            repo = UserRepository(session)
            db_user = await repo.get_by_id(user.id)  # type: ignore[arg-type]
            if db_user is not None:
                await repo.update_level(db_user, level)

        await callback.message.answer(
            f"Тест завершён! \n\nПредполагаемый уровень английского: <b>{level}</b>.",
            reply_markup=main_menu_kb(),
        )
        await state.clear()
        return

    # Есть ещё вопросы – задаём следующий
    await state.update_data(
        answers_correct=answers_correct,
        current_index=next_index,
    )
    await _send_current_quiz_question(callback.message, questions, next_index)


# ---------------------------------------------------------------------------
# Тестирование уровнем по тексту
# ---------------------------------------------------------------------------


@tests_router.message(F.text == "Анализ текста")
async def ask_text_for_level_test(message: Message, state: FSMContext) -> None:
    await state.set_state(LevelTestStates.TEXT_INPUT)
    await message.answer(
        "Отправьте несколько предложений на английском языке – по ним я оценю ваш уровень.",
    )


@tests_router.message(LevelTestStates.TEXT_INPUT)
async def handle_text_level_test(
    message: Message,
    state: FSMContext,
    user: Any,
) -> None:
    text = message.text or ""

    async with async_session_maker() as session:
        service = TestService(
            session=session,
            diff_regressor=DifficultyRegressionService(session),
            question_generator=QuestionGenerator(),
        )
        level = await service.analyze_text_level(text)

        # обновляем уровень пользователя в БД
        from app.db.repositories import UserRepository

        repo = UserRepository(session)
        db_user = await repo.get_by_id(user.id)  # type: ignore[arg-type]
        if db_user is not None:
            await repo.update_level(db_user, level)

    await message.answer(
        f"По анализу текста ваш предполагаемый уровень английского: <b>{level}</b>.",
        reply_markup=main_menu_kb(),
    )
    await state.clear()
