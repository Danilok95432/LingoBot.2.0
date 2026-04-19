from aiogram import Router, F
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery
from aiogram.types import BufferedInputFile  # <-- ВАЖНО

from app.core.ml.question_generator import QuestionGenerator
from app.core.services import LessonService, LessonType, DifficultyRegressionService
from app.db.session import async_session_maker
from app.db.repositories import LessonRepository
from app.keyboards import (
    options_kb,
    lesson_in_progress_kb,
    next_question_kb,
    main_menu_kb,
)
from app.states import LessonStates
from app.utils import render_progress_bar
from app.core.services.tts_service import TextToSpeechService
from app.utils.telegram_retry import tg_retry

lessons_router = Router()
tts_service = TextToSpeechService()


# ===== ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ОТПРАВКИ ВОПРОСА В УРОКЕ =====

async def _send_lesson_question(message: Message, state: FSMContext, qdto):
    """
    Унифицированная отправка текущего вопроса урока в чат.

    Поддерживает типы:
    - grammar / vocabulary: обычный текст + варианты
    - listening: TTS-аудио + варианты
    - speaking: текст + инструкция отправить голосовое (без вариантов)
    """
    question = qdto.question
    progress = render_progress_bar(qdto.index, qdto.total)
    await state.update_data(current_question_id=question.id)

    # --- АУДИРОВАНИЕ ---
    if question.type == "listening":
        options = question.payload["options"]
        target_sentence = options[question.correct_option_index]

        audio_bytes = await tts_service.synthesize_to_ogg_bytes(target_sentence)
        # aiogram 3 требует InputFile, используем BufferedInputFile
        voice_file = BufferedInputFile(audio_bytes, filename="lesson_listening.ogg")

        # Голосовое с подписью и прогрессом
        await message.answer_voice(
            voice=voice_file,
            caption=f"{question.payload['text']}\n\n{progress}",
            reply_markup=lesson_in_progress_kb(),
        )

        # Варианты ответа
        await tg_retry(message.answer,
            "Выберите ответ:",
            reply_markup=options_kb(options, question_id=question.id),
        )
        await state.set_state(LessonStates.IN_LESSON)
        return

    # --- ПРОИЗНОШЕНИЕ ---
    if question.type == "speaking":
        # Сохраняем ожидаемую фразу в FSM — её возьмёт pronunciation_handler
        await state.update_data(expected_pronunciation=question.pronunciation_phrase)

        await tg_retry(message.answer,
            f"{question.payload['text']}\n\n{progress}",
            reply_markup=lesson_in_progress_kb(),
        )
        await message.answer(
            "Запишите голосовое сообщение с этой фразой и отправьте его в чат.",
        )
        await state.set_state(LessonStates.IN_LESSON)
        return

    # --- Обычные вопросы (grammar / vocabulary и другие) ---
    await tg_retry(message.answer,
        f"{question.payload['text']}\n\n{progress}",
        reply_markup=lesson_in_progress_kb(),
    )
    await tg_retry(message.answer,
        "Выберите ответ:",
        reply_markup=options_kb(question.payload["options"], question_id=question.id),
    )
    await state.set_state(LessonStates.IN_LESSON)


# ===== СТАРТ УРОКА-ТЕСТА =====

@lessons_router.message(F.text == "📚 Урок-тест")
async def start_lesson_test(message: Message, state: FSMContext, user):
    await tg_retry(message.answer,"Генерирую урок-тест...", reply_markup=lesson_in_progress_kb())
    async with async_session_maker() as session:
        service = LessonService(
            session=session,
            diff_regressor=DifficultyRegressionService(session),
            question_generator=QuestionGenerator(),
        )
        lesson = await service.start_lesson(
            user_id=user.id,
            level=user.level,
            lesson_type=LessonType.LESSON_TEST,
            last_message_id=message.message_id,
        )
        await state.set_state(LessonStates.IN_LESSON)
        await state.update_data(lesson_id=lesson.id)

        qdto = await service.get_current_question(lesson)
        await _send_lesson_question(message, state, qdto)


# ===== СТАРТ АУДИО-УРОКА =====

@lessons_router.message(F.text == "🎧 Аудио-урок")
async def start_audio_lesson(message: Message, state: FSMContext, user):
    await tg_retry(message.answer,"Генерирую аудио-урок...", reply_markup=lesson_in_progress_kb())
    async with async_session_maker() as session:
        service = LessonService(
            session=session,
            diff_regressor=DifficultyRegressionService(session),
            question_generator=QuestionGenerator(),
        )
        lesson = await service.start_lesson(
            user_id=user.id,
            level=user.level,
            lesson_type=LessonType.AUDIO_LESSON,
            last_message_id=message.message_id,
        )
        await state.set_state(LessonStates.IN_LESSON)
        await state.update_data(lesson_id=lesson.id)

        qdto = await service.get_current_question(lesson)
        await _send_lesson_question(message, state, qdto)


# ===== ОБРАБОТКА ОТВЕТА ПОЛЬЗОВАТЕЛЯ (КНОПКИ) =====

@lessons_router.callback_query(F.data.startswith("answer:"))
async def handle_answer(callback: CallbackQuery, state: FSMContext, user):
    await callback.answer()
    data = await state.get_data()
    lesson_id = data.get("lesson_id")
    if not lesson_id:
        await tg_retry(callback.message.answer, "Сессия урока не найдена. Вернитесь в главное меню.")
        return

    # callback.data: "answer:{question_id}:{option_index}"
    _, qid_str, idx_str = callback.data.split(":")
    answer_idx = int(idx_str)

    async with async_session_maker() as session:
        service = LessonService(
            session=session,
            diff_regressor=DifficultyRegressionService(session),
            question_generator=QuestionGenerator(),
        )
        lesson_repo = LessonRepository(session)

        # Берём активную сессию урока для пользователя
        lesson = await lesson_repo.get_active_session(user.id)
        if not lesson:
            await tg_retry(callback.message.answer, "Сессия урока завершена или не найдена.")
            await state.clear()
            return

        # Оцениваем эмоцию (пока заглушка, но не ломает логику)
        emotion_score = 0.0

        # Фиксируем ответ пользователя
        is_correct, lesson = await service.answer_question(
            lesson=lesson,
            user_id=user.id,
            answer_index=answer_idx,
            emotion_score=emotion_score,
        )

        result_text = "Верно! 🎉" if is_correct else "Неверно 😢"
        qdto = await service.get_current_question(lesson)
        if qdto is None:
            msg = (
                f"{result_text}\n"
                f"Урок завершён! Правильных ответов: {lesson.correct_count}/{lesson.total_questions}"
            )
            await callback.message.answer(msg, reply_markup=main_menu_kb())
            await state.clear()
            return

        # Прогресс показываем по уже отвеченным вопросам
        progress = render_progress_bar(qdto.index - 1, qdto.total)
        await tg_retry(callback.message.answer,
            f"{result_text}\n{progress}",
            reply_markup=next_question_kb(),
        )
        await state.set_state(LessonStates.WAITING_NEXT)
        await state.update_data(lesson_id=lesson.id)


# ===== ПОЛУЧЕНИЕ СЛЕДУЮЩЕГО ВОПРОСА =====

@lessons_router.message(LessonStates.WAITING_NEXT, F.text == "Следующий вопрос")
async def next_question(message: Message, state: FSMContext, user):
    data = await state.get_data()
    lesson_id = data.get("lesson_id")
    if not lesson_id:
        await tg_retry(message.answer,"Сессия урока не найдена.", reply_markup=main_menu_kb())
        await state.clear()
        return

    async with async_session_maker() as session:
        service = LessonService(
            session=session,
            diff_regressor=DifficultyRegressionService(session),
            question_generator=QuestionGenerator(),
        )
        lesson_repo = LessonRepository(session)
        lesson = await lesson_repo.get_active_session(user.id)
        if not lesson:
            await tg_retry(message.answer,"Урок уже завершён.", reply_markup=main_menu_kb())
            await state.clear()
            return

        qdto = await service.get_current_question(lesson)
        if qdto is None:
            await tg_retry(message.answer,
                f"Урок завершён! Правильных ответов: {lesson.correct_count}/{lesson.total_questions}",
                reply_markup=main_menu_kb(),
            )
            await state.clear()
            return

        await _send_lesson_question(message, state, qdto)
