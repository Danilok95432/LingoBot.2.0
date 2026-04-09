from aiogram import Router, F
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery, BufferedInputFile

from datetime import datetime, timezone

from app.core.ml.question_generator import QuestionGenerator
from app.core.services import QuickQuestionService
from app.core.services.tts_service import TextToSpeechService
from app.db.session import async_session_maker
from app.db.models import Question, UserAnswer
from app.db.repositories import QuestionRepository  # пока не используется, но оставляю, если нужно где-то ещё
from app.keyboards import (
    options_kb,
    lesson_in_progress_kb,
    quick_question_kb,
    quick_question_after_answer_kb,
    main_menu_kb,
)
from app.states import QuickQuestionStates
from app.utils.telegram_retry import tg_retry

quick_router = Router()
tts_service = TextToSpeechService()

QUICK_TYPES = {
    "Слова": "vocabulary",
    "Грамматика": "grammar",
    "Аудирование": "listening",
    "Произношение": "speaking",
}


@quick_router.message(F.text == "⚡ Быстрый вопрос")
async def quick_question_menu(message: Message, state: FSMContext):
    """Показать меню выбора типа быстрого вопроса."""
    text = "Выберите тип быстрого вопроса:"
    await message.answer(text, reply_markup=quick_question_kb())
    await state.set_state(QuickQuestionStates.CHOOSING_TYPE)


async def _send_quick_question(message: Message, question: Question):
    """
    Отправка вопроса быстрого типа:
    - для listening — голосовое + варианты
    - для остальных — текст + варианты
    """
    # --- АУДИРОВАНИЕ ---
    if question.type == "listening":
        options = question.payload["options"]
        target_sentence = options[question.correct_option_index]

        # прогресс, пока генерим TTS
        wait_msg = await tg_retry(message.answer,
            "🎧 Готовлю аудио для задания, подождите пару секунд..."
        )

        audio_bytes = await tts_service.synthesize_to_ogg_bytes(target_sentence)
        voice_file = BufferedInputFile(audio_bytes, filename="quick_listening.ogg")

        try:
            await wait_msg.delete()
        except Exception:
            # Если не удалось удалить — не критично
            pass

        await message.answer_voice(
            voice=voice_file,
            caption=question.payload["text"],  # текст задания
            reply_markup=options_kb(
                options,
                question_id=question.id,
                prefix="qq_answer",  # общий prefix для быстрых вопросов
            ),
        )
        return

    # --- ОСТАЛЬНЫЕ ТИПЫ (слова, грамматика и т.п.) ---
    await message.answer(
        question.payload["text"],
        reply_markup=options_kb(
            question.payload["options"],
            question_id=question.id,
            prefix="qq_answer",
        ),
    )


@quick_router.message(QuickQuestionStates.CHOOSING_TYPE, F.text.in_(QUICK_TYPES.keys()))
async def quick_question_generate(message: Message, state: FSMContext, user):
    qtype = QUICK_TYPES[message.text]
    await tg_retry(message.answer,f"Генерирую быстрый вопрос типа: {message.text}")

    async with async_session_maker() as session:
        service = QuickQuestionService(session, QuestionGenerator(use_llm=True))
        question = await service.get_or_generate_question(user.id, user.level, qtype)

    # сохраняем тип и id текущего вопроса
    await state.update_data(qtype=qtype, current_question_id=question.id)

    # --- ПРОИЗНОШЕНИЕ ---
    if qtype == "speaking":
        # Вопрос на произношение
        await message.answer(
            question.payload["text"],
            reply_markup=lesson_in_progress_kb(),
        )
        await state.set_state(QuickQuestionStates.WAITING_ANSWER)
        await state.update_data(expected_pronunciation=question.pronunciation_phrase)
        return

    # --- ВСЕ ОСТАЛЬНЫЕ ТИПЫ (включая listening) ---
    await _send_quick_question(message, question)
    await state.set_state(QuickQuestionStates.WAITING_ANSWER)


@quick_router.callback_query(F.data.startswith("qq_answer:"))
async def handle_quick_answer(callback: CallbackQuery, state: FSMContext, user):
    await callback.answer()

    try:
        _, qid_str, idx_str = callback.data.split(":")
        question_id = int(qid_str)
        answer_idx = int(idx_str)
    except ValueError:
        await tg_retry(callback.message.answer, "Некорректный ответ, попробуйте ещё раз.")
        return

    async with async_session_maker() as session:
        # Берём вопрос из БД
        question: Question | None = await session.get(Question, question_id)
        if not question:
            await tg_retry(callback.message.answer, "Вопрос не найден, попробуйте ещё раз.")
            await state.clear()
            return

        payload = question.payload or {}
        options = payload.get("options") or []
        correct_index = question.correct_option_index  # <-- берём из модели, а не из payload

        # Проверяем корректность индексов
        if correct_index is None or not (0 <= answer_idx < len(options)) or not (0 <= correct_index < len(options)):
            await tg_retry(callback.message.answer, "Ошибка в данных вопроса. Попробуйте другой вопрос.")
            await state.clear()
            return

        is_correct = answer_idx == correct_index
        user_answer_text = options[answer_idx] if 0 <= answer_idx < len(options) else str(answer_idx)
        correct_text = options[correct_index] if 0 <= correct_index < len(options) else str(correct_index)

        # Сохраняем ответ пользователя
        ua = UserAnswer(
            user_id=user.id,
            question_id=question.id,
            session_id=None,  # быстрый вопрос, не привязан к уроку
            is_correct=is_correct,
            user_answer=user_answer_text,
            time_spent_sec=None,
            attempts=1,
            emotion_score=0.0,
            created_at=datetime.now(timezone.utc),
        )
        session.add(ua)
        await session.commit()

    # Ответ пользователю
    if is_correct:
        await tg_retry(callback.message.answer,
            f"✅ Верно! Ответ: {user_answer_text}",
            reply_markup=quick_question_after_answer_kb(),
        )
    else:
        await tg_retry(callback.message.answer,
            f"❌ Неверно.\n"
            f"Вы выбрали: {user_answer_text}\n"
            f"Правильный ответ: {correct_text}",
            reply_markup=quick_question_after_answer_kb(),
        )

    # Остаёмся в контексте быстрого вопроса, ждём, что пользователь решит
    await state.set_state(QuickQuestionStates.AFTER_ANSWER)


@quick_router.message(QuickQuestionStates.AFTER_ANSWER, F.text == "Ещё вопрос")
async def quick_question_more(message: Message, state: FSMContext, user):
    data = await state.get_data()
    qtype = data.get("qtype")  # должен быть сохранён при выборе категории

    if not qtype:
        await tg_retry(message.answer,"Тип вопроса не найден. Пожалуйста, выберите категорию заново из меню.")
        await state.clear()
        return

    async with async_session_maker() as session:
        service = QuickQuestionService(session, QuestionGenerator(use_llm=True))
        question = await service.get_or_generate_question(user.id, user.level, qtype)

    if not question:
        await tg_retry(message.answer,"Не удалось сгенерировать новый вопрос. Попробуйте позже.")
        await state.clear()
        return

    # сохраняем новый текущий вопрос
    await state.update_data(current_question_id=question.id)

    # --- ПРОИЗНОШЕНИЕ ---
    if qtype == "speaking":
        await message.answer(
            question.payload["text"],
            reply_markup=lesson_in_progress_kb(),
        )
        await state.set_state(QuickQuestionStates.WAITING_ANSWER)
        await state.update_data(expected_pronunciation=question.pronunciation_phrase)
        return

    # --- ВСЕ ОСТАЛЬНЫЕ ТИПЫ (включая listening) ---
    await _send_quick_question(message, question)
    await state.set_state(QuickQuestionStates.WAITING_ANSWER)


@quick_router.message(QuickQuestionStates.AFTER_ANSWER, F.text == "В главное меню")
async def quick_question_back_to_menu(message: Message, state: FSMContext):
    await state.clear()
    await tg_retry(message.answer,
        "Вы в главном меню.",
        reply_markup=main_menu_kb(),
    )
