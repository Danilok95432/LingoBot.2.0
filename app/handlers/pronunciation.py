from aiogram import Router, F
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from aiogram.enums import ChatAction
from loguru import logger
from app.db.repositories import LessonRepository
from app.keyboards import next_question_kb, main_menu_kb
from app.states import LessonStates

from app.db.session import async_session_maker
from app.db.models import UserAnswer
from app.core.ml.emotion_analyzer import EmotionAnalyzer
from app.core.ml.pronunciation_evaluator import PronunciationEvaluator
from app.core.services import PronunciationService
from app.core.ml.speech_emotion_analyzer import SpeechEmotionAnalyzer

pronunciation_router = Router()
speech_emotion_analyzer = SpeechEmotionAnalyzer()


@pronunciation_router.message(F.voice)
async def handle_voice(message: Message, state: FSMContext, user):
    data = await state.get_data()
    expected = data.get("expected_pronunciation")
    if not expected:
        await message.answer(
            "Сейчас нет активного задания на произношение. "
            "Получите его через 'Быстрый вопрос' или урок."
        )
        return

    # Сообщение об ожидании оставляем
    await message.bot.send_chat_action(
        chat_id=message.chat.id,
        action=ChatAction.TYPING,
    )
    wait_msg = await message.answer(
        "⏳ Оцениваю ваше произношение, это может занять 10–20 секунд..."
    )

    service = PronunciationService(
        evaluator=PronunciationEvaluator(),
        emotion_analyzer=EmotionAnalyzer(),
    )

    try:
        feedback, emotion_info = await service.evaluate(
            bot=message.bot,
            voice=message.voice,
            expected_text=expected,
            user_level=user.level,
        )

        # ------ базовая часть по произношению ------
        score = getattr(feedback, "score", 0.0)
        threshold = getattr(feedback, "threshold", 0.0)
        passed = getattr(feedback, "passed", False)
        recognized_text = getattr(feedback, "recognized_text", "")
        global_comment = getattr(feedback, "global_comment", "")
        tips = getattr(feedback, "tips", [])

        if passed:
            text = (
                "Отлично! Ваше произношение засчитано ✅\n"
                f"Оценка: {score:.2f} (порог {threshold:.2f})"
            )
        else:
            text = (
                "Пока не засчитано ❌\n"
                f"Оценка: {score:.2f}, нужно хотя бы {threshold:.2f}.\n"
                "Попробуйте ещё раз!"
            )

        if recognized_text:
            text += f"\n\nРаспознанный текст:\n“{recognized_text}”"

        if global_comment:
            text += f"\n\nКомментарий:\n{global_comment}"

        if tips:
            text += "\n\nСоветы:"
            for tip in tips:
                text += f"\n• {tip}"

         # ------ блок про эмоции (устойчивый к tuple/dict/obj) ------
        label = None
        emo_score = None

        if emotion_info is not None:
            # 1) dict-формат
            if isinstance(emotion_info, dict):
                label = emotion_info.get("label")
                emo_score = emotion_info.get("score")

            # 2) tuple/list, как у тебя сейчас: ('sad', 0.47, 0.47)
            elif isinstance(emotion_info, (tuple, list)):
                if len(emotion_info) >= 2:
                    label = emotion_info[0]
                    emo_score = emotion_info[1]

            # 3) объект с атрибутами
            else:
                label = getattr(emotion_info, "label", None)
                emo_score = getattr(emotion_info, "score", None)

        logger.debug(
            "handle_voice: parsed emotion -> label='{}', score={:.3f}",
            label,
            emo_score,
        )

        if label:
            emoji_map = {
                "happy": "😊",
                "sad": "😔",
                "angry": "😠",
                "fear": "😨",
                "disgust": "🤢",
                "surprise": "😲",
                "neutral": "😐",
            }
            emo = emoji_map.get(label, "🙂")

            text += "\n\n"
            text += f"{emo} Похоже, вы говорили с эмоцией: {label}"
            if emo_score is not None:
                text += f" (уверенность модели: {emo_score:.2f})."

            extra_tips = []
            if label in {"sad", "fear", "angry"}:
                extra_tips.append(
                    "Если чувствуете напряжение — сделайте пару глубоких вдохов перед следующей фразой."
                )
            if label == "happy":
                extra_tips.append(
                    "Классно, что вы в хорошем настроении — в таком состоянии учиться легче! 🎉"
                )

            if extra_tips:
                text += "\n" + "\n".join(f"- {t}" for t in extra_tips)

        question_id = data.get("current_question_id")
        lesson_id = data.get("lesson_id")

        if question_id is not None:
            try:
                async with async_session_maker() as session:
                    lesson_repo = LessonRepository(session)

                    lesson = None
                    if lesson_id is not None:
                        lesson = await lesson_repo.get_session(int(lesson_id))

                    # Если это ответ внутри урока — сохраняем его в lesson session
                    if lesson is not None and lesson.state == "in_progress":
                        ua = UserAnswer(
                            user_id=user.id,
                            question_id=int(question_id),
                            session_id=lesson.id,
                            is_correct=bool(passed),
                            user_answer=recognized_text or expected,
                            time_spent_sec=None,
                            attempts=1,
                            emotion_score=float(emo_score) if emo_score is not None else 0.0,
                        )
                        session.add(ua)

                        if passed:
                            lesson.correct_count += 1

                        lesson.current_index += 1

                        is_finished = lesson.current_index >= lesson.total_questions
                        if is_finished:
                            lesson.state = "completed"

                        await session.commit()
                        await session.refresh(lesson)

                        if is_finished:
                            final_text = (
                                f"{text}\n\n"
                                f"Урок завершён! Правильных ответов: "
                                f"{lesson.correct_count}/{lesson.total_questions}"
                            )
                            await wait_msg.edit_text(final_text)
                            await message.answer("Возвращаемся в меню.", reply_markup=main_menu_kb())
                            await state.clear()
                            return

                        await wait_msg.edit_text(text)
                        await message.answer(
                            "Ответ сохранён. Нажмите, чтобы перейти к следующему вопросу.",
                            reply_markup=next_question_kb(),
                        )
                        await state.set_state(LessonStates.WAITING_NEXT)
                        await state.update_data(lesson_id=lesson.id)
                        return

                    # Иначе это обычный quick speaking question
                    ua = UserAnswer(
                        user_id=user.id,
                        question_id=int(question_id),
                        session_id=None,
                        is_correct=bool(passed),
                        user_answer=recognized_text or expected,
                        time_spent_sec=None,
                        attempts=1,
                        emotion_score=float(emo_score) if emo_score is not None else 0.0,
                    )
                    session.add(ua)
                    await session.commit()

            except Exception as db_exc:
                logger.error("Не удалось сохранить UserAnswer для произношения: {}", db_exc)

        await wait_msg.edit_text(text)

    except Exception as e:
        logger.exception("Ошибка при оценке произношения: {}", e)
        try:
            await wait_msg.edit_text(
                "Произошла ошибка при оценке произношения. "
                "Попробуйте ещё раз чуть позже 🙏"
            )
        except Exception:
            logger.exception("Не удалось отправить сообщение об ошибке пользователю")
