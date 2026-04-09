from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Question, UserAnswer, LessonSession
from app.db.repositories import QuestionRepository, LessonRepository
from app.core.services.difficulty_regression_service import DifficultyRegressionService
from app.core.ml.question_generator import QuestionGenerator


class LessonType(str, Enum):
    LESSON_TEST = "lesson_test"
    AUDIO_LESSON = "audio_lesson"


@dataclass
class LessonQuestionDTO:
    question: Question
    index: int
    total: int


class LessonService:
    def __init__(
        self,
        session: AsyncSession,
        diff_regressor: DifficultyRegressionService,
        question_generator: QuestionGenerator,
    ) -> None:
        self.session = session
        self.question_repo = QuestionRepository(session)
        self.lesson_repo = LessonRepository(session)
        self.diff_regressor = diff_regressor
        self.question_generator = question_generator

    async def start_lesson(
        self,
        user_id: int,
        level: str,
        lesson_type: LessonType,
        last_message_id: int | None = None,
    ) -> LessonSession:
        """
        Создаём новую сессию урока:
        - считаем рекомендуемую сложность
        - добиваем пул вопросов до нужного количества (batched generate_and_store_bulk)
        - создаём LessonSession
        """
        # 1) предсказали сложность для этого юзера и уровня
        difficulty = await self.diff_regressor.predict_level_difficulty(
            user_id=user_id,
            level=level,
        )

        # 2) какие типы вопросов хотим в уроке
        if lesson_type == LessonType.LESSON_TEST:
            qtypes = ["grammar", "vocabulary"]
            total_questions = 10
        elif lesson_type == LessonType.AUDIO_LESSON:
            qtypes = ["listening", "speaking"]
            total_questions = 6
        else:
            qtypes = ["grammar", "vocabulary"]
            total_questions = 8

        # 3) добиваем пул вопросов в БД батчами
        await self.question_generator.generate_and_store_bulk(
            session=self.session,
            qtypes=qtypes,
            level=level,
            n=total_questions,
        )

        # 4) выбираем конкретные вопросы для урока
        questions = await self.question_repo.get_random_questions_for_lesson(
            user_id=user_id,
            qtypes=qtypes,
            level=level,
            limit=total_questions,
        )
        question_ids = [q.id for q in questions]

        # 5) создаём саму сессию урока и сохраняем difficulty в meta
        lesson = await self.lesson_repo.create_session(
            user_id=user_id,
            session_type=lesson_type.value,
            question_ids=question_ids,
            last_message_id=last_message_id,
            meta={"difficulty": difficulty},  # <-- вот так, без аргумента difficulty=
        )

        return lesson

    async def get_current_question(self, lesson: LessonSession) -> LessonQuestionDTO | None:
        if lesson.current_index >= lesson.total_questions:
            return None
        q_id = lesson.question_ids[lesson.current_index]
        question = await self.session.get(Question, q_id)
        return LessonQuestionDTO(
            question=question,
            index=lesson.current_index + 1,
            total=lesson.total_questions,
        )

    async def answer_question(
        self,
        lesson: LessonSession,
        user_id: int,
        answer_index: int,
        emotion_score: float = 0.0,
    ) -> tuple[bool, LessonSession]:
        q_id = lesson.question_ids[lesson.current_index]
        question: Question = await self.session.get(Question, q_id)

        is_correct = answer_index == question.correct_option_index

        ua = UserAnswer(
            user_id=user_id,
            question_id=q_id,
            session_id=lesson.id,
            is_correct=is_correct,
            user_answer=str(answer_index),
            emotion_score=emotion_score,
            created_at=datetime.utcnow(),
        )
        self.session.add(ua)

        if is_correct:
            lesson.correct_count += 1
        lesson.current_index += 1

        if lesson.current_index >= lesson.total_questions:
            lesson.state = "completed"

        await self.session.commit()
        await self.session.refresh(lesson)
        return is_correct, lesson
