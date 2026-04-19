from datetime import datetime, timezone
from collections.abc import Awaitable, Callable

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Question, UserAnswer
from app.db.repositories import QuestionRepository
from app.core.ml.question_generator import QuestionGenerator


class QuickQuestionService:
    def __init__(self, session: AsyncSession, question_generator: QuestionGenerator):
        self.session = session
        self.question_repo = QuestionRepository(session)
        self.question_generator = question_generator

    async def get_or_generate_question(
        self,
        user_id: int,
        level: str,
        qtype: str,
        on_generation_start: Callable[[], Awaitable[None]] | None = None,
    ) -> Question | None:
        """
        1. Пытаемся взять свежий вопрос из БД, который пользователь ещё не решал.
        2. Если нет — генерируем пачку через LLM и кладём в БД.
        3. Потом снова пробуем взять вопрос.
        """

        question = await self.question_repo.get_random_fresh_question(
            user_id=user_id,
            level=level,
            qtype=qtype,
        )
        if question:
            return question

        if on_generation_start is not None:
            await on_generation_start()

        batch_size = 10
        logger.info(
            "QuickQuestion: generating {} questions for user {}, type={}, level={}",
            batch_size,
            user_id,
            qtype,
            level,
        )

        await self.question_generator.generate_and_store_bulk(
            session=self.session,
            qtypes=[qtype],
            level=level,
            n=batch_size,
        )

        question = await self.question_repo.get_random_fresh_question(
            user_id=user_id,
            level=level,
            qtype=qtype,
        )
        if question:
            return question

        logger.warning(
            "QuickQuestion: no questions even after generation, returning None (type={}, level={})",
            qtype,
            level,
        )
        return None

    async def save_answer(
        self,
        user_id: int,
        question_id: int,
        is_correct: bool,
        user_answer: str | None = None,
        time_spent_sec: float | None = None,
        emotion_score: float = 0.0,
    ) -> UserAnswer:
        ua = UserAnswer(
            user_id=user_id,
            question_id=question_id,
            session_id=None,
            is_correct=is_correct,
            user_answer=user_answer,
            time_spent_sec=time_spent_sec,
            attempts=1,
            emotion_score=emotion_score,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(ua)
        await self.session.commit()
        await self.session.refresh(ua)
        return ua