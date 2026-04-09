# app/core/services/quick_question_service.py

from datetime import datetime, timezone
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Question, UserAnswer
from app.db.repositories import QuestionRepository
from app.core.ml.question_generator import QuestionGenerator


class QuickQuestionService:
    def __init__(self, session: AsyncSession, question_generator: QuestionGenerator):
        self.session = session
        self.question_repo = QuestionRepository(session)
        # 👇 ВАЖНО: храним генератор тут
        self.question_generator = question_generator

    async def get_or_generate_question(self, user_id: int, level: str, qtype: str) -> Question:
        """
        1. Пытаемся взять свежий вопрос из БД, который пользователь ещё не решал.
        2. Если нет — генерируем пачку через LLM (Ollama) и кладём в БД.
        3. Потом снова пробуем взять вопрос.
        """

        # 1) Пробуем достать уже готовый вопрос
        question = await self.question_repo.get_random_fresh_question(
            user_id=user_id,
            level=level,
            qtype=qtype,
        )
        if question:
            return question

        # 2) Если нет — генерируем пачку (например, 10 штук)
        batch_size = 10
        logger.info(
            "QuickQuestion: generating %s questions for user %s, type=%s, level=%s",
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

        # 3) Ещё раз пробуем достать вопрос
        question = await self.question_repo.get_random_fresh_question(
            user_id=user_id,
            level=level,
            qtype=qtype,
        )
        if question:
            return question

        # 4) Совсем fallback — вернём None, а выше обработаем (или можно сделать встроенный шаблон)
        logger.warning(
            "QuickQuestion: no questions even after generation, returning None (type=%s, level=%s)",
            qtype,
            level,
        )
        return None
