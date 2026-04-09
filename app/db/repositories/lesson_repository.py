from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import LessonSession


class LessonRepository:
    """Хранилище для сессий уроков."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_session(
        self,
        user_id: int,
        session_type: str,
        question_ids: list[int],
        last_message_id: int | None = None,
        meta: dict | None = None,
    ) -> LessonSession:
        """Создать новую сессию урока для пользователя.

        total_questions берём из длины списка question_ids.
        """
        session_obj = LessonSession(
            user_id=user_id,
            session_type=session_type,
            question_ids=question_ids,
            total_questions=len(question_ids),
            current_index=0,
            correct_count=0,
            state="in_progress",
            started_at=datetime.utcnow(),
            last_message_id=last_message_id,
            meta=meta or {},
        )
        self.session.add(session_obj)
        await self.session.commit()
        await self.session.refresh(session_obj)
        return session_obj

    async def get_active_session(self, user_id: int) -> LessonSession | None:
        """Вернуть активную (in_progress) сессию пользователя.

        Если по какой-то причине активных несколько, берём самую позднюю.
        """
        stmt = (
            select(LessonSession)
            .where(
                LessonSession.user_id == user_id,
                LessonSession.state == "in_progress",
            )
            .order_by(LessonSession.started_at.desc())
        )
        result = await self.session.execute(stmt)
        # Берём первую строку без ошибок MultipleResultsFound
        return result.scalars().first()

    async def get_session(self, lesson_id: int) -> LessonSession | None:
        """Получить сессию по её id."""
        return await self.session.get(LessonSession, lesson_id)

    async def update_session(
        self,
        lesson_id: int,
        current_index: int | None = None,
        correct_count: int | None = None,
        last_message_id: int | None = None,
        state: str | None = None,
    ) -> None:
        """Обновить поля сессии урока.

        Если state меняется на "completed", выставляем finished_at.
        """
        session_obj = await self.session.get(LessonSession, lesson_id)
        if session_obj is None:
            return

        if current_index is not None:
            session_obj.current_index = current_index
        if correct_count is not None:
            session_obj.correct_count = correct_count
        if last_message_id is not None:
            session_obj.last_message_id = last_message_id
        if state is not None:
            session_obj.state = state
            if state == "completed":
                session_obj.finished_at = datetime.utcnow()

        await self.session.commit()
