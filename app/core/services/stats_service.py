from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import StatsRepository


class StatsService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = StatsRepository(session)

    async def get_stats_text(self, user_id: int) -> str:
        stats = await self.repo.get_user_stats(user_id)
        return (
            f"Ваш уровень: {stats['level']}\n"
            f"Streak по дням: {stats['streak_days']}\n"
            f"Пройдено уроков: {stats['total_lessons']}\n"
            f"Всего ответов: {stats['total_answers']}\n"
            f"Правильных ответов: {stats['correct_answers']}\n"
            f"Точность: {stats['accuracy_percent']}%"
        )
