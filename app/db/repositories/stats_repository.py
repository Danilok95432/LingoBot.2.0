from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User, LessonSession, UserAnswer


class StatsRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_user_stats(self, user_id: int) -> dict:
        user_stmt = select(User).where(User.id == user_id)
        user_res = await self.session.execute(user_stmt)
        user = user_res.scalar_one()

        lessons_stmt = select(func.count(LessonSession.id)).where(
            LessonSession.user_id == user_id,
            LessonSession.state == "completed",
        )
        lessons_count = (await self.session.execute(lessons_stmt)).scalar_one()

        answers_stmt = select(
            func.count(UserAnswer.id),
            func.sum(
                case((UserAnswer.is_correct, 1), else_=0)
            ),
        ).where(UserAnswer.user_id == user_id)
        total_answers, correct_answers = (await self.session.execute(answers_stmt)).one()
        total_answers = total_answers or 0
        correct_answers = correct_answers or 0
        accuracy = (correct_answers / total_answers * 100) if total_answers else 0.0

        return {
            "level": user.level,
            "streak_days": user.streak_days,
            "total_lessons": lessons_count,
            "total_answers": total_answers,
            "correct_answers": correct_answers,
            "accuracy_percent": round(accuracy, 2),
        }
