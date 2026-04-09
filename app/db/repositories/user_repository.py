from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(self, telegram_id: int, username: str | None, first_name: str | None, last_name: str | None) -> User:
        stmt = select(User).where(User.telegram_id == telegram_id)
        result = await self.session.execute(stmt)
        user = result.scalar_one_or_none()
        if user:
            return user

        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            created_at=datetime.utcnow(),
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def update_activity(self, user: User) -> None:
        today = date.today()
        if user.last_activity_date is None or user.last_activity_date < today:
            if user.last_activity_date == today.fromordinal(today.toordinal() - 1):
                user.streak_days += 1
            else:
                user.streak_days = 1
            user.last_activity_date = today
        await self.session.commit()

    async def update_level(self, user: User, new_level: str) -> None:
        user.level = new_level
        await self.session.commit()

    async def get_by_id(self, user_id: int) -> User | None:
        """Получить пользователя по внутреннему ID."""
        stmt = select(User).where(User.id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
