from typing import Callable, Awaitable, Dict, Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject

from app.db.session import async_session_maker
from app.db.repositories import UserRepository


class UserContextMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        """Подкладываем объект пользователя из БД в kwargs хендлеров как `user`.

        Используем `event_from_user`, который уже подготовлен встроенным
        UserContextMiddleware из aiogram, поэтому работает и для сообщений,
        и для callback_query, и для других апдейтов от пользователя.
        """
        # aiogram уже положил сюда Telegram User
        from_user = data.get("event_from_user")

        user = None
        if from_user is not None:
            async with async_session_maker() as session:
                repo = UserRepository(session)
                user = await repo.get_or_create(
                    telegram_id=from_user.id,
                    username=from_user.username,
                    first_name=from_user.first_name,
                    last_name=from_user.last_name,
                )
                await repo.update_activity(user)

        # Делаем `user` доступным в хендлерах по имени аргумента
        data["user"] = user
        return await handler(event, data)
