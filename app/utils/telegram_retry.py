# app/utils/telegram_retry.py
import asyncio
from typing import Any, Awaitable, Callable, TypeVar

from loguru import logger
from aiogram.exceptions import TelegramNetworkError

T = TypeVar("T")


async def tg_retry(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff: float = 2.0,
    **kwargs: Any,
) -> T:
    """
    Универсальный ретрайер для методов Telegram API.

    Пример:
        await tg_retry(message.answer, "Привет!")
        await tg_retry(message.answer_voice, voice=voice_file, caption="...")
    """
    delay = initial_delay

    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
        except TelegramNetworkError as e:
            if attempt == retries:
                logger.error(
                    "TelegramNetworkError on %s (attempt %s/%s), giving up: %r",
                    getattr(func, "__name__", repr(func)),
                    attempt,
                    retries,
                    e,
                )
                raise

            logger.warning(
                "TelegramNetworkError on %s (attempt %s/%s), retry in %.1fs: %r",
                getattr(func, "__name__", repr(func)),
                attempt,
                retries,
                delay,
                e,
            )
            await asyncio.sleep(delay)
            delay *= backoff
