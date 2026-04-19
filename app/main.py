import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramNetworkError
from aiogram.fsm.storage.memory import MemoryStorage
from loguru import logger

from app.config import get_settings
from app.handlers import (
    common_router,
    lessons_router,
    quick_router,
    stats_router,
    tests_router,
    pronunciation_router,
)
from app.logging_config import setup_logging
from app.middlewares import UserContextMiddleware

async def run_polling_with_retry(bot: Bot, dp: Dispatcher):
    retries = 0
    while True:
        try:
            logger.info("Starting polling...")
            await dp.start_polling(bot)
            break  # если polling завершился нормально — выходим из цикла
        except TelegramNetworkError as e:
            logger.warning(f"Failed to fetch updates - {e}")
            retries += 1
            wait_time = min(60, 2 ** retries)
            logger.warning(
                "Sleep for %.1f seconds and try again... (tryings = %s, bot id = %s)",
                wait_time,
                retries,
                bot.id,
            )
            await asyncio.sleep(wait_time)
        except Exception as e:
            logger.exception(f"Unexpected error during polling: {e}")
            await asyncio.sleep(5)

async def main():
    settings = get_settings()

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=MemoryStorage())

    dp.update.middleware(UserContextMiddleware())

    dp.include_router(common_router)
    dp.include_router(lessons_router)
    dp.include_router(quick_router)
    dp.include_router(stats_router)
    dp.include_router(tests_router)
    dp.include_router(pronunciation_router)

    logger.info("Starting bot")
    await bot.delete_webhook(drop_pending_updates=True)

    await run_polling_with_retry(bot, dp)


if __name__ == "__main__":
    asyncio.run(main())
