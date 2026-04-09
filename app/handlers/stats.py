from aiogram import Router, F
from aiogram.types import Message

from app.core.services import StatsService
from app.db.session import async_session_maker
from app.keyboards import main_menu_kb

stats_router = Router()


@stats_router.message(F.text == "📊 Статистика")
async def show_stats(message: Message, user):
    async with async_session_maker() as session:
        service = StatsService(session)
        text = await service.get_stats_text(user.id)
    await message.answer(text, reply_markup=main_menu_kb())
