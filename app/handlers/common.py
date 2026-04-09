from aiogram import Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext

from app.keyboards import main_menu_kb

common_router = Router()


@common_router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я бот для изучения английского языка. Выберите действие в меню ниже.",
        reply_markup=main_menu_kb(),
    )


@common_router.message(Command("menu"))
async def cmd_menu(message: Message):
    await message.answer("Возврат в главное меню", reply_markup=main_menu_kb())


@common_router.message(F.text == "В главное меню")
async def back_to_main_menu(message: Message, state: FSMContext):
    """Возврат в главное меню из любого состояния."""
    await state.clear()
    await message.answer("Возврат в главное меню", reply_markup=main_menu_kb())
