from loguru import logger


class TTSEngine:
    """Интерфейс для генерации аудио для заданий на аудирование.

    Реализуйте интеграцию с любым TTS (Coqui, Silero, встроенный движок и т.п.)
    и сохранение файла, верните путь/URL к аудио.
    """

    async def synthesize_and_store(self, text: str) -> str | None:
        logger.info(f"TTS placeholder synthesize: '{text}'")
        # Вернуть путь к файлу или URL после интеграции
        return None
