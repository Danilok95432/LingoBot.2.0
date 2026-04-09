# app/core/services/pronunciation_service.py
from __future__ import annotations

from typing import Any, Optional, Dict, Tuple

from aiogram import Bot
from aiogram.types import Voice
from loguru import logger

from app.core.ml.pronunciation_evaluator import PronunciationEvaluator
from app.core.ml.emotion_analyzer import EmotionAnalyzer


class PronunciationService:
    """
    Сервис комплексной оценки голосового ответа:
    - качество произношения (через PronunciationEvaluator / Whisper)
    - эмоция говорящего (через EmotionAnalyzer, по аудио)
    """

    def __init__(
        self,
        evaluator: Optional[PronunciationEvaluator] = None,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
    ) -> None:
        # Whisper-оценщик произношения
        self.evaluator = evaluator or PronunciationEvaluator()
        # Модель эмоций (может быть None, тогда просто не считаем эмоции)
        self.emotion_analyzer = emotion_analyzer

    # -------------------------------------------------
    # Вспомогательное: скачать голосовое из Telegram
    # -------------------------------------------------
    async def _download_voice_bytes(self, bot: Bot, voice: Voice) -> bytes:
        """
        Скачиваем голосовое сообщение и возвращаем «сырые» байты файла.

        Для aiogram 3 достаточно `bot.download(voice)` – он вернёт BytesIO.
        """
        file_like = await bot.download(voice)
        audio_bytes = file_like.read()
        logger.debug("PronunciationService: downloaded %d bytes of audio", len(audio_bytes))
        return audio_bytes

    # -------------------------------------------------
    # Основной публичный метод
    # -------------------------------------------------
    async def evaluate(
        self,
        bot: Bot,
        voice: Voice,
        expected_text: str,
        user_level: str,
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        Возвращает:
            (feedback, emotion_info)

        feedback — это твой PronunciationFeedback:
            .score
            .threshold
            .passed
            .expected_text
            .recognized_text
            .global_comment
            .tips
            .words
            ...

        emotion_info — dict вида:
            {"label": str | None, "score": float | None, "probs": dict | None}
        """
        # 1) Скачиваем аудио
        audio_bytes = await self._download_voice_bytes(bot, voice)

        # 2) Получаем подробный фидбэк по произношению
        feedback = await self.evaluator.evaluate_with_feedback(
            audio_bytes=audio_bytes,
            expected_text=expected_text,
            level=user_level,
        )

        # 3) Эмоции по голосу
        emotion_info: Optional[Dict[str, Any]] = None
        if self.emotion_analyzer is not None:
            try:
                emotion_info = await self.emotion_analyzer.analyze_audio(audio_bytes)
                logger.debug("PronunciationService: emotion_info={}", emotion_info)
            except Exception as e:
                logger.error("PronunciationService: ошибка при анализе эмоций: {}", e)

        return feedback, emotion_info
