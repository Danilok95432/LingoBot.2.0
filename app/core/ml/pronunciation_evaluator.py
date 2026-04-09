import os
import random
import difflib
import re
import unicodedata
import tempfile
from dataclasses import dataclass
from typing import List, Literal, Optional

from loguru import logger

try:
    import whisper  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    whisper = None  # type: ignore


WordStatus = Literal["ok", "mispronounced", "missing", "extra"]


@dataclass
class WordFeedback:
    """Информация о произношении/распознавании отдельного слова."""
    expected: Optional[str]  # слово, которое должно было быть
    actual: Optional[str]    # слово, которое реально распозналось
    similarity: float        # похожесть [0,1]
    status: WordStatus       # ok / mispronounced / missing / extra


@dataclass
class PronunciationFeedback:
    """Полный результат оценки произношения."""
    score: float                    # общий скор [0,1]
    threshold: float                # порог для уровня пользователя
    passed: bool                    # score >= threshold
    expected_text: str              # исходная фраза
    recognized_text: str            # распознанная Whisper'ом фраза
    expected_norm: str              # нормализованный expected_text
    recognized_norm: str            # нормализованный recognized_text
    words: List[WordFeedback]       # поминутная/поштучная инфа по словам
    global_comment: str             # общий вердикт ("Отлично", "Нужно подтянуть ..." и т.п.)
    tips: List[str]                 # список конкретных советов по улучшению


class PronunciationEvaluator:
    """Интерфейс для оценки произношения с реальным ASR (Whisper).

    Архитектура:
    1) transcribe(audio_bytes) -> recognized_text
       - локальная модель Whisper (openai-whisper) обрабатывает файл.
    2) evaluate / evaluate_with_feedback(audio_bytes, expected_text, level)
       - нормализует тексты, считает похожесть и даёт подробный фидбэк.
    3) threshold_for_level(level) -> float
       - порог по уровню языка.
    """

    LEVEL_THRESHOLDS = {
        "A1": 0.4,
        "A2": 0.5,
        "B1": 0.6,
        "B2": 0.7,
        "C1": 0.8,
        "C2": 0.9,
    }

    _whisper_model = None  # кэш модели Whisper

    # ---------- Публичный API ----------

    async def evaluate_placeholder(self, expected_text: str) -> float:
        """Старая заглушка: случайный скор (оставляем для тестов)."""
        return random.uniform(0.4, 0.95)

    def threshold_for_level(self, level: str) -> float:
        """Вернуть порог для уровня пользователя."""
        return self.LEVEL_THRESHOLDS.get(level, 0.6)

    async def evaluate(self, audio_bytes: bytes, expected_text: str, level: str = "B1") -> float:
        """Простая версия: вернуть только числовой скор.

        Для совместимости с уже существующим кодом. Внутри использует
        более продвинутый evaluate_with_feedback.
        """
        feedback = await self.evaluate_with_feedback(audio_bytes, expected_text, level)
        return feedback.score

    async def evaluate_with_feedback(
        self,
        audio_bytes: bytes,
        expected_text: str,
        level: str,
    ) -> PronunciationFeedback:
        """Расширенная версия оценки с детальным фидбэком.

        :param audio_bytes: байты голосового сообщения (ogg/opus от Telegram)
        :param expected_text: фраза, которую пользователь должен произнести
        :param level: CEFR-уровень пользователя (A1–C2)
        :return: PronunciationFeedback
        """
        recognized = await self.transcribe(audio_bytes)
        logger.debug(
            "PronunciationEvaluator: expected='{}', recognized='{}'",
            expected_text,
            recognized,
        )

        threshold = self.threshold_for_level(level)

        if not recognized:
            # Ничего не распознали — сразу очень низкий скор и базовый фидбэк
            expected_norm = self._normalize_text(expected_text)
            return PronunciationFeedback(
                score=0.0,
                threshold=threshold,
                passed=False,
                expected_text=expected_text,
                recognized_text="",
                expected_norm=expected_norm,
                recognized_norm="",
                words=[],
                global_comment="Я почти ничего не смог распознать. "
                               "Попробуйте говорить чуть чётче и ближе к микрофону.",
                tips=[
                    "Говорите немного медленнее.",
                    "Убедитесь, что вокруг не слишком шумно.",
                    "Держите телефон ближе к рту."
                ],
            )

        # Общий скор по нормализованным строкам
        expected_norm = self._normalize_text(expected_text)
        recognized_norm = self._normalize_text(recognized)
        score = self._similarity(expected_norm, recognized_norm)
        passed = score >= threshold

        # Анализ по словам
        words_feedback = self._word_level_feedback(expected_norm, recognized_norm)

        # Текстовый вердикт + советы
        global_comment = self._build_global_comment(score, threshold, level)
        tips = self._build_tips(words_feedback, level)

        return PronunciationFeedback(
            score=score,
            threshold=threshold,
            passed=passed,
            expected_text=expected_text,
            recognized_text=recognized,
            expected_norm=expected_norm,
            recognized_norm=recognized_norm,
            words=words_feedback,
            global_comment=global_comment,
            tips=tips,
        )

    # ---------- ASR (Whisper) ----------

    def _get_whisper_model(self):
        """Ленивая загрузка модели Whisper.

        Модель задаётся переменной окружения WHISPER_MODEL (tiny/base/small/medium/large),
        по умолчанию 'base'.
        """
        if whisper is None:
            logger.error(
                "Модуль 'whisper' не установлен. "
                "Добавьте 'openai-whisper' в requirements.txt."
            )
            return None

        if self.__class__._whisper_model is None:
            model_name = os.getenv("WHISPER_MODEL", "base")
            logger.info("Загрузка модели Whisper: {}", model_name)
            try:
                self.__class__._whisper_model = whisper.load_model(model_name)
            except Exception as exc:  # pragma: no cover - защита от падения
                logger.error("Не удалось загрузить модель Whisper: {}", exc)
                self.__class__._whisper_model = None
        return self.__class__._whisper_model

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Распознать английскую речь из байтов OGG/OPUS с помощью Whisper."""
        model = self._get_whisper_model()
        if model is None:
            return ""

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            result = model.transcribe(tmp_path, language="en")
            text = (result.get("text") or "").strip()
            logger.debug("Whisper raw text: {}", text)
            return text
        except Exception as exc:  # pragma: no cover
            logger.error("Ошибка при распознавании Whisper: {}", exc)
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ---------- Внутренние утилиты ----------

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = unicodedata.normalize("NFKD", text)
        # Оставляем только латиницу и апостроф
        text = re.sub(r"[^a-z']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _similarity(self, expected: str, actual: str) -> float:
        """Метрика похожести по строкам [0,1]."""
        if not expected or not actual:
            return 0.0
        matcher = difflib.SequenceMatcher(a=expected, b=actual)
        return matcher.ratio()

    def _word_level_feedback(
        self,
        expected_norm: str,
        recognized_norm: str,
    ) -> List[WordFeedback]:
        """Грубый анализ по словам через alignment.

        Используем SequenceMatcher по спискам слов и помечаем:
        - ok: слово совпало
        - mispronounced: слово на его месте другое (низкая похожесть)
        - missing: слово не прозвучало
        - extra: лишнее слово у пользователя
        """
        expected_words = expected_norm.split()
        actual_words = recognized_norm.split()

        matcher = difflib.SequenceMatcher(a=expected_words, b=actual_words)
        feedback: List[WordFeedback] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for ew, aw in zip(expected_words[i1:i2], actual_words[j1:j2]):
                    feedback.append(
                        WordFeedback(
                            expected=ew,
                            actual=aw,
                            similarity=1.0,
                            status="ok",
                        )
                    )
            elif tag == "replace":
                # ожидаемые слова заменены другими
                e_chunk = expected_words[i1:i2]
                a_chunk = actual_words[j1:j2]
                for idx in range(max(len(e_chunk), len(a_chunk))):
                    ew = e_chunk[idx] if idx < len(e_chunk) else None
                    aw = a_chunk[idx] if idx < len(a_chunk) else None
                    if ew is None:
                        # лишнее слово
                        feedback.append(
                            WordFeedback(
                                expected=None,
                                actual=aw,
                                similarity=0.0,
                                status="extra",
                            )
                        )
                    elif aw is None:
                        # пропущенное слово
                        feedback.append(
                            WordFeedback(
                                expected=ew,
                                actual=None,
                                similarity=0.0,
                                status="missing",
                            )
                        )
                    else:
                        sim = self._similarity(ew, aw)
                        status: WordStatus = "ok" if sim >= 0.8 else "mispronounced"
                        feedback.append(
                            WordFeedback(
                                expected=ew,
                                actual=aw,
                                similarity=sim,
                                status=status,
                            )
                        )
            elif tag == "delete":
                # ожидаемые слова отсутствуют в распознанном
                for ew in expected_words[i1:i2]:
                    feedback.append(
                        WordFeedback(
                            expected=ew,
                            actual=None,
                            similarity=0.0,
                            status="missing",
                        )
                    )
            elif tag == "insert":
                # лишние слова в распознанном
                for aw in actual_words[j1:j2]:
                    feedback.append(
                        WordFeedback(
                            expected=None,
                            actual=aw,
                            similarity=0.0,
                            status="extra",
                        )
                    )

        return feedback

    def _build_global_comment(self, score: float, threshold: float, level: str) -> str:
        """Сгенерировать краткий вердикт по произношению."""
        # Небольшой буфер вокруг порога
        margin = 0.05

        if score >= threshold + margin:
            return (
                f"Отличное произношение для уровня {level}! "
                f"Общий скор: {score:.2f} (порог: {threshold:.2f})."
            )
        elif score >= threshold:
            return (
                f"В целом хорошо для уровня {level}, но есть что улучшить. "
                f"Общий скор: {score:.2f} (порог: {threshold:.2f})."
            )
        elif score >= threshold - 0.1:
            return (
                f"Немного не дотягивает до ожидаемого уровня {level}. "
                f"Общий скор: {score:.2f} (порог: {threshold:.2f}). "
                "Обратите внимание на выделенные слова и попробуйте ещё раз."
            )
        else:
            return (
                f"Пока далеко от ожидаемого уровня {level}. "
                f"Общий скор: {score:.2f} (порог: {threshold:.2f}). "
                "Советую проговаривать фразу медленнее и чётче, "
                "особенно проблемные слова."
            )

    def _build_tips(self, words_feedback: List[WordFeedback], level: str) -> List[str]:
        """Сгенерировать список конкретных советов по словам."""
        tips: List[str] = []

        mispronounced = [
            wf for wf in words_feedback
            if wf.status == "mispronounced" and wf.expected
        ]
        missing = [wf for wf in words_feedback if wf.status == "missing" and wf.expected]

        # 1) советы по конкретным словам
        if mispronounced:
            bad_words = {wf.expected for wf in mispronounced if wf.expected}
            tips.append(
                "Обратите особое внимание на произношение слов: "
                + ", ".join(sorted(bad_words))
            )

        if missing:
            missing_words = {wf.expected for wf in missing if wf.expected}
            tips.append(
                "Некоторые слова вы, кажется, пропустили: "
                + ", ".join(sorted(missing_words))
            )

        # 2) общие советы по уровню
        if level in {"A1", "A2"}:
            tips.append("Пробуйте говорить медленнее и делать небольшие паузы между словами.")
        elif level in {"B1", "B2"}:
            tips.append("Следите за окончаниями слов и связкой слов в потоке речи.")
        else:  # C1 / C2
            tips.append(
                "На продвинутом уровне попробуйте поработать над интонацией и ритмом, "
                "ближе к естественной речи носителей."
            )

        # Если совсем тихо было — уже учли в global_comment, но можно дублировать:
        if not tips:
            tips.append("В целом всё неплохо, но попробуйте повторить фразу ещё раз для закрепления.")

        return tips
