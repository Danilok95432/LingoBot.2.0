from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import List

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ml.question_generator import QuestionGenerator, GeneratedQuestion
from app.core.services.difficulty_regression_service import DifficultyRegressionService
from app.db.repositories import QuestionRepository, LessonRepository, UserRepository


CEFR_LEVELS: list[str] = ["A1", "A2", "B1", "B2", "C1", "C2"]


class LevelTestMode(str, Enum):
    QUIZ = "quiz"
    TEXT = "text"


@dataclass
class LevelTestQuestion:
    index: int
    level: str
    type: str
    text: str
    options: list[str]
    correct_index: int


class TestService:
    """
    Сервис определения уровня английского.

    Две подсистемы:

    * QUIZ — короткий тест с вариантами ответов по всем CEFR-уровням;
    * TEXT — определение уровня по произвольному тексту пользователя.

    Для уровня мы используем свои вопросы (через QuestionGenerator),
    но сами вопросы для теста уровня в БД не сохраняем — они живут
    только в FSM.
    """

    def __init__(
        self,
        session: AsyncSession,
        diff_regressor: DifficultyRegressionService,
        question_generator: QuestionGenerator,
    ) -> None:
        self.session = session
        self.diff_regressor = diff_regressor
        self.question_generator = question_generator

        # Оставляем репозитории для будущего расширения и обновления пользователя
        self.question_repo = QuestionRepository(session)
        self.lesson_repo = LessonRepository(session)
        self.user_repo = UserRepository(session)

    # ------------------------------------------------------------------
    # QUIZ-тестирование
    # ------------------------------------------------------------------

    async def build_level_quiz(
        self,
        approx_level: str | None,
        questions_per_level: int = 2,
    ) -> list[LevelTestQuestion]:
        """
        Собрать список вопросов для теста уровня.

        Для каждого CEFR-уровня генерируем несколько вопросов
        (грамматика / лексика). Вопросы НЕ записываются в БД.
        """
        levels = CEFR_LEVELS
        questions: list[LevelTestQuestion] = []

        # Постараемся начать ближе к текущему уровню пользователя,
        # но всё равно включим задачи со всех уровней.
        if approx_level in levels:
            start_idx = levels.index(approx_level)
        else:
            start_idx = 1  # A2 по умолчанию

        ordered_levels = levels[start_idx:] + levels[:start_idx]

        idx = 0
        for level in ordered_levels:
            for _ in range(questions_per_level):
                qtype = random.choice(["grammar", "vocabulary"])
                gen: GeneratedQuestion = self.question_generator._generate_single(qtype, level)
                questions.append(
                    LevelTestQuestion(
                        index=idx,
                        level=gen.level,
                        type=gen.type,
                        text=gen.text,
                        options=list(gen.options),
                        correct_index=gen.correct_index,
                    )
                )
                idx += 1

        random.shuffle(questions)
        logger.info(
            "Built level quiz: approx_level={}, total_questions={}",
            approx_level,
            len(questions),
        )
        return questions

    def estimate_level_from_quiz(
        self,
        questions: list[LevelTestQuestion],
        answers_correct: list[bool],
    ) -> str:
        """
        Оценить уровень по результатам квиза.

        Для каждого уровня считаем долю верных ответов и берём
        наивысший уровень, где доля >= 0.6. Если ничего не набрано —
        возвращаем A1.
        """
        if not questions or not answers_correct:
            return "A1"

        per_level_total: dict[str, int] = {lvl: 0 for lvl in CEFR_LEVELS}
        per_level_correct: dict[str, int] = {lvl: 0 for lvl in CEFR_LEVELS}

        for q, is_ok in zip(questions, answers_correct):
            per_level_total[q.level] += 1
            if is_ok:
                per_level_correct[q.level] += 1

        best_level = "A1"
        for level in CEFR_LEVELS:
            total = per_level_total[level]
            if total == 0:
                continue
            ratio = per_level_correct[level] / total
            logger.debug(
                "Level quiz stats: level={} total={} correct={} ratio={}",
                level,
                total,
                per_level_correct[level],
                ratio,
            )
            if ratio >= 0.6:
                best_level = level

        return best_level

    # ------------------------------------------------------------------
    # Анализ произвольного текста
    # ------------------------------------------------------------------

    async def analyze_text_level(self, text: str) -> str:
        """
        Определение уровня по тексту пользователя.

        Используем лёгкую эвристическую «модель», которая смотрит на:
        * длину текста;
        * длину предложений;
        * разнообразие слов;
        * долю длинных слов;
        * наличие продвинутых грамматических конструкций и связок.
        """
        level = self._heuristic_text_level(text)
        logger.info("Text level estimated as {}", level)
        return level

    def _heuristic_text_level(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "A1"

        # Простейшая токенизация
        tokens = [t for t in re.split(r"[^a-zA-Z']+", cleaned.lower()) if t]
        if not tokens:
            return "A1"

        n_tokens = len(tokens)
        unique_tokens = len(set(tokens))

        # Предложения
        sentences = [s for s in re.split(r"[.!?]+", cleaned) if s.strip()]
        n_sent = max(len(sentences), 1)
        avg_sent_len = n_tokens / n_sent

        # Доля длинных слов (>= 8 букв)
        long_words = [t for t in tokens if len(t) >= 8]
        long_ratio = len(long_words) / n_tokens

        # Разнообразие слов
        type_token_ratio = unique_tokens / n_tokens

        text_lower = cleaned.lower()

        advanced_linkers = [
            "however",
            "moreover",
            "nevertheless",
            "furthermore",
            "consequently",
            "whereas",
            "despite",
            "although",
            "in spite of",
            "on the other hand",
        ]
        advanced_grammar = [
            "have been",
            "had been",
            "would have",
            "could have",
            "might have",
            "should have",
            "if i had",
            "if i were",
        ]

        linker_hits = sum(1 for w in advanced_linkers if w in text_lower)
        grammar_hits = sum(1 for w in advanced_grammar if w in text_lower)

        score = 0.0

        # Балл за длину текста
        if n_tokens < 30:
            score += 0
        elif n_tokens < 80:
            score += 1
        elif n_tokens < 150:
            score += 2
        else:
            score += 3

        # Средняя длина предложений
        if avg_sent_len <= 7:
            score += 0
        elif avg_sent_len <= 12:
            score += 1
        elif avg_sent_len <= 18:
            score += 2
        else:
            score += 3

        # Разнообразие слов
        if type_token_ratio <= 0.4:
            score += 0
        elif type_token_ratio <= 0.55:
            score += 1
        elif type_token_ratio <= 0.7:
            score += 2
        else:
            score += 3

        # Длинные слова
        if long_ratio >= 0.25:
            score += 2
        elif long_ratio >= 0.15:
            score += 1

        # Продвинутые связки и грамматика
        score += min(linker_hits, 3) * 0.7
        score += min(grammar_hits, 3) * 0.7

        logger.debug(
            "Text features: tokens=%s, uniq=%s, avg_sent_len=%.1f, "
            "long_ratio=%.2f, TTR=%.2f, linkers=%s, grammar=%s, score=%.2f",
            n_tokens,
            unique_tokens,
            avg_sent_len,
            long_ratio,
            type_token_ratio,
            linker_hits,
            grammar_hits,
            score,
        )

        # Маппинг score -> CEFR
        if score < 2.0:
            return "A1"
        if score < 4.0:
            return "A2"
        if score < 6.0:
            return "B1"
        if score < 8.0:
            return "B2"
        if score < 10.0:
            return "C1"
        return "C2"
