from __future__ import annotations

from typing import Sequence

import numpy as np
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.linear_model import LogisticRegression

from app.db.models import UserAnswer, Question


# Диапазон допустимых сложностей для каждого уровня
LEVEL_BOUNDS: dict[str, tuple[float, float]] = {
    "A1": (0.15, 0.35),
    "A2": (0.25, 0.45),
    "B1": (0.35, 0.55),
    "B2": (0.45, 0.70),
    "C1": (0.60, 0.85),
    "C2": (0.70, 0.95),
}


class DifficultyRegressionService:
    """
    Сервис подбора сложности на основе истории ответов.

    ML-часть:
    - Фичи: difficulty вопроса (одно число)
    - Цель: is_correct (0/1)
    - Модель: LogisticRegression (scikit-learn)
    - Цель оптимизации: подобрать difficulty так, чтобы P(correct) ≈ target_p (0.7)
    """

    def __init__(
        self,
        session: AsyncSession,
        max_history: int = 300,
        min_samples: int = 20,
        target_p: float = 0.7,
    ) -> None:
        self.session = session
        self.max_history = max_history
        self.min_samples = min_samples
        self.target_p = target_p

    # ---------------------- Публичный метод ---------------------------

    async def predict_level_difficulty(
        self,
        user_id: int,
        level: str,
        base: float | None = None,
    ) -> float:
        """
        Предсказать оптимальную сложность для пользователя и уровня.

        :param user_id: id пользователя в БД (НЕ telegram_id)
        :param level: уровень (A1..C2)
        :param base: базовая сложность (если None — берём середину диапазона уровня)
        """
        if base is None:
            base = self._default_base(level)

        logger.debug(
            "DifficultyRegressionService (sklearn): user=%s, level=%s, base=%.3f",
            user_id,
            level,
            base,
        )

        xs, ys, weights = await self._load_history(user_id, level)

        if not xs:
            logger.debug("Нет истории для user=%s, level=%s → base=%.3f", user_id, level, base)
            return base

        acc = self._weighted_accuracy(ys, weights)

        # если данных мало — используем эвристику
        if len(xs) < self.min_samples:
            adjusted = self._heuristic_adjust(base, level, acc)
            logger.debug(
                "Мало данных (n=%s) → эвристика: acc=%.2f, base=%.3f, diff=%.3f",
                len(xs),
                acc,
                base,
                adjusted,
            )
            return adjusted

        # Пытаемся обучить логистическую регрессию
        try:
            difficulty = self._fit_logistic_and_select(xs, ys, weights, level, base)
            logger.info(
                "Predicted difficulty (logreg) for user %s level %s: %.3f (samples=%s, acc=%.2f)",
                user_id,
                level,
                difficulty,
                len(xs),
                acc,
            )
            return difficulty
        except Exception as exc:
            logger.error("Ошибка в логистической регрессии: %s", exc)
            # фоллбек — просто корректируем по точности
            return self._heuristic_adjust(base, level, acc)

    # ---------------------- загрузка данных ---------------------------

    async def _load_history(
        self,
        user_id: int,
        level: str,
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Грузим историю: difficulty вопроса + правильность ответа.
        """
        stmt = (
            select(UserAnswer, Question.difficulty)
            .join(Question, Question.id == UserAnswer.question_id)
            .where(
                UserAnswer.user_id == user_id,
                Question.level == level,
            )
            .order_by(UserAnswer.created_at.desc())
            .limit(self.max_history)
        )
        result = await self.session.execute(stmt)
        rows = result.all()

        xs: list[float] = []
        ys: list[float] = []
        weights: list[float] = []

        for idx, row in enumerate(rows):
            ua: UserAnswer = row[0]
            diff = row[1]

            if diff is None:
                continue

            x = float(diff)
            y = 1.0 if ua.is_correct else 0.0

            # вес по "свежести": новые ответы чуть важнее
            w = 0.9**idx

            xs.append(x)
            ys.append(y)
            weights.append(w)

        return xs, ys, weights

    # ---------------------- вспомогательные методы --------------------

    def _level_bounds(self, level: str) -> tuple[float, float]:
        return LEVEL_BOUNDS.get(level, (0.3, 0.8))

    def _default_base(self, level: str) -> float:
        low, high = self._level_bounds(level)
        return (low + high) / 2

    def _weighted_accuracy(self, ys: Sequence[float], weights: Sequence[float]) -> float:
        if not ys:
            return 0.0
        w_sum = float(sum(weights)) or 1.0
        return float(sum(y * w for y, w in zip(ys, weights)) / w_sum)

    def _heuristic_adjust(self, base: float, level: str, acc: float) -> float:
        """
        Простой фоллбек, если данных мало или модель неадекватна:
        - если user >85% правильных → заметно повышаем сложность
        - если <40% → заметно снижаем
        - остальное — мягкие сдвиги
        """
        low, high = self._level_bounds(level)

        if acc > 0.85:
            target = base + 0.15
        elif acc > 0.70:
            target = base + 0.08
        elif acc < 0.40:
            target = base - 0.12
        elif acc < 0.55:
            target = base - 0.05
        else:
            target = base

        return max(low, min(high, target))

    def _fit_logistic_and_select(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        weights: Sequence[float],
        level: str,
        base: float,
    ) -> float:
        """
        Обучаем логистическую регрессию P(correct | difficulty)
        и ищем сложность, при которой P ~= target_p.
        """
        X = np.array(xs, dtype=float).reshape(-1, 1)
        y = np.array(ys, dtype=int)
        w = np.array(weights, dtype=float)

        # если все ответы одинаковы (все 0 или все 1),
        # LogisticRegression формально обучится, но целевая точность
        # будет либо 0, либо 1. На всякий случай заранее проверим.
        if np.all(y == 0) or np.all(y == 1):
            acc = float(np.mean(y))
            return self._heuristic_adjust(base, level, acc)

        # C побольше, чтобы кривая не была слишком "плоской"
        model = LogisticRegression(
            C=10.0,
            solver="lbfgs",
        )
        model.fit(X, y, sample_weight=w)

        coef = float(model.coef_[0, 0])
        intercept = float(model.intercept_[0])

        # Если коэффициент почти нулевой — кривая почти плоская,
        # сложность не влияет → фоллбек к эвристике.
        if abs(coef) < 1e-3:
            # оценим среднюю точность
            acc = float(np.average(y, weights=w))
            return self._heuristic_adjust(base, level, acc)

        # logit(p) = intercept + coef * d
        logit_target = float(np.log(self.target_p / (1 - self.target_p)))
        d_star = (logit_target - intercept) / coef

        low, high = self._level_bounds(level)
        raw = max(low, min(high, d_star))

        # лёгкое сглаживание, чтобы не прыгало
        smoothed = 0.5 * base + 0.5 * raw
        return float(smoothed)
