from typing import Iterable, Sequence

from sqlalchemy import select, func, and_, not_, exists
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Question, UserAnswer


class QuestionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_questions(self, questions: Iterable[Question]) -> None:
        self.session.add_all(list(questions))
        await self.session.commit()

    async def get_random_fresh_question(
        self,
        user_id: int,
        level: str,
        qtype: str,
    ) -> Question | None:
        ua_subq = (
            select(UserAnswer.question_id)
            .where(UserAnswer.user_id == user_id)
            .subquery()
        )

        stmt = (
            select(Question)
            .where(
                Question.type == qtype,
                Question.level == level,
                not_(Question.id.in_(ua_subq)),
            )
            .order_by(func.random())
            .limit(1)
        )

        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def get_random_questions_for_lesson(
        self,
        *args,
        **kwargs,
    ) -> list[Question]:
        user_id = kwargs.get("user_id", None)
        level = kwargs.get("level", None)
        qtypes: Sequence[str] | None = (
            kwargs.get("qtypes")
            or kwargs.get("types")
            or kwargs.get("question_types")
        )
        n: int = kwargs.get("n") or kwargs.get("limit") or 10
        target_diff: float | None = (
            kwargs.get("difficulty") or kwargs.get("target_difficulty")
        )

        if args:
            if level is None and len(args) >= 1:
                level = args[0]
            if qtypes is None and len(args) >= 2:
                qtypes = args[1]
            if n == 10 and len(args) >= 3:
                n = args[2]
            if target_diff is None and len(args) >= 4:
                target_diff = args[3]

        if qtypes is None:
            qtypes = ("grammar", "vocabulary")

        conditions = [Question.type.in_(list(qtypes))]

        if level is not None:
            conditions.append(Question.level == level)

        if target_diff is not None:
            delta = 0.15
            conditions.append(
                Question.difficulty.between(target_diff - delta, target_diff + delta)
            )

        if user_id is not None:
            ua_subq = (
                select(UserAnswer.question_id)
                .where(UserAnswer.user_id == user_id)
                .subquery()
            )
            conditions.append(not_(Question.id.in_(ua_subq)))

        stmt = (
            select(Question)
            .where(*conditions)
            .order_by(func.random())
            .limit(n)
        )

        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def get_random_question(
        self,
        user_id: int,
        qtype: str,
        level: str,
    ) -> Question | None:
        ua_subq = select(UserAnswer.question_id).where(UserAnswer.user_id == user_id).subquery()

        stmt = (
            select(Question)
            .where(
                Question.type == qtype,
                Question.level == level,
                not_(Question.id.in_(ua_subq)),
            )
            .order_by(func.random())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def count_fresh(
            self,
            user_id: int,
            qtypes: list[str],
            level: str,
    ) -> int:
        ua_subq = (
            select(UserAnswer.question_id)
            .where(UserAnswer.user_id == user_id)
            .subquery()
        )

        stmt = (
            select(func.count(Question.id))
            .where(
                Question.type.in_(qtypes),
                Question.level == level,
                ~Question.id.in_(select(ua_subq.c.question_id)),
            )
        )

        res = await self.session.execute(stmt)
        return res.scalar_one() or 0

    async def get_fresh_questions(
            self,
            user_id: int,
            qtypes: list[str],
            level: str,
            limit: int,
    ) -> list[Question]:
        ua_subq = (
            select(UserAnswer.question_id)
            .where(UserAnswer.user_id == user_id)
            .subquery()
        )

        stmt = (
            select(Question)
            .where(
                Question.type.in_(qtypes),
                Question.level == level,
                ~Question.id.in_(select(ua_subq.c.question_id)),
            )
            .order_by(Question.created_at.desc())
            .limit(limit)
        )

        res = await self.session.execute(stmt)
        return list(res.scalars().all())

