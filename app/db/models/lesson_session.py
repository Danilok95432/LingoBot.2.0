from datetime import datetime

from sqlalchemy import Integer, String, DateTime, ForeignKey, BigInteger
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class LessonSession(Base):
    __tablename__ = "lesson_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    user = relationship("User")

    session_type: Mapped[str] = mapped_column(String(32))  # lesson_test / audio_lesson / quick / level_test
    state: Mapped[str] = mapped_column(String(32), default="in_progress")

    question_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    current_index: Mapped[int] = mapped_column(Integer, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, default=0)
    total_questions: Mapped[int] = mapped_column(Integer)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    last_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    meta: Mapped[dict] = mapped_column(JSONB, default=dict)
