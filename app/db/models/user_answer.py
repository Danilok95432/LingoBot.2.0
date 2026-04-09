from datetime import datetime

from sqlalchemy import Integer, String, Boolean, Float, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class UserAnswer(Base):
    __tablename__ = "user_answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    user = relationship("User")

    question_id: Mapped[int] = mapped_column(Integer, ForeignKey("questions.id", ondelete="CASCADE"))
    question = relationship("Question")

    session_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("lesson_sessions.id", ondelete="CASCADE"), nullable=True)
    session = relationship("LessonSession")

    is_correct: Mapped[bool] = mapped_column(Boolean)
    user_answer: Mapped[str | None] = mapped_column(String(512), nullable=True)
    time_spent_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    emotion_score: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
