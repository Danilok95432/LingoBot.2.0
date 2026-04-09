from datetime import datetime

from sqlalchemy import Integer, String, Float, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(32))  # grammar / vocabulary / listening / speaking
    level: Mapped[str] = mapped_column(String(2))
    topic: Mapped[str] = mapped_column(String(64))

    # payload хранит json с текстом задания, вариантами ответов и т.д.
    payload: Mapped[dict] = mapped_column(JSONB)

    correct_option_index: Mapped[int] = mapped_column(Integer)
    audio_file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    pronunciation_phrase: Mapped[str | None] = mapped_column(String(512), nullable=True)

    difficulty: Mapped[float] = mapped_column(Float, default=0.5)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
