"""Patient model."""

from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import String, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from common.database import Base


class Patient(Base):
    """Patient model for storing anonymized patient data."""

    __tablename__ = "patients"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    anonymous_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    created_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    age: Mapped[int | None] = mapped_column(Integer)
    gender: Mapped[str | None] = mapped_column(String(20))
    medical_history: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
